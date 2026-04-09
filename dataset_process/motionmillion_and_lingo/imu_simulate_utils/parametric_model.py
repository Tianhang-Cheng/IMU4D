"""
Simplified SMPL-X ParametricModel implementation using only body joints (first 22 joints).
Removes hand, face, expression, and PCA-related functionality.

References:
- https://github.com/vchoutas/smplx/blob/main/smplx/body_models.py
- https://github.com/vchoutas/smplx/blob/main/smplx/lbs.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import pickle
import os
from typing import Optional, Dict, Tuple, List


class SimplifiedSMPLX(nn.Module):
    """
    Simplified SMPL-X model using only body joints (first 22 joints). Hand and face joints are set to zeros.
    
    Args:
        model_path: Path to the official model to be loaded.
        gender: 'male', 'female', or 'neutral'
        num_betas: Number of shape parameters (default 10)
    """
    
    def __init__(self,
                 model_path: str = '',
                 gender: str = 'neutral',
                 num_betas: int = 10,
                 vert_mask: List[int] = None,
                 device: str = 'cuda'):
        super().__init__()
        
        self.name = 'SimplifiedSMPLX'
        self.gender = gender
        self.vert_mask = vert_mask
        self.device = device
        
        # Load SMPL-X model data
        model_path = os.path.join(model_path, f'SMPLX_{gender.upper()}.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f, encoding='latin1')
        
        # Store model data as buffers
        self.register_buffer('v_template', torch.tensor(model_data['v_template'], dtype=torch.float32, device=device))
        self.register_buffer('weights', torch.tensor(model_data['weights'], dtype=torch.float32, device=device))
        self.register_buffer('faces', torch.tensor(model_data['f'].astype(np.int32), dtype=torch.long, device=device))

        # Pose parameters
        num_pose_basis = model_data['posedirs'].shape[-1]
        posedirs = np.reshape(model_data['posedirs'], [-1, num_pose_basis]).T   # (V, 3, P) -> (P, V*3)
        self.register_buffer('posedirs', torch.tensor(posedirs, dtype=torch.float32, device=device))

        # Shape parameters
        shapedirs = model_data['shapedirs']
        self.num_betas = min(num_betas, shapedirs.shape[-1])
        shapedirs = shapedirs[:, :, :self.num_betas]
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=torch.float32, device=device))

        # Handle J_regressor - it might be sparse or dense
        if hasattr(model_data['J_regressor'], 'todense'):
            J_regressor = model_data['J_regressor'].todense()
        else:
            J_regressor = model_data['J_regressor']
        self.register_buffer('J_regressor', torch.tensor(J_regressor, dtype=torch.float32, device=device))

        # Kinematic tree
        self.parents = model_data['kintree_table'][0].astype(np.int32)
        self.parents[0] = -1
        self.register_buffer('parents_tensor', torch.tensor(self.parents, dtype=torch.long, device=device))

        # Joint counts
        self._num_joints = 55       # Total SMPL-X joints (required for proper skinning)
        self._num_body_joints = 22  # Only first 22 are actively controlled
        self._num_vertices = self.v_template.shape[0]
        
        # Joint names for the 22 body joints
        self.joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
        ]
    
    @property
    def num_joints(self):
        return self._num_joints
    
    @property
    def num_body_joints(self):
        return self._num_body_joints
    
    @property
    def num_vertices(self):
        return self._num_vertices
    
    def forward(self,
                pose: torch.Tensor,
                betas: Optional[torch.Tensor] = None,
                transl: Optional[torch.Tensor] = None,
                return_vertices: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute joint positions, rotations, and vertices.
        
        Args:
            pose: Body pose parameters [B, 66] or [B, 22, 3]
                  Format: [global_orient(3), body_pose(63)]
            betas: Shape (beta) parameters [B, num_betas]
            transl: Global translation [B, 3]
            return_vertices: Whether to compute and return vertices
            
        Returns:
            Dictionary containing:
                - joints_pos: All joint positions [B, 55, 3]
                - joints_rot: Global rotation matrices for all joints [B, 55, 3, 3]
                - body_joints_pos: Body joint positions only [B, 22, 3]
                - body_joints_rot: Global rotation matrices for body joints [B, 22, 3, 3]
                - vertices: Vertex positions [B, V, 3] (if return_vertices=True)
        """
        batch_size = pose.shape[0]
        device = pose.device
        dtype = pose.dtype
        
        # Parse pose parameters in [B, 22, 3] format
        if pose.dim() == 2 and pose.shape[1] == 66:
            pose = pose.view(batch_size, 22, 3)  # Reshape to [B, 22, 3]
        elif pose.dim() == 3 and pose.shape[1] == 22:
            pass
        else:
            raise ValueError(f"Expected pose shape [B, 66] or [B, 22, 3], got {pose.shape}")
        
        # Create zero poses for hands and face (33 joints: 3 face + 30 hand)
        zero_hand_face = torch.zeros([batch_size, 33, 3], dtype=dtype, device=device)
        
        # Concatenate all pose parameters: body (22) + face/hands (33) = 55 joints
        full_pose = torch.cat([
            pose,               # 22 joints (global_orient + body_pose)
            zero_hand_face      # 33 joints (jaw, leye, reye, 15 left hand, 15 right hand)
        ], dim=1)  # [B, 55, 3]
        
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas], dtype=dtype, device=device)

        # Add shape contribution
        blend_shape = torch.einsum('bl,mkl->bmk', [betas, self.shapedirs])
        v_shaped = self.v_template + blend_shape
        
        # Get the joints (N x J x 3)
        J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor])
        
        # Add pose blend shapes (N x J x 3 x 3)
        ident = torch.eye(3, dtype=dtype, device=device)
        rot_mats = batch_axis_angle_to_matrix(full_pose.view(-1, 3)).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, self.posedirs).view(batch_size, -1, 3)
        
        v_posed = pose_offsets + v_shaped
        
        # Get the global joint locations
        J_transformed, A = batch_rigid_transform(rot_mats, J, self.parents_tensor)
        
        if transl is not None:
            J_transformed = J_transformed + transl.unsqueeze(1)
        
        output = {
            'joints_pos': J_transformed,
            'joints_rot': A[..., :3, :3],                                   # All rotations [B, 55, 3, 3]
            'body_joints_pos': J_transformed[:, :self.num_body_joints, :],  # Body joint positions [B, 22, 3]
            'body_joints_rot': A[:, :self.num_body_joints, :3, :3]          # Body joint rotations [B, 22, 3, 3]
        }
        
        # Perform skinning
        if return_vertices:
            if self.vert_mask is not None:
                v_posed_skinning = v_posed[:, self.vert_mask]
                weights = self.weights[self.vert_mask]
            else:
                v_posed_skinning = v_posed
                weights = self.weights

            W = weights.unsqueeze(0).expand(batch_size, -1, -1)

            # (N x V x J) x (N x J x 16) -> N x V x 4 x 4
            T = torch.matmul(W, A.view(batch_size, self.num_joints, 16)).view(batch_size, -1, 4, 4)

            homogen_coord = torch.ones([batch_size, v_posed_skinning.shape[1], 1], dtype=dtype, device=device)
            v_posed_homo = torch.cat([v_posed_skinning, homogen_coord], dim=2)
            v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
            vertices = v_homo[:, :, :3, 0]
            
            if transl is not None:
                vertices = vertices + transl.unsqueeze(1)
            
            output['vertices'] = vertices
        
        return output


def batch_axis_angle_to_matrix(
    rot_vecs: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_Rt_to_T(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    dtype=torch.float32
) -> Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_Rt_to_T(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


# Example usage
if __name__ == "__main__":

    model = SimplifiedSMPLX(
        model_path="/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smplx",
        gender="neutral",
        num_betas=10,
        vert_mask=list(range(1000))
    )
    
    # Example input
    batch_size = 2
    pose = torch.zeros(batch_size, 66)   # global_orient(3) + body_pose(63)
    transl = torch.zeros(batch_size, 3)  # Translation
    
    output = model(
        pose=pose,
        shape=None,
        transl=transl,
        return_vertices=True
    )
    print(f"Body joints shape: {output['body_joints_pos'].shape}")  # [2, 22, 3]
    print(f"Body rotations shape: {output['body_joints_rot'].shape}")  # [2, 22, 3, 3]
    print(f"Vertices shape: {output['vertices'].shape}")  # [2, 10475, 3]