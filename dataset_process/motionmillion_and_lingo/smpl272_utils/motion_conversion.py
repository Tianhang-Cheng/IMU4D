"""
This script includes conversion functions between human motion representations.
"""
import copy
import numpy as np
import torch

from dataset_process.motionmillion_and_lingo.smpl272_utils.face_z_align_util import *
import dataset_process.motionmillion_and_lingo.smpl272_utils.custom_272_utils as c_utils

def smpl85_to_smpl272(smpl_85, smplx_model, reset_position=True):
    """
    Convert SMPL-85 pose to SMPL-272 pose.

    References: https://github.com/Li-xingXiao/272-dim-Motion-Representation 
    """
    # (1) The motion height is offset by estimating the floor as zero rather than centering it on the first frame.
    # (2) Skip heading correction on +z axis as assuming it is already done for smpl_85 input.

    # Get default pelvis position
    default_smplx_output = smplx_model()
    with torch.no_grad():
        rest_pelvis = default_smplx_output.Jtr[0, 0]
    
    ##### Get global joint positions from SMPL-85 #####
    smpl_85_cuda = torch.from_numpy(smpl_85).cuda()
    with torch.no_grad():
        joints_global = smplx_model(
            pose_body=smpl_85_cuda[:,3:66], 
            root_orient=smpl_85_cuda[:,:3], 
            trans=smpl_85_cuda[:, 72:75] - rest_pelvis[None, :],
            betas=smpl_85_cuda[:, 75:]
        ).Jtr.cpu().numpy()
    joints_global = joints_global[:, :22, :3]   # keep only the first 22 joints

    ##### Convert to 272-dim representation #####
    n_frames, n_joints, _ = joints_global.shape
    root_idx = 0

    # get smpl rotations
    smpl_rot_3x3 = quaternion_to_matrix_np(
        expmap_to_quaternion(
            smpl_85[:, :66].reshape(n_frames, n_joints, 3)
        )
    )

    # set root at origin and place human on floor
    if reset_position:
        origin = copy.deepcopy(joints_global[0, root_idx])  # first frame root position
        origin[1] = np.min(joints_global[:, :, 1])
        joints_global = joints_global - origin

    # get root velocities
    root_vel = joints_global[1:, root_idx, :] - joints_global[:-1, root_idx, :]

    # get foot contacts (not used for now)
    # contacts = c_utils.foot_detect(joints_global, 0.15 / 100)  # 0.15 cm threshold

    # get root heading difference
    root_heading = -np.arctan2(smpl_rot_3x3[:, root_idx, 0, 2], smpl_rot_3x3[:, root_idx, 2, 2])
    root_heading_diff = root_heading[1:] - root_heading[:-1]
    root_heading_diff_rot = np.array([c_utils.rot_yaw(x) for x in root_heading_diff])
    root_heading_diff_6d = matrix_to_rotation_6d(
        torch.from_numpy(root_heading_diff_rot)
    ).numpy()

    # transform root xz velocities, positions, velocities, and rotations to local coordinates
    root_heading_rot = np.array([c_utils.rot_yaw(x) for x in root_heading])
    
    local_root_vel_xz = np.matmul(
        root_heading_rot[:-1],
        root_vel[..., None]
    ).squeeze()[..., [0, 2]]
    
    local_pos = np.matmul(
        np.repeat(root_heading_rot[:, None, :, :], n_joints, axis=1),
        joints_global[..., None]
    ).squeeze(-1)
    
    local_vel = local_pos[1:] - local_pos[:-1]

    local_rot = copy.deepcopy(smpl_rot_3x3)
    local_rot[:, 0, ...] = np.matmul(root_heading_rot, local_rot[:, 0, ...])  # apply heading rotation to root joint

    # aggregate all into final representation
    size_frame = 8 + n_joints * 3 + n_joints * 3 + n_joints * 6
    smpl_272 = np.zeros((n_frames, size_frame))

    smpl_272[0, 2] = 1  # set first frame root rotation to identity
    smpl_272[0, 6] = 1
    smpl_272[1:, :2] = local_root_vel_xz
    smpl_272[1:, 2:8] = root_heading_diff_6d
    smpl_272[:, 8:8+3*n_joints] = np.reshape(local_pos, (n_frames, -1))
    smpl_272[1:, 8+3*n_joints:8+6*n_joints] = np.reshape(local_vel, (n_frames-1, -1))
    smpl_272[:, 8+6*n_joints:8+12*n_joints] = np.reshape(local_rot[..., :, :2, :], (n_frames, -1)) # take 6D rotation

    return smpl_272