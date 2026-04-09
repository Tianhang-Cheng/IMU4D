
import torch
import numpy as np
from utils.rotation import convert_rotation
from jaxtyping import Float
from torch import Tensor
from utils.egoallo.transforms import SO3, SE3
from utils.human import load_smplx_model

smplx_model = load_smplx_model()

def mse(original, decoded):
    return np.mean((original - decoded) ** 2)


def psnr(original, decoded, data_range=2.0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) in dB.
    
    Args:
        original: Original data
        decoded: Decoded data
        data_range: Maximum possible value range (default: 2.0 for [-1, 1] normalized data)
    
    Returns:
        PSNR value in dB
    """
    mse_val = mse(original, decoded)
    if mse_val == 0:
        return float('inf')
    
    psnr_val = 20 * np.log10(data_range / np.sqrt(mse_val))
    return psnr_val


def orientation_error(
    label_Ts: Float[Tensor, "batch 7"],
    pred_Ts: Float[Tensor, "batch 7"],
) -> np.ndarray:
    """Adapted from EgoAllo project"""
    matrix_errors = (
        SO3(pred_Ts[:, :4]).as_matrix()
        @ SO3(label_Ts[:, :4]).inverse().as_matrix()
    ) - torch.eye(3, device=label_Ts.device)
    assert matrix_errors.shape == (pred_Ts.shape[0], 3, 3)

    return torch.mean(
        torch.linalg.norm(matrix_errors.reshape((pred_Ts.shape[0], 9)), dim=-1),
        dim=-1,
    ).numpy(force=True)


def translation_error(
    label_Ts: Float[Tensor, "batch 7"],
    pred_Ts: Float[Tensor, "batch 7"],
) -> np.ndarray:
    """Adapted from EgoAllo project"""
    errors = pred_Ts[:, 4:7] - label_Ts[:, 4:7]
    assert errors.shape == (pred_Ts.shape[0], 3)

    return torch.mean(
        torch.linalg.norm(errors, dim=-1),
        dim=-1,
    ).numpy(force=True)


def mpjpe_error(
    label_T_world_root: Float[Tensor, "batch 7"],
    label_Ts_world_joint: Float[Tensor, "batch 21 7"],
    pred_T_world_root: Float[Tensor, "batch 7"],
    pred_Ts_world_joint: Float[Tensor, "batch 21 7"],
    per_frame_procrustes_align: bool = False,
) -> np.ndarray:
    """Adapted from EgoAllo project"""
    bs, _, _ = pred_Ts_world_joint.shape

    # Concatenate the world root to the joints.
    label_Ts_world_joint = torch.cat(
        [label_T_world_root[..., None, :], label_Ts_world_joint], dim=-2
    )
    pred_Ts_world_joint = torch.cat(
        [pred_T_world_root[..., None, :], pred_Ts_world_joint], dim=-2
    )
    del label_T_world_root, pred_T_world_root

    pred_joint_positions = pred_Ts_world_joint[:, :, 4:7]
    label_joint_positions = label_Ts_world_joint[:, :, 4:7]

    if per_frame_procrustes_align:
        pass  # TODO: add proscutes alignment here if needed

    position_diff = pred_joint_positions - label_joint_positions
    assert position_diff.shape == (bs, 22, 3)

    # Per-joint position errors, in millimeters.
    pjpe = torch.linalg.norm(position_diff, dim=-1) * 1000.0
    assert pjpe.shape == (bs, 22)

    # Mean per-joint position errors.
    mpjpe = torch.mean(pjpe.reshape((bs, -1)), dim=-1)
    assert mpjpe.shape == (bs,)

    return mpjpe.cpu().numpy()

def compute_mpjpe(sample_dict, return_joints=False, max_frame_length=None):
    """
    """
    pred_dict = sample_dict['pred']
    gt_dict = sample_dict['gt']

    orient = gt_dict['orient'].reshape(-1, 3) # [N, 3]
    transl = gt_dict['transl'].reshape(-1, 3) # [N, 3]
    pose = gt_dict['pose'].reshape(-1, 63) # [N, 63]

    output_orient = pred_dict['orient'].reshape(-1, 3)
    output_transl = pred_dict['transl'].reshape(-1, 3)
    output_pose = pred_dict['pose'].reshape(-1, 63)

    if max_frame_length is not None:
        pose = pose[:max_frame_length]
        output_pose = output_pose[:max_frame_length]
        orient = orient[:max_frame_length]
        output_orient = output_orient[:max_frame_length]
        transl = transl[:max_frame_length]
        output_transl = output_transl[:max_frame_length]

    # metrics (3d trajectory)
    label_Ts = torch.cat([
        convert_rotation(torch.from_numpy(orient).float(), 'aa', 'quat'), 
        torch.from_numpy(transl).float()
    ], dim=-1)
    pred_Ts = torch.cat([
        convert_rotation(torch.from_numpy(output_orient).float(), 'aa', 'quat'), 
        torch.from_numpy(output_transl).float()
    ], dim=-1)
    orient_error = orientation_error(label_Ts, pred_Ts).item()
    transl_error = translation_error(label_Ts, pred_Ts).item()

    # metrics (human poses)
    original_pose_aa = pose.reshape(-1, 21, 3)
    decoded_pose_aa = output_pose.reshape(-1, 21, 3)

    original_pose_quat_wxyz = convert_rotation(
        torch.from_numpy(original_pose_aa).float(), 'aa', 'quat'  # (N, 21, 4)
    ).numpy()
    decoded_pose_quat_wxyz = convert_rotation(
        torch.from_numpy(decoded_pose_aa).float(), 'aa', 'quat'   # (N, 21, 4)
    ).numpy()

    dummy_transl = np.zeros((pose.shape[0], 21, 3))
    original_Ts = torch.from_numpy(
        np.concatenate((original_pose_quat_wxyz, dummy_transl), axis=-1)  # (N*21, 7)
    ).float().reshape(-1, 7)
    decoded_Ts = torch.from_numpy(
        np.concatenate((decoded_pose_quat_wxyz, dummy_transl), axis=-1)  # (N*21, 7)
    ).float().reshape(-1, 7)
    ori_error = orientation_error(original_Ts, decoded_Ts).item()

    num_poses = pose.shape[0]
    original_T_world_root = torch.zeros((num_poses, 7), dtype=torch.float32)
    original_Ts_world_joint = torch.zeros((num_poses, 21, 7), dtype=torch.float32)
    decoded_T_world_root = torch.zeros((num_poses, 7), dtype=torch.float32)
    decoded_Ts_world_joint = torch.zeros((num_poses, 21, 7), dtype=torch.float32)

    dummy_rot_quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32)[None].expand(num_poses, -1)  # (num_poses, 4)

    dummy_root_orient = torch.zeros((num_poses, 3), dtype=torch.float32).cuda()
    dummy_trans = torch.zeros((num_poses, 3), dtype=torch.float32).cuda()
    dummy_betas = torch.zeros((num_poses, 10), dtype=torch.float32).cuda()

    with torch.no_grad():
        # original
        smplex_original_output = smplx_model(
            pose_body=torch.from_numpy(original_pose_aa).float().cuda().reshape(-1, 63), 
            root_orient=dummy_root_orient,
            trans=dummy_trans,
            betas=dummy_betas
        )

        original_joints_global = smplex_original_output.Jtr.cpu()
        original_joints_global = original_joints_global[:, :22, :3]   # (chunk_size, 22, 3)

        original_vertices = smplex_original_output.v.cpu()

        original_T_world_root = torch.cat([
            dummy_rot_quat, 
            original_joints_global[:, 0, :]
        ], dim=-1)  # (chunk_size, 7)
        original_Ts_world_joint = torch.cat([
            dummy_rot_quat.unsqueeze(1).expand(-1, 21, -1), 
            original_joints_global[:, 1:22, :]
        ], dim=-1)  # (chunk_size, 21, 7)

        # decoded
        smplex_decoded_output = smplx_model(
            pose_body=torch.from_numpy(decoded_pose_aa).float().cuda().reshape(-1, 63), 
            root_orient=dummy_root_orient,
            trans=dummy_trans,
            betas=dummy_betas
        )

        decoded_joints_global = smplex_decoded_output.Jtr.cpu()
        decoded_joints_global = decoded_joints_global[:, :22, :3]

        decoded_vertices = smplex_decoded_output.v.cpu()

        decoded_T_world_root = torch.cat([
            dummy_rot_quat, 
            decoded_joints_global[:, 0, :]
        ], dim=-1)
        decoded_Ts_world_joint = torch.cat([
            dummy_rot_quat.unsqueeze(1).expand(-1, 21, -1), 
            decoded_joints_global[:, 1:22, :]
        ], dim=-1)

    mpjpe = mpjpe_error(
        original_T_world_root, original_Ts_world_joint,
        decoded_T_world_root, decoded_Ts_world_joint,
    ).mean().item()  # already in mm

    if return_joints:
        return mpjpe, original_joints_global, decoded_joints_global, original_vertices, decoded_vertices

    return mpjpe, ori_error, orient_error, transl_error