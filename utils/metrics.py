
import torch
import numpy as np
from utils.rotation import convert_rotation
from jaxtyping import Float
from torch import Tensor
from utils.egoallo.transforms import SO3, SE3
from utils.human import load_smplx_model

_smplx_model = None
_smplx_device = None


def get_smplx_model():
    """Return the process-local SMPL-X model, building it on first use.

    This used to run at import time, which is before accelerate assigns each rank
    its device, so every rank's body model ended up on cuda:0. Deferring the build
    to the first call puts it on the rank's own device.
    """
    global _smplx_model, _smplx_device
    if _smplx_model is None:
        _smplx_device = (torch.device(f"cuda:{torch.cuda.current_device()}")
                         if torch.cuda.is_available() else torch.device("cpu"))
        _smplx_model = load_smplx_model(_smplx_device)
    return _smplx_model


def get_smplx_device():
    """Device the body model sits on.

    BodyModel registers its tensors as buffers and exposes no parameters, so
    ``next(model.parameters())`` raises StopIteration -- the device is recorded
    when the model is built instead.
    """
    get_smplx_model()
    return _smplx_device

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


def decode_smplx_motion_geometry(
    sample_dict,
    max_frame_length=None,
    *,
    include_vertices=True,
    to_cpu=True,
):
    """Decode saved GT/pred body poses into root-local SMPL-X geometry.

    This is the repository-specific adapter used by both online diagnostics and
    the canonical, model-independent metrics in :mod:`metric.motion`.  It does
    no metric calculation; callers decide which numerical protocol to apply to
    the returned joints and vertices.

    Returns:
        ``(gt_joints, pred_joints, gt_vertices, pred_vertices)``. Vertices are
        ``None`` when ``include_vertices`` is false. Tensors remain on the body
        model device when ``to_cpu`` is false.
    """

    pred_dict = sample_dict['pred']
    gt_dict = sample_dict['gt']
    smplx_model = get_smplx_model()
    device = get_smplx_device()

    def _pose(value):
        if torch.is_tensor(value):
            value = value.detach()
        else:
            value = torch.from_numpy(np.asarray(value))
        value = value.reshape(-1, 63)
        if max_frame_length is not None:
            value = value[:max_frame_length]
        return value.to(device=device, dtype=torch.float32)

    gt_pose = _pose(gt_dict['pose'])
    pred_pose = _pose(pred_dict['pose'])
    if gt_pose.shape != pred_pose.shape:
        raise ValueError(
            f"GT/pred pose shape mismatch: {tuple(gt_pose.shape)} vs "
            f"{tuple(pred_pose.shape)}"
        )
    num_poses = gt_pose.shape[0]
    dummy_root_orient = torch.zeros((2 * num_poses, 3), dtype=torch.float32, device=device)
    dummy_trans = torch.zeros((2 * num_poses, 3), dtype=torch.float32, device=device)
    dummy_betas = torch.zeros((2 * num_poses, 10), dtype=torch.float32, device=device)
    with torch.no_grad():
        output = smplx_model(
            pose_body=torch.cat([gt_pose, pred_pose], dim=0),
            root_orient=dummy_root_orient,
            trans=dummy_trans,
            betas=dummy_betas,
        )
    joints = output.Jtr[:, :22, :3]
    values = (
        joints[:num_poses],
        joints[num_poses:],
        output.v[:num_poses] if include_vertices else None,
        output.v[num_poses:] if include_vertices else None,
    )
    return tuple(value.cpu() if to_cpu and value is not None else value for value in values)


def compute_mpjpe(sample_dict, return_joints=False, max_frame_length=None):
    """MPJPE (mm) plus root / joint orientation and root translation errors.

    Everything runs on the body model's device, and the ground truth and the
    prediction go through one stacked SMPL-X pass. The previous version
    converted rotations on the CPU (tiny tensors, so torch's per-op overhead
    dominated), ran the body model twice and copied both vertex sets back to
    the host although only the joints are used; after the caption-decoding
    fix it was a quarter of the per-sample evaluation time. Values match the
    old path to float32 rounding.
    """
    pred_dict = sample_dict['pred']
    gt_dict = sample_dict['gt']

    device = get_smplx_device()

    def _prep(value, width):
        if torch.is_tensor(value):
            value = value.detach()
        else:
            value = torch.from_numpy(np.asarray(value))
        value = value.reshape(-1, width)
        if max_frame_length is not None:
            value = value[:max_frame_length]
        return value.to(device=device, dtype=torch.float32)

    orient, output_orient = _prep(gt_dict['orient'], 3), _prep(pred_dict['orient'], 3)
    transl, output_transl = _prep(gt_dict['transl'], 3), _prep(pred_dict['transl'], 3)
    pose, output_pose = _prep(gt_dict['pose'], 63), _prep(pred_dict['pose'], 63)
    num_poses = pose.shape[0]

    # metrics (3d trajectory)
    label_Ts = torch.cat([convert_rotation(orient, 'aa', 'quat'), transl], dim=-1)
    pred_Ts = torch.cat([convert_rotation(output_orient, 'aa', 'quat'), output_transl], dim=-1)
    orient_error = orientation_error(label_Ts, pred_Ts).item()
    transl_error = translation_error(label_Ts, pred_Ts).item()

    # metrics (human poses): per-joint rotation error, translation zeroed
    original_pose_quat = convert_rotation(pose.reshape(-1, 21, 3), 'aa', 'quat')  # (N, 21, 4)
    decoded_pose_quat = convert_rotation(output_pose.reshape(-1, 21, 3), 'aa', 'quat')
    dummy_transl = torch.zeros((num_poses, 21, 3), dtype=torch.float32, device=device)
    original_Ts = torch.cat([original_pose_quat, dummy_transl], dim=-1).reshape(-1, 7)  # (N*21, 7)
    decoded_Ts = torch.cat([decoded_pose_quat, dummy_transl], dim=-1).reshape(-1, 7)
    ori_error = orientation_error(original_Ts, decoded_Ts).item()

    # Joints from the body pose alone: root at the origin, identity heading.
    (
        original_joints_global,
        decoded_joints_global,
        original_vertices,
        decoded_vertices,
    ) = decode_smplx_motion_geometry(
        sample_dict,
        max_frame_length,
        include_vertices=return_joints,
        to_cpu=False,
    )

    dummy_rot_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device
    )[None].expand(num_poses, -1)  # (N, 4)

    def _world_transforms(joints):
        T_world_root = torch.cat([dummy_rot_quat, joints[:, 0, :]], dim=-1)  # (N, 7)
        Ts_world_joint = torch.cat(
            [dummy_rot_quat.unsqueeze(1).expand(-1, 21, -1), joints[:, 1:22, :]], dim=-1
        )  # (N, 21, 7)
        return T_world_root, Ts_world_joint

    original_T_world_root, original_Ts_world_joint = _world_transforms(original_joints_global)
    decoded_T_world_root, decoded_Ts_world_joint = _world_transforms(decoded_joints_global)

    mpjpe = mpjpe_error(
        original_T_world_root, original_Ts_world_joint,
        decoded_T_world_root, decoded_Ts_world_joint,
    ).mean().item()  # already in mm

    if return_joints:
        # The seam / visualization tools consume host tensors, vertices included.
        return (
            mpjpe,
            original_joints_global.cpu(),
            decoded_joints_global.cpu(),
            original_vertices.cpu(),
            decoded_vertices.cpu(),
        )

    return mpjpe, ori_error, orient_error, transl_error
