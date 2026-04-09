import torch
import numpy as np

from utils.metrics import compute_mpjpe as compute_mpjpe_showo
from utils.rotation import convert_rotation
from utils.eval_utils import eval_recon
from dataset_process.motionmillion_and_lingo.convert_lingo_dataset import align_poses_to_frame, transform_aligned_to_reference

def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat, (scale, R, t)


def compute_mpjpe(preds,
                  target,
                  valid_mask=None,
                  pck_joints=None,
                  sample_wise=True):
    """
    Mean per-joint position error (i.e. mean Euclidean distance)
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape, print(preds.shape,
                                              target.shape)  # BxJx3
    mpjpe = torch.norm(preds - target, p=2, dim=-1)  # BxJ

    if pck_joints is None:
        if sample_wise:
            mpjpe_seq = ((mpjpe * valid_mask.float()).sum(-1) /
                         valid_mask.float().sum(-1)
                         if valid_mask is not None else mpjpe.mean(-1))
        else:
            mpjpe_seq = mpjpe[valid_mask] if valid_mask is not None else mpjpe
        return mpjpe_seq
    else:
        mpjpe_pck_seq = mpjpe[:, pck_joints]
        return mpjpe_pck_seq


def calc_pampjpe(preds, target, sample_wise=True, return_transform_mat=False):
    # Expects BxJx3
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    target, preds = target.float(), preds.float()
    # extracting the keypoints that all samples have valid annotations
    # valid_mask = (target[:, :, 0] != -2.).sum(0) == len(target)
    # preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(preds[:, valid_mask], target[:, valid_mask])
    # pa_mpjpe_each = compute_mpjpe(preds_tranformed, target[:, valid_mask], sample_wise=sample_wise)

    preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(
        preds, target)
    pa_mpjpe_each = compute_mpjpe(preds_tranformed,
                                  target,
                                  sample_wise=sample_wise)

    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each


def mpjve_error(gt_verts: torch.Tensor,
                pred_verts: torch.Tensor,
                fps: float | None = 30) -> torch.Tensor:
    """
    Mean Per Joint Vertex Error (MPJVE): Measure of mean Euclidean distance error across all root aligned vertices of the
    SMPL body mesh vertices.
    """
    assert gt_verts.shape == pred_verts.shape

    vel_err = torch.linalg.norm(pred_verts - gt_verts, dim=-1)

    # mean over time and joints -> scalar
    return vel_err.mean()

def compute_mjpre(pred_joints: torch.Tensor,
               target_joints: torch.Tensor,
               reduction: str = "mean") -> torch.Tensor:
    """
    Mean Joint Position Reconstruction Error (MJPRE)
    = MPJPE after pelvis alignment.

    Args:
        pred_joints:  [..., J, 3]  预测关节坐标
        target_joints: [..., J, 3] GT 关节坐标
        reduction: "mean" | "sum" | "none"

    Returns:
        标量 (mean/sum) 或逐关节误差张量 [..., J]
    """
    if isinstance(pred_joints, np.ndarray):
        pred_joints = torch.from_numpy(pred_joints)
    if isinstance(target_joints, np.ndarray):
        target_joints = torch.from_numpy(target_joints)
    pelvis_index = 0 # SMPL-X pelvis index

    if pred_joints.shape != target_joints.shape:
        raise ValueError(f"Shape mismatch: pred {pred_joints.shape}, target {target_joints.shape}")
    if pred_joints.shape[-1] != 3:
        raise ValueError(f"Expected last dim = 3 (xyz), got {pred_joints.shape[-1]}")

    # 对 pelvis 做平移对齐
    pred_root = pred_joints[..., pelvis_index:pelvis_index+1, :]   # [..., 1, 3]
    gt_root   = target_joints[..., pelvis_index:pelvis_index+1, :] # [..., 1, 3]

    pred_aligned = pred_joints - pred_root
    gt_aligned   = target_joints - gt_root

    # L2 距离：[..., J]
    error = torch.linalg.norm(pred_aligned - gt_aligned, dim=-1)

    if reduction == "mean":
        return error.mean()
    elif reduction == "sum":
        return error.sum()
    elif reduction in ("none", None):
        return error
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

import argparse
import glob
import os
import numpy as np
import tqdm
import math

"""
srun --account=benk-delta-gpu --partition=gpuA100x4-interactive --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem 60g --gpus=1 --pty /bin/bash
conda activate show
cd /scratch/benk/tcheng1/code/imu-human-mllm/third_party/Showo

python /scratch/benk/tcheng1/code/imu-human-mllm/third_party/Showo/calculate_metric/get_metric.py
"""
def rotation_angle_error_deg(
    R1: torch.Tensor,  # [B, 3, 3]
    R2: torch.Tensor,  # [B, 3, 3]
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute rotation angle difference between batches of rotation matrices.
    
    Inputs:
        R1, R2: [B, 3, 3] rotation matrices
    Output:
        angles_deg: [B] tensor of angular differences in degrees
    """
    # Relative rotation R_rel = R2 * R1^T
    if not isinstance(R1, torch.Tensor):
        R1 = torch.from_numpy(R1)
    if not isinstance(R2, torch.Tensor):
        R2 = torch.from_numpy(R2)
    R_rel = R2 @ R1.transpose(-1, -2)   # [B, 3, 3]

    # Trace per matrix -> [B]
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)

    # cos(theta) = (trace - 1) / 2
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)

    # Angle in radians
    theta = torch.acos(cos_theta)   # [B]

    # Convert to degrees
    theta_deg = theta * (180.0 / math.pi)
    return theta_deg

import numpy as np

class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        # 计算变化率 (Derivative)
        dx = (x - self.x_prev) * self.freq
        edx = self._low_pass_filter(dx, self.dx_prev, self._alpha(self.d_cutoff))
        self.dx_prev = edx

        # 计算自适应截止频率
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        
        # 计算平滑后的值
        out = self._low_pass_filter(x, self.x_prev, self._alpha(cutoff))
        self.x_prev = out
        return out

    def _low_pass_filter(self, x, x_prev, alpha):
        return alpha * x + (1.0 - alpha) * x_prev

def smooth_motion_data(data, freq=30, min_cutoff=0.3, beta=0.007):
    """
    针对 [t, N] 形状的数据进行平滑
    """
    t, n = data.shape
    one_euro = OneEuroFilter(freq, min_cutoff, beta)
    smoothed_data = np.zeros_like(data)
    
    for i in range(t):
        smoothed_data[i] = one_euro.filter(data[i])
        
    return smoothed_data

def get_number(
    model,
    dataset,
    result_folder,
    log_dir,
    apply_shited_window_avg=False,
    eval_frame_length=60,
):

    if not os.path.exists(result_folder):
        return
    
    _base = os.path.basename(result_folder.rstrip("/"))
    assert _base.startswith("viz_test_generate_number"), (
        f"Result folder basename must start with 'viz_test_generate_number', got {_base}"
    )

    mpjpes = []
    mpjpes_old = []
    pa_mpjpes = []
    mjpres = []
    mpjves = []
    mtes = []
    result_list = glob.glob(os.path.join(result_folder, '*.npy'))
    result_list.sort(key=lambda x: int(x.split('/')[-1].split('_')[1]))
    assert len(result_list) > 0, f"No result found in {result_folder}"

    for result_path in tqdm.tqdm(result_list):

        print(f"Processing {result_path}")

        sample = np.load(result_path, allow_pickle=True).item()

        # MPJPE
        mpjpe, original_joints_global, decoded_joints_global, original_vertices, decoded_vertices = \
            compute_mpjpe_showo(sample, return_joints=True, max_frame_length=eval_frame_length)
        print(f"MPJPE: {mpjpe:.2f} mm")
        mpjpes_old.append(mpjpe)

        if apply_shited_window_avg and (model == 'Ours' or model == 'GPT2'):

            if os.path.exists(result_path.replace('viz_test_generate_number_shifted_0', 'viz_test_generate_number_shifted_2')):
                sample_shifted = np.load(result_path.replace('viz_test_generate_number_shifted_0', 'viz_test_generate_number_shifted_2'), allow_pickle=True).item()
                cut_len = min(sample['pred']['pose'].shape[0]-2, sample_shifted['pred']['pose'].shape[0])
                # Align shifted seq's frame 0 to sample's frame 2, then avg orient/transl/pose
                orient_shifted = sample_shifted['pred']['orient'][:cut_len]
                transl_shifted = sample_shifted['pred']['transl'][:cut_len]
                pose_shifted = sample_shifted['pred']['pose'][:cut_len]
                aligned_orient, aligned_transl, aligned_pose, _, _ = align_poses_to_frame(
                    orient_shifted, transl_shifted, pose_shifted, ref_frame=0
                )
                ref_orient = sample['pred']['orient'][2]
                ref_transl = sample['pred']['transl'][2]
                orient_shifted_aligned, transl_shifted_aligned, pose_shifted_aligned = transform_aligned_to_reference(
                    aligned_orient, aligned_transl, aligned_pose, ref_orient, ref_transl
                )
                # Average in axis-angle space first
                orient_avg = (sample['pred']['orient'][2:2+cut_len] + orient_shifted_aligned) / 2.0
                transl_avg = (sample['pred']['transl'][2:2+cut_len] + transl_shifted_aligned) / 2.0
                pose_avg = (sample['pred']['pose'][2:2+cut_len] + pose_shifted_aligned) / 2.0
                sample['pred']['orient'][2:2+cut_len] = orient_avg
                sample['pred']['transl'][2:2+cut_len] = transl_avg
                sample['pred']['pose'][2:2+cut_len] = pose_avg

        # MPJPE
        mpjpe_new, _, _, _, _ = \
            compute_mpjpe_showo(sample, return_joints=True, max_frame_length=eval_frame_length)
        print(f"New MPJPE: {mpjpe_new:.2f} mm")
        # compute_mpjpe_ref = compute_mpjpe(original_joints_global, decoded_joints_global)
        # print(f"Compute MPJPE: {compute_mpjpe_ref:.2f} mm")

        preds = original_joints_global[:eval_frame_length] # [T, 22, 3] global joints
        target = decoded_joints_global[:eval_frame_length] # [T, 22, 3]

        #PA MPJPE
        # import pdb; pdb.set_trace()
        pa_mpjpe_each = calc_pampjpe(preds, target)
        pa_mpjpe_each_mean = pa_mpjpe_each.mean() * 1000.0
        print(f"PA MPJPE: {pa_mpjpe_each_mean:.2f} mm")

        # MJPRE
        pred_pose = sample['pred']['pose'][:eval_frame_length]
        gt_pose = sample['gt']['pose'][:eval_frame_length]
        pred_pose = torch.from_numpy(pred_pose).reshape(-1, 3)
        gt_pose = torch.from_numpy(gt_pose).reshape(-1, 3)
        pred_pose = convert_rotation(pred_pose, 'aa', 'mat').reshape(-1, 3, 3)
        gt_pose = convert_rotation(gt_pose, 'aa', 'mat').reshape(-1, 3, 3)

        mjpre_each = rotation_angle_error_deg(pred_pose, gt_pose)
        mjpre_each_mean = mjpre_each.mean()
        print(f"MJPRE: {mjpre_each_mean:.2f} degrees")

        # MPJVE
        mpjve_each = mpjve_error(original_vertices, decoded_vertices) * 1000
        mpjve_each_mean = mpjve_each.sum()
        print(f"MPJVE: {mpjve_each_mean:.2f} mm")

        # MTE
        pred_motion = np.concatenate([sample['pred']['orient'], sample['pred']['transl']], axis=-1)[:eval_frame_length]
        gt_motion = np.concatenate([sample['gt']['orient'], sample['gt']['transl']], axis=-1)[:eval_frame_length]
        mte_each = eval_recon(gt_motion[None], pred_motion[None], use_6d_rotation=False)['ATE'] * 1000
        mte_each_mean = mte_each
        print(f"MTE: {mte_each_mean:.2f} mm")

        mpjpes.append(mpjpe_new)
        pa_mpjpes.append(pa_mpjpe_each_mean)
        mjpres.append(mjpre_each_mean)
        mpjves.append(mpjve_each_mean)
        mtes.append(mte_each_mean)
    mean_mpjpe = np.mean(mpjpes)
    mean_pa_mpjpe = np.mean(pa_mpjpes)
    mean_mjpre = np.mean(mjpres)
    mean_mpjve = np.mean(mpjves)
    mean_mte = np.mean(mtes)
    std_mpjpe = np.std(mpjpes)
    std_pa_mpjpe = np.std(pa_mpjpes)
    std_mjpre = np.std(mjpres)
    std_mpjve = np.std(mpjves)
    std_mte = np.std(mtes)

    metric_dict = {
        'mean': {
            'MPJPE': mean_mpjpe,
            'PA MPJPE': mean_pa_mpjpe,
            'MJPRE': mean_mjpre,
            'MPJVE': mean_mpjve,
            'MTE': mean_mte
        },
        'std': {
            'MPJPE': std_mpjpe,
            'PA MPJPE': std_pa_mpjpe,
            'MJPRE': std_mjpre,
            'MPJVE': std_mpjve,
            'MTE': std_mte
        },
        'raw': {
            'MPJPE': mpjpes,
            'PA MPJPE': pa_mpjpes,
            'MJPRE': mjpres,
            'MPJVE': mpjves,
            'MTE': mtes
        }
    }
    if apply_shited_window_avg:
        save_txt_path = os.path.join(log_dir, f'{model}_{dataset}_{eval_frame_length}frame_avg_motion_metric.txt')
        save_npy_path = os.path.join(log_dir, f'{model}_{dataset}_{eval_frame_length}frame_avg_motion_metric.npy')
    else:
        save_txt_path = os.path.join(log_dir, f'{model}_{dataset}_{eval_frame_length}frame_raw_motion_metric.txt')
        save_npy_path = os.path.join(log_dir, f'{model}_{dataset}_{eval_frame_length}frame_raw_motion_metric.npy')
    print('########################################################')
    print(f'{model} {dataset} motion metric on {len(result_list)} samples with frame length {eval_frame_length}:')
    print(f"MPJPE: {mean_mpjpe:.2f} mm")
    print(f"Old MPJPE: {np.mean(mpjpes_old):.2f} mm")
    print(f"PA MPJPE: {mean_pa_mpjpe:.2f} mm")
    print(f"MJPRE: {mean_mjpre:.2f} degrees")
    print(f"MPJVE: {mean_mpjve:.2f} mm")
    print(f"MTE: {mean_mte:.2f} mm")
    with open(save_txt_path, 'w') as f:
        f.write(f"Apply shifted window average: {apply_shited_window_avg}\n")
        f.write(f"Eval frame length: {eval_frame_length}\n")
        f.write("# Mean metrics:\n")
        f.write(f"MPJPE: {mean_mpjpe:.2f} mm\n")
        # f.write(f"Old MPJPE: {np.mean(mpjpes_old):.2f} mm\n")
        f.write(f"PA MPJPE: {mean_pa_mpjpe:.2f} mm\n")
        f.write(f"MJPRE: {mean_mjpre:.2f} degrees\n")
        f.write(f"MPJVE: {mean_mpjve:.2f} mm\n")
        f.write(f"MTE: {mean_mte:.2f} mm\n")
        f.write("# Std metrics:\n")
        f.write(f"MPJPE: {std_mpjpe:.2f} mm\n")
        f.write(f"PA MPJPE: {std_pa_mpjpe:.2f} mm\n")
        f.write(f"MJPRE: {std_mjpre:.2f} degrees\n")
        f.write(f"MPJVE: {std_mpjve:.2f} mm\n")
        f.write(f"MTE: {std_mte:.2f} mm\n")
    
    np.save(save_npy_path, metric_dict)
    print(f"Saved metrics to {save_txt_path} and {save_npy_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate motion metrics from viz_test_generate_number npy outputs."
    )
    parser.add_argument(
        "--result-folder",
        type=str,
        required=True,
        help="Directory ending with viz_test_generate_number containing *.npy predictions.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "Where to write *_motion_metric.txt and *.npy summaries. "
            "If omitted, uses <result-folder>/evaluation (created if missing)."
        ),
    )
    parser.add_argument("--model", type=str, default="Ours")
    parser.add_argument("--dataset", type=str, default="LINGO")
    parser.add_argument(
        "--apply-shifted-window-avg",
        action="store_true",
        help="Average with shifted-window predictions when available (Ours/GPT2).",
    )
    parser.add_argument(
        "--eval-frame-length", type=int, default=60,
        help="Only evaluate the first N frames."
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = os.path.join(args.result_folder, "evaluation")

    os.makedirs(log_dir, exist_ok=True)
    get_number(
        args.model,
        args.dataset,
        args.result_folder,
        log_dir,
        apply_shited_window_avg=args.apply_shifted_window_avg,
        eval_frame_length=args.eval_frame_length,
    )