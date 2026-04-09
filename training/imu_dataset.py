from typing import Any, Callable, Optional

import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import torch
import sys
import os
import random
import json
import glob
import random
from typing import Tuple
from bisect import bisect_left, bisect_right
from dataset_process.custom_path import humoto_root, parahome_root, imuposer_root, dipimu_root

def smooth_avg(acc, s=3):
    """
    Smooth data using a centered moving average.
    
    Args:
        acc: numpy array of shape [time, n_device, n_value]
        s: window size (should be odd for symmetric smoothing)
    
    Returns:
        Smoothed array of same shape as input
    """
    nan_array = np.full((s // 2, acc.shape[1], acc.shape[2]), np.nan)
    acc = np.concatenate((nan_array, acc, nan_array), axis=0)
    arrays = []
    for i in range(s):
        L = acc.shape[0]
        arrays.append(acc[i:L-(s-i-1)])
    smoothed = np.nanmean(np.stack(arrays, axis=0), axis=0)
    return smoothed

def intersecting_indices(intervals, A, B, *, inclusive=False):
    """
    Return indices of intervals that intersect with (A, B) (open) by default.
    If inclusive=True, treat it as [A, B] (closed) for endpoint-touching.

    Assumes intervals are sorted, non-overlapping: a0<=b0<=a1<=b1<=...
    """
    if A >= B:
        return []

    a = [x[0] for x in intervals]
    b = [x[1] for x in intervals]

    if inclusive:
        # Overlap if a_i <= B and b_i >= A
        left = bisect_left(b, A)        # first i with b_i >= A
        right = bisect_right(a, B) - 1  # last  i with a_i <= B
    else:
        # Overlap if a_i < B and b_i > A (open interval)
        left = bisect_right(b, A)       # first i with b_i > A
        right = bisect_left(a, B) - 1   # last  i with a_i < B

    return list(range(left, right + 1)) if left <= right else []

def random_contiguous_subarray_bounds(N, A):
    return _random_contiguous_subarray_bounds(N, A, A)

def _random_contiguous_subarray_bounds(N: int, A: int, B: int, *, rng: random.Random | None = None) -> Tuple[int, int]:
    """
    Return (start_idx, end_idx) inclusive for a random contiguous subarray.

    - N: sequence length
    - A, B: integer length bounds (inclusive)
    - rng: optional random.Random instance for reproducibility
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}.")
    if not (isinstance(A, int) and isinstance(B, int)):
        raise TypeError("A and B must be integers.")
    if A <= 0 or B <= 0:
        raise ValueError("A and B must be positive.")

    # if sequence too short, return the full range
    if N < A:
        return 0, N - 1

    # clamp B to at most N
    B = min(B, N)

    r = rng if rng is not None else random
    M = r.randint(A, B)
    start = r.randint(0, N - M)
    end = start + M - 1
    return start, end

# sys.path.append('/scratch/benk/tcheng1/code/imu-human-mllm/')
from imu_synthesis.get_imu_readings import simulate_imu_readings
from imu_synthesis.utils.rotation import convert_rotation
from dataset_process.motionmillion_and_lingo.convert_lingo_dataset import align_poses_to_first_frame

def align_translation(inv_rotation, inv_translation, translation):
    aligned_transl = (inv_rotation @ (translation + inv_translation).T).T  # Apply inverse rotation and translation
    return aligned_transl

def align_orientation(inv_rotation, orientation):
    aligned_orient = R.from_rotvec(orientation).as_matrix()
    aligned_orient = np.einsum('ij,njk->nik', inv_rotation, aligned_orient)
    aligned_orient = R.from_matrix(aligned_orient).as_rotvec()
    aligned_orient = R.from_rotvec(aligned_orient).as_rotvec()
    return aligned_orient

# use max's split
def filter_by_targets(all_strings, target_datasets):
    before_len = len(all_strings)
    filtered_strings = [
        s for s in all_strings 
        if any(t in s for t in target_datasets)
    ]
    after_len = len(filtered_strings)
    print(f"Filtered {before_len - after_len} out of {before_len} data")
    return filtered_strings

def add_velocity_scaled_noise(x, eps=0.01):
    """
    Noise magnitude follows local velocity.
    Supports np.ndarray and torch.Tensor.
    """
    if torch.is_tensor(x):
        # torch version
        vel = torch.diff(x, dim=0, prepend=x[:1])
        noise = torch.randn_like(x)
        noise = noise.clip(-1, 1)
        return x + eps * vel * noise
    else:
        # numpy version
        vel = np.diff(x, axis=0, prepend=x[:1])
        noise = np.random.randn(*x.shape)
        noise = np.clip(noise, -1, 1)
        return x + eps * vel * noise


def get_angular_velocity(R_sim: np.ndarray, fps: float) -> np.ndarray:
    r"""
    Compute sensor-local angular velocity from rotation matrices.

    Parameters
    ----------
    R_sim : np.ndarray
        Rotation matrices with shape [T, ..., 3, 3],
        where ... represents arbitrary batch dimensions.
    fps : float
        Sampling frequency (frames per second).

    Returns
    -------
    np.ndarray
        Angular velocity with shape [T, ..., 3].
    """
    # Central differences for interior points
    Rdot_mid = R_sim[2:] - R_sim[:-2]

    # Forward / backward differences for boundaries
    Rdot0 = -3 * R_sim[0] + 4 * R_sim[1] - R_sim[2]
    Rdot1 =  3 * R_sim[-1] - 4 * R_sim[-2] + R_sim[-3]

    # Concatenate along time dimension
    Rdot = np.concatenate(
        (
            Rdot0[np.newaxis],
            Rdot_mid,
            Rdot1[np.newaxis],
        ),
        axis=0,
    )

    # Scale by timestep
    Rdot = Rdot * (fps / 2.0)

    # Body-frame angular velocity hat matrix
    Rt = np.swapaxes(R_sim, -1, -2)
    w_hat = Rt @ Rdot

    # Enforce skew-symmetry
    w_hat = 0.5 * (w_hat - np.swapaxes(w_hat, -1, -2))

    # vee operator
    w = np.stack(
        (
            w_hat[..., 2, 1],
            w_hat[..., 0, 2],
            w_hat[..., 1, 0],
        ),
        axis=-1,
    )

    return w

def process_imuposer_data(sample, random_cut, random_mask_text, cut_length, shift=0, 
                        filter_short_text=True, add_ground_data=False, rot_rep='6d', split=None,
                        data_source=None, dynamic_object=False, fps=30, sample_idx=None, IMUSEQMAXLEN=1e6,
                        smooth_imu_acc=True):

    n_time = min(len(sample['motion_smpl']), len(sample['imu_data']))
    assert cut_length is not None
    if n_time < cut_length and split == 'train':
        return None

    sample['motion_smpl'] = sample['motion_smpl'][:n_time]
    sample['imu_data'] = sample['imu_data'][:n_time]


    motion_smpl = sample['motion_smpl'] # [n, 85]
    smpl_orient = motion_smpl[:, 0:3].copy()
    smpl_pose = motion_smpl[:, 3:66].copy()
    smpl_transl = motion_smpl[:, 66:69].copy()

    if random_cut and split == 'train':
        assert n_time >= cut_length , "cut_length must be greater than or equal to n_time"
        start_cut_idx, end_cut_idx = random_contiguous_subarray_bounds(n_time, cut_length) 
    else:
        start_cut_idx = 0
        end_cut_idx = min(n_time, cut_length)
    
    start_cut_idx = start_cut_idx + shift
    end_cut_idx = end_cut_idx + shift

    smpl_orient = smpl_orient[start_cut_idx:end_cut_idx]
    smpl_pose = smpl_pose[start_cut_idx:end_cut_idx]
    smpl_transl = smpl_transl[start_cut_idx:end_cut_idx]

    # align the smpl poses to the first frame
    smpl_orient_new, smpl_transl_new, smpl_pose_new, inv_rotation, inv_translation = \
        align_poses_to_first_frame(smpl_orient, smpl_transl, smpl_pose)

    imu_data = sample['imu_data'][start_cut_idx:end_cut_idx] # [n, 30]
    a_sim = imu_data[:, :15].reshape(-1, 5, 3)  # [n, 5, 3]
    R_sim = imu_data[:, 15:].reshape(-1, 5, 3, 3) # [n, 5, 3, 3]

    # do smooth
    if smooth_imu_acc:
        a_sim = smooth_avg(a_sim, s=3)

    # change sequence
    """
    ['left_hip', 'right_hip', 'left_ear', 'right_ear', 'left_elbow', 'right_elbow']
    ['左髋', '右髋', '左耳', '右耳', '左肘', '右肘']
    0 左手腕 1 右手腕 2 左前口袋 3 右前口袋 4 头部
    """
    permute_idx = [2, 3, 4, 4, 0, 1]
    a_sim = a_sim[:, permute_idx, :] # [n, 6, 3]
    R_sim = R_sim[:, permute_idx, :, :] # [n, 6, 3, 3]

    a_sim = np.einsum('ij,bnj->bni', inv_rotation, a_sim)
    R_sim = np.einsum('ij,bnjk->bnik', inv_rotation, R_sim)
    w_sim = get_angular_velocity(R_sim, fps=fps)

    a_sim = a_sim + np.array((0, -9.8, 0), dtype=np.float32) # add gravity back

    obj_pose_dict = {}

    if 'objects' not in sample:
        sample['objects'] = {}

    if add_ground_data:
        ground_height = 0
        if 'ground' in sample['objects']:
            raise ValueError("Ground data already exists")
        sample['objects']['ground'] = np.array([1.0, 0.0, 0.0, 0.0, 
                                                0.0, ground_height, 0.0,
                                                1.0, 1.0, 1.0], dtype=np.float32)

    if len(sample['objects']) > 0:
        obj_pose = sample['objects']
        for obj in obj_pose.keys():
            obj_quat = obj_pose[obj][0:4].copy().astype(np.float32)
            obj_mat = convert_rotation(torch.from_numpy(obj_quat), 'quat', 'mat').float().cpu().numpy() # [3,3]
            obj_mat = np.einsum('ij,jk->ik', inv_rotation, obj_mat) # [3,3]
            obj_6d = convert_rotation(torch.from_numpy(obj_mat), 'mat', '6d').float().cpu().numpy() # [6]

            obj_transl = obj_pose[obj][4:7].copy()
            obj_transl = np.einsum('ij,j->i', inv_rotation, obj_transl + inv_translation)
            obj_bbox = np.ones((3,), dtype=np.float32) # placeholder for bbox
            obj_name_filtered = obj.split('.')[0] # remove the suffix
            assert obj == obj_name_filtered

            if obj == 'ground' and data_source != 'parahome':
                # set the ground to be at the origin, only keep the y-axis
                # but for parahome, we keep the original transformation
                plane_y = obj_transl[1]
                obj_transl = np.array([0.0, plane_y, 0.0], dtype=np.float32)

            obj_pose_dict[obj_name_filtered] = {'rot': obj_6d, 'transl': obj_transl, 'bbox': obj_bbox}

    smpl_orient_new = torch.tensor(smpl_orient_new).float()
    smpl_transl_new = torch.tensor(smpl_transl_new).float()
    smpl_pose_new = torch.tensor(smpl_pose_new).float()

    # a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(
    #     p, R, fps=fps,
    #     noise_raw_traj=False,
    #     noise_syn_imu=False,
    #     noise_est_orient=False,
    #     skip_ESKF=True,
    #     device='cpu'
    # )
    R_sim = R_sim.reshape(R_sim.shape[0], R_sim.shape[1], 9) # flatten the rotation matrix to 9d

    a_sim = torch.tensor(a_sim).float()
    w_sim = torch.tensor(w_sim).float()
    R_sim = torch.tensor(R_sim).float()
    imu_data = torch.cat([a_sim, w_sim, R_sim], dim=-1) # [n, 5, 15]

    # import pdb; pdb.set_trace()

    add_small_noise = False
    if add_small_noise:
        imu_data = add_velocity_scaled_noise(imu_data)

    if 'text' in sample:
        sample['description'] = sample['text']
    elif 'texts' in sample:
        sample['description'] = sample['texts']
    elif 'description' not in sample:
        sample['description'] = []

    gt_text_list = []
    # filter out too short descriptions
    if sample['description'] is not None and len(sample['description']) > 0:
        for desc in sample['description']:
            # cut too long descriptions
            words = desc.split(' ')
            if len(words) > 40:
                # random cut the description to 40 words and append ellipsis
                start = random.randint(0, len(words) - 40)
                desc = ' '.join(words[start:start + 40]) + '...'
            gt_text_list.append(desc)
            # if len(desc.split(' ')) >= 7 or not filter_short_text: # at least 7 words; if filter_text is False, then don't filter the text
            #     gt_text_list.append(desc)

    # if random_mask_text:
    #     # mask_prob = 0.0 # FIXME
    #     mask_prob = 0.2 # FIXME
    #     # if p < mask_prob, then set the gt_text_list to an empty list
    #     if random.random() < mask_prob:
    #         gt_text_list = []
    # gt_text_list = []

    if rot_rep == '6d':
        orient = convert_rotation(smpl_orient_new, 'aa', '6d').reshape(-1, 6).float()
        pose = convert_rotation(smpl_pose_new.reshape(-1, 3), 'aa', '6d').reshape(-1, 21*6).float()
        transl = smpl_transl_new
    elif rot_rep == 'aa':
        orient = smpl_orient_new.reshape(-1, 3).float()
        pose = smpl_pose_new.reshape(-1, 21*3).float()
        transl = smpl_transl_new
    else:
        raise ValueError(f"Invalid rotation representation: {rot_rep}")

    sample_output = {
        'imu_data': imu_data,
        'orient': orient, # [n, 6]
        'transl': transl,
        'pose': pose, # [n, 21*6]
        'description': gt_text_list,
        'objects': obj_pose_dict,
        # 'aS': aS,
        # 'wS': wS,
    }

    length = len(sample_output['imu_data'])
    length = length - (length % 4)  # make length a multiple of 4
    length = min(length, IMUSEQMAXLEN)  # cut to max length to avoid OOM

    for k, v in sample_output.items():
        if k in ['description', 'scene_name', 'scene_2d_layout', 'scene_mesh', 'scene_occ_grid']:
            # skip non-numeric data
            continue
        elif k == 'objects' and dynamic_object:
            for obj_name, obj_data in sample_output[k].items():
                obj_data['rot'] = obj_data['rot'][:length]
                obj_data['transl'] = obj_data['transl'][:length]
                obj_data['bbox'] = obj_data['bbox'][:length]
        elif k != 'objects':
            sample_output[k] = sample_output[k][:length]
    
    if not (len(sample_output['imu_data']) == len(sample_output['orient']) == len(sample_output['transl']) == len(sample_output['pose'])):
        print('len(imu_data): ', len(sample_output['imu_data']))
        print('len(orient): ', len(sample_output['orient']))
        print('len(transl): ', len(sample_output['transl']))
        print('len(pose): ', len(sample_output['pose']))
        print('sample_idx: ', sample_idx)
        import pdb; pdb.set_trace()
        _=1

    # sample_output = {k: pad_to_length(v, self.MAXLEN) for k, v in sample.items()} # [MAXLEN, 3, nvars]
    return sample_output
    

def rotation_matrix_to_axis_angle(r: torch.Tensor):
    r"""
    Turn rotation matrices into axis-angles. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    import cv2
    result = [cv2.Rodrigues(_)[0] for _ in r.clone().detach().cpu().view(-1, 3, 3).numpy()]
    result = torch.from_numpy(np.stack(result)).float().squeeze(-1).to(r.device)
    return result

def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False, avoid_nan=False):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :param avoid_nan: If True, return zeros if norm is 0.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    if avoid_nan:
        normalized_x[torch.isnan(normalized_x)] = 0
    return normalized_x if not return_norm else (normalized_x, norm)

def hat(v: torch.Tensor):
    r"""
    Return the 3x3 skew-symmetric matrix of the 3D vector. (torch, batch)

    :param v: Tensor in shape [..., 3].
    :return: Tensor in shape [..., 3, 3].
    """
    return torch.stack((torch.zeros_like(v[..., 0]), -v[..., 2], v[..., 1],
                        v[..., 2], torch.zeros_like(v[..., 0]), -v[..., 0],
                        -v[..., 1], v[..., 0], torch.zeros_like(v[..., 0]),), dim=-1).view(*v.shape[:-1], 3, 3)

def axis_angle_to_rotation_matrix(a: torch.Tensor):
    r"""
    Turn axis-angles into rotation matrices. (torch, batch)

    :param a: Axis-angle tensor that can reshape to [batch_size, 3].
    :return: Rotation matrix of shape [batch_size, 3, 3].
    """
    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis) | torch.isinf(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * hat(axis)
    return r

def process_dipimu_data(sample, random_cut, random_mask_text, cut_length, shift=0, 
                        filter_short_text=True, add_ground_data=False, rot_rep='6d', split=None,
                        data_source=None, dynamic_object=False, fps=30, sample_idx=None, IMUSEQMAXLEN=1e6,
                        acc_scale=1.0, gyro_scale=1.0, smooth_imu_acc=True):

    n_time = min(len(sample['motion_smpl']), len(sample['imu_acc']))

    assert cut_length is not None
    if n_time < cut_length and split == 'train':
        return None

    sample['motion_smpl'] = sample['motion_smpl'][:n_time]
    sample['imu_acc'] = sample['imu_acc'][:n_time]
    sample['imu_ori'] = sample['imu_ori'][:n_time]

    g = torch.tensor([0, -9.798, 0])
    ori = torch.tensor(sample['imu_ori']).float()
    acc = torch.tensor(sample['imu_acc']).float()
    w = rotation_matrix_to_axis_angle(ori[:-1].transpose(2, 3).matmul(ori[1:])).view(-1, ori.shape[1], 3) * 60
    w = torch.cat((w, torch.zeros_like(w[:1])))
    m = ori.transpose(2, 3).matmul(torch.tensor([1, 0, 0.]).unsqueeze(-1)).squeeze(-1)
    a = ori.transpose(2, 3).matmul((acc - g).unsqueeze(-1)).squeeze(-1)

    aS = a
    wS = w
    mS = m

    if smooth_imu_acc:
        aS = smooth_avg(aS, s=3)
    aS = torch.tensor(aS).float()

    # simulate IMU ESKF
    N = len(wS)
    # R_sim = torch.empty(N, 6, 3, 3)
    R_sim = torch.empty(N, 6, 3, 3)
    R_sim[0] = torch.eye(3).float()
    # angular velocity integration, much faster for approximate training
    dR = axis_angle_to_rotation_matrix(wS / 60).view(-1, 6, 3, 3).cpu()
    for i in range(1, N):
        R_sim[i] = R_sim[i - 1].matmul(dR[i])
        # print(R_sim[i])

    # a_sim = R_sim.matmul(aS.unsqueeze(-1)).squeeze(-1) + torch.tensor((0, -9.8, 0), device=device)
    a_sim = R_sim.matmul(aS.unsqueeze(-1)).squeeze(-1)
    w_sim = R_sim.matmul(wS.unsqueeze(-1)).squeeze(-1)

    motion_smpl = sample['motion_smpl'] # [n, 85]
    smpl_orient = motion_smpl[:, 0:3].copy()
    smpl_pose = motion_smpl[:, 3:66].copy()
    smpl_transl = motion_smpl[:, 66:69].copy()

    if random_cut and split == 'train':
        assert n_time >= cut_length , "cut_length must be greater than or equal to n_time"
        start_cut_idx, end_cut_idx = random_contiguous_subarray_bounds(n_time, cut_length) 
    else:
        start_cut_idx = 0
        end_cut_idx = min(n_time, cut_length)
    
    start_cut_idx = start_cut_idx + shift
    end_cut_idx = end_cut_idx + shift

    smpl_orient = smpl_orient[start_cut_idx:end_cut_idx]
    smpl_pose = smpl_pose[start_cut_idx:end_cut_idx]
    smpl_transl = smpl_transl[start_cut_idx:end_cut_idx]

    # align the smpl poses to the first frame
    smpl_orient_new, smpl_transl_new, smpl_pose_new, inv_rotation, inv_translation = \
        align_poses_to_first_frame(smpl_orient, smpl_transl, smpl_pose)

    # change sequence
    """
    ['left_hip', 'right_hip', 'left_ear', 'right_ear', 'left_elbow', 'right_elbow']
    (0: left wrist, 1: right wrist, 2: left thigh, 3: right thigh, 4: head, 5: pelvis)
    0 左手腕 1 右手腕
    """
    a_sim = a_sim[start_cut_idx:end_cut_idx]
    R_sim = R_sim[start_cut_idx:end_cut_idx]
    w_sim = w_sim[start_cut_idx:end_cut_idx]

    permute_idx = [2, 3, 4, 4, 0, 1]
    a_sim = a_sim[:, permute_idx, :] # [n, 6, 3]
    R_sim = R_sim[:, permute_idx, :, :] # [n, 6, 3, 3]

    a_sim = np.einsum('ij,bnj->bni', inv_rotation, a_sim)
    R_sim = np.einsum('ij,bnjk->bnik', inv_rotation, R_sim)
    w_sim = get_angular_velocity(R_sim, fps=fps)

    a_sim = a_sim + np.array((0, -9.8, 0), dtype=np.float32) # add gravity back

    obj_pose_dict = {}

    if 'objects' not in sample:
        sample['objects'] = {}

    if add_ground_data:
        ground_height = 0
        if 'ground' in sample['objects']:
            raise ValueError("Ground data already exists")
        sample['objects']['ground'] = np.array([1.0, 0.0, 0.0, 0.0, 
                                                0.0, ground_height, 0.0,
                                                1.0, 1.0, 1.0], dtype=np.float32)

    if len(sample['objects']) > 0:
        obj_pose = sample['objects']
        for obj in obj_pose.keys():
            obj_quat = obj_pose[obj][0:4].copy().astype(np.float32)
            obj_mat = convert_rotation(torch.from_numpy(obj_quat), 'quat', 'mat').float().cpu().numpy() # [3,3]
            obj_mat = np.einsum('ij,jk->ik', inv_rotation, obj_mat) # [3,3]
            obj_6d = convert_rotation(torch.from_numpy(obj_mat), 'mat', '6d').float().cpu().numpy() # [6]

            obj_transl = obj_pose[obj][4:7].copy()
            obj_transl = np.einsum('ij,j->i', inv_rotation, obj_transl + inv_translation)
            obj_bbox = np.ones((3,), dtype=np.float32) # placeholder for bbox
            obj_name_filtered = obj.split('.')[0] # remove the suffix
            assert obj == obj_name_filtered

            if obj == 'ground' and data_source != 'parahome':
                # set the ground to be at the origin, only keep the y-axis
                # but for parahome, we keep the original transformation
                plane_y = obj_transl[1]
                obj_transl = np.array([0.0, plane_y, 0.0], dtype=np.float32)

            obj_pose_dict[obj_name_filtered] = {'rot': obj_6d, 'transl': obj_transl, 'bbox': obj_bbox}

    smpl_orient_new = torch.tensor(smpl_orient_new).float()
    smpl_transl_new = torch.tensor(smpl_transl_new).float()
    smpl_pose_new = torch.tensor(smpl_pose_new).float()

    # a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(
    #     p, R, fps=fps,
    #     noise_raw_traj=False,
    #     noise_syn_imu=False,
    #     noise_est_orient=False,
    #     skip_ESKF=True,
    #     device='cpu'
    # )
    R_sim = R_sim.reshape(R_sim.shape[0], R_sim.shape[1], 9) # flatten the rotation matrix to 9d

    a_sim = torch.tensor(a_sim).float() / acc_scale
    w_sim = torch.tensor(w_sim).float() / gyro_scale
    R_sim = torch.tensor(R_sim).float()
    imu_data = torch.cat([a_sim, w_sim, R_sim], dim=-1) # [n, 5, 15]

    # import pdb; pdb.set_trace()

    add_small_noise = False
    if add_small_noise:
        imu_data = add_velocity_scaled_noise(imu_data)

    if 'text' in sample:
        sample['description'] = sample['text']
    elif 'texts' in sample:
        sample['description'] = sample['texts']
    elif 'description' not in sample:
        sample['description'] = []

    gt_text_list = []
    # filter out too short descriptions
    if sample['description'] is not None and len(sample['description']) > 0:
        for desc in sample['description']:
            if len(desc.split(' ')) >= 7 or not filter_short_text: # at least 7 words; if filter_text is False, then don't filter the text
                gt_text_list.append(desc)

    if random_mask_text:
        # mask_prob = 0.0 # FIXME
        mask_prob = 0.2 # FIXME
        # if p < mask_prob, then set the gt_text_list to an empty list
        if random.random() < mask_prob:
            gt_text_list = []
    
    gt_text_list = []

    if rot_rep == '6d':
        orient = convert_rotation(smpl_orient_new, 'aa', '6d').reshape(-1, 6).float()
        pose = convert_rotation(smpl_pose_new.reshape(-1, 3), 'aa', '6d').reshape(-1, 21*6).float()
        transl = smpl_transl_new
    elif rot_rep == 'aa':
        orient = smpl_orient_new.reshape(-1, 3).float()
        pose = smpl_pose_new.reshape(-1, 21*3).float()
        transl = smpl_transl_new
    else:
        raise ValueError(f"Invalid rotation representation: {rot_rep}")

    sample_output = {
        'imu_data': imu_data,
        'orient': orient, # [n, 6]
        'transl': transl,
        'pose': pose, # [n, 21*6]
        'description': gt_text_list,
        'objects': obj_pose_dict,
        # 'aS': aS,
        # 'wS': wS,
    }

    length = len(sample_output['imu_data'])
    length = length - (length % 4)  # make length a multiple of 4
    length = min(length, IMUSEQMAXLEN)  # cut to max length to avoid OOM

    for k, v in sample_output.items():
        if k in ['description', 'scene_name', 'scene_2d_layout', 'scene_mesh', 'scene_occ_grid']:
            # skip non-numeric data
            continue
        elif k == 'objects' and dynamic_object:
            for obj_name, obj_data in sample_output[k].items():
                obj_data['rot'] = obj_data['rot'][:length]
                obj_data['transl'] = obj_data['transl'][:length]
                obj_data['bbox'] = obj_data['bbox'][:length]
        elif k != 'objects':
            sample_output[k] = sample_output[k][:length]
    
    if not (len(sample_output['imu_data']) == len(sample_output['orient']) == len(sample_output['transl']) == len(sample_output['pose'])):
        print('len(imu_data): ', len(sample_output['imu_data']))
        print('len(orient): ', len(sample_output['orient']))
        print('len(transl): ', len(sample_output['transl']))
        print('len(pose): ', len(sample_output['pose']))
        print('sample_idx: ', sample_idx)
        import pdb; pdb.set_trace()
        _=1

    # sample_output = {k: pad_to_length(v, self.MAXLEN) for k, v in sample.items()} # [MAXLEN, 3, nvars]
    return sample_output

def process_imu_data(sample, random_cut, random_mask_text, cut_length, shift=0, 
                     filter_short_text=True, add_ground_data=False, rot_rep='6d',
                     motion_only=False, scene_only=False, IMUSEQMAXLEN=1e6,
                     data_source=None, dynamic_object=False, fps=30, split=None, add_imu_noise=False, 
                     acc_scale=1.0, gyro_scale=1.0):

    assert data_source in ['parahome', 'humoto', 'other_dataset'], f"Invalid data source: {data_source}"

    if 'motion_smpl' in sample.keys():
        motion_smpl = sample['motion_smpl'] # [n, 85]
    elif 'motion_data_smpl85' in sample.keys():
        motion_smpl = sample['motion_data_smpl85'] # [n, 85]
    else:
        raise ValueError(f"Invalid motion smpl data: {sample.keys()}")
    imu_traj = sample['imu_traj'] # [n, 6, 6]

    n_time = len(motion_smpl)
    assert cut_length is not None
    if n_time < cut_length and split == 'train':
        return None
    
    smpl_orient = motion_smpl[:, 0:3].copy()
    smpl_pose = motion_smpl[:, 3:66].copy()
    smpl_transl = motion_smpl[:, 72:75].copy()

    imu_traj_data = imu_traj.copy() # [n, 6, 6]
    imu_rot = imu_traj_data[:, :, 0:3] # [n, 6, 3]
    imu_rot = (
        convert_rotation(
            torch.from_numpy(imu_rot.reshape(-1, 3)).float(), 'aa', 'mat'
        ).view(n_time, -1, 3, 3).numpy().astype(np.float32)
    ) # [n, 6, 3, 3]
    imu_position = imu_traj_data[:, :, 3:6] # [n, 6, 3]

    if random_cut and split == 'train': #FIXME
        assert n_time >= cut_length , "cut_length must be greater than or equal to n_time"
        start_cut_idx, end_cut_idx = random_contiguous_subarray_bounds(n_time, cut_length)
    else:
        start_cut_idx = 0
        end_cut_idx = min(n_time, cut_length)
    
    start_cut_idx = start_cut_idx + shift
    end_cut_idx = end_cut_idx + shift

    imu_rot = imu_rot[start_cut_idx:end_cut_idx]
    imu_position = imu_position[start_cut_idx:end_cut_idx]

    smpl_orient = smpl_orient[start_cut_idx:end_cut_idx]
    smpl_pose = smpl_pose[start_cut_idx:end_cut_idx]
    smpl_transl = smpl_transl[start_cut_idx:end_cut_idx]

    # align the smpl poses to the first frame
    smpl_orient_new, smpl_transl_new, smpl_pose_new, inv_rotation, inv_translation = \
        align_poses_to_first_frame(smpl_orient, smpl_transl, smpl_pose)

    # and apply the same inverse transformation to the imu data
    # inv_rotation: [3, 3], inv_translation: [3,]
    # [3, 3] @ ([n, 6, 3] + [1, 1, 3]) -> [n, 6, 3]
    imu_position_new = np.einsum('ij,bnj->bni', inv_rotation, imu_position + inv_translation[None, None])
    # [3, 3] @ [n, 6, 3, 3] -> [n, 6, 3, 3]
    imu_rot_new = np.einsum('ij,bnjk->bnik', inv_rotation, imu_rot)

    # smpl_transl_compare = np.einsum('ij,bj->bi', inv_rotation, smpl_transl + inv_translation) should be the same as smpl_transl_new
    # smpl_orient_compare = np.einsum('ij,njk->nik', inv_rotation, R.from_rotvec(smpl_orient).as_matrix())
    # smpl_orient_compare = R.from_matrix(smpl_orient_compare).as_rotvec() # should be the same as smpl_orient_new

    obj_pose_dict = {}

    if 'objects' not in sample:
        sample['objects'] = {}

    if add_ground_data:
        ground_height = 0
        if 'ground' in sample['objects']:
            raise ValueError("Ground data already exists")
        sample['objects']['ground'] = np.array([1.0, 0.0, 0.0, 0.0, 
                                                 0.0, ground_height, 0.0,
                                                 1.0, 1.0, 1.0], dtype=np.float32)

    if len(sample['objects']) > 0:
        obj_pose = sample['objects']
        for obj in obj_pose.keys():

            if dynamic_object:
        
                obj_quat = obj_pose[obj][:, 0:4].copy().astype(np.float32)
                obj_mat = convert_rotation(torch.from_numpy(obj_quat), 'quat', 'mat').float().cpu().numpy() # [t,3,3]
                obj_mat = np.einsum('ij,tjk->tik', inv_rotation, obj_mat) # [t,3,3]
                obj_6d = convert_rotation(torch.from_numpy(obj_mat), 'mat', '6d').float().cpu().numpy() # [t,6]

                t = obj_6d.shape[0]
                obj_transl = obj_pose[obj][:,4:7].copy()
                obj_transl = np.einsum('ij,tj->ti', inv_rotation, obj_transl + inv_translation)
                obj_bbox = np.ones((t, 3), dtype=np.float32) # placeholder for bbox

                obj_name_filtered = obj.split('.')[0] # remove the suffix
                assert obj == obj_name_filtered

                if obj == 'ground' and data_source != 'parahome': 
                    # set the ground to be at the origin, only keep the y-axis
                    # but for parahome, we keep the original transformation
                    obj_transl[:, 0] = 0.0
                    obj_transl[:, 2] = 0.0

                obj_pose_dict[obj_name_filtered] = {'rot': obj_6d, 'transl': obj_transl, 'bbox': obj_bbox}
                # import pdb; pdb.set_trace()
            else:
                obj_quat = obj_pose[obj][0:4].copy().astype(np.float32)
                obj_mat = convert_rotation(torch.from_numpy(obj_quat), 'quat', 'mat').float().cpu().numpy() # [3,3]
                obj_mat = np.einsum('ij,jk->ik', inv_rotation, obj_mat) # [3,3]
                obj_6d = convert_rotation(torch.from_numpy(obj_mat), 'mat', '6d').float().cpu().numpy() # [6]

                obj_transl = obj_pose[obj][4:7].copy()
                obj_transl = np.einsum('ij,j->i', inv_rotation, obj_transl + inv_translation)
                obj_bbox = obj_pose[obj][7:10].copy()

                obj_name_filtered = obj.split('.')[0] # remove the suffix
                assert obj == obj_name_filtered

                if obj == 'ground' and data_source != 'parahome':
                    # set the ground to be at the origin, only keep the y-axis
                    # but for parahome, we keep the original transformation
                    plane_y = obj_transl[1]
                    obj_transl = np.array([0.0, plane_y, 0.0], dtype=np.float32)

                obj_pose_dict[obj_name_filtered] = {'rot': obj_6d, 'transl': obj_transl, 'bbox': obj_bbox}

    p = torch.tensor(imu_position_new).float()
    R = torch.tensor(imu_rot_new).float()
    smpl_orient_new = torch.tensor(smpl_orient_new).float()
    smpl_transl_new = torch.tensor(smpl_transl_new).float()
    smpl_pose_new = torch.tensor(smpl_pose_new).float()

    a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(
        p, R, fps=fps,
        noise_raw_traj=add_imu_noise,
        noise_syn_imu=add_imu_noise,
        noise_est_orient=add_imu_noise,
        skip_ESKF=True,
        device='cpu'
    )
    a_sim = a_sim / acc_scale
    w_sim = w_sim / gyro_scale
    R_sim = R_sim.reshape(R_sim.shape[0], R_sim.shape[1], 9) # flatten the rotation matrix to 9d
    imu_data = torch.cat([a_sim, w_sim, R_sim], dim=-1) # [n, 6, 15]

    # add_small_noise = False
    # if add_small_noise:
    #     imu_data = add_velocity_scaled_noise(imu_data)

    if 'text' in sample:
        sample['description'] = sample['text']
    elif 'texts' in sample:
        sample['description'] = sample['texts']
    elif 'description' not in sample:
        sample['description'] = []

    gt_text_list = []

    if data_source == 'parahome':
        frame_bounds_str = list(list(sample['description'].keys()))
        frame_bounds_list = []
        for bound in frame_bounds_str:
            frame_bounds = bound.split(' ')
            frame_bounds = (int(frame_bounds[0]), int(frame_bounds[1]))
            frame_bounds_list.append(frame_bounds)
        
        valid_text_indices = intersecting_indices(frame_bounds_list, start_cut_idx, end_cut_idx)
        gt_text = ''
        for idx in valid_text_indices:
            gt_text += (sample['description'][frame_bounds_str[idx]] + ' ')
        gt_text_list = [gt_text]
    
    else:
        # filter out too short descriptions
        if sample['description'] is not None and len(sample['description']) > 0:
            for desc in sample['description']:
                if len(desc.split(' ')) >= 7 or not filter_short_text: # at least 7 words; if filter_text is False, then don't filter the text
                    gt_text_list.append(desc)

        if random_mask_text:
            # mask_prob = 0.0 # FIXME
            mask_prob = 0.2 # FIXME
            # if p < mask_prob, then set the gt_text_list to an empty list
            if random.random() < mask_prob:
                gt_text_list = []
    
    if motion_only:
        gt_text_list = []

    if rot_rep == '6d':
        orient = convert_rotation(smpl_orient_new, 'aa', '6d').reshape(-1, 6).float()
        pose = convert_rotation(smpl_pose_new.reshape(-1, 3), 'aa', '6d').reshape(-1, 21*6).float()
        transl = smpl_transl_new
    elif rot_rep == 'aa':
        orient = smpl_orient_new.reshape(-1, 3).float()
        pose = smpl_pose_new.reshape(-1, 21*3).float()
        transl = smpl_transl_new
    else:
        raise ValueError(f"Invalid rotation representation: {rot_rep}")

    sample_output = {
        'imu_data': imu_data,
        'orient': orient, # [n, 6]
        'transl': transl,
        'pose': pose, # [n, 21*6]
        'description': gt_text_list,
        'objects': obj_pose_dict,
        # 'aS': aS,
        # 'wS': wS,
    }

    length = len(sample_output['imu_data'])
    length = length - (length % 4)  # make length a multiple of 4
    length = min(length, IMUSEQMAXLEN)  # cut to max length to avoid OOM

    for k, v in sample_output.items():
        if k in ['description', 'scene_name', 'scene_2d_layout', 'scene_mesh', 'scene_occ_grid']:
            # skip non-numeric data
            continue
        elif k == 'objects' and dynamic_object:
            for obj_name, obj_data in sample_output[k].items():
                obj_data['rot'] = obj_data['rot'][:length]
                obj_data['transl'] = obj_data['transl'][:length]
                obj_data['bbox'] = obj_data['bbox'][:length]
        elif k != 'objects':
            sample_output[k] = sample_output[k][:length]

    # sample_output = {k: pad_to_length(v, self.MAXLEN) for k, v in sample.items()} # [MAXLEN, 3, nvars]

    return sample_output
    

class IMUDataset():
    def __init__(
        self,
        root=None,
        split: Optional[str] = 'train', # 'train', 'val'， 'test'
        seed: int = 42, # New argument for reproducibility
        overfit: bool = False, # for debugging
        motion_only: bool = False, # whether to only use motion data
        text_only: bool = False, # whether to only use text data
        scene_only: bool = False, # whether to only use scene data
        random_cut: bool = False, # whether to randomly cut the sequence
        random_mask_text: bool = False, # whether to randomly mask the text
        add_humoto_data: bool = True, # whether to add humoto data
        add_motiongv_data: bool = True, # whether to add motiongv data
        shift: int = 0, # whether to shift the imu data
        selected_dataset: Optional[str] = None, # whether to evaluate on the full dataset
        shuffle_list: bool = True, # whether to shuffle the data
        return_path_only: bool = False, # whether to return the path only
        dynamic_object: bool = False, # whether to use dynamic object
        fps: int = 30, # fps of the imu data
        add_imu_noise: bool = False, # whether to add noise to the imu data
        IMUSEQMAXLEN: int = 1e6, # cut max length to avoid OOM (out of memory) issues
        acc_scale: float = 1.0, # scale the acceleration data
        gyro_scale: float = 1.0, # scale the gyroscope data
        **kwargs,
    ):
        # self.root = '/scratch/benk/hhsu2/imu-humans/final_data_per_sequence' # hardcode for now
        self.root = root
        assert self.root is not None, "Please specify the root directory of the dataset."

        # self.humoto_root = kwargs.get('humoto_root', None)
        # self.parahome_root = kwargs.get('parahome_root', None)
        # self.imuposer_root = kwargs.get('imuposer_root', None)
        # self.dipimu_root = kwargs.get('dipimu_root', None)

        self.split = split
        self.seed = seed
        self.motion_only = motion_only
        self.text_only = text_only
        self.scene_only = scene_only
        self.random_cut = random_cut
        self.random_mask_text = random_mask_text
        self.shift = shift
        self.dynamic_object = dynamic_object
        self.fps = fps
        self.add_imu_noise = add_imu_noise
        self.acc_scale = acc_scale
        self.gyro_scale = gyro_scale
        self.cut_length = IMUSEQMAXLEN
        self.IMUSEQMAXLEN = IMUSEQMAXLEN
        assert not add_imu_noise, "add_imu_noise is not supported for full dataset."
        assert IMUSEQMAXLEN is not None
        
        assert self.root is not None, "Please specify the root directory of the dataset."
        assert split in ['train', 'val', 'test']

        if split != 'train':
            assert not random_cut, "only do random cut for train split."
        # Load IMU data here
        # all_data_list = [v for v in all_data_dict.values()]
        # sample_nums = {'train': 702270, 'val': 45236, 'test': 131097}
        # sample_num = sample_nums[split]
        # all_data_list = np.arange(sample_num)

        target_datasets = [
            'LINGO', 
            'BABEL', 
            'Mirror_BABEL', 
            'PhantomDanceDatav1.1', 
            'Mirror_PhantomDanceDatav1.1',
            # 'MotionGV',
            'MotionLLAMA', 
            'MotionUnion',
            # 'Mirror_MotionGV',
            'Mirror_MotionLLAMA', 
            'Mirror_MotionUnion',
        ]

        if add_humoto_data:
            target_datasets.append('HUMOTO')
        if add_motiongv_data:
            target_datasets.append('MotionGV')
            target_datasets.append('Mirror_MotionGV')
        
        self.selected_dataset = selected_dataset
        if selected_dataset is not None:
            assert selected_dataset in ['HUMOTO', 'LINGO', 'ParaHome', 'humanml', 'imuposer', 'dipimu']
            target_datasets = [selected_dataset]

        # Other dataset
        split_file_1 = f"{self.root}/splits/t2m_{self.split}.txt"
        all_data_list_1 = [line.strip() for line in open(split_file_1).readlines()]
        all_data_list_1 = filter_by_targets(all_data_list_1, target_datasets)
        split_file_2 = f"{self.root}/splits/tokenizer_{self.split}.txt"
        all_data_list_2 = [line.strip() for line in open(split_file_2).readlines()]
        all_data_list_2 = filter_by_targets(all_data_list_2, target_datasets)

        # Use sorted() for deterministic order across runs (set iteration order is undefined in Python)
        all_data_list = sorted(set(all_data_list_1 + all_data_list_2))
        print(f'Total number of data: {len(all_data_list)}')
        all_data_list = [data.replace('/', '_') for data in all_data_list]
 
        # Humoto dataset
        if 'HUMOTO' in target_datasets:
            self.humoto_root = humoto_root
            humoto_all_data_list = np.load(f"{self.humoto_root}/{self.split}_indices.npy").tolist()
            humoto_all_data_list = ['humoto_{}'.format(k) for k in humoto_all_data_list]
            all_data_list = all_data_list + humoto_all_data_list
        if 'ParaHome' in target_datasets:
            self.parahome_root = parahome_root
            parahome_all_data_list = np.load(f"{self.parahome_root}/{self.split}_split.npy").tolist()
            parahome_all_data_list = ['parahome_{}'.format(k) for k in parahome_all_data_list]
            all_data_list = all_data_list + parahome_all_data_list
        if 'imuposer' in target_datasets:
            self.imuposer_root = imuposer_root
            imuposer_all_data_list = []
            if self.split == 'train':
                for i in range(0, 9):
                    files = glob.glob(f"{self.imuposer_root}/P{i}/*.pkl")
                    imuposer_all_data_list.extend(files)
            else:
                for i in range(9, 11):
                    files = glob.glob(f"{self.imuposer_root}/P{i}/*.pkl")
                    imuposer_all_data_list.extend(files)
            all_data_list = all_data_list + imuposer_all_data_list
        if 'dipimu' in target_datasets:
            self.dipimu_root = dipimu_root
            dipimu_all_data_list = []
            if self.split == 'train':
                for i in range(0, 9):
                    files = glob.glob(f"{self.dipimu_root}/s_{str(i).zfill(2)}/*.pkl")
                    dipimu_all_data_list.extend(files)
            else:
                for i in range(9, 11):
                    files = glob.glob(f"{self.dipimu_root}/s_{str(i).zfill(2)}/*.pkl")
                    dipimu_all_data_list.extend(files)
            # import pdb; pdb.set_trace()
            all_data_list = all_data_list + dipimu_all_data_list

        if selected_dataset == 'HUMOTO':
            assert 'HUMOTO' in target_datasets, "Humoto data is not added. Please add humoto data."
            all_data_list = humoto_all_data_list
        if selected_dataset == 'ParaHome':
            assert 'ParaHome' in target_datasets, "ParaHome data is not added. Please add ParaHome data."
            all_data_list = parahome_all_data_list
        if selected_dataset == 'imuposer':
            assert 'imuposer' in target_datasets
            all_data_list = imuposer_all_data_list
        if selected_dataset == 'dipimu':
            assert 'dipimu' in target_datasets
            all_data_list = dipimu_all_data_list

        # Shuffle the data for random splitting
        if shuffle_list:
            np.random.seed(self.seed) # Set seed for reproducibility
            np.random.shuffle(all_data_list)
        else:
            all_data_list = sorted(all_data_list) # sort the data list

        # assert not overfit, "Overfit is not supported for full dataset."
        # Calculate split index
        # import pdb; pdb.set_trace()
        if overfit:
            sample_num = 1
            self.random_cut = False
            self.data = all_data_list[:sample_num]
        else:
            self.data = all_data_list
        
        self.rest_pelvis = np.array([ 0.00312326, -0.35140744,  0.01203655], dtype=np.float32)

        print(f"IMU dataset loaded. Split: '{self.split}', Number of samples: {len(self.data)}")

        self.return_path_only = return_path_only

    def __len__(self):
        return len(self.data)
    
    def _set_cut_length(self, cut_length: int):
        self.cut_length = cut_length
    
    def enable_random_cut(self):
        assert self.split == 'train', "only enable random cut for train split."
        self.random_cut = True
    
    def disable_random_cut(self):
        self.random_cut = False
    
    def __getitem__(self, idx):

        sample_idx = self.data[idx]
        if self.random_cut:
            assert self.cut_length is not None, "cut_length must be set when random_cut is True"

        # try:
        if isinstance(sample_idx, str) and 'humoto' in sample_idx:
            # load from humoto path
            data_source = 'humoto'
            sample_idx = int(sample_idx.split('_')[1])
            if self.dynamic_object:
                sample_path = f"{self.humoto_root}/all_time/{sample_idx:07d}.pkl" 
            else:
                sample_path = f"{self.humoto_root}/all/{sample_idx:07d}.pkl" 
            add_ground_data = False
            with open(sample_path, 'rb') as f:
                sample = pickle.load(f)

            # some processing
            sample['motion_smpl'][:, 72:75] = sample['motion_smpl'][:, 72:75] + self.rest_pelvis
            sample['imu_traj'][:, :, 3:6] = sample['imu_traj'][:, :, 3:6] + self.rest_pelvis
            gt_objects = sample['objects']
            gt_objects_new = gt_objects.copy()
            if self.dynamic_object:
                for obj_name, obj_data in gt_objects.items():
                    obj_data = obj_data.astype(np.float32)
                    obj_data[:, 5] = obj_data[:, 5] + self.rest_pelvis[1]
                    gt_objects_new[obj_name] = obj_data
            else:
                for obj_name, obj_data in gt_objects.items():
                    obj_data = obj_data.astype(np.float32)
                    obj_data[5] = obj_data[5] + self.rest_pelvis[1]
                    gt_objects_new[obj_name] = obj_data
            sample['objects'] = gt_objects_new

        elif isinstance(sample_idx, str) and 'parahome' in sample_idx:
            # load from parahome path
            data_source = 'parahome'
            sample_idx = sample_idx.split('_')[1].split('.')[0]
            add_ground_data = True

            # import pdb; pdb.set_trace()
            imu_traj_path = f"{self.parahome_root}/imu_traj/{sample_idx}.npy"
            imu_traj = np.load(imu_traj_path, allow_pickle=True)
            motion_smpl_path = f"{self.parahome_root}/motions_smpl85/{sample_idx}.npy"
            motion_smpl = np.load(motion_smpl_path, allow_pickle=True)
            text_path = f"{self.parahome_root}/text_annotations/{sample_idx}.json"
            text = json.load(open(text_path, 'r'))

            sample = {'objects': {}} # since we learn a global transformation, no need to add other objects
            sample['motion_smpl'] = motion_smpl # [n, 85]
            sample['imu_traj'] = imu_traj # [n, 6, 6]
            sample['text'] = text # a dictionary of text 
            
        elif isinstance(sample_idx, str) and 'imuposer' in sample_idx:
            data_source = 'imuposer'
            add_ground_data = True
            with open(sample_idx, 'rb') as f:
                data = pickle.load(f)
            
            sample = {}
            smplx_params = data['smplx_params']
            _orient = smplx_params['global_orient']  # (N, 3)
            _pose = smplx_params['body_pose'] # (N, 63)
            _transl = smplx_params['transl']  # (N, 3)
            sample['motion_smpl'] = np.concatenate([_orient, _pose, _transl], axis=-1)  # (N, 69)
            assert sample['motion_smpl'].shape[1] == 69, f"Invalid motion_smpl shape: {sample['motion_smpl'].shape}"
            sample['imu_data'] = data['imu_data']

            # import pdb; pdb.set_trace()
        
        elif isinstance(sample_idx, str) and 'DIP_IMU' in sample_idx:
            data_source = 'dipimu'
            add_ground_data = True

            with open(sample_idx, 'rb') as f:
                data = pickle.load(f)
            
            sample = {}
            # print(data.keys())
            smplx_params = data['smplx_params']
            smpl_orient = smplx_params['global_orient'] # (N, 3)
            smpl_transl = smplx_params['transl'] # (N, 3)
            smpl_pose = smplx_params['body_pose'] # (N, 63)
            sample['motion_smpl'] = np.concatenate([smpl_orient, smpl_pose, smpl_transl], axis=-1) # (N, 69)
            assert sample['motion_smpl'].shape[1] == 69, f"Invalid motion_smpl shape: {sample['motion_smpl'].shape}"
            sample['imu_acc'] = data['imu_acc']
            sample['imu_ori'] = data['imu_ori']
            # import pdb; pdb.set_trace()
            
        else:
            data_source = 'other_dataset'
            sample_path = os.path.join(self.root, 'motion_data', self.split, sample_idx + '.pkl')
            if not os.path.exists(sample_path):
                sample_path = os.path.join(self.root, self.split, sample_idx + '.pkl')
            add_ground_data = True
        
            with open(sample_path, 'rb') as f:
                sample = pickle.load(f)
 

        if data_source == 'imuposer':
            assert not self.motion_only
            assert not self.scene_only
            assert not self.add_imu_noise, 'realworld data already has noise'
            smooth_imu_acc = True
            sample_output = process_imuposer_data(sample, self.random_cut, self.random_mask_text, 
                                                self.cut_length, shift=self.shift, 
                                                add_ground_data=add_ground_data,
                                                filter_short_text=(self.selected_dataset is None),
                                                data_source=data_source, 
                                                dynamic_object=self.dynamic_object,
                                                fps=self.fps, 
                                                split=self.split,
                                                IMUSEQMAXLEN=self.IMUSEQMAXLEN,
                                                sample_idx=sample_idx,
                                                smooth_imu_acc=smooth_imu_acc)
        elif data_source == 'dipimu':
            assert not self.motion_only
            assert not self.scene_only
            assert not self.add_imu_noise, 'realworld data already has noise'
            smooth_imu_acc = True
            sample_output = process_dipimu_data(sample, self.random_cut, self.random_mask_text, 
                                                self.cut_length, shift=self.shift, 
                                                add_ground_data=add_ground_data,
                                                filter_short_text=(self.selected_dataset is None),
                                                data_source=data_source, 
                                                dynamic_object=self.dynamic_object,
                                                fps=self.fps,
                                                split=self.split,
                                                IMUSEQMAXLEN=self.IMUSEQMAXLEN,
                                                sample_idx=sample_idx,
                                                acc_scale=self.acc_scale,
                                                gyro_scale=self.gyro_scale,
                                                smooth_imu_acc=smooth_imu_acc)
        else:
            sample_output = process_imu_data(sample, self.random_cut, self.random_mask_text, 
                                            self.cut_length, shift=self.shift, 
                                            add_ground_data=add_ground_data,
                                            filter_short_text=(self.selected_dataset is None),
                                            motion_only=self.motion_only,
                                            scene_only=self.scene_only,
                                            data_source=data_source, 
                                            dynamic_object=self.dynamic_object,
                                            fps=self.fps,
                                            split=self.split,
                                            IMUSEQMAXLEN=self.IMUSEQMAXLEN,
                                            add_imu_noise=self.add_imu_noise,
                                            acc_scale=self.acc_scale,
                                            gyro_scale=self.gyro_scale)
        
        if sample_output is None:
            # not meeting the minimum length requirement
            return self.__getitem__((idx + 1) % len(self))

        sample_output['sample_idx'] = sample_idx
        return sample_output
    
    def collate_fn(self, batch):
        return batch

def set_cut_length(dataloader, cut_min_length: int, cut_max_length: int):
    cut_length = random.randint(cut_min_length, cut_max_length)
    dataloader.dataset._set_cut_length(cut_length)

if __name__ == "__main__":

    selected_dataset = 'LINGO'
    dataset = IMUDataset(
        split='train',
        root='/shared/perception/datasets/imu_data/final_data_per_sequence',
        seed=42,
        overfit=False,
        random_cut=False,
        selected_dataset=selected_dataset,
        return_path_only=True,
        shift=2,
        dynamic_object=False,
        IMUSEQMAXLEN=50,
    )

    print(len(dataset))

    imu_data_all = []

    import tqdm
    for i in tqdm.tqdm(range(len(dataset))):
        sample_path = dataset[i]
        sample = dataset.__getitem__(i)

        # import pdb; pdb.set_trace()

        imu_data = sample['imu_data']
        imu_data_all.append(imu_data)

        # print(sample_path)
        print(f"IMU data shape: {sample['imu_data'].shape}")
        print(f"Orientation shape: {sample['orient'].shape}")
        print(f"Translation shape: {sample['transl'].shape}")

        if i > 10:
            break
    imu_data_all = np.concatenate(imu_data_all, axis=0)
    for i in range(15):
        v = imu_data_all[..., i]
        print(f"IMU data {i} shape: {v.shape}")
        print(f"IMU data {i} mean: {v.mean()}")
        print(f"IMU data {i} std: {v.std()}")
        print(f"IMU data {i} min: {v.min()}")
        print(f"IMU data {i} max: {v.max()}")
        print(f"IMU data {i} median: {np.median(v)}")

    import pdb; pdb.set_trace()
    # imu_std = np.std(imu_data_all, axis=0, keepdims=True)
    # np.save(f'/scratch/bfyo/tcheng1/imu_std.npy', imu_std)
    #     # gt_text[i] = sample_path
    #     print(f"Description: {sample['description']}")
    #     # gt_text[i] = sample['description']
    #     # print(f"IMU data shape: {sample['imu_data'].shape}, dtype: {sample['imu_data'].dtype}, device: {sample['imu_data'].device}")
    #     # print(f"Orientation shape: {sample['orient'].shape}, dtype: {sample['orient'].dtype}, device: {sample['orient'].device}")
    #     # print(f"Translation shape: {sample['transl'].shape}, dtype: {sample['transl'].dtype}, device: {sample['transl'].device}")
    #     # print(f"Pose shape: {sample['pose'].shape}, dtype: {sample['pose'].dtype}, device: {sample['pose'].device}")
    #     # print(f"Objects: {sample['objects'].keys()}")

    #     # for obj_key in sample['objects'].keys():
    #     #     print(obj_key)
    #         # print(f"Object {obj_key}: rot dtype: {sample['objects'][obj_key]['rot'].dtype}, transl dtype: {sample['objects'][obj_key]['transl'].dtype}, bbox dtype: {sample['objects'][obj_key]['bbox'].dtype}")
    #         # print(f"Object {obj_key}: rot device: {sample['objects'][obj_key]['rot'].device}, transl device: {sample['objects'][obj_key]['transl'].device}, bbox device: {sample['objects'][obj_key]['bbox'].device}")
    #         # break

    # # # save the gt_text to a pickle file
    # # with open(f'/scratch/benk/tcheng1/gt_text/{selected_dataset}_gt_text.pkl', 'wb') as f:
    # #     pickle.dump(gt_text, f)
    # # print(f"Saved gt_text to /scratch/benk/tcheng1/gt_text/{selected_dataset}_gt_text.pkl")