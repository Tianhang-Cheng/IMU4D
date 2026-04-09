import os
import numpy as np
import torch
import gc
from scipy.spatial.transform import Rotation as R

from utils.rotation import convert_rotation

def get_norm_params(data, normalization_type="quantile"):
    """
    Compute normalization parameters for the given data.

    Args:
        data (np.ndarray): Input data to compute normalization parameters.
        normalization_type (str): "standard", "minmax" or "quantile".

    Returns:
        dict: Normalization parameters.
    """
    if normalization_type == "minmax":
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        return {"min_vals": min_vals, "max_vals": max_vals, "type": "minmax"}
    elif normalization_type == "quantile":
        p1_vals = np.percentile(data, 1, axis=0)
        p99_vals = np.percentile(data, 99, axis=0)
        return {"p1_vals": p1_vals, "p99_vals": p99_vals, "type": "quantile"}
    elif normalization_type == "standard":
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        return {"mean_vals": mean_vals, "std_vals": std_vals, "type": "standard"}
    else:
        raise ValueError("normalization_type must be 'minmax', 'quantile' or 'standard'")


def normalize_data(data, norm_params, eps=1e-8):
    """
    Normalize data to [-1, 1] range using precomputed normalization parameters.

    Args:
        data (np.ndarray): Input data to normalize.
        norm_params (dict): Precomputed normalization parameters.

    Returns:
        np.ndarray: Normalized data.
    """
    if norm_params["type"] == "minmax":
        min_vals = norm_params["min_vals"]
        max_vals = norm_params["max_vals"]
        range_vals = max_vals - min_vals
        return 2 * (data - min_vals) / (range_vals + eps) - 1
    elif norm_params["type"] == "quantile":
        p1_vals = norm_params["p1_vals"]
        p99_vals = norm_params["p99_vals"]
        range_vals = p99_vals - p1_vals
        return 2 * (data - p1_vals) / (range_vals + eps) - 1
    elif norm_params["type"] == "standard":
        mean_vals = norm_params["mean_vals"]
        std_vals = norm_params["std_vals"]
        return (data - mean_vals) / (std_vals + eps)
    else:
        raise ValueError(f"Unknown normalization type: {norm_params['type']}")


def denormalize_data(data, norm_params):
    """
    Denormalize data from [-1, 1] range back to original scale.

    Args:
        data (np.ndarray): Normalized data.
        norm_params (dict): Normalization parameters.

    Returns:
        np.ndarray: Denormalized data.
    """
    if norm_params["type"] == "minmax":
        min_vals = norm_params["min_vals"]
        max_vals = norm_params["max_vals"]
        range_vals = max_vals - min_vals
        return (data + 1) / 2 * range_vals + min_vals
    elif norm_params["type"] == "quantile":
        p1_vals = norm_params["p1_vals"]
        p99_vals = norm_params["p99_vals"]
        range_vals = p99_vals - p1_vals
        return (data + 1) / 2 * range_vals + p1_vals
    elif norm_params["type"] == "standard":
        mean_vals = norm_params["mean_vals"]
        std_vals = norm_params["std_vals"]
        return data * std_vals + mean_vals
    else:
        raise ValueError(f"Unknown normalization type: {norm_params['type']}")


def chunk_sequences(sequences, chunk_length, pad_value=None):
    """
    Split sequences into chunks of specified length with optional padding.

    Args:
        sequences (list): List of numpy arrays representing sequences.
        chunk_length (int): Length of each chunk.
        pad_value (float): Value to use for padding (default: None).

    Returns:
        list: List of chunked sequences.
    """
    chunked_sequences = []
    for seq in sequences:
        seq_len, feature_dim = seq.shape
        if seq_len < chunk_length:
            continue
        num_chunks = (seq_len + chunk_length - 1) // chunk_length
        for i in range(num_chunks):
            start_idx = i * chunk_length
            end_idx = min(start_idx + chunk_length, seq_len)
            chunk = seq[start_idx:end_idx]
            if chunk.shape[0] < chunk_length and pad_value is not None:
                padding = np.full((chunk_length - chunk.shape[0], feature_dim), pad_value, dtype=chunk.dtype)
                chunk = np.concatenate([chunk, padding], axis=0)
            if chunk.shape[0] == chunk_length:  # Only append if chunk is valid
                chunked_sequences.append(chunk)
    return chunked_sequences


def chunk_sequences_overlap(sequences, chunk_length, overlap=0.5):
    """
    Split sequences into overlapping chunks of specified length.

    Args:
        sequences (list): List of numpy arrays representing sequences.
        chunk_length (int): Length of each chunk.
        overlap (float): Fraction of overlap between consecutive chunks.

    Returns:
        list: List of chunked sequences with specified overlap.
    """
    chunked_sequences = []
    for seq in sequences:
        seq_len, feature_dim = seq.shape
        if seq_len < chunk_length:
            continue
        step_size = max(1, int(chunk_length * (1 - overlap)))
        for start_idx in range(0, seq_len - chunk_length + 1, step_size):
            end_idx = start_idx + chunk_length
            chunk = seq[start_idx:end_idx]
            chunked_sequences.append(chunk)
    return chunked_sequences


def align_poses_to_first_frame(orient, transl, pose):
    """
    Align poses to the first frame and return transformation parameters
    
    Args:
        orient (np.ndarray): Orientation data (N, 3)
        transl (np.ndarray): Translation data (N, 3)
        pose (np.ndarray): Pose data (N, pose_dim)
    
    Returns:
        tuple: (aligned_orient, aligned_transl, aligned_pose, inv_rotation, inv_translation)
    """
    # Get first frame transformation for alignment
    first_orient = orient[0]
    first_transl = transl[0]
    first_orient_matrix = R.from_rotvec(first_orient).as_matrix()

    # Extract the yaw component of the first frame's orientation
    yaw = np.arctan2(first_orient_matrix[0, 2], first_orient_matrix[2, 2])
    yaw_matrix = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    first_orient_matrix = yaw_matrix

    # Transformation to make first frame pelvis the origin and orient toward Z+
    inv_translation = -first_transl
    inv_rotation = np.linalg.inv(first_orient_matrix)

    # Transform translations
    aligned_transl = inv_rotation @ (transl + inv_translation[np.newaxis, :]).T
    aligned_transl = aligned_transl.T
    
    # Transform orientations
    orient_matrices = R.from_rotvec(orient).as_matrix()
    transformed_matrices = np.einsum('ij,njk->nik', inv_rotation, orient_matrices)
    aligned_orient = R.from_matrix(transformed_matrices).as_rotvec()

    # Pose remains unchanged in this alignment (only global transformation)
    aligned_pose = pose.copy()

    return aligned_orient, aligned_transl, aligned_pose, inv_rotation, inv_translation


def load_dataset(data_dir, split="train", split_ratio=0.9, norm_options='quantile', group_std=False, align_first_frame=True, use_6d_rotation=True):
    """
    Load the original lingomotion dataset and perform train-val split.

    Args:
        data_dir (str): Path to the IMU dataset directory.
        split (str): "train" or "val" to specify the dataset split.
        split_ratio (float): Ratio of data to use for training (default: 0.9).
        norm_options (str): Normalization options for the motion data ("quantile", "standard", "minmax").
        group_std (bool): Whether to use group standard deviation for similar motion features.
        align_first_frame (bool): Whether to align each sequence to its first frame.
        use_6d_rotation (bool): Whether to convert rotation vectors to 6D representation.

    Returns:
        tuple: (sequences, normalization_params) for the specified split.
    """
    start_idx = np.load(os.path.join(data_dir, 'start_idx.npy'))
    end_idx = np.load(os.path.join(data_dir, 'end_idx.npy'))
    orient = np.load(os.path.join(data_dir, 'human_orient.npy'))
    transl = np.load(os.path.join(data_dir, 'transl_aligned.npy'))
    pose = np.load(os.path.join(data_dir, 'human_pose.npy'))

    # Align all sequences to their first frame if requested
    aligned_sequences = []
    for st, en in zip(start_idx, end_idx):
        seq_orient = orient[st:en]
        seq_transl = transl[st:en]
        seq_pose = pose[st:en]
        
        if align_first_frame:
            seq_orient, seq_transl, seq_pose, _, _ = align_poses_to_first_frame(seq_orient, seq_transl, seq_pose)
        
        if use_6d_rotation:
            seq_orient = torch.from_numpy(seq_orient).float()
            seq_orient = convert_rotation(seq_orient, 'aa', '6d').numpy()
            seq_pose = torch.from_numpy(seq_pose.reshape(-1, 21, 3)).float()
            seq_pose = convert_rotation(seq_pose, 'aa', '6d').numpy().reshape(-1, 126)
        else:
            seq_orient = R.from_rotvec(seq_orient).as_rotvec() / np.pi
            seq_pose = R.from_rotvec(seq_pose).as_rotvec() / np.pi

        aligned_sequences.append({
            'orient': seq_orient,
            'transl': seq_transl,
            'pose': seq_pose
        })

    if group_std and norm_options != "standard":
        raise ValueError("group_std can only be used with standard normalization")
    
    all_seq_data = []
    seq_len_list = []
    for seq in aligned_sequences:
        seq_data = np.concatenate([seq['orient'], seq['transl'], seq['pose']], axis=1)
        all_seq_data.append(seq_data)
        seq_len = seq['orient'].shape[0]
        seq_len_list.append(seq_len)
    all_seq_data = np.concatenate(all_seq_data, axis=0).astype(np.float32)

    if use_6d_rotation:
        assert all_seq_data.shape[1] == 135, "Each sequence must have 135 features (6 for orient, 3 for transl, 126 for pose)"
    else:
        assert all_seq_data.shape[1] == 69, "Each sequence must have 69 features (3 for orient, 3 for transl, 63 for pose)"

    # Compute normalization parameters
    orient_norm_params = get_norm_params(all_seq_data[:, :6], normalization_type=norm_options)   # TODO: now assume 6D rotation is used
    transl_norm_params = get_norm_params(all_seq_data[:, 6:9], normalization_type=norm_options)
    pose_norm_params = get_norm_params(all_seq_data[:, 9:], normalization_type=norm_options)

    if group_std:
        orient_norm_params['std_vals'] = np.full(orient_norm_params['std_vals'].shape[0], np.mean(orient_norm_params['std_vals']) / 1.0)
        transl_norm_params['std_vals'] = np.full(transl_norm_params['std_vals'].shape[0], np.mean(transl_norm_params['std_vals']) / 1.0)
        pose_norm_params['std_vals'] = np.full(pose_norm_params['std_vals'].shape[0], np.mean(pose_norm_params['std_vals']) / 1.0)

    all_seq_data[:, :6] = normalize_data(all_seq_data[:, :6], orient_norm_params)
    all_seq_data[:, 6:9] = normalize_data(all_seq_data[:, 6:9], transl_norm_params)
    all_seq_data[:, 9:] = normalize_data(all_seq_data[:, 9:], pose_norm_params)

    all_seq = []
    start_idx = 0
    for seq_len in seq_len_list:
        seq_data = all_seq_data[start_idx:start_idx+seq_len]
        all_seq.append(seq_data)
        start_idx += seq_len

    norm_params = {
        "orient": orient_norm_params,
        "transl": transl_norm_params,
        "pose": pose_norm_params
    }

    # Split data
    split_idx = int(len(all_seq) * split_ratio)
    if split == "train":
        return all_seq[:split_idx], norm_params
    elif split == "val":
        return all_seq[split_idx:], norm_params
    else:
        raise ValueError("split must be 'train' or 'val'")


def get_dummy_norm_params(dims):
    return {"mean_vals": np.zeros(dims), "std_vals": np.ones(dims), "type": "standard"}


def load_dataset_smpl_272(data_dir, split='train', split_ratio=0.9, norm_options='quantile', norm_all=False, group_std=False, motion_type='full'):
    """
    Load the processed lingomotion dataset in smpl-272 format and perform train-val split.

    Args:
        data_dir (str): Path to the dataset directory.
        split (str): "train" or "val" to specify the dataset split.
        split_ratio (float): Ratio of data to use for training (default: 0.9).
        norm_options (str): Normalization options for the motion data ("quantile", "standard", "minmax").
        norm_all (bool): Whether to normalize all features or only on non-rotational features.
        group_std (bool): Whether to use group standard deviation for similar motion features.
        motion_type (str): Type of motion representation to use (e.g., "minimal", "full").

    Returns:
        tuple: (sequences, normalization_params) for the specified split.
    """
    # SMPL_272 Representation:
    # - velocity_xz: 2
    # - root_rot_diff: 6 (relative rotation in 6d)
    # - positions_local: 66 (22 joints * 3)
    # - velocities_local: 66 (22 joints * 3)
    # - rotations_local: 132 (22 joints * 6)

    smpl272_f_list = sorted(os.listdir(data_dir))
    
    all_seq_data = []
    seq_len_list = []
    for f_name in smpl272_f_list:
        smpl272_data = np.load(os.path.join(data_dir, f_name))
        all_seq_data.append(smpl272_data)
        seq_len_list.append(smpl272_data.shape[0])
    all_seq_data = np.concatenate(all_seq_data, axis=0).astype(np.float32)

    assert all_seq_data.shape[1] == 272, "Each sequence must have 272 features"

    if group_std and norm_options != "standard":
        raise ValueError("group_std can only be used with standard normalization")

    # Normalization
    joints_num = 22

    velocity_xz_norm_params = get_norm_params(all_seq_data[:, :2], normalization_type=norm_options)
    positions_local_norm_params = get_norm_params(all_seq_data[:, 8:8+66], normalization_type=norm_options)
    velocities_local_norm_params = get_norm_params(all_seq_data[:, 74:74+66], normalization_type=norm_options)

    # Include rotational features for normalization
    if norm_all:
        root_rot_diff_norm_params = get_norm_params(all_seq_data[:, 2:8], normalization_type=norm_options)
        rotations_local_norm_params = get_norm_params(all_seq_data[:, 8+6*joints_num:8+12*joints_num], normalization_type=norm_options)
    else:
        # Use 0-mean, 1-std normalization
        root_rot_diff_norm_params = get_dummy_norm_params(6)
        rotations_local_norm_params = get_dummy_norm_params(132)

    if group_std:
        # Group standard deviation for similar motion features (adopted from https://github.com/Li-xingXiao/272-dim-Motion-Representation/blob/master/cal_mean_std.py)
        velocity_xz_norm_params['std_vals'] = np.full(velocity_xz_norm_params['std_vals'].shape[0], np.mean(velocity_xz_norm_params['std_vals']) / 1.0)
        root_rot_diff_norm_params['std_vals'] = np.full(root_rot_diff_norm_params['std_vals'].shape[0], np.mean(root_rot_diff_norm_params['std_vals']) / 1.0)
        if motion_type == 'minimal':
            positions_local_norm_params['std_vals'] = np.full(positions_local_norm_params['std_vals'].shape[0], positions_local_norm_params['std_vals'][1] / 1.0)  # use local y height
        else:
            positions_local_norm_params['std_vals'] = np.full(positions_local_norm_params['std_vals'].shape[0], np.mean(positions_local_norm_params['std_vals']) / 1.0)
        velocities_local_norm_params['std_vals'] = np.full(velocities_local_norm_params['std_vals'].shape[0], np.mean(velocities_local_norm_params['std_vals']) / 1.0)
        rotations_local_norm_params['std_vals'] = np.full(rotations_local_norm_params['std_vals'].shape[0], np.mean(rotations_local_norm_params['std_vals']) / 1.0)
        
    # root xz velocity, local positions, local velocities
    all_seq_data[:, :2] = normalize_data(all_seq_data[:, :2], velocity_xz_norm_params)
    all_seq_data[:, 8:8+3*joints_num] = normalize_data(all_seq_data[:, 8:8+3*joints_num], positions_local_norm_params)
    all_seq_data[:, 8+3*joints_num:8+6*joints_num] = normalize_data(all_seq_data[:, 8+3*joints_num:8+6*joints_num], velocities_local_norm_params)

    # root rotation difference, local rotations
    all_seq_data[:, 2:8] = normalize_data(all_seq_data[:, 2:8], root_rot_diff_norm_params)
    all_seq_data[:, 8+6*joints_num:8+12*joints_num] = normalize_data(all_seq_data[:, 8+6*joints_num:8+12*joints_num], rotations_local_norm_params)

    norm_params = {
        "velocity_xz": velocity_xz_norm_params,
        "root_rot_diff": root_rot_diff_norm_params,
        "positions_local": positions_local_norm_params,
        "velocities_local": velocities_local_norm_params,
        "rotations_local": rotations_local_norm_params
    }

    all_seq = []
    start_idx = 0
    for seq_len in seq_len_list:
        seq_data = all_seq_data[start_idx:start_idx+seq_len]
        all_seq.append(seq_data)
        start_idx += seq_len

    assert len(all_seq) == len(smpl272_f_list), "Mismatch in number of sequences"

    gc.collect()

    # Split data
    split_idx = int(len(all_seq) * split_ratio)
    if split == "train":
        return all_seq[:split_idx], norm_params
    elif split == "val":
        return all_seq[split_idx:], norm_params
    else:
        raise ValueError("split must be 'train' or 'val'")
    

def load_dataset_smpl_272_minimal(data_dir, split='train', split_ratio=0.9, norm_options='quantile', group_std=False):
    """
    Load the processed lingomotion dataset in the minimal version of smpl-272 format and perform train-val split.

    Args:
        data_dir (str): Path to the dataset directory.
        split (str): "train" or "val" to specify the dataset split.
        split_ratio (float): Ratio of data to use for training (default: 0.9).
        norm_options (str): Normalization options for the motion data ("quantile", "standard", "minmax").
        group_std (bool): Whether to use group standard deviation for similar motion features.

    Returns:
        tuple: (sequences, normalization_params) for the specified split.
    """
    # SMPL_272 Representation:
    # - velocity_xz: 2
    # - root_rot_diff: 6 (relative rotation in 6d)
    # - positions_local: 66 (22 joints * 3)
    # - velocities_local: 66 (22 joints * 3)
    # - rotations_local: 132 (22 joints * 6)

    # Minimal Representation:
    # - velocity_xz: 2
    # - root_rot_diff: 1 (in θ format)
    # - local_y_height: 1
    # - root_rotations_local: 6 (in 6D format)
    # - rotations_local: 126 (21 joints * 6)

    smpl272_f_list = sorted(os.listdir(data_dir))
    
    all_seq_data = []
    seq_len_list = []
    for f_name in smpl272_f_list:
        smpl272_data = np.load(os.path.join(data_dir, f_name))

        velocity_xz = smpl272_data[:, :2]  # root xz velocity
        root_rot_diff_theta = np.arctan2(-smpl272_data[:, 4], smpl272_data[:, 2]) / np.pi * 180.0  # the 6d root_rot_diff be like: [cos(θ), 0, -sin(θ); 0, 1, 0]
        local_y_height = smpl272_data[:, 9:10]  # local y height
        root_rotations_local = smpl272_data[:, 140:146]  # root local rotations (6-dim)
        rotations_local = smpl272_data[:, 146:]  # local rotations (21 joints * 6)

        smpl272_minimal_data = np.concatenate([velocity_xz, root_rot_diff_theta[:, np.newaxis], local_y_height, root_rotations_local, rotations_local], axis=1)

        all_seq_data.append(smpl272_minimal_data)
        seq_len_list.append(smpl272_minimal_data.shape[0])
    all_seq_data = np.concatenate(all_seq_data, axis=0).astype(np.float32)

    assert all_seq_data.shape[1] == 136, "Each sequence must have 136 features"

    if group_std and norm_options != "standard":
        raise ValueError("group_std can only be used with standard normalization")

    # Normalization
    joints_num = 22

    velocity_xz_norm_params = get_norm_params(all_seq_data[:, :2], normalization_type=norm_options)
    root_rot_diff_theta_norm_params = get_norm_params(all_seq_data[:, 2:3], normalization_type=norm_options)
    local_y_height_norm_params = get_norm_params(all_seq_data[:, 3:4], normalization_type=norm_options)
    root_rotations_local_norm_params = get_norm_params(all_seq_data[:, 4:10], normalization_type=norm_options)
    rotations_local_norm_params = get_norm_params(all_seq_data[:, 10:], normalization_type=norm_options)

    if group_std:
        # Group standard deviation for similar motion features (adopted from https://github.com/Li-xingXiao/272-dim-Motion-Representation/blob/master/cal_mean_std.py)
        # ignore root_rot_diff_theta and local_y_height since they only have one dimension
        velocity_xz_norm_params['std_vals'] = np.full(velocity_xz_norm_params['std_vals'].shape[0], np.mean(velocity_xz_norm_params['std_vals']) / 1.0)
        root_rotations_local_norm_params['std_vals'] = np.full(root_rotations_local_norm_params['std_vals'].shape[0], np.mean(root_rotations_local_norm_params['std_vals']) / 1.0)
        rotations_local_norm_params['std_vals'] = np.full(rotations_local_norm_params['std_vals'].shape[0], np.mean(rotations_local_norm_params['std_vals']) / 1.0)

    # root xz velocity, local positions, local velocities
    all_seq_data[:, :2] = normalize_data(all_seq_data[:, :2], velocity_xz_norm_params)
    all_seq_data[:, 2:3] = normalize_data(all_seq_data[:, 2:3], root_rot_diff_theta_norm_params)
    all_seq_data[:, 3:4] = normalize_data(all_seq_data[:, 3:4], local_y_height_norm_params)
    all_seq_data[:, 4:10] = normalize_data(all_seq_data[:, 4:10], root_rotations_local_norm_params)
    all_seq_data[:, 10:] = normalize_data(all_seq_data[:, 10:], rotations_local_norm_params)

    norm_params = {
        "velocity_xz": velocity_xz_norm_params,
        "root_rot_diff_theta": root_rot_diff_theta_norm_params,
        "local_y_height": local_y_height_norm_params,
        "root_rotations_local": root_rotations_local_norm_params,
        "rotations_local": rotations_local_norm_params
    }

    all_seq = []
    start_idx = 0
    for seq_len in seq_len_list:
        seq_data = all_seq_data[start_idx:start_idx+seq_len]
        all_seq.append(seq_data)
        start_idx += seq_len

    assert len(all_seq) == len(smpl272_f_list), "Mismatch in number of sequences"

    gc.collect()

    # Split data
    split_idx = int(len(all_seq) * split_ratio)
    if split == "train":
        return all_seq[:split_idx], norm_params
    elif split == "val":
        return all_seq[split_idx:], norm_params
    else:
        raise ValueError("split must be 'train' or 'val'")