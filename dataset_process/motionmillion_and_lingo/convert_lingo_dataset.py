"""
This script converts the Lingo dataset into a format suitable for training.
"""
import os
import numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from dataset_process.motionmillion_and_lingo.smpl272_utils.motion_conversion import smpl85_to_smpl272
from utils.human import load_smplx_model

def load_lingo_dataset(data_dir):

    start_idx = np.load(os.path.join(data_dir, 'start_idx.npy'))
    end_idx = np.load(os.path.join(data_dir, 'end_idx.npy'))       # exclusive

    orient = np.load(os.path.join(data_dir, 'human_orient.npy'))
    pose = np.load(os.path.join(data_dir, 'human_pose.npy'))
    transl = np.load(os.path.join(data_dir, 'transl_aligned.npy'))
    texts = pickle.load(open(os.path.join(data_dir, 'text_aug.pkl'), 'rb'))  # action descriptions

    # Load data
    all_seq = []
    for st, en, text in zip(start_idx, end_idx, texts):

        orient_seq = orient[st:en]
        pose_seq = pose[st:en]
        transl_seq = transl[st:en]

        all_seq.append({
            'orient': orient_seq,
            'pose': pose_seq,
            'transl': transl_seq,
            'description': text,
        })

    return all_seq


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

    # original value
    # original_transl = (first_orient_matrix @ aligned_transl.T).T - inv_translation[np.newaxis, :]
    
    # Transform orientations
    orient_matrices = R.from_rotvec(orient).as_matrix()
    transformed_matrices = np.einsum('ij,njk->nik', inv_rotation, orient_matrices)
    aligned_orient = R.from_matrix(transformed_matrices).as_rotvec()

    # Pose remains unchanged in this alignment (only global transformation)
    aligned_pose = pose.copy()

    return aligned_orient, aligned_transl, aligned_pose, inv_rotation, inv_translation


def align_poses_to_frame(orient, transl, pose, ref_frame=0):
    """
    Align poses to a specific reference frame (same logic as align_poses_to_first_frame).
    
    Args:
        orient (np.ndarray): Orientation data (N, 3)
        transl (np.ndarray): Translation data (N, 3)
        pose (np.ndarray): Pose data (N, pose_dim)
        ref_frame (int): Index of the reference frame to align to
    
    Returns:
        tuple: (aligned_orient, aligned_transl, aligned_pose, inv_rotation, inv_translation)
    """
    ref_orient = orient[ref_frame]
    ref_transl = transl[ref_frame]
    ref_orient_matrix = R.from_rotvec(ref_orient).as_matrix()

    yaw = np.arctan2(ref_orient_matrix[0, 2], ref_orient_matrix[2, 2])
    yaw_matrix = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    ref_orient_matrix = yaw_matrix

    inv_translation = -ref_transl
    inv_rotation = np.linalg.inv(ref_orient_matrix)

    aligned_transl = inv_rotation @ (transl + inv_translation[np.newaxis, :]).T
    aligned_transl = aligned_transl.T

    orient_matrices = R.from_rotvec(orient).as_matrix()
    transformed_matrices = np.einsum('ij,njk->nik', inv_rotation, orient_matrices)
    aligned_orient = R.from_matrix(transformed_matrices).as_rotvec()

    aligned_pose = pose.copy()

    return aligned_orient, aligned_transl, aligned_pose, inv_rotation, inv_translation


def transform_aligned_to_reference(aligned_orient, aligned_transl, aligned_pose, ref_orient, ref_transl):
    """
    Transform poses from aligned (local) coordinates back to a reference frame.
    Inverse of align_poses_to_frame.
    
    Args:
        aligned_orient (np.ndarray): Orientation in local frame (N, 3)
        aligned_transl (np.ndarray): Translation in local frame (N, 3)
        aligned_pose (np.ndarray): Pose data (N, pose_dim)
        ref_orient (np.ndarray): Reference orientation (3,)
        ref_transl (np.ndarray): Reference translation (3,)
    
    Returns:
        tuple: (orient, transl, pose) in reference frame
    """
    ref_orient_matrix = R.from_rotvec(ref_orient).as_matrix()
    yaw = np.arctan2(ref_orient_matrix[0, 2], ref_orient_matrix[2, 2])
    ref_orient_matrix = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    transl = (ref_orient_matrix @ aligned_transl.T).T + ref_transl[np.newaxis, :]

    orient_matrices = R.from_rotvec(aligned_orient).as_matrix()
    transformed_matrices = np.einsum('ij,njk->nik', ref_orient_matrix, orient_matrices)
    orient = R.from_matrix(transformed_matrices).as_rotvec()

    return orient, transl, aligned_pose.copy()


if __name__ == '__main__':

    lingo_data_dir = '/projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/data/dataset'
    lingo_dataset = load_lingo_dataset(lingo_data_dir)

    dataset_out_dir = './motionmillion_smpl85/LINGO'
    os.makedirs(dataset_out_dir, exist_ok=True)

    smplx_model = load_smplx_model()

    start_end_dict = {}
    text_description_dict = {}
    current_idx = 0
    full_data_smpl85 = []
    full_data_smpl141 = []

    for i, seq_data in tqdm(enumerate(lingo_dataset), total=len(lingo_dataset), desc='Processing Lingo dataset'):

        # Convert each sequence to SMPL85 format
        nfrm = seq_data['orient'].shape[0]
        orient = seq_data['orient']       # (nfrm, 3)
        pose = seq_data['pose']           # (nfrm, 63)
        transl = seq_data['transl']       # (nfrm, 3)
        text = seq_data['description']

        orient, transl, pose, _, _ = align_poses_to_first_frame(orient, transl, pose)

        data_smpl85 = np.concatenate([orient, pose, np.zeros((nfrm, 6)), transl, np.zeros((nfrm, 10))], axis=-1).astype(np.float32)

        data_smpl272 = smpl85_to_smpl272(data_smpl85, smplx_model)
        data_smpl141 = np.concatenate([
            data_smpl272[:, 0:2],            # velo_xz
            data_smpl272[:, 2:8],            # head_diff
            data_smpl272[:, 9:10],           # height
            data_smpl272[:, 8+6*22:8+12*22], # local rotations (6*22=132)
        ], axis=1)

        f_name = f'LINGO/{i:05d}'

        full_data_smpl85.append(data_smpl85)
        full_data_smpl141.append(data_smpl141)
        start_end_dict[f_name] = (current_idx, current_idx + data_smpl85.shape[0])   # non-inclusive end index
        current_idx += data_smpl85.shape[0]

        # Store text description
        text_description_dict[f_name] = text

    full_data_smpl85 = np.concatenate(full_data_smpl85, axis=0).astype(np.float32)
    full_data_smpl141 = np.concatenate(full_data_smpl141, axis=0).astype(np.float32)
    print(f"Total frames in Lingo dataset: {full_data_smpl85.shape[0]}")
    
    assert full_data_smpl141.shape[0] == full_data_smpl85.shape[0]
    assert full_data_smpl85.shape[1] == 85
    assert full_data_smpl141.shape[1] == 141

    # save the processed data
    np.save(os.path.join(dataset_out_dir, 'motion_smpl85.npy'), full_data_smpl85)
    np.save(os.path.join(dataset_out_dir, 'motion_smpl141.npy'), full_data_smpl141)
    # np.save(os.path.join(dataset_out_dir, 'orient.npy'), full_data_smpl85[:, 0:3])
    # np.save(os.path.join(dataset_out_dir, 'transl.npy'), full_data_smpl85[:, 72:75])
    # np.save(os.path.join(dataset_out_dir, 'pose.npy'), full_data_smpl85[:, 3:66])
    with open(os.path.join(dataset_out_dir, 'start_end.pkl'), 'wb') as f:
        pickle.dump(start_end_dict, f)
    with open(os.path.join(dataset_out_dir, 'texts.pkl'), 'wb') as f:
        pickle.dump(text_description_dict, f)