"""
This script extracts 6DoF IMU trajectories (in 6 locations) using SMPL-85 data from the MotionMillion + LINGO dataset.
"""

import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
import argparse
from dataclasses import dataclass

from imu_simulate_utils.parametric_model import SimplifiedSMPLX
from imu_simulate_utils.rotation import convert_rotation


joint_names_smplx = {
    0: 'pelvis',
    1: 'left_hip',
    2: 'right_hip',
    3: 'spine1',
    4: 'left_knee',
    5: 'right_knee',
    6: 'spine2',
    7: 'left_ankle',
    8: 'right_ankle',
    9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
}


class IMU_Sensor_Config:
    """Configuration for IMU sensors."""
    def __init__(self):
        self.vi_mask = torch.tensor([4576, 7313, 3677, 6438, 9003, 5493])  # Vertices indices for IMU sensors
        self.ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])              # Joint indices for IMU sensors
        self.imu_names = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee', 'head', 'pelvis']


def simulate_imu_trajectory(dataset_dir, motion_output_dir, simplified_smplx_model, rest_pelvis, vi_mask, ji_mask):

    # motion_data = np.load(os.path.join(dataset_dir, 'motion_smpl85.npy'))
    motion_data = np.load(os.path.join(dataset_dir, 'motion_smpl85_from_smpl141.npy'))
    start_end_dict = pickle.load(open(os.path.join(dataset_dir, 'start_end.pkl'), 'rb'))

    imu_trajs = []

    for f_name, (start_idx, end_idx) in tqdm(start_end_dict.items(), desc=f"Processing {dataset_dir.split('/')[-1]}", dynamic_ncols=True):

        data_smpl85 = motion_data[start_idx:end_idx]

        # Adjust the pelvis translation for correct SMPL meshes
        traj_corrected = data_smpl85[:, 72:75] - rest_pelvis

        # Simulate IMU trajectory
        smpl_85_cuda = torch.from_numpy(data_smpl85).cuda()
        with torch.no_grad():
            smplx_output = simplified_smplx_model(
                pose=smpl_85_cuda[:, :66],
                betas=torch.zeros((smpl_85_cuda.shape[0], 10), device='cuda'),  # assume zero betas, otherwise using smpl_85_cuda[:, 75:]
                transl=torch.from_numpy(traj_corrected).cuda(),
            )
            vertices = smplx_output['vertices'].detach().cpu().numpy()    # (B, N_imus, 3)
            joints_rot_3x3 = smplx_output['body_joints_rot'][:, ji_mask]  # (B, N_imus, 3, 3)
            joints_rot_aa = convert_rotation(joints_rot_3x3, 'mat', 'aa') # (B, N_imus, 3)
            joints_rot_aa = joints_rot_aa.detach().cpu().numpy()

        imu_traj = np.concatenate([joints_rot_aa, vertices], axis=-1).astype(np.float32)  # (B, N_imus, 6)
        imu_trajs.append(imu_traj)

    imu_trajs = np.concatenate(imu_trajs, axis=0).astype(np.float32)  # (total_frames, N_imus, 6)

    assert imu_trajs.shape[0] == motion_data.shape[0], \
        f"IMU trajectory frames {imu_trajs.shape[0]} do not match motion data frames {motion_data.shape[0]}"
    
    # Save
    np.save(os.path.join(motion_output_dir, 'imu_traj_from_smpl141_pip_setup.npy'), imu_trajs)


if __name__ == "__main__":

    # motion_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data/motion_data'

    # single_datasets = [
    #     'LINGO',      # text annotations too short
    #     'BABEL',
    #     'Mirror_BABEL', 
    #     'PhantomDanceDatav1.1', 
    #     'Mirror_PhantomDanceDatav1.1'
    # ]

    # multi_datasets = [
    #     'MotionGV',   # too large for memory
    #     'MotionLLAMA', 
    #     'MotionUnion', 
    #     'Mirror_MotionGV', 
    #     'Mirror_MotionLLAMA', 
    #     'Mirror_MotionUnion'
    # ]

    # # multi datasets have sub-datasets
    # expanded_multi_datasets = []
    # for dataset_name in multi_datasets:
    #     dataset_path = os.path.join(motion_dir, dataset_name)
    #     subdirs = [
    #         f for f in sorted(os.listdir(dataset_path)) 
    #         if os.path.isdir(os.path.join(dataset_path, f))
    #     ]
    #     expanded_multi_datasets.extend([f"{dataset_name}/{subdir}" for subdir in subdirs])

    # all_dataset_names = single_datasets + expanded_multi_datasets

    # for name in all_dataset_names:
    #     print(f"{name}")



    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_dir', type=str, required=True, help='Path to the dataset directory to process')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the IMU trajectory')
    parser.add_argument('--dataset_name', type=str, required=True, help='Sub-dataset name to process')
    args = parser.parse_args()

    motion_root_dir = args.motion_dir
    dataset_name = args.dataset_name

    motion_output_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(motion_output_dir, exist_ok=True)

    imu_sensor_config = IMU_Sensor_Config()
    vi_mask = imu_sensor_config.vi_mask
    ji_mask = imu_sensor_config.ji_mask
    imu_names = imu_sensor_config.imu_names

    # Load SMPL-X model
    smplx_model_path = '/projects/benk/hhsu2/imu-humans/body_models/human_model_files/smplx'
    simplified_smplx_model = SimplifiedSMPLX(
        model_path=smplx_model_path,
        gender='neutral',
        num_betas=10,
        vert_mask=vi_mask.tolist(),
        device='cuda',
    )

    smplx_output = simplified_smplx_model(pose=torch.zeros((1, 66), device='cuda'))
    rest_pelvis = smplx_output['joints_pos'][0, 0].detach().cpu().numpy()

    dataset_dir = os.path.join(motion_root_dir, dataset_name)

    simulate_imu_trajectory(
        dataset_dir,
        motion_output_dir,
        simplified_smplx_model,
        rest_pelvis,
        vi_mask,
        ji_mask
    )



    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to process',
    #                        choices=['BABEL', 'Mirror_BABEL', 'PhantomDanceDatav1.1', 'Mirror_PhantomDanceDatav1.1',
    #                                 'MotionGV', 'MotionLLAMA', 'MotionUnion', 'Mirror_MotionGV', 'Mirror_MotionLLAMA', 'Mirror_MotionUnion', 'LINGO'])
    # args = argparser.parse_args()

    # single_dataset_names = ['BABEL', 'Mirror_BABEL', 'PhantomDanceDatav1.1', 'Mirror_PhantomDanceDatav1.1', 'LINGO']
    # multi_dataset_names = ['MotionGV', 'MotionLLAMA', 'MotionUnion', 'Mirror_MotionGV', 'Mirror_MotionLLAMA', 'Mirror_MotionUnion']

    # dataset_name = args.dataset_name
    # is_multi_dataset = True if dataset_name in multi_dataset_names else False

    # motion_root_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data/motion_data'

    # imu_sensor_config = IMU_Sensor_Config()
    # vi_mask = imu_sensor_config.vi_mask
    # ji_mask = imu_sensor_config.ji_mask
    # imu_names = imu_sensor_config.imu_names

    # # Load SMPL-X model
    # smplx_model_path = '/projects/benk/hhsu2/imu-humans/body_models/human_model_files/smplx'
    # simplified_smplx_model = SimplifiedSMPLX(
    #     model_path=smplx_model_path,
    #     gender='neutral',
    #     num_betas=10,
    #     vert_mask=vi_mask.tolist(),
    #     device='cuda',
    # )

    # # Get pelvis offset in the rest pose to correct the global trajectory
    # smplx_output = simplified_smplx_model(pose=torch.zeros((1, 66), device='cuda'))
    # rest_pelvis = smplx_output['joints_pos'][0, 0].detach().cpu().numpy()

    # if is_multi_dataset:
    #     # For multi-dataset, we need to process each dataset separately
    #     dataset_dir = os.path.join(motion_root_dir, dataset_name)
    #     for sub_dataset_name in os.listdir(dataset_dir):
    #         sub_dataset_dir = os.path.join(dataset_dir, sub_dataset_name)
    #         simulate_imu_trajectory(
    #             sub_dataset_dir,
    #             simplified_smplx_model,
    #             rest_pelvis,
    #             vi_mask,
    #             ji_mask
    #         )
    # else:
    #     dataset_dir = os.path.join(motion_root_dir, dataset_name)
    #     simulate_imu_trajectory(
    #         dataset_dir,
    #         simplified_smplx_model,
    #         rest_pelvis,
    #         vi_mask,
    #         ji_mask
    #     )