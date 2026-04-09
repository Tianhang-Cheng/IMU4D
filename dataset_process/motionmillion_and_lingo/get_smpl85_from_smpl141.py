"""
This script converts SMPL-141 into SMPL-85 but disables pelvis_at_origin option.

Compared with get_smpl85.py, here we do NOT set pelvis_at_origin=True, so that the pelvis height indicates real-world height. We operate on Delta cluster.
"""
import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
import argparse

from smpl272_utils.motion_process import recover_from_local_rotation

import sys
sys.path.append('/projects/benk/hhsu2/imu-humans/code/imu-human-mllm/third_party/MotionGPT')
from utils.evaluate import smpl141_to_smpl272, smpl272_to_smpl141


if __name__ == '__main__':

    root_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data'
    motion_data_dir = os.path.join(root_dir, 'motion_data')

    single_dataset_names = ['LINGO', 'BABEL', 'Mirror_BABEL', 'PhantomDanceDatav1.1', 'Mirror_PhantomDanceDatav1.1']
    multi_dataset_names = ['MotionGV', 'MotionLLAMA', 'MotionUnion', 'Mirror_MotionGV', 'Mirror_MotionLLAMA', 'Mirror_MotionUnion']

    # multi datasets have sub-datasets
    expanded_multi_datasets = []
    for dataset_name in multi_dataset_names:
        dataset_path = os.path.join(motion_data_dir, dataset_name)
        subdirs = [
            f for f in sorted(os.listdir(dataset_path)) 
            if os.path.isdir(os.path.join(dataset_path, f))
        ]
        expanded_multi_datasets.extend([f"{dataset_name}/{subdir}" for subdir in subdirs])

    all_dataset_names = single_dataset_names + expanded_multi_datasets

    for dataset_name in all_dataset_names:
        print(f"Processing dataset: {dataset_name}...")

        dataset_dir = os.path.join(motion_data_dir, dataset_name)
        motion_smpl141_all = np.load(os.path.join(dataset_dir, 'motion_smpl141.npy'))
        with open(os.path.join(dataset_dir, 'start_end.pkl'), 'rb') as f:
            start_end_dict = pickle.load(f)

        full_data_smpl85 = []

        for f_name, (start_idx, end_idx) in tqdm(start_end_dict.items(), desc=f"Converting {dataset_name}", dynamic_ncols=True):
            data_smpl141 = motion_smpl141_all[start_idx:end_idx]
            data_smpl272 = smpl141_to_smpl272(data_smpl141)
            data_smpl85 = recover_from_local_rotation(data_smpl272, njoint=22, pelvis_at_origin=False)
            full_data_smpl85.append(data_smpl85)

        full_data_smpl85 = np.concatenate(full_data_smpl85, axis=0).astype(np.float32)
        print(f"Total frames in {dataset_name}: {full_data_smpl85.shape[0]}")

        assert full_data_smpl85.shape[0] == motion_smpl141_all.shape[0], f"Frame count mismatch in {dataset_name}"

        np.save(os.path.join(dataset_dir, 'motion_smpl85_from_smpl141.npy'), full_data_smpl85)