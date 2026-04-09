"""
This script converts every 272-dim motion representation file into a 85-dim one.
"""

import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
import argparse

from smpl272_utils.motion_process import recover_from_local_rotation


if __name__ == '__main__':

    root_dir = '/projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/MotionMillion'
    motion_272rpr_dir = os.path.join(root_dir, 'motion_272rpr')

    out_dir = './motionmillion_smpl85'
    os.makedirs(out_dir, exist_ok=True)


    # Get all possible files to convert
    full_f_list = []

    t2m_split_dir = os.path.join(root_dir, 'split/version1/t2m_60_300')
    for split in ['train', 'val', 'test']:
        t2m_txt = os.path.join(t2m_split_dir, f'{split}.txt')
        t2m_f_list = [line.strip() for line in open(t2m_txt).readlines()]
        t2m_f_list = sorted(t2m_f_list)
        full_f_list.extend(t2m_f_list)

    tokenizer_split_dir = os.path.join(root_dir, 'split/version1/tokenizer_96')
    for split in ['train', 'val', 'test']:
        tokenizer_txt = os.path.join(tokenizer_split_dir, f'{split}.txt')
        tokenizer_f_list = [line.strip() for line in open(tokenizer_txt).readlines()]
        tokenizer_f_list = sorted(tokenizer_f_list)
        full_f_list.extend(tokenizer_f_list)

    full_f_list = sorted(set(full_f_list))
    print(f"Total files to convert: {len(full_f_list)}")


    single_dataset_names = ['BABEL', 'Mirror_BABEL', 'PhantomDanceDatav1.1', 'Mirror_PhantomDanceDatav1.1']
    multi_dataset_names = ['MotionGV', 'MotionLLAMA', 'MotionUnion', 'Mirror_MotionGV', 'Mirror_MotionLLAMA', 'Mirror_MotionUnion']


    # Process single dataset
    for dataset_name in single_dataset_names:

        f_list = [f for f in full_f_list if f.startswith(dataset_name)]
        print(f"Processing {dataset_name}: {len(f_list)} files")

        dataset_out_dir = os.path.join(out_dir, dataset_name)
        os.makedirs(dataset_out_dir, exist_ok=True)

        start_end_dict = {}
        current_idx = 0
        full_data_smpl85 = []

        # convert 272-dim to 85-dim
        for f_name in tqdm(f_list, desc=f"Converting {dataset_name}", dynamic_ncols=True):
            data_smpl272 = np.load(os.path.join(motion_272rpr_dir, f_name + '.npy'))
            data_smpl85 = recover_from_local_rotation(data_smpl272, njoint=22, pelvis_at_origin=True)

            full_data_smpl85.append(data_smpl85)
            start_end_dict[f_name] = (current_idx, current_idx + data_smpl85.shape[0])   # non-inclusive end index
            current_idx += data_smpl85.shape[0]

        full_data_smpl85 = np.concatenate(full_data_smpl85, axis=0).astype(np.float32)
        print(f"Total frames in {dataset_name}: {full_data_smpl85.shape[0]}")

        # save the processed data
        np.save(os.path.join(dataset_out_dir, 'motion_smpl85.npy'), full_data_smpl85)
        # np.save(os.path.join(dataset_out_dir, 'orient.npy'), full_data_smpl85[:, 0:3])
        # np.save(os.path.join(dataset_out_dir, 'transl.npy'), full_data_smpl85[:, 72:75])
        # np.save(os.path.join(dataset_out_dir, 'pose.npy'), full_data_smpl85[:, 3:66])
        with open(os.path.join(dataset_out_dir, 'start_end.pkl'), 'wb') as f:
            pickle.dump(start_end_dict, f)


    # Process multi dataset
    for dataset_name in multi_dataset_names:

        dataset_dir = os.path.join(motion_272rpr_dir, dataset_name)
        sub_dataset_names = [f for f in sorted(os.listdir(dataset_dir)) if os.path.isdir(os.path.join(dataset_dir, f))]

        for sub_dataset_name in sub_dataset_names:

            sub_dataset_dir = os.path.join(dataset_dir, sub_dataset_name)
            sub_dataset_out_dir = os.path.join(out_dir, dataset_name, sub_dataset_name)
            os.makedirs(sub_dataset_out_dir, exist_ok=True)

            f_list = [f for f in full_f_list if f.startswith(f'{dataset_name}/{sub_dataset_name}/')]
            print(f"Processing {dataset_name}/{sub_dataset_name}: {len(f_list)} files")

            start_end_dict = {}
            current_idx = 0
            full_data_smpl85 = []

            # convert 272-dim to 85-dim
            for f_name in tqdm(f_list, desc=f"Converting {dataset_name}/{sub_dataset_name}", dynamic_ncols=True):
                data_smpl272 = np.load(os.path.join(motion_272rpr_dir, f_name + '.npy'))
                data_smpl85 = recover_from_local_rotation(data_smpl272, njoint=22, pelvis_at_origin=True)

                full_data_smpl85.append(data_smpl85)
                start_end_dict[f_name] = (current_idx, current_idx + data_smpl85.shape[0])   # non-inclusive end index
                current_idx += data_smpl85.shape[0]

            full_data_smpl85 = np.concatenate(full_data_smpl85, axis=0).astype(np.float32)
            print(f"Total frames in {dataset_name}/{sub_dataset_name}: {full_data_smpl85.shape[0]}")

            # save the processed data
            np.save(os.path.join(sub_dataset_out_dir, 'motion_smpl85.npy'), full_data_smpl85)
            # np.save(os.path.join(sub_dataset_out_dir, 'orient.npy'), full_data_smpl85[:, 0:3])
            # np.save(os.path.join(sub_dataset_out_dir, 'transl.npy'), full_data_smpl85[:, 72:75])
            # np.save(os.path.join(sub_dataset_out_dir, 'pose.npy'), full_data_smpl85[:, 3:66])
            with open(os.path.join(sub_dataset_out_dir, 'start_end.pkl'), 'wb') as f:
                pickle.dump(start_end_dict, f)