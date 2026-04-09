"""
This script aggregates all text files from a dataset into a single pickle file.
"""

import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
import argparse


if __name__ == '__main__':

    root_dir = '/projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/MotionMillion'
    motion_272rpr_dir = os.path.join(root_dir, 'texts')

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

    ##### Only text-to-motion files have text descriptions
    # tokenizer_split_dir = os.path.join(root_dir, 'split/version1/tokenizer_96')
    # for split in ['train', 'val', 'test']:
    #     tokenizer_txt = os.path.join(tokenizer_split_dir, f'{split}.txt')
    #     tokenizer_f_list = [line.strip() for line in open(tokenizer_txt).readlines()]
    #     tokenizer_f_list = sorted(tokenizer_f_list)
    #     full_f_list.extend(tokenizer_f_list)

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

        text_description_dict = {}

        # convert every text description in .txt file into a list
        for f_name in tqdm(f_list, desc=f"Converting {dataset_name}", dynamic_ncols=True):
            description_txt = os.path.join(motion_272rpr_dir, f_name + '.txt')
            text_data = [line.strip() for line in open(description_txt).readlines()]
            text_description_dict[f_name] = text_data

        # save the text descriptions
        with open(os.path.join(dataset_out_dir, 'texts.pkl'), 'wb') as f:
            pickle.dump(text_description_dict, f)


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

            text_description_dict = {}

            # convert every text description in .txt file into a list
            for f_name in tqdm(f_list, desc=f"Converting {dataset_name}/{sub_dataset_name}", dynamic_ncols=True):
                description_txt = os.path.join(motion_272rpr_dir, f_name + '.txt')
                text_data = [line.strip() for line in open(description_txt).readlines()]
                text_description_dict[f_name] = text_data

            # save the text descriptions
            with open(os.path.join(sub_dataset_out_dir, 'texts.pkl'), 'wb') as f:
                pickle.dump(text_description_dict, f)
