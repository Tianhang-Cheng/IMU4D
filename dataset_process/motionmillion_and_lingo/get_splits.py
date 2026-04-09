"""
This script generates the list of all files depending on the dataset splits.
"""

import os
import numpy as np
import pickle


if __name__ == '__main__':

    root_dir = '/projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/MotionMillion'
    out_dir = './motionmillion_splits'
    os.makedirs(out_dir, exist_ok=True)


    # Get lingo dataset splits
    with open('./motionmillion_smpl85/LINGO/start_end.pkl', 'rb') as f:
        lingo_data = pickle.load(f)
    lingo_f_list = list(lingo_data.keys())
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1


    t2m_split_dir = os.path.join(root_dir, 'split/version1/t2m_60_300')
    for split in ['train', 'val', 'test']:
        t2m_txt = os.path.join(t2m_split_dir, f'{split}.txt')
        t2m_f_list = [line.strip() for line in open(t2m_txt).readlines()]
        t2m_f_list = sorted(t2m_f_list)

        # Append lingo files to t2m train set
        if split == 'train':
            t2m_f_list.extend(lingo_f_list[:int(len(lingo_f_list)*train_ratio)])
        elif split == 'val':
            t2m_f_list.extend(lingo_f_list[int(len(lingo_f_list)*train_ratio):int(len(lingo_f_list)*(train_ratio+val_ratio))])
        elif split == 'test':
            t2m_f_list.extend(lingo_f_list[int(len(lingo_f_list)*(train_ratio+val_ratio)):])

        # save
        with open(os.path.join(out_dir, f't2m_{split}.txt'), 'w') as f:
            for item in t2m_f_list:
                f.write(f"{item}\n")


    tokenizer_split_dir = os.path.join(root_dir, 'split/version1/tokenizer_96')
    for split in ['train', 'val', 'test']:
        tokenizer_txt = os.path.join(tokenizer_split_dir, f'{split}.txt')
        tokenizer_f_list = [line.strip() for line in open(tokenizer_txt).readlines()]
        tokenizer_f_list = sorted(tokenizer_f_list)
        
        # Append lingo files to tokenizer train set
        if split == 'train':
            tokenizer_f_list.extend(lingo_f_list[:int(len(lingo_f_list)*train_ratio)])
        elif split == 'val':
            tokenizer_f_list.extend(lingo_f_list[int(len(lingo_f_list)*train_ratio):int(len(lingo_f_list)*(train_ratio+val_ratio))])
        elif split == 'test':
            tokenizer_f_list.extend(lingo_f_list[int(len(lingo_f_list)*(train_ratio+val_ratio)):])

        # save
        with open(os.path.join(out_dir, f'tokenizer_{split}.txt'), 'w') as f:
            for item in tokenizer_f_list:
                f.write(f"{item}\n")