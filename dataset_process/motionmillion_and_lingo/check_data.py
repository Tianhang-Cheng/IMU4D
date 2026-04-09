"""
This script checks if the files listed in the train.txt, val.txt, and test.txt exist in the directory.
"""

import os
import sys


root_dir = '/projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/MotionMillion'
motion_272rpr_dir = os.path.join(root_dir, 'motion_272rpr')

out_dir = 'valid_data'
os.makedirs(out_dir, exist_ok=True)

# T2M
t2m_split_dir = os.path.join(root_dir, 'split/version1/t2m_60_300')
for split in ['train', 'val', 'test']:
    t2m_txt = os.path.join(t2m_split_dir, f'{split}.txt')
    t2m_f_list = [line.strip() for line in open(t2m_txt).readlines()]
    t2m_f_list = sorted(t2m_f_list)
    print(f"Checking t2m {split} files: {len(t2m_f_list)}")
    
    valid_f_list = []
    missing_f_list = []
    for f in t2m_f_list:
        if not os.path.exists(os.path.join(motion_272rpr_dir, f'{f}.npy')):
            print(f"Missing file: {f}")
            missing_f_list.append(f)
        else:
            valid_f_list.append(f)
    print(f"==========================================")
    print(f"Valid {split} files: {len(valid_f_list)}")
    print(f"Missing {split} files: {len(missing_f_list)}")
    print(f"==========================================")

    valid_f_list = sorted(valid_f_list)
    missing_f_list = sorted(missing_f_list)

    with open(os.path.join(out_dir, f"t2m_{split}_valid.txt"), 'w') as out_file:
        for f in valid_f_list:
            out_file.write(f + '\n')
    
    with open(os.path.join(out_dir, f"t2m_{split}_missing.txt"), 'w') as out_file:
        for f in missing_f_list:
            out_file.write(f + '\n')


# Motion tokenizer
tokenizer_split_dir = os.path.join(root_dir, 'split/version1/tokenizer_96')
for split in ['train', 'val', 'test']:
    tokenizer_txt = os.path.join(tokenizer_split_dir, f'{split}.txt')
    tokenizer_f_list = [line.strip() for line in open(tokenizer_txt).readlines()]
    tokenizer_f_list = sorted(tokenizer_f_list)
    print(f"Checking tokenizer {split} files: {len(tokenizer_f_list)}")
    
    valid_f_list = []
    missing_f_list = []
    for f in tokenizer_f_list:
        if not os.path.exists(os.path.join(motion_272rpr_dir, f'{f}.npy')):
            print(f"Missing file: {f}")
            missing_f_list.append(f)
        else:
            valid_f_list.append(f)
    print(f"==========================================")
    print(f"Valid {split} files: {len(valid_f_list)}")
    print(f"Missing {split} files: {len(missing_f_list)}")
    print(f"==========================================")

    valid_f_list = sorted(valid_f_list)
    missing_f_list = sorted(missing_f_list)

    with open(os.path.join(out_dir, f"tokenizer_{split}_valid.txt"), 'w') as out_file:
        for f in valid_f_list:
            out_file.write(f + '\n')

    with open(os.path.join(out_dir, f"tokenizer_{split}_missing.txt"), 'w') as out_file:
        for f in missing_f_list:
            out_file.write(f + '\n')
