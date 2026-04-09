import os
import random
import shutil


keep_dataset_names = [
    'BABEL', 'Mirror_BABEL', 
    'PhantomDanceDatav1.1', 'Mirror_PhantomDanceDatav1.1',
    'LINGO',
    'MotionGV', 'Mirror_MotionGV',
    'MotionLLAMA', 'Mirror_MotionLLAMA',
    'MotionUnion/EgoBody', 'MotionUnion/haa500', 'MotionUnion/humanml', 'MotionUnion/idea400', 'MotionUnion/kungfu', 'MotionUnion/music',
    'Mirror_MotionUnion/EgoBody', 'Mirror_MotionUnion/haa500', 'Mirror_MotionUnion/humanml', 'Mirror_MotionUnion/idea400', 'Mirror_MotionUnion/kungfu', 'Mirror_MotionUnion/music',
]


if __name__ == '__main__':

    random.seed(42)

    root_motion_dir = './motionmillion_smpl85'
    root_splits_dir = './motionmillion_splits'

    out_dir = './final_data'
    os.makedirs(out_dir, exist_ok=True)

    out_motion_data_dir = os.path.join(out_dir, 'motion_data')
    os.makedirs(out_motion_data_dir, exist_ok=True)

    # Step 1: copy selected datasets to final_data directory
    for item in keep_dataset_names:
        src_dir = os.path.join(root_motion_dir, item)
        dst_dir = os.path.join(out_motion_data_dir, item)
        if os.path.exists(src_dir):
            shutil.copytree(src_dir, dst_dir)
            print(f"Copied {src_dir} to {dst_dir}")
        else:
            print(f"Source directory {src_dir} does not exist.")

    # Step 2: adjust split files
    out_split_dir = os.path.join(out_dir, 'splits')
    os.makedirs(out_split_dir, exist_ok=True)

    # text-to-motion data
    for split in ['train', 'val', 'test']:
        t2m_txt = os.path.join(root_splits_dir, f't2m_{split}.txt')
        t2m_f_list = [line.strip() for line in open(t2m_txt).readlines()]
        new_t2m_f_list = []
        for t2m_f_name in t2m_f_list:
            if any(keep_name in t2m_f_name for keep_name in keep_dataset_names):
                new_t2m_f_list.append(t2m_f_name)
        random.shuffle(new_t2m_f_list)
        with open(os.path.join(out_split_dir, f't2m_{split}.txt'), 'w') as f:
            for item in new_t2m_f_list:
                f.write(f"{item}\n")
        print(f"Filtered t2m {split} set: {len(t2m_f_list)} -> {len(new_t2m_f_list)}")

    # motion tokenizer data
    for split in ['train', 'val', 'test']:
        tokenizer_txt = os.path.join(root_splits_dir, f'tokenizer_{split}.txt')
        tokenizer_f_list = [line.strip() for line in open(tokenizer_txt).readlines()]
        new_tokenizer_f_list = []
        for tokenizer_f_name in tokenizer_f_list:
            if any(keep_name in tokenizer_f_name for keep_name in keep_dataset_names):
                new_tokenizer_f_list.append(tokenizer_f_name)
        random.shuffle(new_tokenizer_f_list)
        with open(os.path.join(out_split_dir, f'tokenizer_{split}.txt'), 'w') as f:
            for item in new_tokenizer_f_list:
                f.write(f"{item}\n")
        print(f"Filtered tokenizer {split} set: {len(tokenizer_f_list)} -> {len(new_tokenizer_f_list)}")