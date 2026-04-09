"""
This script is used to split the large motion dataset into multiple files, one for each sequence.
"""
import numpy as np
from tqdm import tqdm
import os
import pickle
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed


def save_sequence(id, split_dir, dataset_name, start_end_dict, texts_dict, 
                  motion_data_smpl141_all, motion_data_smpl85_all, imu_traj_all):
    """Process and save a single sequence."""
    output_file = os.path.join(split_dir, f"{id.replace('/', '_')}.pkl")
    
    # Check if this ID belongs to current dataset
    if not id.startswith(dataset_name) or id not in start_end_dict:
        return id, 'not_in_dataset'
    
    # Skip if already exists
    # if os.path.exists(output_file):
    #     return id, 'skipped'
    
    try:
        start, end = start_end_dict[id]
        motion_data_smpl141 = motion_data_smpl141_all[start:end]   # (T, 141)
        motion_data_smpl85 = motion_data_smpl85_all[start:end]     # (T, 85)
        imu_traj = imu_traj_all[start:end]                         # (T, 6, 6)
        
        texts = texts_dict.get(id) if texts_dict is not None else None
        
        # Save as .pkl file
        with open(output_file, 'wb') as f:
            pickle.dump({
                'motion_data_smpl141': motion_data_smpl141,
                'motion_data_smpl85': motion_data_smpl85,
                'imu_traj': imu_traj,
                'texts': texts
            }, f)
        
        return id, 'success'
    except Exception as e:
        return id, f'error: {str(e)}'


if __name__ == "__main__":

    MAX_WORKERS = 16  # Adjust based on your system

    root_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data'
    motion_data_dir = os.path.join(root_dir, 'motion_data')

    out_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data_per_sequence'
    os.makedirs(out_dir, exist_ok=True)
    motion_out_dir = os.path.join(out_dir, 'motion_data')
    os.makedirs(motion_out_dir, exist_ok=True)
    
    motion_out_dir_train = os.path.join(motion_out_dir, 'train')
    os.makedirs(motion_out_dir_train, exist_ok=True)
    motion_out_dir_val = os.path.join(motion_out_dir, 'val')
    os.makedirs(motion_out_dir_val, exist_ok=True)
    motion_out_dir_test = os.path.join(motion_out_dir, 'test')
    os.makedirs(motion_out_dir_test, exist_ok=True)


    # Copy folders 'mean_std_smpl141' and 'splits' directly to out_dir
    shutil.copytree(os.path.join(root_dir, 'mean_std_smpl141'), os.path.join(out_dir, 'mean_std_smpl141'), dirs_exist_ok=True)
    shutil.copytree(os.path.join(root_dir, 'splits'), os.path.join(out_dir, 'splits'), dirs_exist_ok=True)


    # Get all ids of each split (train/val/test) from each task (t2m/tokenizer)
    train_id_list, val_id_list, test_id_list = [], [], []
    for task in ['t2m', 'tokenizer']:
        task_train_txt = os.path.join(out_dir, 'splits', f'{task}_train.txt')
        task_train_f_list = [line.strip() for line in open(task_train_txt).readlines()]
        train_id_list.extend(task_train_f_list)
        task_val_txt = os.path.join(out_dir, 'splits', f'{task}_val.txt')
        task_val_f_list = [line.strip() for line in open(task_val_txt).readlines()]
        val_id_list.extend(task_val_f_list)
        task_test_txt = os.path.join(out_dir, 'splits', f'{task}_test.txt')
        task_test_f_list = [line.strip() for line in open(task_test_txt).readlines()]
        test_id_list.extend(task_test_f_list)
    train_id_list = sorted(set(train_id_list))
    val_id_list = sorted(set(val_id_list))
    test_id_list = sorted(set(test_id_list))
    print(f"Total train/val/test sequences: {len(train_id_list)}/{len(val_id_list)}/{len(test_id_list)}")


    # Get all dataset names
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


    # Process each dataset
    for dataset_name in all_dataset_names:
        print(f"Processing dataset: {dataset_name}...")

        dataset_dir = os.path.join(motion_data_dir, dataset_name)

        with open(os.path.join(dataset_dir, 'start_end.pkl'), 'rb') as f:
            start_end_dict = pickle.load(f)
        texts_dict = None
        if os.path.exists(os.path.join(dataset_dir, 'texts.pkl')):
            with open(os.path.join(dataset_dir, 'texts.pkl'), 'rb') as f:
                texts_dict = pickle.load(f)
        motion_data_smpl141_all = np.load(os.path.join(dataset_dir, 'motion_smpl141.npy'))
        motion_data_smpl85_all = np.load(os.path.join(dataset_dir, 'motion_smpl85_from_smpl141.npy'))
        imu_traj_all = np.load(os.path.join(dataset_dir, 'imu_traj_from_smpl141.npy'))


        # Process train split with multi-threading
        print(f"  Processing train split...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for id in train_id_list:
                future = executor.submit(save_sequence, id, motion_out_dir_train, dataset_name,
                                       start_end_dict, texts_dict, motion_data_smpl141_all,
                                       motion_data_smpl85_all, imu_traj_all)
                futures.append(future)
            
            # Wait for all to complete with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Train"):
                result = future.result()

        
        # Process val split with multi-threading
        print(f"  Processing val split...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for id in val_id_list:
                future = executor.submit(save_sequence, id, motion_out_dir_val, dataset_name,
                                       start_end_dict, texts_dict, motion_data_smpl141_all,
                                       motion_data_smpl85_all, imu_traj_all)
                futures.append(future)
            
            # Wait for all to complete with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Val"):
                result = future.result()


        # Process test split with multi-threading
        print(f"  Processing test split...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for id in test_id_list:
                future = executor.submit(save_sequence, id, motion_out_dir_test, dataset_name,
                                       start_end_dict, texts_dict, motion_data_smpl141_all,
                                       motion_data_smpl85_all, imu_traj_all)
                futures.append(future)
            
            # Wait for all to complete with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Test"):
                result = future.result()


        # Train split
        # for id in train_id_list:
        #     if id.startswith(dataset_name) and id in start_end_dict:
        #         if os.path.exists(os.path.join(motion_out_dir_train, f"{id.replace('/', '_')}.pkl")):
        #             continue
        #         start, end = start_end_dict[id]
        #         motion_data_smpl141 = motion_data_smpl141_all[start:end]   # (T, 141)
        #         motion_data_smpl85 = motion_data_smpl85_all[start:end]     # (T, 85)
        #         imu_traj = imu_traj_all[start:end]                         # (T, 6, 6)

        #         texts = texts_dict.get(id) if texts_dict is not None else None

        #         # save as .pkl file
        #         with open(os.path.join(motion_out_dir_train, f"{id.replace('/', '_')}.pkl"), 'wb') as f:
        #             pickle.dump({
        #                 'motion_data_smpl141': motion_data_smpl141,
        #                 'motion_data_smpl85': motion_data_smpl85,
        #                 'imu_traj': imu_traj,
        #                 'texts': texts
        #             }, f)


        # Val split
        # for id in val_id_list:
        #     if id.startswith(dataset_name) and id in start_end_dict:
        #         if os.path.exists(os.path.join(motion_out_dir_val, f"{id.replace('/', '_')}.pkl")):
        #             continue
        #         start, end = start_end_dict[id]
        #         motion_data_smpl141 = motion_data_smpl141_all[start:end]   # (T, 141)
        #         motion_data_smpl85 = motion_data_smpl85_all[start:end]     # (T, 85)
        #         imu_traj = imu_traj_all[start:end]                         # (T, 6, 6)

        #         texts = texts_dict.get(id) if texts_dict is not None else None

        #         # save as .pkl file
        #         with open(os.path.join(motion_out_dir_val, f"{id.replace('/', '_')}.pkl"), 'wb') as f:
        #             pickle.dump({
        #                 'motion_data_smpl141': motion_data_smpl141,
        #                 'motion_data_smpl85': motion_data_smpl85,
        #                 'imu_traj': imu_traj,
        #                 'texts': texts
        #             }, f)


        # Test split
        # for id in test_id_list:
        #     if id.startswith(dataset_name) and id in start_end_dict:
        #         if os.path.exists(os.path.join(motion_out_dir_test, f"{id.replace('/', '_')}.pkl")):
        #             continue
        #         start, end = start_end_dict[id]
        #         motion_data_smpl141 = motion_data_smpl141_all[start:end]   # (T, 141)
        #         motion_data_smpl85 = motion_data_smpl85_all[start:end]     # (T, 85)
        #         imu_traj = imu_traj_all[start:end]                         # (T, 6, 6)

        #         texts = texts_dict.get(id) if texts_dict is not None else None

        #         # save as .pkl file
        #         with open(os.path.join(motion_out_dir_test, f"{id.replace('/', '_')}.pkl"), 'wb') as f:
        #             pickle.dump({
        #                 'motion_data_smpl141': motion_data_smpl141,
        #                 'motion_data_smpl85': motion_data_smpl85,
        #                 'imu_traj': imu_traj,
        #                 'texts': texts
        #             }, f)