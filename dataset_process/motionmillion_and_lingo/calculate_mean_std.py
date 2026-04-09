import numpy as np
import os
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


if __name__ == "__main__":

    ##### This is the original code for mean & std calculation #####

    # root_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data/'
    # root_motion_data_dir = os.path.join(root_dir, 'motion_data')

    # motion_format = 'smpl141'  # 'smpl85', 'smpl141', 'smpl272'
    # n_joints = 22

    # out_mean_std_dir = os.path.join(root_dir, f'mean_std_{motion_format}')
    # os.makedirs(out_mean_std_dir, exist_ok=True)

    # # get all motion files
    # motion_files = []
    # for root, dirs, files in os.walk(root_motion_data_dir, followlinks=True):
    #     for file in files:
    #         if file.endswith(f'motion_{motion_format}.npy'):
    #             motion_files.append(os.path.join(root, file))

    # print(f"Found {len(motion_files)} motion files.")
    # for i, file in enumerate(motion_files):
    #     print(f"\t {i}: {file}")

    # motion_data_list = []
    # for file in tqdm(motion_files):
    #     data = np.load(file)
    #     if np.isnan(data).any():
    #         print(file)
    #         continue
    #     motion_data_list.append(data)

    # motion_data = np.concatenate(motion_data_list, axis=0)
    # print("Motion data shape:", motion_data.shape)

    # assert motion_data.shape[1] == int(motion_format.split('smpl')[-1])

    # mean = data.mean(axis=0)
    # std = data.std(axis=0)

    # # average std trick from HumanML3D
    # if motion_format == 'smpl85':
    #     pass
    # elif motion_format == 'smpl141':
    #     std[:2] = std[:2].mean() / 1.0
    #     std[2:8] = std[2:8].mean() / 1.0
    #     std[8:9] = std[8:9].mean() / 1.0
    #     std[9:9+6*n_joints] = std[9:9+6*n_joints].mean() / 1.0
    # elif motion_format == 'smpl272':
    #     std[:2] = std[:2].mean() / 1.0
    #     std[2:8] = std[2:8].mean() / 1.0
    #     std[8:8+3*n_joints] = std[8:8+3*n_joints].mean() / 1.0
    #     std[8+3*n_joints:8+6*n_joints] = std[8+3*n_joints:8+6*n_joints].mean() / 1.0
    #     std[8+6*n_joints:8+12*n_joints] = std[8+6*n_joints:8+12*n_joints].mean() / 1.0
    # else:
    #     raise NotImplementedError

    # np.save(os.path.join(out_mean_std_dir, 'mean.npy'), mean)
    # np.save(os.path.join(out_mean_std_dir, 'std.npy'), std)


    ##### This is the updated version for mean & std calculation #####

    def process_file_for_mean(file):
        """Process a single file to compute sum and frame count for mean calculation"""
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            data_smpl141 = data['motion_data_smpl141']
            return data_smpl141.sum(axis=0), data_smpl141.shape[0]
        except Exception as e:
            print(f"Error processing {file}: {e}")
            return None, 0


    def process_file_for_std(file, mean):
        """Process a single file to compute squared differences for std calculation"""
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            data_smpl141 = data['motion_data_smpl141']
            squared_diff = (data_smpl141 - mean) ** 2
            return squared_diff.sum(axis=0)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            return None


    root_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data_per_sequence'
    root_motion_data_dir = os.path.join(root_dir, 'motion_data')

    n_joints = 22

    out_mean_std_dir = os.path.join(root_dir, f'mean_std_smpl141_2025-11-03')
    os.makedirs(out_mean_std_dir, exist_ok=True)

    # get all motion files
    motion_files = []
    for root, dirs, files in os.walk(root_motion_data_dir, followlinks=True):
        for file in files:
            if file.endswith('.pkl'):
                motion_files.append(os.path.join(root, file))

    print(f"Found {len(motion_files)} motion files.")

    # determine the number of worker threads
    max_workers = min(64, os.cpu_count() * 2)    # Use 2x CPU cores, max 32 threads
    print(f"Using {max_workers} worker threads")

    # First pass to compute the mean (multithreaded)
    print("Computing mean...")
    total_sum = np.zeros(141)
    frame_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file_for_mean, file): file 
                          for file in motion_files}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(motion_files)):
            file_sum, file_frames = future.result()
            if file_sum is not None:
                total_sum += file_sum
                frame_count += file_frames
    
    mean = total_sum / frame_count
    print(f"Mean computed from {frame_count} frames")

    # Second pass to compute the std (multithreaded)
    print("Computing std...")
    total_squared_diff = np.zeros(141)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file_for_std, file, mean): file 
                          for file in motion_files}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(motion_files)):
            file_squared_diff = future.result()
            if file_squared_diff is not None:
                total_squared_diff += file_squared_diff
    
    std = np.sqrt(total_squared_diff / frame_count)

    # average std trick from HumanML3D
    std[:2] = std[:2].mean() / 1.0
    std[2:8] = std[2:8].mean() / 1.0
    std[8:9] = std[8:9].mean() / 1.0
    std[9:9+6*n_joints] = std[9:9+6*n_joints].mean() / 1.0

    np.save(os.path.join(out_mean_std_dir, 'mean.npy'), mean)
    np.save(os.path.join(out_mean_std_dir, 'std.npy'), std)