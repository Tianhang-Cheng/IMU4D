import numpy as np
import os
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


if __name__ == "__main__":

    ##### This code would save the motion length of each sequence to a single dict #####

    def process_file_for_length(file):
        """Process a single file to get motion length"""
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            data_smpl141 = data['motion_data_smpl141']
            return file, data_smpl141.shape[0]
        except Exception as e:
            print(f"Error processing {file}: {e}")
            return file, None

    root_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data_per_sequence'
    root_motion_data_dir = os.path.join(root_dir, 'motion_data')

    # get all motion files
    motion_files = []
    for root, dirs, files in os.walk(root_motion_data_dir, followlinks=True):
        for file in files:
            if file.endswith('.pkl'):
                motion_files.append(os.path.join(root, file))

    print(f"Found {len(motion_files)} motion files.")

    # determine the number of worker threads
    max_workers = min(64, os.cpu_count() * 2)    # Use 2x CPU cores, max 64 threads
    print(f"Using {max_workers} worker threads")

    # Process files in parallel
    motion_len_dict = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file_for_length, file): file 
                          for file in motion_files}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(motion_files)):
            file_path, length = future.result()
            if length is not None:
                rel_path = os.path.relpath(file_path, root_motion_data_dir)
                motion_len_dict[rel_path] = length

    out_motion_len_file = os.path.join(root_dir, 'motion_len_dict.pkl')
    with open(out_motion_len_file, 'wb') as f:
        pickle.dump(motion_len_dict, f)
    
    print(f"Saved motion lengths for {len(motion_len_dict)} files to {out_motion_len_file}")