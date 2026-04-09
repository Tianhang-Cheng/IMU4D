import numpy as np
import os
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


if __name__ == "__main__":

    ##### This code would save number of text descriptions of each sequence to a single dict #####

    def process_file_for_text_count(file):
        """Process a single file to count text descriptions"""
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            if data['texts'] is None:
                return file, 0
            else:
                return file, len(data['texts'])
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
    texts_count_dict = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file_for_text_count, file): file
                          for file in motion_files}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(motion_files)):
            file_path, count = future.result()
            if count is not None:
                rel_path = os.path.relpath(file_path, root_motion_data_dir)
                texts_count_dict[rel_path] = count

    out_texts_count_file = os.path.join(root_dir, 'texts_count_dict.pkl')
    with open(out_texts_count_file, 'wb') as f:
        pickle.dump(texts_count_dict, f)

    print(f"Saved texts counts for {len(texts_count_dict)} files to {out_texts_count_file}")