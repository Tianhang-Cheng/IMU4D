"""
This script is used to convert float32 numpy arrays to bfloat16 format.
"""
import numpy as np
import os
import torch


def convert_file_to_bfloat16(file_path: str, motion_format: str) -> None:
    array = np.load(file_path)

    if array.dtype != np.float32:
        array = array.astype(np.float32)

    bf16_array = array.astype(np.bfloat16)
    target_name = f"motion_{motion_format}_bf16.npy"
    target_path = os.path.join(os.path.dirname(file_path), target_name)

    np.save(target_path, bf16_array)
    print(f"Saved bfloat16 motion to {target_path}")


if __name__ == "__main__":

    # root_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data/'   # Delta cluster
    root_dir = '/projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/data/imu-humans/final_data'   # Campus cluster
    root_motion_data_dir = os.path.join(root_dir, 'motion_data')

    motion_format = 'smpl141'

    # get all motion files
    motion_files = []
    for root, dirs, files in os.walk(root_motion_data_dir, followlinks=True):
        for file in files:
            if file.endswith(f'motion_{motion_format}.npy') and not file.endswith("_bf16.npy"):
                motion_files.append(os.path.join(root, file))


    # for file_path in motion_files:
    #     arr = np.load(file_path)
    #     arr_f32 = arr.astype(np.float32)
    #     arr_bf16 = torch.from_numpy(arr_f32).bfloat16()
    #     target_path = os.path.join(os.path.dirname(file_path), f"motion_{motion_format}_bf16.pt")
    #     torch.save(arr_bf16, target_path)
    #     print(f"Converted and saved bfloat16 motion to {target_path}")


    BYTES_IN_GB = 1024 ** 3
    total_bytes = 0
    for file_path in motion_files:
        size_bytes = os.path.getsize(file_path)
        size_gb = size_bytes / BYTES_IN_GB
        print(f"{file_path}: {size_gb:.4f} GB")
        total_bytes += size_bytes
    print(f"Total size of all motion files: {total_bytes / BYTES_IN_GB:.4f} GB")


    BYTES_IN_GB = 1024 ** 3
    total_bytes = 0
    for file_path in motion_files:
        size_bytes = os.path.getsize(file_path.replace('.npy', '_bf16.pt'))
        size_gb = size_bytes / BYTES_IN_GB
        print(f"{file_path}: {size_gb:.4f} GB")
        total_bytes += size_bytes
    print(f"Total size of all motion files: {total_bytes / BYTES_IN_GB:.4f} GB")