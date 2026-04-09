import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import List
import random
import tqdm
import pickle
import os

from rotation2 import convert_rotation

def find_files(root_folder, target_filename) -> List[str]:
    """
    Recursively find all 'imu_traj.npy' files under the given root folder.

    Args:
        root_folder (str): The root directory to search in.

    Returns:
        list[str]: A list of full paths to files named 'imu_traj.npy'.
    """
    matches = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == target_filename:
                matches.append(os.path.join(dirpath, filename))
    return matches

class VariableLengthArrayDataset(Dataset):
    def __init__(self, chunk_size: int):
        """
        Dataset for handling arrays of variable lengths with padding and chunking.
        
        Args:
            arrays: List of numpy arrays, each with shape [n_i, n_variable]
            chunk_size: The K value - threshold for padding/chunking
        """

        parent_folder = '/work/hdd/benk/hhsu2/imu-humans/final_data/motion_data'

        imu_traj_files = find_files(parent_folder, 'imu_traj.npy')

        good_list = ['EgoBody','Haa500','HumanML3D','Idea400','Kungfu','Music']

        data_arrays = []
        count = 0
        max_count = 20
        for imu_traj_file in tqdm.tqdm(imu_traj_files, desc="Loading files"):

            if 'MotionUnion' in imu_traj_file and not any(good in imu_traj_file for good in good_list):
                continue

            if count > max_count:
                break
            count += 1

            imu_traj_data = np.load(imu_traj_file, allow_pickle=True).reshape(-1, 36) # Reshape to (n_i, 36)
            motion_data = np.load(imu_traj_file.replace('imu_traj.npy', 'motion_smpl85.npy'), allow_pickle=True).reshape(-1, 85)
            motion_data_clipped0 = motion_data[:, :66] # orient + pose
            motion_data_clipped1 = motion_data[:, 72:75]
            motion_data = np.concatenate([motion_data_clipped0, motion_data_clipped1], axis=-1) # [n_i, 69]
            # to 6d representation
            motion_data_rot6d = motion_data.copy() # [n_i, 69]

            motion_data_rot6d = motion_data_rot6d.reshape(-1, 3) # [n_i*23, 3]
            motion_data_rot6d = torch.from_numpy(motion_data_rot6d).float()
            motion_data_rot6d = convert_rotation(motion_data_rot6d, src_rep='aa', tgt_rep='6d') 
            motion_data_rot6d = motion_data_rot6d.numpy().reshape(-1, 23*6)

            data = np.concatenate([imu_traj_data, motion_data], axis=-1)
            data_arrays.append(data)

        self.data_arrays = np.concatenate(data_arrays, axis=0) # [len, n_var]
        self.chunk_size = chunk_size
    
    def __len__(self):
        return len(self.data_arrays) - self.chunk_size + 1

    def __getitem__(self, idx):
        return self.data_arrays[idx:idx + self.chunk_size]

def collate_fn(batch):
    """
    Custom collate function to stack data into batches.
    """
    data = np.stack(batch, axis=0)  # Shape: [batch_size, seq_length, n_variable]
    data = torch.tensor(data, dtype=torch.float32)
    return data

def create_dataloaders(
    seq_length: int, 
    batch_size: int = 32,
    random_seed: int = 42,
    val_ratio: float = 0.05, 
    test_ratio: float = 0.0,
):
    """
    Create train, validation, and test DataLoaders for variable length arrays.
    
    Args:
        arrays: List of numpy arrays, each with shape [n_i, n_variable]
        seq_length_size: the length of the sequence to visualize (K value)
        batch_size: Number of samples per batch
        val_ratio: Portion of data to use for validation (default: 0.05 = 5%)
        test_ratio: Portion of data to use for testing (default: 0.05 = 5%)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Create the full dataset
    full_dataset = VariableLengthArrayDataset(seq_length)
    
    # Calculate sizes for train, validation, and test splits
    dataset_size = len(full_dataset)
    val_size = int(val_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        ) if train_size > 0 else None,
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
    }
    
    return dataloaders

# Example usage:
if __name__ == "__main__":
    dataloaders = create_dataloaders(seq_length=20, batch_size=32)

    # Print information about the splits
    for split_name, dataloader in dataloaders.items():
        print(f"{split_name.capitalize()} set:")
        print(f"  Number of batches: {len(dataloader)}")
        total_samples = len(dataloader.dataset)
        print(f"  Number of samples: {total_samples}")
        
        # Get a sample batch
        if len(dataloader) > 0:
            data_batch = next(iter(dataloader))
            print(f"  Batch data shape: {data_batch.shape}")
        print()