"""
This script is used to train a DCT + BPE tokenizer on 3d trajectories (orientation & translation) from our custom IMU data.

- FAST repo (https://huggingface.co/physical-intelligence/fast)

- Example usage:
python train_fast_tokenizer.py --data_dir ../../../dataset/ --save_dir ./ckpts/ --use_6d_rotation --chunk_length 60 --scale 10.0 --vocab_size 2048 --overlap 0.0
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import json
import torch
from scipy.spatial.transform import Rotation as R
import pickle

from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation, Unit

from utils.rotation import convert_rotation
from utils.metrics import mse, psnr, orientation_error, translation_error, mpjpe_error
from utils.data_utils import normalize_data, denormalize_data

def xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]

def eval_recon(original, decoded, use_6d_rotation=False, norm_params=None):
    """
    Evaluate reconstruction quality for orientation and translation.

    Args:
        original (np.ndarray): Original data with shape (batch, seq_len, features)
        decoded (np.ndarray): Decoded data with same shape as original
        use_6d_rotation (bool): Whether 6D rotation representation is used
        norm_params (dict): Normalization parameters for denormalization

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    orient_dim = 6 if use_6d_rotation else 3
    
    original_orient = original[:, :, :orient_dim]
    original_transl = original[:, :, orient_dim:orient_dim+3]
    
    decoded_orient = decoded[:, :, :orient_dim]
    decoded_transl = decoded[:, :, orient_dim:orient_dim+3]

    # MSR and PSNR for translation
    translation_mse = mse(original_transl, decoded_transl)
    translation_psnr = psnr(original_transl, decoded_transl, data_range=2.0)

    # 6d -> aa
    if use_6d_rotation:
        original_rotvec = convert_rotation(
            torch.from_numpy(original_orient).float(), '6d', 'aa').numpy()
        decoded_rotvec = convert_rotation(
            torch.from_numpy(decoded_orient).float(), '6d', 'aa').numpy()
    else:
        original_rotvec = original_orient * np.pi
        decoded_rotvec = decoded_orient * np.pi

    # MSR and PSNR for aa rotation
    orientation_mse = mse(original_rotvec, decoded_rotvec)
    orientation_psnr = psnr(original_rotvec, decoded_rotvec, data_range=2*np.pi)

    batch_size, seq_len, _ = original.shape
    total_frames = batch_size * seq_len

    # Denormalize translation data for trajectory evaluation only
    if norm_params and norm_params.get("transl"):
        original_transl_denorm = denormalize_data(original_transl, norm_params["transl"])
        decoded_transl_denorm = denormalize_data(decoded_transl, norm_params["transl"])
    else:
        original_transl_denorm = original_transl
        decoded_transl_denorm = decoded_transl

    # Flatten for trajectory evaluation
    original_rotvec_flat = original_rotvec.reshape(total_frames, 3)
    decoded_rotvec_flat = decoded_rotvec.reshape(total_frames, 3)
    original_transl_flat = original_transl_denorm.reshape(total_frames, 3)
    decoded_transl_flat = decoded_transl_denorm.reshape(total_frames, 3)

    # Convert to quaternions for trajectory evaluation
    original_rot = R.from_rotvec(original_rotvec_flat)
    decoded_rot = R.from_rotvec(decoded_rotvec_flat)

    original_quat_wxyz = xyzw_to_wxyz(original_rot.as_quat())  # (N, 4) - wxyz format
    decoded_quat_wxyz = xyzw_to_wxyz(decoded_rot.as_quat())  # (N, 4) - wxyz format

    timestamps = np.arange(total_frames, dtype=np.float64)

    traj_ref = PoseTrajectory3D(
        positions_xyz=original_transl_flat,
        orientations_quat_wxyz=original_quat_wxyz,
        timestamps=timestamps
    )
    
    traj_est = PoseTrajectory3D(
        positions_xyz=decoded_transl_flat,
        orientations_quat_wxyz=decoded_quat_wxyz,
        timestamps=timestamps
    )

    # Trajectory evaluation metrics
    ATE = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=False, correct_scale=False)
    RPEt = main_rpe.rpe(traj_ref, traj_est, est_name='traj', align=False, correct_scale=False,
        pose_relation=PoseRelation.translation_part, delta=1, delta_unit=Unit.frames, all_pairs=True)
    RPEr = main_rpe.rpe(traj_ref, traj_est, est_name='traj', align=False, correct_scale=False,
        pose_relation=PoseRelation.rotation_angle_deg, delta=1, delta_unit=Unit.frames, all_pairs=True)

    # Orientation and translation errors
    original_Ts = np.concatenate((original_quat_wxyz, original_transl_flat), axis=-1)  # (N, 7)
    original_Ts = torch.from_numpy(original_Ts).float()
    decoded_Ts = np.concatenate((decoded_quat_wxyz, decoded_transl_flat), axis=-1)  # (N, 7)
    decoded_Ts = torch.from_numpy(decoded_Ts).float()
    ori_error = orientation_error(original_Ts, decoded_Ts).item()
    trans_error = translation_error(original_Ts, decoded_Ts).item()

    metrics = {
        "MSE_trans": translation_mse,
        "PSNR_trans": translation_psnr,
        "MSE_orient": orientation_mse,
        "PSNR_orient": orientation_psnr,
        "orient_error": ori_error,
        "trans_error": trans_error,
        "ATE": ATE.stats['rmse'],
        "RPEt": RPEt.stats['rmse'],
        "RPEr": RPEr.stats['rmse']
    }

    return metrics


def main():
    """
    Main function to train or evaluate the IMU tokenizer.
    """
    parser = argparse.ArgumentParser(description="Train a tokenizer on IMU dataset")
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to the IMU dataset directory')
    parser.add_argument('--save_dir', type=str, default='.', 
                       help='Directory to save the trained tokenizer model')
    parser.add_argument('--eval', action='store_true', 
                       help='Evaluate from the checkpoint')
    parser.add_argument('--use_6d_rotation', action='store_true', 
                       help='Use 6D rotation representation instead of 3D rotvec')
    parser.add_argument('--chunk_length', type=int, default=60, 
                       help='Length of each chunk (fps * seconds)')
    parser.add_argument('--scale', type=float, default=10.0, 
                       help='Increase the scale for less lossy compression')
    parser.add_argument('--vocab_size', type=int, default=2048, 
                       help='Vocabulary size for the tokenizer')
    parser.add_argument('--overlap', type=float, default=0.5, 
                       help='Overlap fraction for chunking sequences (default: 0.5)')
    # parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
    #                     help='Dataset split to use (train or val)')
    
    args = parser.parse_args()

    # args.eval = True  # Uncomment to force evaluation mode

    args.split = 'train' if not args.eval else 'val'

    # Load the IMU dataset
    print("\n=== Loading Dataset ===")
    imu_data, imu_norm_params = load_dataset(
        args.data_dir, 
        use_6d_rotation=args.use_6d_rotation,
        split=args.split,
        data_type="motion",
        align_first_frame=True
    )
    print(f"Loaded {len(imu_data)} sequences for {args.split} split")

    if args.eval:
        evaluate_tokenizer(args, imu_data, imu_norm_params)
    else:
        print("=== IMU Tokenizer Training ===")
        print(f"Data directory: {args.data_dir}")
        print(f"Save directory: {args.save_dir}")
        print(f"Use 6D rotation: {args.use_6d_rotation}")
        print(f"Chunk length: {args.chunk_length}")
        print(f"Scale: {args.scale}")
        print(f"Vocab size: {args.vocab_size}")
        print(f"Overlap: {args.overlap}")

        os.makedirs(args.save_dir, exist_ok=True)

        hparams = {
            'data_dir': args.data_dir,
            'use_6d_rotation': args.use_6d_rotation,
            'chunk_length': args.chunk_length,
            'scale': args.scale,
            'vocab_size': args.vocab_size,
            'overlap': args.overlap
        }

        hparams_path = os.path.join(args.save_dir, 'hparams.json')
        with open(hparams_path, 'w') as f:
            json.dump(hparams, f, indent=4)
        print(f"Hyperparameters saved to: {hparams_path}")
        
        norm_params_path = os.path.join(args.save_dir, 'imu_norm_params.pkl')
        with open(norm_params_path, 'wb') as f:
            pickle.dump(imu_norm_params, f)
        print(f"Normalization parameters saved to: {norm_params_path}")

        train_tokenizer(args, imu_data)


if __name__ == "__main__":
    main()
