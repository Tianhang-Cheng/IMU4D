"""
This script simulates IMU readings from IMU trajectories.

Reference:
- https://github.com/Xinyu-Yi/GlobalPose/blob/main/imu_synthesis.py
"""

import os
import numpy as np
import pickle
import torch
from tqdm import tqdm

from .utils.simulation import IMUSimulator
from .utils.rotation import convert_rotation

class Constants:
    gW = (0, -9.8, 0)
    mW = (1., 0, 0)

def _walking_noise(shape, std):
    return torch.cumsum(torch.normal(torch.zeros(shape), std), dim=0)


def generate_random_rotation_matrix(n=1):
    r"""
    Generate random rotation matrices. (torch, batch)

    :param n: Number of rotation matrices to generate.
    :return: Random rotation matrices of shape [n, 3, 3].
    """
    q = torch.zeros(n, 4)
    while True:
        n = q.norm(dim=1)
        mask = (n == 0) | (n > 1)
        if q[mask].shape[0] == 0:
            break
        q[mask] = torch.rand_like(q[mask]) * 2 - 1
    q = q / q.norm(dim=1, keepdim=True)
    return convert_rotation(q, 'quat', 'mat')


def simulate_imu_readings(p, R, fps=30, noise_raw_traj=True, noise_syn_imu=True, noise_est_orient=True, skip_ESKF=True, device='cuda'):
    """
    Simulate IMU readings from IMU trajectories.

    Args:
        p: (N, 6, 3) tensor, IMU positions
        R: (N, 6, 3, 3) tensor, IMU orientations
        fps: int, default 30, IMU sampling rate
        noise_raw_traj: bool, whether to add noise to raw IMU trajectory
        noise_syn_imu: bool, whether to add noise to synthesized IMU readings
        noise_est_orient: bool, whether to add noise to estimated orientation
        skip_ESKF: bool, True to use angular velocity integration (faster); otherwise use ESKF (more accurate)

    Returns:
        a_sim: (N, 6, 3) tensor, accelerometer readings in global frame
        w_sim: (N, 6, 3) tensor, gyroscope readings in global frame
        R_sim: (N, 6, 3, 3) tensor, estimated IMU orientations in global frame
        aS: (N, 6, 3) tensor, accelerations readings in sensor frame
        wS: (N, 6, 3) tensor, angular velocities readings in sensor frame
        p_sim: (N, 6, 3) tensor, simulated IMU positions (use for DEBUGGING)
    """
    N = len(p)
    k = np.sqrt(np.pi / 8)

    # simulate IMU trajectory
    if noise_raw_traj:
        # Model calibration error
        # - RBS for static bone-to-sensor mounting error
        # - dR for time-varying calibration error (i.e., sensor drift, mounting looseness)
        RBS = generate_random_rotation_matrix(6).to(device)
        dp = _walking_noise(shape=(N, 6, 3), std=1e-3 * k * np.sqrt(1 / fps)) + torch.randn(6, 3) * 1e-2 * k
        dw = _walking_noise(shape=(N, 6, 3), std=1e-2 * k * np.sqrt(1 / fps)) + torch.randn(6, 3) * 1e-1 * k
        dR = convert_rotation(dw, 'aa', 'mat').view(-1, 6, 3, 3)
        p = p + R.matmul(dp.unsqueeze(-1).to(device)).squeeze(-1)
        R = R.matmul(RBS).matmul(dR.to(device))
    else:
        RBS = torch.eye(3, device=device).expand(6, 3, 3)

    # simulate IMU signals
    imu_simulator = IMUSimulator()
    imu_simulator.set_trajectory(p, R, fps=fps)
    aS = imu_simulator.get_acceleration(gW=Constants.gW)
    wS = imu_simulator.get_angular_velocity()
    mS = imu_simulator.get_magnetic_field(mW=Constants.mW)

    # simulate IMU noise
    if noise_syn_imu:
        # white noise + random walk noise
        aS = torch.normal(aS, std=5e-2) + _walking_noise(shape=(N, 6, 3), std=1e-4 * np.sqrt(1 / fps)).view_as(aS).to(device)
        wS = torch.normal(wS, std=5e-3) + _walking_noise(shape=(N, 6, 3), std=1e-5 * np.sqrt(1 / fps)).view_as(wS).to(device)
        mS = torch.normal(mS, std=5e-3) + _walking_noise(shape=(N, 6, 3), std=1e-5 * np.sqrt(1 / fps)).view_as(mS).to(device)

    # simulate IMU ESKF
    R_sim = torch.empty(N, 6, 3, 3)
    if not skip_ESKF:
        pass
        ### my current implementation is still buggy, skip ESKF for now
        # for i in range(6):
        #     ##### EKF implementation with AHRS package (https://ahrs.readthedocs.io/en/latest/) #####
        #     # TODO: ERROR orientation is not aligned with the original IMU trajectory
        #     # TODO: walking noise is not added
    else:
        # angular velocity integration, much faster for approximate training
        dR = convert_rotation(wS / fps, 'aa', 'mat').view(-1, 6, 3, 3).cpu()
        R_sim[0] = R[0].cpu()
        for i in range(1, N):
            R_sim[i] = R_sim[i - 1].matmul(dR[i])

    # add Gaussian noise
    if noise_est_orient:
        nR = convert_rotation(torch.randn(N, 6, 3) * 0.1 * k, 'aa', 'mat').view(-1, 6, 3, 3)
        R_sim = R_sim.matmul(nR)

    # simulate T-pose calibration
    R_sim = R_sim.to(device)
    a_sim = R_sim.matmul(aS.unsqueeze(-1)).squeeze(-1) + torch.tensor(Constants.gW, device=device)
    w_sim = R_sim.matmul(wS.unsqueeze(-1)).squeeze(-1)
    R_sim = R_sim.matmul(RBS.transpose(1, 2))
    return a_sim, w_sim, R_sim, aS, wS, p



if __name__ == '__main__':

    # Example usage
    data_path = '/home/haoyuyh3/Downloads/data/motionmillion_processed/motionmillion_smpl85/LINGO'
    imu_traj_data = np.load(os.path.join(data_path, 'imu_traj.npy')).astype(np.float32)
    motion_data = np.load(os.path.join(data_path, 'motion_smpl85.npy')).astype(np.float32)
    start_end_dict = pickle.load(open(os.path.join(data_path, 'start_end.pkl'), 'rb'))
    
    # Load sequence data
    seq_idx = 3267
    start, end = start_end_dict[f'LINGO/{seq_idx:05d}']
    data = motion_data[start:end]
    imu_joints_rot = imu_traj_data[start:end, :, 0:3]
    imu_vertices = imu_traj_data[start:end, :, 3:6]
    seq_len = data.shape[0]
    imu_joints_rot = (
        convert_rotation(
            torch.from_numpy(imu_joints_rot.reshape(-1, 3)).float(), 'aa', 'mat'
        ).view(seq_len, -1, 3, 3).numpy().astype(np.float32)
    )

    p = torch.tensor(imu_vertices).cuda()
    R = torch.tensor(imu_joints_rot).cuda()

    noise_raw_traj = True
    noise_syn_imu = True
    noise_est_orient = True
    skip_ESKF = True

    # Simulate IMU readings (accelerometer, gyroscope, magnetometer)
    a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(
        p, R, fps=30,
        noise_raw_traj=noise_raw_traj,
        noise_syn_imu=noise_syn_imu,
        noise_est_orient=noise_est_orient,
        skip_ESKF=skip_ESKF
    )

    R = R.cpu().numpy()          # IMU orient before perturb
    p = p.cpu().numpy()          # IMU position before perturb
    R_sim = R_sim.cpu().numpy()  # IMU orient after perturb
    p_sim = p_sim.cpu().numpy()  # IMU position after perturb

    # Visualize
    # import rerun as rr
    # from .viewer import Viewer

    # viewer = Viewer()

    # dt = 1.0 / viewer.sample_fps
    # seq_len = p_imu.shape[0]
    # IMU_device_names = ['left_hip', 'right_hip', 'left_ear', 'right_ear', 'left_elbow', 'right_elbow']

    # label = 'imu_gt'
    # for imu_idx in range(p_imu.shape[1]):
    #     if IMU_device_names[imu_idx] not in ['left_elbow']:
    #         continue
    #     # Log IMU trajectories
    #     viewer.log_trajectory(p_imu[:, imu_idx], seq_idx, label=label + f'_{IMU_device_names[imu_idx]}', color_idx=imu_idx+3)
    #     # Log IMU orientations
    #     for frame_idx in tqdm(range(seq_len), desc="Logging IMU orientations", dynamic_ncols=True):
    #         rr.set_time_sequence("frames", frame_idx)
    #         rr.set_time_seconds("sensor_time", frame_idx * dt)
    #         transl = p[frame_idx, imu_idx]
    #         orient = R[frame_idx, imu_idx]
    #         viewer._log_orientations(orient, transl, seq_idx, label=label + f'_{IMU_device_names[imu_idx]}', arrow_scale=0.1)

    # label = 'imu_estimated'
    # for imu_idx in range(p_imu.shape[1]):
    #     if IMU_device_names[imu_idx] not in ['left_elbow']:
    #         continue
    #     # Log IMU trajectories
    #     viewer.log_trajectory(p_imu[:, imu_idx], seq_idx, label=label + f'_{IMU_device_names[imu_idx]}', color_idx=imu_idx+9)
    #     # Log IMU orientations
    #     for frame_idx in tqdm(range(seq_len), desc="Logging IMU orientations", dynamic_ncols=True):
    #         rr.set_time_sequence("frames", frame_idx)
    #         rr.set_time_seconds("sensor_time", frame_idx * dt)
    #         transl = p_sim[frame_idx, imu_idx]
    #         orient = R_sim[frame_idx, imu_idx]
    #         viewer._log_orientations(orient, transl, seq_idx, label=label + f'_{IMU_device_names[imu_idx]}', arrow_scale=0.1)