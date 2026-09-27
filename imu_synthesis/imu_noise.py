"""Sensor-realism augmentation for synthetic (virtual) IMU readings.

Motivation (dataset_process/ncsa/README_ncsa_imu.md, "Real-vs-synthetic
diagnosis"): the pretraining IMUs were noise-free second differences of the
mocap trajectory (75-80 % of the acceleration energy above 4 Hz, no mounting /
attitude / bias error), while phone / watch / earbud readings are low-passed on
the device, carry a slowly drifting yaw and a few degrees of mounting error.
This module makes the synthetic side look like a consumer device:

* bandwidth      - per-sensor zero-phase Butterworth low-pass of the sensor-frame
                   acceleration and angular velocity (random cutoff for training,
                   a fixed cutoff for evaluation so metrics stay deterministic);
* ori_static_bias - constant per-sensor bone-to-sensor mounting error (sensor side);
* yaw_drift      - world-side heading random walk of the attitude estimate;
* ori_jitter     - AR(1) attitude-estimation error (as in get_imu_readings);
* acc_white / acc_bias, gyro_white / gyro_bias - additive sensor noise;
* time_shift     - integer per-sensor / IMU-vs-motion frame misalignment
                   (applied by ``training.imu_dataset.process_imu_data`` through
                   :func:`crop_margin` and :func:`sample_time_shifts`);
* heading_conjugation - per-sensor yaw conjugation ``Y(t) R Y(-t)`` / ``Y(t) a``:
                   the exact effect of a T-pose-only calibration on a device with an
                   arbitrary yaw reference; the model must infer ``t`` from motion.

Every stochastic magnitude is multiplied by one per-sample factor
``U(0, 1)`` when ``random_magnitude`` is on, so the model keeps seeing
near-clean inputs too.  The attitude error rotates the world-frame
acceleration as well (gravity leakage), which is where most real acceleration
bias comes from.

All settings live under ``training.imu_noise`` (see configs/defaults.yaml).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
from scipy.signal import butter, filtfilt, lfilter

from .get_imu_readings import Constants
from .utils.rotation import convert_rotation
from .utils.simulation import IMUSimulator

DEFAULT_IMU_NOISE: dict[str, Any] = {
    "enabled": False,
    # One U(0, 1) factor per sample scaling every stochastic magnitude below.
    "random_magnitude": True,
    # Constant per-sensor bone-to-sensor rotation error, per-axis std (deg).
    "ori_static_bias_deg": 5.0,
    # World-side yaw random walk of the attitude estimate: std reached after
    # one minute (deg).
    "yaw_drift_deg_per_min": 4.0,
    # AR(1) attitude jitter: stationary per-axis std (deg) and coefficient.
    "ori_jitter_deg": 1.5,
    "ori_jitter_rho": 0.95,
    # Sensor-frame accelerometer white noise (m/s^2) and constant world-frame
    # per-sensor bias (per-axis std, m/s^2).
    "acc_white_std": 0.15,
    "acc_bias_std": 0.2,
    # Sensor-frame gyroscope white noise and constant bias (rad/s).
    "gyro_white_std": 0.015,
    "gyro_bias_std": 0.005,
    # Integer frame misalignments: per sensor in [-k, k] and IMU-vs-motion in
    # [-k, k] (30 Hz frames).
    "time_shift_sensor_frames": 1,
    "time_shift_gt_frames": 1,
    # Per-sensor heading conjugation R -> Y(t) R Y(-t), a -> Y(t) a, with
    # t ~ U(-deg, deg), applied to each sensor independently with this
    # probability. Models a device whose arbitrary yaw reference was removed on
    # the sensor side by a T-pose-only calibration (NCSA phone / watch / earbud);
    # the network has to infer t from motion consistency. 0 disables.
    "heading_conjugation_prob": 0.5,
    "heading_conjugation_deg": 180.0,
    # Training low-pass cutoff range (Hz, uniform per sensor).  Null disables.
    "lowpass_hz": [3.0, 10.0],
    # Fixed cutoff for every non-train split (Hz).  Null disables.
    "eval_lowpass_hz": 8.0,
    "lowpass_order": 2,
}


def resolve_imu_noise_cfg(cfg: Any) -> Optional[dict[str, Any]]:
    """Merge ``cfg`` with :data:`DEFAULT_IMU_NOISE`; ``None`` when disabled."""
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        # OmegaConf DictConfig or any mapping.
        cfg = {str(k): v for k, v in dict(cfg).items()}
    merged = dict(DEFAULT_IMU_NOISE)
    unknown = set(cfg) - set(merged)
    if unknown:
        raise KeyError(f"Unknown training.imu_noise keys: {sorted(unknown)}")
    merged.update(cfg)
    if not bool(merged["enabled"]):
        return None
    for key in ("lowpass_hz",):
        if merged[key] is not None:
            values = [float(v) for v in merged[key]]
            if len(values) != 2 or values[0] <= 0 or values[1] < values[0]:
                raise ValueError(f"training.imu_noise.{key} must be [low, high] Hz, got {merged[key]}")
            merged[key] = values
    return merged


def crop_margin(cfg: Optional[dict[str, Any]]) -> int:
    """Extra frames to crop on each side so time shifts stay inside the clip."""
    if cfg is None:
        return 0
    return int(cfg["time_shift_sensor_frames"]) + int(cfg["time_shift_gt_frames"])


def sample_time_shifts(cfg: dict[str, Any], rng: np.random.Generator, num_sensors: int = 6) -> np.ndarray:
    """Per-sensor frame offsets of the IMU window relative to the motion window."""
    k_gt = int(cfg["time_shift_gt_frames"])
    k_sensor = int(cfg["time_shift_sensor_frames"])
    gt_shift = int(rng.integers(-k_gt, k_gt + 1)) if k_gt > 0 else 0
    sensor_shift = rng.integers(-k_sensor, k_sensor + 1, size=num_sensors) if k_sensor > 0 else np.zeros(num_sensors, dtype=int)
    return (gt_shift + sensor_shift).astype(int)


def lowpass_filter(x: np.ndarray, fps: float, cutoffs_hz: np.ndarray, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0 of ``x`` [N, S, C], one cutoff per sensor."""
    n = x.shape[0]
    out = np.array(x, dtype=np.float64, copy=True)
    nyquist = 0.5 * float(fps)
    for s in range(x.shape[1]):
        cutoff = float(cutoffs_hz[s])
        if not math.isfinite(cutoff) or cutoff >= 0.98 * nyquist:
            continue  # nothing to remove at 30 Hz
        b, a = butter(int(order), cutoff / nyquist, btype="low")
        padlen = min(3 * max(len(a), len(b)), n - 1)
        if padlen < 1:
            continue
        out[:, s] = filtfilt(b, a, out[:, s], axis=0, padlen=padlen)
    return out.astype(np.float32)


def _yaw_matrices(phi: torch.Tensor) -> torch.Tensor:
    """Rotation about +Y (world up) for angles ``phi`` [...]."""
    c, s = torch.cos(phi), torch.sin(phi)
    z, o = torch.zeros_like(phi), torch.ones_like(phi)
    return torch.stack(
        [torch.stack([c, z, s], -1), torch.stack([z, o, z], -1), torch.stack([-s, z, c], -1)], -2
    )


def simulate_noisy_imu_readings(
    p: torch.Tensor,
    R: torch.Tensor,
    fps: float,
    cfg: dict[str, Any],
    train: bool,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Virtual IMU with device-like imperfections.

    Args:
        p: [N, S, 3] sensor positions (world, first-frame aligned).
        R: [N, S, 3, 3] true sensor orientations (world <- sensor).
        fps: sampling rate.
        cfg: resolved config (:func:`resolve_imu_noise_cfg`, not None).
        train: stochastic terms on; otherwise only the fixed eval low-pass.
        rng: numpy generator (seed it from ``random`` inside dataloader workers).

    Returns:
        a_sim [N, S, 3] world-frame gravity-free acceleration,
        w_sim [N, S, 3] world-frame angular velocity,
        R_sim [N, S, 3, 3] estimated sensor orientation.
    """
    p = p.float().cpu()
    R = R.float().cpu()
    n, num_sensors = R.shape[0], R.shape[1]
    g_w = torch.tensor(Constants.gW, dtype=torch.float32)

    simulator = IMUSimulator()
    simulator.set_trajectory(p, R, fps=fps)
    a_s = simulator.get_acceleration(gW=Constants.gW)  # sensor frame, with gravity
    w_s = simulator.get_angular_velocity()  # sensor frame

    # --- bandwidth -----------------------------------------------------------
    cutoffs = None
    if train and cfg["lowpass_hz"] is not None:
        cutoffs = rng.uniform(cfg["lowpass_hz"][0], cfg["lowpass_hz"][1], size=num_sensors)
    elif not train and cfg["eval_lowpass_hz"] is not None:
        cutoffs = np.full(num_sensors, float(cfg["eval_lowpass_hz"]))
    if cutoffs is not None:
        a_s = torch.from_numpy(lowpass_filter(a_s.numpy(), fps, cutoffs, cfg["lowpass_order"]))
        w_s = torch.from_numpy(lowpass_filter(w_s.numpy(), fps, cutoffs, cfg["lowpass_order"]))

    if not train:
        a_sim = R.matmul(a_s.unsqueeze(-1)).squeeze(-1) + g_w
        w_sim = R.matmul(w_s.unsqueeze(-1)).squeeze(-1)
        return a_sim, w_sim, R

    scale = float(rng.uniform()) if bool(cfg["random_magnitude"]) else 1.0

    def normal(std: float, *shape: int) -> torch.Tensor:
        return torch.from_numpy(rng.normal(0.0, std * scale, size=shape).astype(np.float32))

    # --- additive sensor noise (sensor frame) ---------------------------------
    w_s = w_s + normal(cfg["gyro_white_std"], n, num_sensors, 3) + normal(cfg["gyro_bias_std"], 1, num_sensors, 3)
    a_s = a_s + normal(cfg["acc_white_std"], n, num_sensors, 3)

    # --- attitude estimation error --------------------------------------------
    rho = float(cfg["ori_jitter_rho"])
    jitter_std = math.radians(float(cfg["ori_jitter_deg"])) * scale
    eps = rng.normal(0.0, jitter_std * math.sqrt(max(1.0 - rho * rho, 1e-6)), size=(n, num_sensors, 3))
    err_aa = lfilter([1.0], [1.0, -rho], eps, axis=0).astype(np.float32)
    jitter = convert_rotation(torch.from_numpy(err_aa).reshape(-1, 3), "aa", "mat").reshape(n, num_sensors, 3, 3)

    drift_step = math.radians(float(cfg["yaw_drift_deg_per_min"])) * scale / math.sqrt(60.0 * float(fps))
    phi = torch.from_numpy(np.cumsum(rng.normal(0.0, drift_step, size=(n, num_sensors)), axis=0).astype(np.float32))
    yaw = _yaw_matrices(phi)  # [N, S, 3, 3]

    r_est = yaw.matmul(R).matmul(jitter)

    # --- constant mounting error (orientation channel only) ------------------
    bias_aa = normal(math.radians(float(cfg["ori_static_bias_deg"])), num_sensors, 3)
    mounting = convert_rotation(bias_aa, "aa", "mat").reshape(1, num_sensors, 3, 3)

    a_sim = r_est.matmul(a_s.unsqueeze(-1)).squeeze(-1) + g_w + normal(cfg["acc_bias_std"], 1, num_sensors, 3)
    w_sim = r_est.matmul(w_s.unsqueeze(-1)).squeeze(-1)
    r_sim = r_est.matmul(mounting)

    # --- T-pose-only heading calibration (per-sensor yaw conjugation) --------
    prob = float(cfg["heading_conjugation_prob"])
    if prob > 0.0:
        hit = rng.uniform(size=num_sensors) < prob
        theta = rng.uniform(-1.0, 1.0, size=num_sensors) * math.radians(float(cfg["heading_conjugation_deg"]))
        theta = torch.from_numpy(np.where(hit, theta, 0.0).astype(np.float32))
        yaw_c = _yaw_matrices(theta).unsqueeze(0)  # [1, S, 3, 3]
        r_sim = yaw_c.matmul(r_sim).matmul(yaw_c.transpose(-1, -2))
        a_sim = yaw_c.matmul(a_sim.unsqueeze(-1)).squeeze(-1)  # gravity is along Y: unchanged
        w_sim = yaw_c.matmul(w_sim.unsqueeze(-1)).squeeze(-1)
    return a_sim, w_sim, r_sim
