"""Offline rotation smoothing for decoded evaluation poses."""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation


def lowpass_pose(pose, fps, cutoff_hz=3.0, order=4):
    """Filter axis-angle rotations along time, returning float32, same shape.

    Quaternion sign continuity avoids axis-angle wrap artifacts. Filtering is
    forward/backward (uses future frames), then normalized onto unit quaternions.
    Short clips reduce padding; clips with fewer than two frames are unchanged.
    """
    pose = np.asarray(pose)
    if pose.ndim < 2 or np.prod(pose.shape[1:]) % 3:
        raise ValueError(f"Expected [time, ..., 3*n] rotations, got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError("Pose contains non-finite rotations")
    if not np.isfinite(fps) or not np.isfinite(cutoff_hz) or not 0 < cutoff_hz < fps / 2:
        raise ValueError(f"Expected 0 < cutoff_hz < fps/2, got {cutoff_hz=}, {fps=}")
    if not isinstance(order, (int, np.integer)) or order <= 0:
        raise ValueError("Filter order must be a positive integer")
    if len(pose) < 2:
        return pose.astype(np.float32, copy=True)

    q = Rotation.from_rotvec(pose.reshape(-1, 3)).as_quat().reshape(len(pose), -1, 4)
    signs = np.where(np.sum(q[1:] * q[:-1], axis=-1) < 0, -1., 1.)
    q[1:] *= np.cumprod(signs, axis=0)[..., None]
    sos = butter(order, cutoff_hz, fs=fps, output="sos")
    default_padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    filtered = sosfiltfilt(sos, q, axis=0, padlen=min(default_padlen, len(pose)-1))
    # Match the float32 filtered quaternion path used in the paired study.
    filtered = filtered.astype(np.float32).astype(np.float64)
    norm = np.linalg.norm(filtered, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("Degenerate filtered quaternion")
    filtered /= norm
    return Rotation.from_quat(filtered.reshape(-1, 4)).as_rotvec().reshape(pose.shape).astype(np.float32)
