from typing import Any, Callable, Optional

import numpy as np
from dataset_process.asset_frames import asset_key, canonicalize_track
from dataset_process.ground_plane import resolve_ground_plane
import pickle
from scipy.spatial.transform import Rotation as R
import torch
import sys
import os
import random
import json
import glob
import random
from typing import Tuple
from bisect import bisect_left, bisect_right
from dataset_process.custom_path import humoto_root, parahome_root, imuposer_root, dipimu_root

def smooth_avg(acc, s=3):
    """
    Smooth data using a centered moving average.
    
    Args:
        acc: numpy array of shape [time, n_device, n_value]
        s: window size (should be odd for symmetric smoothing)
    
    Returns:
        Smoothed array of same shape as input
    """
    nan_array = np.full((s // 2, acc.shape[1], acc.shape[2]), np.nan)
    acc = np.concatenate((nan_array, acc, nan_array), axis=0)
    arrays = []
    for i in range(s):
        L = acc.shape[0]
        arrays.append(acc[i:L-(s-i-1)])
    smoothed = np.nanmean(np.stack(arrays, axis=0), axis=0)
    return smoothed

def intersecting_indices(intervals, A, B, *, inclusive=False):
    """
    Return indices of intervals that intersect with (A, B) (open) by default.
    If inclusive=True, treat it as [A, B] (closed) for endpoint-touching.

    Assumes intervals are sorted, non-overlapping: a0<=b0<=a1<=b1<=...
    """
    if A >= B:
        return []

    a = [x[0] for x in intervals]
    b = [x[1] for x in intervals]

    if inclusive:
        # Overlap if a_i <= B and b_i >= A
        left = bisect_left(b, A)        # first i with b_i >= A
        right = bisect_right(a, B) - 1  # last  i with a_i <= B
    else:
        # Overlap if a_i < B and b_i > A (open interval)
        left = bisect_right(b, A)       # first i with b_i > A
        right = bisect_left(a, B) - 1   # last  i with a_i < B

    return list(range(left, right + 1)) if left <= right else []

def random_contiguous_subarray_bounds(N, A):
    """Return Python slice bounds [start, end) for A frames (or all N if shorter)."""
    return _random_contiguous_subarray_bounds(N, A, A)

def _random_contiguous_subarray_bounds(N: int, A: int, B: int, *, rng: random.Random | None = None) -> Tuple[int, int]:
    """
    Return (start_idx, end_idx) with an exclusive end for a random subarray.

    - N: sequence length
    - A, B: integer length bounds (inclusive)
    - rng: optional random.Random instance for reproducibility
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}.")
    if not (isinstance(A, int) and isinstance(B, int)):
        raise TypeError("A and B must be integers.")
    if A <= 0 or B <= 0:
        raise ValueError("A and B must be positive.")

    # if sequence too short, return the full range
    if N < A:
        return 0, N

    # clamp B to at most N
    B = min(B, N)

    r = rng if rng is not None else random
    M = r.randint(A, B)
    start = r.randint(0, N - M)
    end = start + M
    return start, end

# sys.path.append('/scratch/benk/tcheng1/code/imu-human-mllm/')
from imu_synthesis.get_imu_readings import simulate_imu_readings
from imu_synthesis.imu_noise import (
    crop_margin,
    resolve_imu_noise_cfg,
    sample_time_shifts,
    simulate_noisy_imu_readings,
)
from imu_synthesis.utils.rotation import convert_rotation
from dataset_process.motionmillion_and_lingo.convert_lingo_dataset import align_poses_to_first_frame

# Eval-only virtual-IMU replacement (evaluation/imu_body_shape.py): a pickle
# {"imu_traj": {sample_id: [T, 6, 6]}} re-simulated on another body shape.
# Passed by environment so forked loader workers see it; unset = off.
ENV_EVAL_IMU_TRAJ = "IMU4D_EVAL_IMU_TRAJ"
_EVAL_IMU_TRAJ = None


def _eval_imu_traj_override(sample, split):
    global _EVAL_IMU_TRAJ
    path = os.environ.get(ENV_EVAL_IMU_TRAJ, "").strip()
    if not path:
        return sample['imu_traj']
    if split == 'train':
        raise ValueError(f"{ENV_EVAL_IMU_TRAJ} is evaluation-only")
    if _EVAL_IMU_TRAJ is None:
        with open(path, 'rb') as f:
            _EVAL_IMU_TRAJ = pickle.load(f)['imu_traj']
    sid = sample.get('id')
    if sid not in _EVAL_IMU_TRAJ:
        raise KeyError(f"{ENV_EVAL_IMU_TRAJ}: no imu_traj for sample {sid!r} in {path}")
    traj = _EVAL_IMU_TRAJ[sid]
    if traj.shape != sample['imu_traj'].shape:
        raise ValueError(f"{ENV_EVAL_IMU_TRAJ}: {sid!r} has {traj.shape}, sample has {sample['imu_traj'].shape}")
    return traj

def align_translation(inv_rotation, inv_translation, translation):
    aligned_transl = (inv_rotation @ (translation + inv_translation).T).T  # Apply inverse rotation and translation
    return aligned_transl

def align_orientation(inv_rotation, orientation):
    aligned_orient = R.from_rotvec(orientation).as_matrix()
    aligned_orient = np.einsum('ij,njk->nik', inv_rotation, aligned_orient)
    aligned_orient = R.from_matrix(aligned_orient).as_rotvec()
    aligned_orient = R.from_rotvec(aligned_orient).as_rotvec()
    return aligned_orient

# use max's split
def filter_by_targets(all_strings, target_datasets):
    before_len = len(all_strings)
    filtered_strings = [
        s for s in all_strings 
        if any(t in s for t in target_datasets)
    ]
    after_len = len(filtered_strings)
    print(f"Filtered {before_len - after_len} out of {before_len} data")
    return filtered_strings

def add_velocity_scaled_noise(x, eps=0.01):
    """
    Noise magnitude follows local velocity.
    Supports np.ndarray and torch.Tensor.
    """
    if torch.is_tensor(x):
        # torch version
        vel = torch.diff(x, dim=0, prepend=x[:1])
        noise = torch.randn_like(x)
        noise = noise.clip(-1, 1)
        return x + eps * vel * noise
    else:
        # numpy version
        vel = np.diff(x, axis=0, prepend=x[:1])
        noise = np.random.randn(*x.shape)
        noise = np.clip(noise, -1, 1)
        return x + eps * vel * noise


def _yaw_rot_matrix_y(phi_deg: float) -> np.ndarray:
    """3x3 rotation about the world vertical (+Y axis) by ``phi_deg`` degrees."""
    a = np.deg2rad(float(phi_deg))
    c, s = np.cos(a), np.sin(a)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _rot_vec3_y(v: np.ndarray, Ry: np.ndarray) -> np.ndarray:
    """Premultiply world-frame 3-vectors (last dim == 3) by Ry."""
    return np.einsum("ij,...j->...i", Ry, v, optimize=True)


def _rot_rot_rep_y(x: np.ndarray, rep: str, Ry: np.ndarray) -> np.ndarray:
    """Rotate a world-frame orientation stored as aa(3) / 6d(6) / flat-9 about +Y.

    ``Ry`` premultiplies world coordinates, matching how ``align_poses_to_first_frame``
    and the IMU simulation rotate orientation matrices in this pipeline. ``rep`` is one
    of "aa", "6d", "mat9" (``mat9`` = 3x3 matrix flattened to 9).
    """
    if rep == "mat9":
        m = x.reshape(*x.shape[:-1], 3, 3)
        m = np.einsum("ij,...jk->...ik", Ry, m, optimize=True)
        return m.reshape(*x.shape[:-1], 9).astype(x.dtype)
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    mat = convert_rotation(t, rep, "mat")
    mat = torch.einsum("ij,...jk->...ik", torch.as_tensor(Ry), mat)
    out = convert_rotation(mat, "mat", rep)
    return out.numpy().astype(x.dtype)


def _rotate_imu_data_y(imu_data: "torch.Tensor", Ry: np.ndarray) -> "torch.Tensor":
    """Rotate the canonical IMU channels [a, w, R(9)] about +Y by Ry."""
    x = imu_data.numpy()
    xc = x.copy()
    xc[..., :3] = _rot_vec3_y(x[..., :3], Ry)
    xc[..., 3:6] = _rot_vec3_y(x[..., 3:6], Ry)
    xc[..., 6:15] = _rot_rot_rep_y(x[..., 6:15], "mat9", Ry)
    return torch.from_numpy(xc).float()


DEFAULT_REAL_HEADING_AUG: dict = {
    "enabled": False,
    # Probability that a given physical device is conjugated, and the range of
    # the angle (uniform in +-deg).  Same semantics as
    # ``training.imu_noise.heading_conjugation_*`` on the synthetic side.
    "prob": 0.5,
    "deg": 180.0,
    # Which samples the conjugation may touch:
    #   "all"                  historical behaviour, every real-IMU sample.
    #   "unsupervised_orient"  only samples whose world heading / trajectory
    #                          label is masked anyway (``motion_supervise``).
    # The augmentation randomises the world frame each *device* reports in, so
    # on a sample whose orient/traj label IS supervised it feeds the model an
    # input whose heading no longer matches the target it is asked to predict.
    # That was invisible while NCSA was scored by root-relative MPJPE (which
    # zeroes the root orientation) and only became measurable with the
    # meeting-room global metrics -- see README_ncsa_imu.md.
    "scope": "all",
}
REAL_HEADING_AUG_SCOPES = ("all", "unsupervised_orient")


def _device_slot_groups(sample: dict, num_sensors: int) -> list[list[int]]:
    """Model slots grouped by the physical device that produced them.

    Real captures duplicate one device into two slots (the head/earbud is copied
    to both ear slots, ``imu_duplicated_sensor_slot``), and a heading error is a
    property of the *device*: giving the two copies different angles would be
    physically impossible and teaches the model an inconsistency.  All three real
    datasets carry ``imu_source_devices``; without it every slot is its own group.
    """
    devices = sample.get('imu_source_devices')
    if not devices:
        return [[slot] for slot in range(num_sensors)]
    groups: dict[str, list[int]] = {}
    for slot, device in enumerate(devices[:num_sensors]):
        if str(device) == 'missing':
            continue
        groups.setdefault(str(device), []).append(slot)
    return list(groups.values())


def apply_real_heading_aug(a_sim: np.ndarray, w_sim: np.ndarray, R_sim: np.ndarray,
                           sample: dict, cfg: dict, motion_supervise: dict | None = None) -> np.ndarray:
    """Random per-device heading conjugation of measured readings (in place-ish).

    A device whose arbitrary yaw reference was calibrated away on the *sensor*
    side reports ``Y(t) R Y(-t)`` / ``Y(t) a`` with an unknown ``t`` (see
    ``dataset_process/ncsa/README_ncsa_imu.md``).  Without this augmentation the
    real-IMU fine-tune sees each training session's own fixed ``t`` over and over
    and can memorise it per session, which is why an oracle-calibrated fine-tune
    beats the uncalibrated one by ~8 mm and why a model fine-tuned on
    uncalibrated data degrades when it is later given calibrated input.
    Randomising ``t`` per device makes the heading something the model has to
    infer from motion consistency instead.  Returns the sampled angles (deg).

    ``motion_supervise`` is the sample's resolved label-trust flag; with
    ``cfg["scope"] == "unsupervised_orient"`` the conjugation is skipped for
    samples whose world heading / trajectory is actually supervised, because
    there the randomised input no longer matches the unchanged label.
    """
    prob = float(cfg.get("prob", 0.5))
    deg = float(cfg.get("deg", 180.0))
    scope = str(cfg.get("scope", "all"))
    if scope not in REAL_HEADING_AUG_SCOPES:
        raise ValueError(f"real_imu_heading_aug.scope must be one of {REAL_HEADING_AUG_SCOPES}, got {scope!r}")
    angles = np.zeros(a_sim.shape[1], dtype=np.float32)
    if prob <= 0.0 or deg == 0.0:
        return angles
    if scope == "unsupervised_orient":
        supervise = motion_supervise or {}
        if bool(supervise.get("orient", True)) or bool(supervise.get("traj", True)):
            return angles
    for slots in _device_slot_groups(sample, a_sim.shape[1]):
        if random.random() >= prob:
            continue
        theta = random.uniform(-deg, deg)
        Ry = _yaw_rot_matrix_y(theta)
        for slot in slots:
            angles[slot] = theta
            a_sim[:, slot] = _rot_vec3_y(a_sim[:, slot], Ry)
            w_sim[:, slot] = _rot_vec3_y(w_sim[:, slot], Ry)
            R_sim[:, slot] = np.einsum('ij,njk,lk->nil', Ry, R_sim[:, slot], Ry)
    return angles


def _yaw_aug_enabled(yaw_aug) -> bool:
    """Resolve the train-time yaw augmentation switch (default OFF; opt in per stage).

    Either an explicit ``yaw_aug`` argument (``True`` enables), or the
    ``IMU4D_YAW_AUG`` environment flag (``1``/``true`` enables). Unset means OFF:
    the random world heading is a stage-2 (device-realism) ingredient, so stage 1
    trains on the GT-pelvis canonical heading and only the stage-2 scripts export
    the flag. Only synthetic WDS training flows through ``process_imu_data`` with
    ``split == 'train'``, so this never touches evaluation. Note: the flag/param
    is read inside data workers, so a running training keeps its launch-time
    setting until restarted.
    """
    if yaw_aug is not None:
        return bool(yaw_aug)
    return os.environ.get("IMU4D_YAW_AUG", "0") in ("1", "true", "True")


def get_angular_velocity(R_sim: np.ndarray, fps: float) -> np.ndarray:
    r"""
    Compute sensor-local angular velocity from rotation matrices.

    Parameters
    ----------
    R_sim : np.ndarray
        Rotation matrices with shape [T, ..., 3, 3],
        where ... represents arbitrary batch dimensions.
    fps : float
        Sampling frequency (frames per second).

    Returns
    -------
    np.ndarray
        Angular velocity with shape [T, ..., 3].
    """
    # Central differences for interior points
    Rdot_mid = R_sim[2:] - R_sim[:-2]

    # Forward / backward differences for boundaries
    Rdot0 = -3 * R_sim[0] + 4 * R_sim[1] - R_sim[2]
    Rdot1 =  3 * R_sim[-1] - 4 * R_sim[-2] + R_sim[-3]

    # Concatenate along time dimension
    Rdot = np.concatenate(
        (
            Rdot0[np.newaxis],
            Rdot_mid,
            Rdot1[np.newaxis],
        ),
        axis=0,
    )

    # Scale by timestep
    Rdot = Rdot * (fps / 2.0)

    # Body-frame angular velocity hat matrix
    Rt = np.swapaxes(R_sim, -1, -2)
    w_hat = Rt @ Rdot

    # Enforce skew-symmetry
    w_hat = 0.5 * (w_hat - np.swapaxes(w_hat, -1, -2))

    # vee operator
    w = np.stack(
        (
            w_hat[..., 2, 1],
            w_hat[..., 0, 2],
            w_hat[..., 1, 0],
        ),
        axis=-1,
    )

    return w

REAL_IMU_SOURCES = ('imuposer', 'dipimu', 'ncsa')
REAL_IMU_GRAVITY = np.array((0, -9.8, 0), dtype=np.float32)


def _real_imu_object_targets(objects, inv_rotation, inv_translation, start_cut_idx, end_cut_idx,
                             dynamic_object, data_source, object_metadata=None, source=None):
    """Object annotations -> {'rot', 'transl', 'bbox'} targets (same rules as process_imu_data).

    Real-world samples only carry the synthetic ground plane, but the helper
    accepts the generic ``[D]`` / ``[T, D]`` layouts so mixed batches stay uniform.
    """
    n_frames = end_cut_idx - start_cut_idx
    obj_pose_dict = {}
    for obj in objects.keys():
        obj_name_filtered = obj.split('.')[0]  # remove the suffix
        assert obj == obj_name_filtered
        obj_track = np.asarray(objects[obj])
        if source is not None and obj != 'ground':
            # Per-asset upright frame (dataset_process/asset_canonical_frames.json);
            # identity for assets without an entry. World geometry is unchanged.
            obj_track = canonicalize_track(
                obj_track, asset_key(source, obj, (object_metadata or {}).get(obj))
            )
        if dynamic_object:
            if obj_track.ndim == 1:
                obj_track = np.repeat(obj_track[None], n_frames, axis=0)
            elif obj_track.ndim == 2:
                obj_track = obj_track[start_cut_idx:end_cut_idx]
            else:
                raise ValueError(f"Object {obj!r} must have shape [D] or [T,D], got {obj_track.shape}")
            if len(obj_track) != n_frames:
                raise ValueError(f"Object/body crop mismatch for {obj!r}: {len(obj_track)} != {n_frames}")
            obj_quat = obj_track[:, 0:4].copy().astype(np.float32)
            obj_mat = convert_rotation(torch.from_numpy(obj_quat), 'quat', 'mat').float().cpu().numpy()  # [t,3,3]
            obj_mat = np.einsum('ij,tjk->tik', inv_rotation, obj_mat)
            obj_6d = convert_rotation(torch.from_numpy(obj_mat), 'mat', '6d').float().cpu().numpy()  # [t,6]
            obj_transl = obj_track[:, 4:7].copy()
            obj_transl = np.einsum('ij,tj->ti', inv_rotation, obj_transl + inv_translation)
            obj_bbox = (
                obj_track[:, 7:10].copy().astype(np.float32)
                if obj_track.shape[1] >= 10
                else np.ones((n_frames, 3), dtype=np.float32)
            )
            if obj == 'ground' and data_source != 'parahome':
                obj_transl[:, 0] = 0.0
                obj_transl[:, 2] = 0.0
        else:
            if obj_track.ndim == 2:
                obj_track = obj_track[start_cut_idx]
            elif obj_track.ndim != 1:
                raise ValueError(f"Object {obj!r} must have shape [D] or [T,D], got {obj_track.shape}")
            obj_quat = obj_track[0:4].copy().astype(np.float32)
            obj_mat = convert_rotation(torch.from_numpy(obj_quat), 'quat', 'mat').float().cpu().numpy()  # [3,3]
            obj_mat = np.einsum('ij,jk->ik', inv_rotation, obj_mat)
            obj_6d = convert_rotation(torch.from_numpy(obj_mat), 'mat', '6d').float().cpu().numpy()  # [6]
            obj_transl = obj_track[4:7].copy()
            obj_transl = np.einsum('ij,j->i', inv_rotation, obj_transl + inv_translation)
            obj_bbox = (
                obj_track[7:10].copy().astype(np.float32)
                if obj_track.shape[0] >= 10
                else np.ones((3,), dtype=np.float32)
            )
            if obj == 'ground' and data_source != 'parahome':
                # set the ground to be at the origin, only keep the y-axis
                plane_y = obj_transl[1]
                obj_transl = np.array([0.0, plane_y, 0.0], dtype=np.float32)
        obj_pose_dict[obj_name_filtered] = {'rot': obj_6d, 'transl': obj_transl, 'bbox': obj_bbox}
    return obj_pose_dict


DEFAULT_SHORT_WINDOW_GLOBAL_SUPERVISION: dict = {
    "enabled": False,
    # Windows of at most this many frames restore the world heading / trajectory
    # target on samples whose label carries it masked (``ncsa/v3``). The
    # evaluator canonicalises every window to its own first frame, so a static
    # camera's *constant* heading error is already gone; what the mask exists
    # for is the error that ACCUMULATES inside the window (GVHMR static-camera
    # leak, measured at 2-4 deg/s on v3 -> 4-8 deg over 60 frames, 30-60 deg
    # over 480). Short windows are therefore usable supervision.
    "max_frames": 60,
}


def process_real_imu_data(sample, random_cut, random_mask_text, cut_length, shift=0,
                          filter_short_text=True, add_ground_data=False, rot_rep='6d', split=None,
                          data_source=None, dynamic_object=False, fps=30, sample_idx=None, IMUSEQMAXLEN=1e6,
                          acc_scale=1.0, gyro_scale=1.0, smooth_imu_acc=True, motion_only=False,
                          heading_aug=None, short_window_global_supervision=None):
    """Crop / align / featurise one real-IMU sample (IMUPoser, DIP-IMU).

    ``sample`` follows ``dataset_process/realworld/realworld_io.py``:
    ``motion_data_smpl85`` [T, 85] (or the legacy ``motion_smpl`` [T, 69]),
    ``imu_acc`` [T, 6, 3] and ``imu_ori`` [T, 6, 3, 3] already in model sensor
    order, plus ``imu_acc_smoothed``. Measured readings are used as-is (no
    virtual-IMU simulation, no synthetic noise); the crop is aligned to its
    first frame like every other dataset, the same rotation is applied to the
    sensor readings, the angular velocity is differentiated from the orientation
    and the ``(0, -9.8, 0)`` gravity offset is added back, matching the historical
    ``process_imuposer_data`` / ``process_dipimu_data`` numerics exactly.
    """
    assert data_source in REAL_IMU_SOURCES, f"Invalid real-IMU data source: {data_source}"

    if 'motion_data_smpl85' in sample:
        motion_smpl = np.asarray(sample['motion_data_smpl85'])
    elif 'motion_smpl' in sample:
        motion_smpl = np.asarray(sample['motion_smpl'])
    else:
        raise ValueError(f"Invalid motion smpl data: {sample.keys()}")
    transl_slice = slice(72, 75) if motion_smpl.shape[1] == 85 else slice(66, 69)
    imu_acc = np.asarray(sample['imu_acc'])  # [T, 6, 3]
    imu_ori = np.asarray(sample['imu_ori'])  # [T, 6, 3, 3]

    n_time = min(len(motion_smpl), len(imu_acc), len(imu_ori))
    assert cut_length is not None
    if n_time < cut_length and split == 'train':
        return None

    smpl_orient = motion_smpl[:n_time, 0:3].copy()
    smpl_pose = motion_smpl[:n_time, 3:66].copy()
    smpl_transl = motion_smpl[:n_time, transl_slice].copy()

    if random_cut and split == 'train':
        assert n_time >= cut_length, "cut_length must be greater than or equal to n_time"
        start_cut_idx, end_cut_idx = random_contiguous_subarray_bounds(n_time, cut_length)
    else:
        start_cut_idx = 0
        end_cut_idx = min(n_time, cut_length)

    start_cut_idx = start_cut_idx + shift
    end_cut_idx = end_cut_idx + shift

    smpl_orient = smpl_orient[start_cut_idx:end_cut_idx]
    smpl_pose = smpl_pose[start_cut_idx:end_cut_idx]
    smpl_transl = smpl_transl[start_cut_idx:end_cut_idx]

    # align the smpl poses to the first frame
    smpl_orient_new, smpl_transl_new, smpl_pose_new, inv_rotation, inv_translation = \
        align_poses_to_first_frame(smpl_orient, smpl_transl, smpl_pose)

    a_sim = imu_acc[start_cut_idx:end_cut_idx]  # [n, 6, 3]
    R_sim = imu_ori[start_cut_idx:end_cut_idx]  # [n, 6, 3, 3]
    if smooth_imu_acc and not bool(sample.get('imu_acc_smoothed', False)):
        a_sim = smooth_avg(a_sim, s=3)

    # apply the same first-frame alignment to the measured readings
    a_sim = np.einsum('ij,bnj->bni', inv_rotation, a_sim)
    R_sim = np.einsum('ij,bnjk->bnik', inv_rotation, R_sim)
    w_sim = get_angular_velocity(R_sim, fps=fps)

    if bool(sample.get('imu_acc_add_gravity', True)):
        a_sim = a_sim + REAL_IMU_GRAVITY  # add gravity back (historical numerics; see ncsa README diagnosis)

    # The legacy NCSA v1 shards predate ``motion_supervise``. Their PromptHMR
    # global translation / heading are not valid supervision; retain only the
    # local joint-pose target. Newer meeting-room NCSA samples carry an explicit
    # all-True flag and therefore keep full supervision.  Resolved here because
    # the heading augmentation below is gated on it.
    motion_supervise = sample.get('motion_supervise')
    if motion_supervise is None and data_source == 'ncsa':
        motion_supervise = {'traj': False, 'orient': False, 'pose': True}
    motion_supervise = dict(motion_supervise or {'traj': True, 'orient': True, 'pose': True})

    # Short windows can keep the global targets even when the clip's label has
    # them masked: the crop is canonicalised to its own first frame, so only the
    # error accumulated *inside* the window survives (see
    # DEFAULT_SHORT_WINDOW_GLOBAL_SUPERVISION). Train split only -- evaluation
    # must keep scoring what the stored flag says.
    _short = short_window_global_supervision or {}
    if bool(_short.get("enabled", False)) and split == 'train':
        _window_frames = end_cut_idx - start_cut_idx
        if _window_frames <= int(_short.get("max_frames", 60)):
            motion_supervise = {**motion_supervise, 'traj': True, 'orient': True}

    # Train-time per-device heading conjugation (gravity is along Y, so the
    # offset above is unaffected by the yaw rotation).
    heading_aug_angles = None
    if heading_aug and bool(heading_aug.get("enabled", False)) and split == 'train':
        heading_aug_angles = apply_real_heading_aug(
            a_sim, w_sim, R_sim, sample, heading_aug, motion_supervise=motion_supervise
        )

    objects = dict(sample.get('objects') or {})
    ground_record = None
    if add_ground_data:
        if 'ground' in objects:
            raise ValueError("Ground data already exists")
        ground_record = resolve_ground_plane(sample, motion_smpl, data_source, fps)
        objects['ground'] = np.array([1.0, 0.0, 0.0, 0.0,
                                      0.0, ground_record['height_y'], 0.0,
                                      1.0, 1.0, 1.0], dtype=np.float32)
    obj_pose_dict = _real_imu_object_targets(
        objects, inv_rotation, inv_translation, start_cut_idx, end_cut_idx, dynamic_object, data_source,
        object_metadata=sample.get('object_metadata'), source=sample.get('source'),
    )

    smpl_orient_new = torch.tensor(smpl_orient_new).float()
    smpl_transl_new = torch.tensor(smpl_transl_new).float()
    smpl_pose_new = torch.tensor(smpl_pose_new).float()

    R_sim = R_sim.reshape(R_sim.shape[0], R_sim.shape[1], 9)  # flatten the rotation matrix to 9d
    a_sim = torch.tensor(a_sim).float() / acc_scale
    w_sim = torch.tensor(w_sim).float() / gyro_scale
    R_sim = torch.tensor(R_sim).float()
    imu_data = torch.cat([a_sim, w_sim, R_sim], dim=-1)  # [n, 6, 15]

    description = None
    for key in ('texts', 'text', 'description'):
        if sample.get(key) is not None:
            description = sample[key]
            break
    gt_text_list = [] if (motion_only or description is None) else [str(desc) for desc in description]

    if rot_rep == '6d':
        orient = convert_rotation(smpl_orient_new, 'aa', '6d').reshape(-1, 6).float()
        pose = convert_rotation(smpl_pose_new.reshape(-1, 3), 'aa', '6d').reshape(-1, 21*6).float()
        transl = smpl_transl_new
    elif rot_rep == 'aa':
        orient = smpl_orient_new.reshape(-1, 3).float()
        pose = smpl_pose_new.reshape(-1, 21*3).float()
        transl = smpl_transl_new
    else:
        raise ValueError(f"Invalid rotation representation: {rot_rep}")

    sample_output = {
        'imu_data': imu_data,
        # Real NCSA variants can omit a physical device (for example the
        # headphone in meeting_room_2pt_mv).  Keep this per-sample metadata so
        # the input-layout sampler can mask those slots rather than treating
        # their zero/identity padding as a measured IMU.
        'imu_missing_slots': [int(slot) for slot in sample.get('imu_missing_slots', [])],
        'orient': orient,  # [n, 6]
        'transl': transl,
        'pose': pose,  # [n, 21*6]
        'description': gt_text_list,
        'objects': obj_pose_dict,
        # Which motion channel groups the label is trusted for. Video pseudo-labels
        # under a static-camera assumption corrupt world heading and root trajectory
        # but not the local joint angles (README_ncsa_imu.md, 2026-09-14); the
        # trainer masks the untrusted groups' token labels with -100.
        'motion_supervise': dict(motion_supervise),
    }

    length = len(sample_output['imu_data'])
    length = length - (length % 4)  # make length a multiple of 4
    length = min(length, IMUSEQMAXLEN)  # cut to max length to avoid OOM

    for k in list(sample_output.keys()):
        if k in ('description', 'imu_missing_slots'):
            continue
        elif k == 'objects':
            if dynamic_object:
                for obj_data in sample_output[k].values():
                    obj_data['rot'] = obj_data['rot'][:length]
                    obj_data['transl'] = obj_data['transl'][:length]
                    obj_data['bbox'] = obj_data['bbox'][:length]
        else:
            if isinstance(sample_output[k], dict):  # per-sample flags (e.g. motion_supervise) are not sequences
                continue
            sample_output[k] = sample_output[k][:length]

    if not (len(sample_output['imu_data']) == len(sample_output['orient']) == len(sample_output['transl']) == len(sample_output['pose'])):
        raise RuntimeError(
            f"real-IMU sample {sample_idx}: inconsistent lengths "
            f"{len(sample_output['imu_data'])}/{len(sample_output['orient'])}/"
            f"{len(sample_output['transl'])}/{len(sample_output['pose'])}"
        )
    sample_output['object_anchor_valid'] = dict(sample.get('object_anchor_valid', {}))
    if ground_record is not None:
        sample_output['object_anchor_valid']['ground'] = ground_record['valid']
        sample_output['ground_plane'] = ground_record
    return sample_output


def _legacy_motion_to_smplx_params(motion_smpl):
    motion_smpl = np.asarray(motion_smpl)
    transl_slice = slice(72, 75) if motion_smpl.shape[1] == 85 else slice(66, 69)
    return {
        'global_orient': motion_smpl[:, 0:3],
        'body_pose': motion_smpl[:, 3:66],
        'transl': motion_smpl[:, transl_slice],
    }


def process_imuposer_data(sample, random_cut, random_mask_text, cut_length, shift=0,
                          filter_short_text=True, add_ground_data=False, rot_rep='6d', split=None,
                          data_source=None, dynamic_object=False, fps=30, sample_idx=None, IMUSEQMAXLEN=1e6,
                          smooth_imu_acc=True):
    """Legacy map-style entry: ``{'motion_smpl' [T,69], 'imu_data' [T,60]}`` (see process_real_imu_data)."""
    from dataset_process.realworld.realworld_io import imuposer_legacy_to_sample

    unified = imuposer_legacy_to_sample(
        {'smplx_params': _legacy_motion_to_smplx_params(sample['motion_smpl']), 'imu_data': sample['imu_data']},
        motion_id=str(sample_idx), actor_id='legacy',
    )
    unified['objects'] = sample.get('objects', {})
    unified['texts'] = sample.get('text', sample.get('texts', sample.get('description', [])))
    for key in ('id', '__url__', 'ground_plane', 'object_anchor_valid'):
        if key in sample:
            unified[key] = sample[key]
    return process_real_imu_data(
        unified, random_cut, random_mask_text, cut_length, shift=shift,
        filter_short_text=filter_short_text, add_ground_data=add_ground_data, rot_rep=rot_rep, split=split,
        data_source='imuposer', dynamic_object=dynamic_object, fps=fps, sample_idx=sample_idx,
        IMUSEQMAXLEN=IMUSEQMAXLEN, smooth_imu_acc=smooth_imu_acc,
    )


def process_dipimu_data(sample, random_cut, random_mask_text, cut_length, shift=0,
                        filter_short_text=True, add_ground_data=False, rot_rep='6d', split=None,
                        data_source=None, dynamic_object=False, fps=30, sample_idx=None, IMUSEQMAXLEN=1e6,
                        acc_scale=1.0, gyro_scale=1.0, smooth_imu_acc=True):
    """Legacy map-style entry: ``{'motion_smpl' [T,69], 'imu_acc' [T,6,3], 'imu_ori' [T,6,3,3]}``.

    The gravity removal / smoothing / orientation re-integration now live in
    ``dataset_process.realworld.realworld_io.dipimu_legacy_to_sample`` (they do
    not depend on the crop); the rest is ``process_real_imu_data``.
    """
    from dataset_process.realworld.realworld_io import dipimu_legacy_to_sample

    unified = dipimu_legacy_to_sample(
        {
            'smplx_params': _legacy_motion_to_smplx_params(sample['motion_smpl']),
            'imu_acc': sample['imu_acc'],
            'imu_ori': sample['imu_ori'],
        },
        motion_id=str(sample_idx), actor_id='legacy',
    )
    unified['objects'] = sample.get('objects', {})
    # The historical DIP-IMU path never emitted captions.
    unified['texts'] = []
    for key in ('id', '__url__', 'ground_plane', 'object_anchor_valid'):
        if key in sample:
            unified[key] = sample[key]
    return process_real_imu_data(
        unified, random_cut, random_mask_text, cut_length, shift=shift,
        filter_short_text=filter_short_text, add_ground_data=add_ground_data, rot_rep=rot_rep, split=split,
        data_source='dipimu', dynamic_object=dynamic_object, fps=fps, sample_idx=sample_idx,
        IMUSEQMAXLEN=IMUSEQMAXLEN, acc_scale=acc_scale, gyro_scale=gyro_scale,
        smooth_imu_acc=(smooth_imu_acc and False),  # smoothing already applied at conversion
    )

def rotation_matrix_to_axis_angle(r: torch.Tensor):
    r"""
    Turn rotation matrices into axis-angles. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    import cv2
    result = [cv2.Rodrigues(_)[0] for _ in r.clone().detach().cpu().view(-1, 3, 3).numpy()]
    result = torch.from_numpy(np.stack(result)).float().squeeze(-1).to(r.device)
    return result

def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False, avoid_nan=False):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :param avoid_nan: If True, return zeros if norm is 0.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    if avoid_nan:
        normalized_x[torch.isnan(normalized_x)] = 0
    return normalized_x if not return_norm else (normalized_x, norm)

def hat(v: torch.Tensor):
    r"""
    Return the 3x3 skew-symmetric matrix of the 3D vector. (torch, batch)

    :param v: Tensor in shape [..., 3].
    :return: Tensor in shape [..., 3, 3].
    """
    return torch.stack((torch.zeros_like(v[..., 0]), -v[..., 2], v[..., 1],
                        v[..., 2], torch.zeros_like(v[..., 0]), -v[..., 0],
                        -v[..., 1], v[..., 0], torch.zeros_like(v[..., 0]),), dim=-1).view(*v.shape[:-1], 3, 3)

def axis_angle_to_rotation_matrix(a: torch.Tensor):
    r"""
    Turn axis-angles into rotation matrices. (torch, batch)

    :param a: Axis-angle tensor that can reshape to [batch_size, 3].
    :return: Rotation matrix of shape [batch_size, 3, 3].
    """
    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis) | torch.isinf(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * hat(axis)
    return r


def _apply_sensor_time_shifts(tensors, base, shifts, n_window):
    """Slice ``[N_ext, S, ...]`` tensors to ``[n_window, S, ...]`` with a per-sensor start offset."""
    out = []
    for tensor in tensors:
        sliced = torch.empty((n_window,) + tuple(tensor.shape[1:]), dtype=tensor.dtype)
        for s in range(tensor.shape[1]):
            start = int(base + shifts[s])
            sliced[:, s] = tensor[start:start + n_window, s]
        out.append(sliced)
    return tuple(out)


def process_imu_data(sample, random_cut, random_mask_text, cut_length, shift=0,
                     filter_short_text=True, add_ground_data=False, rot_rep='6d',
                     motion_only=False, scene_only=False, IMUSEQMAXLEN=1e6,
                     data_source=None, dynamic_object=False, fps=30, split=None, add_imu_noise=False,
                     acc_scale=1.0, gyro_scale=1.0, yaw_aug=None, imu_noise_cfg=None):
    """Crop / align / synthesise one virtual-IMU sample.

    ``imu_noise_cfg`` (``training.imu_noise``, see imu_synthesis/imu_noise.py)
    switches the virtual IMU to the device-realism simulator: on the train
    split it adds bandwidth / attitude / bias / timing imperfections, on every
    other split it only applies the fixed evaluation low-pass. ``None`` (or
    ``enabled: false``) keeps the historical noise-free path bit for bit.
    """

    assert data_source in ['parahome', 'humoto', 'other_dataset'], f"Invalid data source: {data_source}"

    if 'motion_smpl' in sample.keys():
        motion_smpl = sample['motion_smpl'] # [n, 85]
    elif 'motion_data_smpl85' in sample.keys():
        motion_smpl = sample['motion_data_smpl85'] # [n, 85]
    else:
        raise ValueError(f"Invalid motion smpl data: {sample.keys()}")
    imu_traj = _eval_imu_traj_override(sample, split) # [n, 6, 6]

    n_time = len(motion_smpl)
    assert cut_length is not None
    if n_time < cut_length and split == 'train':
        return None
    
    smpl_orient = motion_smpl[:, 0:3].copy()
    smpl_pose = motion_smpl[:, 3:66].copy()
    smpl_transl = motion_smpl[:, 72:75].copy()

    imu_traj_data = imu_traj.copy() # [n, 6, 6]
    imu_rot = imu_traj_data[:, :, 0:3] # [n, 6, 3]
    imu_rot = (
        convert_rotation(
            torch.from_numpy(imu_rot.reshape(-1, 3)).float(), 'aa', 'mat'
        ).view(n_time, -1, 3, 3).numpy().astype(np.float32)
    ) # [n, 6, 3, 3]
    imu_position = imu_traj_data[:, :, 3:6] # [n, 6, 3]
    if sample.get('source', data_source) == 'humoto':
        # Older tar files / legacy pickles predate the converter correction.
        # Only virtual-sensor positions have this offset; objects do not.
        from dataset_process.humoto.humoto_io import world_imu_positions
        imu_position = world_imu_positions(imu_position, sample)

    if random_cut and split == 'train': #FIXME
        assert n_time >= cut_length , "cut_length must be greater than or equal to n_time"
        start_cut_idx, end_cut_idx = random_contiguous_subarray_bounds(n_time, cut_length)
    else:
        start_cut_idx = 0
        end_cut_idx = min(n_time, cut_length)
    
    start_cut_idx = start_cut_idx + shift
    end_cut_idx = end_cut_idx + shift

    # Device-realism augmentation: crop the IMU trajectory with a margin so the
    # per-sensor / IMU-vs-motion time shifts stay inside the sequence. The
    # motion targets keep the exact [start, end) window.
    noise_cfg = resolve_imu_noise_cfg(imu_noise_cfg)
    noise_train = noise_cfg is not None and split == 'train'
    margin = crop_margin(noise_cfg) if noise_train else 0
    margin_lo = min(margin, start_cut_idx)
    margin_hi = min(margin, n_time - end_cut_idx)
    imu_rot = imu_rot[start_cut_idx - margin_lo:end_cut_idx + margin_hi]
    imu_position = imu_position[start_cut_idx - margin_lo:end_cut_idx + margin_hi]

    smpl_orient = smpl_orient[start_cut_idx:end_cut_idx]
    smpl_pose = smpl_pose[start_cut_idx:end_cut_idx]
    smpl_transl = smpl_transl[start_cut_idx:end_cut_idx]

    # align the smpl poses to the first frame
    smpl_orient_new, smpl_transl_new, smpl_pose_new, inv_rotation, inv_translation = \
        align_poses_to_first_frame(smpl_orient, smpl_transl, smpl_pose)

    # and apply the same inverse transformation to the imu data
    # inv_rotation: [3, 3], inv_translation: [3,]
    # [3, 3] @ ([n, 6, 3] + [1, 1, 3]) -> [n, 6, 3]
    imu_position_new = np.einsum('ij,bnj->bni', inv_rotation, imu_position + inv_translation[None, None])
    # [3, 3] @ [n, 6, 3, 3] -> [n, 6, 3, 3]
    imu_rot_new = np.einsum('ij,bnjk->bnik', inv_rotation, imu_rot)

    # smpl_transl_compare = np.einsum('ij,bj->bi', inv_rotation, smpl_transl + inv_translation) should be the same as smpl_transl_new
    # smpl_orient_compare = np.einsum('ij,njk->nik', inv_rotation, R.from_rotvec(smpl_orient).as_matrix())
    # smpl_orient_compare = R.from_matrix(smpl_orient_compare).as_rotvec() # should be the same as smpl_orient_new

    obj_pose_dict = {}

    objects = dict(sample.get('objects') or {})
    ground_record = None
    if add_ground_data:
        if 'ground' in objects:
            raise ValueError("Ground data already exists")
        source = sample.get('source', 'motionmillion' if data_source == 'other_dataset' else data_source)
        ground_record = resolve_ground_plane(sample, motion_smpl, source, fps)
        objects['ground'] = np.array([1.0, 0.0, 0.0, 0.0,
                                     0.0, ground_record['height_y'], 0.0,
                                     1.0, 1.0, 1.0], dtype=np.float32)

    if len(objects) > 0:
        obj_pose = objects
        for obj in obj_pose.keys():

            if dynamic_object:

                obj_track = np.asarray(obj_pose[obj])
                if obj_track.ndim == 1:
                    # Promote HuMOTO-style static objects (and synthetic ground)
                    # to the shared temporal object representation.
                    obj_track = np.repeat(
                        obj_track[None], end_cut_idx - start_cut_idx, axis=0
                    )
                elif obj_track.ndim == 2:
                    # Object, body, and IMU must use the exact same random crop.
                    obj_track = obj_track[start_cut_idx:end_cut_idx]
                else:
                    raise ValueError(
                        f"Object {obj!r} must have shape [D] or [T,D], "
                        f"got {obj_track.shape}"
                    )
                if len(obj_track) != len(smpl_orient_new):
                    raise ValueError(
                        f"Object/body crop mismatch for {obj!r}: "
                        f"{len(obj_track)} != {len(smpl_orient_new)}"
                    )

                obj_quat = obj_track[:, 0:4].copy().astype(np.float32)
                obj_mat = convert_rotation(torch.from_numpy(obj_quat), 'quat', 'mat').float().cpu().numpy() # [t,3,3]
                obj_mat = np.einsum('ij,tjk->tik', inv_rotation, obj_mat) # [t,3,3]
                obj_6d = convert_rotation(torch.from_numpy(obj_mat), 'mat', '6d').float().cpu().numpy() # [t,6]

                t = obj_6d.shape[0]
                obj_transl = obj_track[:,4:7].copy()
                obj_transl = np.einsum('ij,tj->ti', inv_rotation, obj_transl + inv_translation)
                obj_bbox = (
                    obj_track[:, 7:10].copy().astype(np.float32)
                    if obj_track.shape[1] >= 10
                    else np.ones((t, 3), dtype=np.float32)
                )

                obj_name_filtered = obj.split('.')[0] # remove the suffix
                assert obj == obj_name_filtered

                if obj == 'ground' and data_source != 'parahome': 
                    # set the ground to be at the origin, only keep the y-axis
                    # but for parahome, we keep the original transformation
                    obj_transl[:, 0] = 0.0
                    obj_transl[:, 2] = 0.0

                obj_pose_dict[obj_name_filtered] = {'rot': obj_6d, 'transl': obj_transl, 'bbox': obj_bbox}
                # import pdb; pdb.set_trace()
            else:
                # HiPHI/OMOMO store object trajectories as [T, 10].  Static
                # object mode deliberately supervises only O0, so collapse a
                # temporal track to the crop's first frame before processing it.
                obj_static = np.asarray(obj_pose[obj])
                if obj_static.ndim == 2:
                    obj_static = obj_static[start_cut_idx]
                elif obj_static.ndim != 1:
                    raise ValueError(
                        f"Object {obj!r} must have shape [D] or [T,D], "
                        f"got {obj_static.shape}"
                    )
                obj_quat = obj_static[0:4].copy().astype(np.float32)
                obj_mat = convert_rotation(torch.from_numpy(obj_quat), 'quat', 'mat').float().cpu().numpy() # [3,3]
                obj_mat = np.einsum('ij,jk->ik', inv_rotation, obj_mat) # [3,3]
                obj_6d = convert_rotation(torch.from_numpy(obj_mat), 'mat', '6d').float().cpu().numpy() # [6]

                obj_transl = obj_static[4:7].copy()
                obj_transl = np.einsum('ij,j->i', inv_rotation, obj_transl + inv_translation)
                obj_bbox = (
                    obj_static[7:10].copy()
                    if obj_static.shape[0] >= 10
                    else np.ones(3, dtype=np.float32)
                )

                obj_name_filtered = obj.split('.')[0] # remove the suffix
                assert obj == obj_name_filtered

                if obj == 'ground' and data_source != 'parahome':
                    # set the ground to be at the origin, only keep the y-axis
                    # but for parahome, we keep the original transformation
                    plane_y = obj_transl[1]
                    obj_transl = np.array([0.0, plane_y, 0.0], dtype=np.float32)

                obj_pose_dict[obj_name_filtered] = {'rot': obj_6d, 'transl': obj_transl, 'bbox': obj_bbox}

    p = torch.tensor(imu_position_new).float()
    R = torch.tensor(imu_rot_new).float()
    smpl_orient_new = torch.tensor(smpl_orient_new).float()
    smpl_transl_new = torch.tensor(smpl_transl_new).float()
    smpl_pose_new = torch.tensor(smpl_pose_new).float()

    if noise_cfg is None:
        a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(
            p, R, fps=fps,
            noise_raw_traj=add_imu_noise,
            noise_syn_imu=add_imu_noise,
            noise_est_orient=add_imu_noise,
            skip_ESKF=True,
            device='cpu'
        )
    else:
        # numpy's global RNG is not re-seeded per dataloader worker; derive the
        # generator from python's ``random`` (which is).
        rng = np.random.default_rng(random.getrandbits(64))
        a_sim, w_sim, R_sim = simulate_noisy_imu_readings(
            p, R, fps=fps, cfg=noise_cfg, train=noise_train, rng=rng
        )
        if noise_train:
            n_window = end_cut_idx - start_cut_idx
            shifts = np.clip(
                sample_time_shifts(noise_cfg, rng, R.shape[1]), -margin_lo, margin_hi
            )
            a_sim, w_sim, R_sim = _apply_sensor_time_shifts(
                (a_sim, w_sim, R_sim), margin_lo, shifts, n_window
            )
            imu_position_new = imu_position_new[margin_lo:margin_lo + n_window]
    a_sim = a_sim / acc_scale
    w_sim = w_sim / gyro_scale
    R_sim = R_sim.reshape(R_sim.shape[0], R_sim.shape[1], 9) # flatten the rotation matrix to 9d
    imu_data = torch.cat([a_sim, w_sim, R_sim], dim=-1) # [n, 6, 15]

    if split == "train" and _yaw_aug_enabled(yaw_aug):
        # Random world-yaw augmentation: canonicalize to a uniformly random
        # horizontal heading instead of the GT-pelvis one, and rotate every
        # world-frame field (IMU channels, GT root orient/transl, objects) by
        # the SAME Ry so the input/target relabeling stays consistent. The GT
        # supervision is first-frame-relative, so this only removes the
        # privileged dependence on the absolute heading, it does not change
        # the underlying motion. Pose (local joint rotations) is untouched.
        _phi = random.uniform(0.0, 360.0)
        _Ry = _yaw_rot_matrix_y(_phi)
        imu_data = _rotate_imu_data_y(imu_data, _Ry)
        imu_position_new = _rot_vec3_y(imu_position_new, _Ry).astype(np.float32)
        smpl_transl_new = torch.from_numpy(
            _rot_vec3_y(smpl_transl_new.numpy(), _Ry).astype(np.float32)
        )
        smpl_orient_new = torch.from_numpy(
            _rot_rot_rep_y(smpl_orient_new.numpy(), "aa", _Ry).astype(np.float32)
        )
        for _k, _d in obj_pose_dict.items():
            if _d.get("rot") is not None:
                _d["rot"] = _rot_rot_rep_y(
                    np.asarray(_d["rot"], dtype=np.float32), "6d", _Ry
                )
            if _d.get("transl") is not None:
                _d["transl"] = _rot_vec3_y(
                    np.asarray(_d["transl"], dtype=np.float32), _Ry
                )

    # add_small_noise = False
    # if add_small_noise:
    #     imu_data = add_velocity_scaled_noise(imu_data)

    if 'text' in sample:
        sample['description'] = sample['text']
    elif 'texts' in sample:
        sample['description'] = sample['texts']
    elif 'description' not in sample:
        sample['description'] = []

    gt_text_list = []

    if data_source == 'parahome':
        frame_bounds_str = list(list(sample['description'].keys()))
        frame_bounds_list = []
        for bound in frame_bounds_str:
            frame_bounds = bound.split(' ')
            frame_bounds = (int(frame_bounds[0]), int(frame_bounds[1]))
            frame_bounds_list.append(frame_bounds)
        
        valid_text_indices = intersecting_indices(frame_bounds_list, start_cut_idx, end_cut_idx)
        gt_text = ''
        for idx in valid_text_indices:
            gt_text += (sample['description'][frame_bounds_str[idx]] + ' ')
        gt_text_list = [gt_text]
    
    else:
        # filter out too short descriptions
        if sample['description'] is not None and len(sample['description']) > 0:
            for desc in sample['description']:
                if len(desc.split(' ')) >= 7 or not filter_short_text: # at least 7 words; if filter_text is False, then don't filter the text
                    gt_text_list.append(desc)

        if random_mask_text:
            # mask_prob = 0.0 # FIXME
            mask_prob = 0.2 # FIXME
            # if p < mask_prob, then set the gt_text_list to an empty list
            if random.random() < mask_prob:
                gt_text_list = []
    
    if motion_only or scene_only:
        # motion_only trains the motion heads alone; scene_only (M2S) predicts
        # objects from ground-truth motion with no language conditioning, so
        # the caption is removed from the sequence in both cases.
        gt_text_list = []

    if rot_rep == '6d':
        orient = convert_rotation(smpl_orient_new, 'aa', '6d').reshape(-1, 6).float()
        pose = convert_rotation(smpl_pose_new.reshape(-1, 3), 'aa', '6d').reshape(-1, 21*6).float()
        transl = smpl_transl_new
    elif rot_rep == 'aa':
        orient = smpl_orient_new.reshape(-1, 3).float()
        pose = smpl_pose_new.reshape(-1, 21*3).float()
        transl = smpl_transl_new
    else:
        raise ValueError(f"Invalid rotation representation: {rot_rep}")

    sample_output = {
        'imu_data': imu_data,
        'orient': orient, # [n, 6]
        'transl': transl,
        'pose': pose, # [n, 21*6]
        'description': gt_text_list,
        'objects': obj_pose_dict,
        # synthetic data: exact GT, every group supervised
        'motion_supervise': {'traj': True, 'orient': True, 'pose': True},
    }

    length = len(sample_output['imu_data'])
    length = length - (length % 4)  # make length a multiple of 4
    length = min(length, IMUSEQMAXLEN)  # cut to max length to avoid OOM

    for k, v in sample_output.items():
        if k in ['description', 'scene_name', 'scene_2d_layout', 'scene_mesh', 'scene_occ_grid']:
            # skip non-numeric data
            continue
        elif k == 'objects' and dynamic_object:
            for obj_name, obj_data in sample_output[k].items():
                obj_data['rot'] = obj_data['rot'][:length]
                obj_data['transl'] = obj_data['transl'][:length]
                obj_data['bbox'] = obj_data['bbox'][:length]
        elif k != 'objects':
            if isinstance(sample_output[k], dict):  # per-sample flags (e.g. motion_supervise) are not sequences
                continue
            sample_output[k] = sample_output[k][:length]

    # Keep aligned sensor locations for full-eval visualization. The model
    # continues to consume only ``imu_data``.
    sample_output['imu_positions'] = torch.from_numpy(
        imu_position_new[:length]
    ).float()

    # Preserve HiPHI supervision metadata without changing the tensor interface
    # consumed by existing HuMOTO/MotionMillion training code.
    for mask_key in ('object_valid_mask', 'object_motion_mask'):
        if mask_key in sample:
            sample_output[mask_key] = {
                name: np.asarray(mask)[start_cut_idx:end_cut_idx][:length].copy()
                for name, mask in sample[mask_key].items()
            }
    for metadata_key in ('object_metadata', 'task_mode', 'annotation_scope'):
        if metadata_key in sample:
            sample_output[metadata_key] = sample[metadata_key]

    sample_output['object_anchor_valid'] = dict(sample.get('object_anchor_valid', {}))
    if ground_record is not None:
        sample_output['object_anchor_valid']['ground'] = ground_record['valid']
        sample_output['ground_plane'] = ground_record

    # sample_output = {k: pad_to_length(v, self.MAXLEN) for k, v in sample.items()} # [MAXLEN, 3, nvars]

    return sample_output
    

class IMUDataset():
    def __init__(
        self,
        root=None,
        split: Optional[str] = 'train', # 'train', 'val'， 'test'
        seed: int = 42, # New argument for reproducibility
        overfit: bool = False, # for debugging
        motion_only: bool = False, # whether to only use motion data
        text_only: bool = False, # whether to only use text data
        scene_only: bool = False, # whether to only use scene data
        random_cut: bool = False, # whether to randomly cut the sequence
        random_mask_text: bool = False, # whether to randomly mask the text
        add_humoto_data: bool = False, # whether to add humoto data
        # MotionGV is recovered from monocular video: many clips barely move and their
        # captions describe actions the recovered pose never performs. Excluded by
        # default here and, since 2026-09-10, in the streaming loader's eval path too
        # (wds_loader.DEFAULT_EXCLUDED_DATASETS).
        add_motiongv_data: bool = False,
        shift: int = 0, # whether to shift the imu data
        selected_dataset: Optional[str] = None, # whether to evaluate on the full dataset
        shuffle_list: bool = True, # whether to shuffle the data
        return_path_only: bool = False, # whether to return the path only
        dynamic_object: bool = False, # whether to use dynamic object
        fps: int = 30, # fps of the imu data
        add_imu_noise: bool = False, # whether to add noise to the imu data
        IMUSEQMAXLEN: int = 1e6, # cut max length to avoid OOM (out of memory) issues
        acc_scale: float = 1.0, # scale the acceleration data
        gyro_scale: float = 1.0, # scale the gyroscope data
        selected_imu_seq: Optional[str] = None,  # absolute or cwd-relative path to one .pkl (eval)
        **kwargs,
    ):
        # self.root = '/scratch/benk/hhsu2/imu-humans/final_data_per_sequence' # hardcode for now
        self.root = root
        assert self.root is not None, "Please specify the root directory of the dataset."

        # self.humoto_root = kwargs.get('humoto_root', None)
        # self.parahome_root = kwargs.get('parahome_root', None)
        # self.imuposer_root = kwargs.get('imuposer_root', None)
        # self.dipimu_root = kwargs.get('dipimu_root', None)

        self.split = split
        self.seed = seed
        self.motion_only = motion_only
        self.text_only = text_only
        self.scene_only = scene_only
        self.random_cut = random_cut
        self.random_mask_text = random_mask_text
        self.shift = shift
        self.dynamic_object = dynamic_object
        self.fps = fps
        self.add_imu_noise = add_imu_noise
        self.acc_scale = acc_scale
        self.gyro_scale = gyro_scale
        self.cut_length = IMUSEQMAXLEN
        self.IMUSEQMAXLEN = IMUSEQMAXLEN
        self.selected_imu_seq_path: Optional[str] = None
        assert not add_imu_noise, "add_imu_noise is not supported for full dataset."
        assert IMUSEQMAXLEN is not None
        
        assert self.root is not None, "Please specify the root directory of the dataset."
        assert split in ['train', 'val', 'test']

        if selected_imu_seq is not None:
            p = os.path.abspath(os.path.normpath(os.path.expanduser(selected_imu_seq)))
            if not os.path.isfile(p):
                raise FileNotFoundError(f"selected_imu_seq not found: {p}")
            self.selected_dataset = selected_dataset
            self.selected_imu_seq_path = p
            self.data = [0]
            self.rest_pelvis = np.array([ 0.00312326, -0.35140744,  0.01203655], dtype=np.float32)
            self.return_path_only = return_path_only
            print(f"IMU dataset loaded. Single sequence file: {self.selected_imu_seq_path} (1 sample)")
            return

        if split != 'train':
            assert not random_cut, "only do random cut for train split."
        # Load IMU data here
        # all_data_list = [v for v in all_data_dict.values()]
        # sample_nums = {'train': 702270, 'val': 45236, 'test': 131097}
        # sample_num = sample_nums[split]
        # all_data_list = np.arange(sample_num)

        target_datasets = [
            'LINGO', 
            'BABEL', 
            'Mirror_BABEL', 
            'PhantomDanceDatav1.1', 
            'Mirror_PhantomDanceDatav1.1',
            # 'MotionGV',
            'MotionLLAMA', 
            'MotionUnion',
            # 'Mirror_MotionGV',
            'Mirror_MotionLLAMA', 
            'Mirror_MotionUnion',
        ]

        if add_humoto_data:
            target_datasets.append('HUMOTO')
        if add_motiongv_data:
            target_datasets.append('MotionGV')
            target_datasets.append('Mirror_MotionGV')
        
        self.selected_dataset = selected_dataset
        if selected_dataset is not None:
            assert selected_dataset in ['HUMOTO', 'LINGO', 'ParaHome', 'humanml', 'imuposer', 'dipimu', 'ncsa']
            target_datasets = [selected_dataset]

        # Other dataset
        split_file_1 = f"{self.root}/splits/t2m_{self.split}.txt"
        all_data_list_1 = [line.strip() for line in open(split_file_1).readlines()]
        all_data_list_1 = filter_by_targets(all_data_list_1, target_datasets)
        split_file_2 = f"{self.root}/splits/tokenizer_{self.split}.txt"
        all_data_list_2 = [line.strip() for line in open(split_file_2).readlines()]
        all_data_list_2 = filter_by_targets(all_data_list_2, target_datasets)

        # Use sorted() for deterministic order across runs (set iteration order is undefined in Python)
        all_data_list = sorted(set(all_data_list_1 + all_data_list_2))
        print(f'Total number of data: {len(all_data_list)}')
        all_data_list = [data.replace('/', '_') for data in all_data_list]
 
        # Humoto dataset
        if 'HUMOTO' in target_datasets:
            self.humoto_root = humoto_root
            humoto_all_data_list = np.load(f"{self.humoto_root}/{self.split}_indices.npy").tolist()
            humoto_all_data_list = ['humoto_{}'.format(k) for k in humoto_all_data_list]
            all_data_list = all_data_list + humoto_all_data_list
        if 'ParaHome' in target_datasets:
            self.parahome_root = parahome_root
            parahome_all_data_list = np.load(f"{self.parahome_root}/{self.split}_split.npy").tolist()
            parahome_all_data_list = ['parahome_{}'.format(k) for k in parahome_all_data_list]
            all_data_list = all_data_list + parahome_all_data_list
        if 'imuposer' in target_datasets:
            self.imuposer_root = imuposer_root
            imuposer_all_data_list = []
            if self.split == 'train':
                for i in range(0, 9):
                    files = glob.glob(f"{self.imuposer_root}/P{i}/*.pkl")
                    imuposer_all_data_list.extend(files)
            else:
                for i in range(9, 11):
                    files = glob.glob(f"{self.imuposer_root}/P{i}/*.pkl")
                    imuposer_all_data_list.extend(files)
            all_data_list = all_data_list + imuposer_all_data_list
        if 'dipimu' in target_datasets:
            self.dipimu_root = dipimu_root
            dipimu_all_data_list = []
            if self.split == 'train':
                for i in range(0, 9):
                    files = glob.glob(f"{self.dipimu_root}/s_{str(i).zfill(2)}/*.pkl")
                    dipimu_all_data_list.extend(files)
            else:
                for i in range(9, 11):
                    files = glob.glob(f"{self.dipimu_root}/s_{str(i).zfill(2)}/*.pkl")
                    dipimu_all_data_list.extend(files)
            # import pdb; pdb.set_trace()
            all_data_list = all_data_list + dipimu_all_data_list

        if selected_dataset == 'HUMOTO':
            assert 'HUMOTO' in target_datasets, "Humoto data is not added. Please add humoto data."
            all_data_list = humoto_all_data_list
        if selected_dataset == 'ParaHome':
            assert 'ParaHome' in target_datasets, "ParaHome data is not added. Please add ParaHome data."
            all_data_list = parahome_all_data_list
        if selected_dataset == 'imuposer':
            assert 'imuposer' in target_datasets
            all_data_list = imuposer_all_data_list
        if selected_dataset == 'dipimu':
            assert 'dipimu' in target_datasets
            all_data_list = dipimu_all_data_list

        # Shuffle the data for random splitting
        if shuffle_list:
            np.random.seed(self.seed) # Set seed for reproducibility
            np.random.shuffle(all_data_list)
        else:
            all_data_list = sorted(all_data_list) # sort the data list

        # assert not overfit, "Overfit is not supported for full dataset."
        # Calculate split index
        # import pdb; pdb.set_trace()
        if overfit:
            sample_num = 1
            self.random_cut = False
            self.data = all_data_list[:sample_num]
        else:
            self.data = all_data_list
        
        self.rest_pelvis = np.array([ 0.00312326, -0.35140744,  0.01203655], dtype=np.float32)

        print(f"IMU dataset loaded. Split: '{self.split}', Number of samples: {len(self.data)}")

        self.return_path_only = return_path_only

    def __len__(self):
        return len(self.data)
    
    def _set_cut_length(self, cut_length: int):
        self.cut_length = cut_length
    
    def enable_random_cut(self):
        assert self.split == 'train', "only enable random cut for train split."
        self.random_cut = True
    
    def disable_random_cut(self):
        self.random_cut = False
    
    def __getitem__(self, idx):

        if self.random_cut:
            assert self.cut_length is not None, "cut_length must be set when random_cut is True"

        sample_idx = self.data[idx]
        if self.selected_imu_seq_path is not None:
            assert idx == 0, "Single-sequence dataset has exactly one item."
            sample_path = self.selected_imu_seq_path
            data_source = 'other_dataset'
            add_ground_data = True
            with open(sample_path, 'rb') as f:
                sample = pickle.load(f)
            sample_idx = os.path.splitext(os.path.basename(sample_path))[0]
        # try:
        elif isinstance(sample_idx, str) and 'humoto' in sample_idx:
            # load from humoto path
            data_source = 'humoto'
            sample_idx = int(sample_idx.split('_')[1])
            if self.dynamic_object:
                sample_path = f"{self.humoto_root}/all_time/{sample_idx:07d}.pkl" 
            else:
                sample_path = f"{self.humoto_root}/all/{sample_idx:07d}.pkl" 
            add_ground_data = False
            with open(sample_path, 'rb') as f:
                sample = pickle.load(f)

            # HuMOTO motion_smpl stores the SMPL-X translation parameter;
            # convert only that field to the global pelvis position expected by
            # process_imu_data. That function separately corrects the legacy
            # virtual-sensor offset; world-space objects stay unchanged.
            sample['motion_smpl'][:, 72:75] = sample['motion_smpl'][:, 72:75] + self.rest_pelvis

        elif isinstance(sample_idx, str) and 'parahome' in sample_idx:
            # load from parahome path
            data_source = 'parahome'
            sample_idx = sample_idx.split('_')[1].split('.')[0]
            add_ground_data = True

            # import pdb; pdb.set_trace()
            imu_traj_path = f"{self.parahome_root}/imu_traj/{sample_idx}.npy"
            imu_traj = np.load(imu_traj_path, allow_pickle=True)
            motion_smpl_path = f"{self.parahome_root}/motions_smpl85/{sample_idx}.npy"
            motion_smpl = np.load(motion_smpl_path, allow_pickle=True)
            text_path = f"{self.parahome_root}/text_annotations/{sample_idx}.json"
            text = json.load(open(text_path, 'r'))

            sample = {'objects': {}} # since we learn a global transformation, no need to add other objects
            sample['motion_smpl'] = motion_smpl # [n, 85]
            sample['imu_traj'] = imu_traj # [n, 6, 6]
            sample['text'] = text # a dictionary of text 
            
        elif isinstance(sample_idx, str) and 'imuposer' in sample_idx:
            data_source = 'imuposer'
            add_ground_data = True
            with open(sample_idx, 'rb') as f:
                data = pickle.load(f)
            
            sample = {}
            smplx_params = data['smplx_params']
            _orient = smplx_params['global_orient']  # (N, 3)
            _pose = smplx_params['body_pose'] # (N, 63)
            _transl = smplx_params['transl']  # (N, 3)
            sample['motion_smpl'] = np.concatenate([_orient, _pose, _transl], axis=-1)  # (N, 69)
            assert sample['motion_smpl'].shape[1] == 69, f"Invalid motion_smpl shape: {sample['motion_smpl'].shape}"
            sample['imu_data'] = data['imu_data']

            # import pdb; pdb.set_trace()
        
        elif isinstance(sample_idx, str) and 'DIP_IMU' in sample_idx:
            data_source = 'dipimu'
            add_ground_data = True

            with open(sample_idx, 'rb') as f:
                data = pickle.load(f)
            
            sample = {}
            # print(data.keys())
            smplx_params = data['smplx_params']
            smpl_orient = smplx_params['global_orient'] # (N, 3)
            smpl_transl = smplx_params['transl'] # (N, 3)
            smpl_pose = smplx_params['body_pose'] # (N, 63)
            sample['motion_smpl'] = np.concatenate([smpl_orient, smpl_pose, smpl_transl], axis=-1) # (N, 69)
            assert sample['motion_smpl'].shape[1] == 69, f"Invalid motion_smpl shape: {sample['motion_smpl'].shape}"
            sample['imu_acc'] = data['imu_acc']
            sample['imu_ori'] = data['imu_ori']
            # import pdb; pdb.set_trace()
            
        else:
            data_source = 'other_dataset'
            sample_path = os.path.join(self.root, 'motion_data', self.split, sample_idx + '.pkl')
            if not os.path.exists(sample_path):
                sample_path = os.path.join(self.root, self.split, sample_idx + '.pkl')
            add_ground_data = True
        
            with open(sample_path, 'rb') as f:
                sample = pickle.load(f)
 

        sample.setdefault('id', str(sample_idx))
        sample.setdefault('source', 'motionmillion' if data_source == 'other_dataset' else data_source)
        filter_short_text = (self.selected_dataset is None) and (self.selected_imu_seq_path is None)

        if data_source == 'imuposer':
            assert not self.motion_only
            assert not self.scene_only
            assert not self.add_imu_noise, 'realworld data already has noise'
            smooth_imu_acc = True
            sample_output = process_imuposer_data(sample, self.random_cut, self.random_mask_text, 
                                                self.cut_length, shift=self.shift, 
                                                add_ground_data=add_ground_data,
                                                filter_short_text=filter_short_text,
                                                data_source=data_source, 
                                                dynamic_object=self.dynamic_object,
                                                fps=self.fps, 
                                                split=self.split,
                                                IMUSEQMAXLEN=self.IMUSEQMAXLEN,
                                                sample_idx=sample_idx,
                                                smooth_imu_acc=smooth_imu_acc)
        elif data_source == 'dipimu':
            assert not self.motion_only
            assert not self.scene_only
            assert not self.add_imu_noise, 'realworld data already has noise'
            smooth_imu_acc = True
            sample_output = process_dipimu_data(sample, self.random_cut, self.random_mask_text, 
                                                self.cut_length, shift=self.shift, 
                                                add_ground_data=add_ground_data,
                                                filter_short_text=filter_short_text,
                                                data_source=data_source, 
                                                dynamic_object=self.dynamic_object,
                                                fps=self.fps,
                                                split=self.split,
                                                IMUSEQMAXLEN=self.IMUSEQMAXLEN,
                                                sample_idx=sample_idx,
                                                acc_scale=self.acc_scale,
                                                gyro_scale=self.gyro_scale,
                                                smooth_imu_acc=smooth_imu_acc)
        else:
            sample_output = process_imu_data(sample, self.random_cut, self.random_mask_text, 
                                            self.cut_length, shift=self.shift, 
                                            add_ground_data=add_ground_data,
                                            filter_short_text=filter_short_text,
                                            motion_only=self.motion_only,
                                            scene_only=self.scene_only,
                                            data_source=data_source, 
                                            dynamic_object=self.dynamic_object,
                                            fps=self.fps,
                                            split=self.split,
                                            IMUSEQMAXLEN=self.IMUSEQMAXLEN,
                                            add_imu_noise=self.add_imu_noise,
                                            acc_scale=self.acc_scale,
                                            gyro_scale=self.gyro_scale)
        
        if sample_output is None:
            if self.selected_imu_seq_path is not None:
                raise RuntimeError(
                    f"process_imu_data returned None (e.g. too short) for {self.selected_imu_seq_path}"
                )
            return self.__getitem__((idx + 1) % len(self))

        sample_output['sample_idx'] = sample_idx
        return sample_output
    
    def collate_fn(self, batch):
        return batch

def set_cut_length(dataloader, cut_min_length: int, cut_max_length: int):
    cut_length = random.randint(cut_min_length, cut_max_length)
    dataloader.dataset._set_cut_length(cut_length)

if __name__ == "__main__":

    selected_dataset = 'LINGO'
    dataset = IMUDataset(
        split='train',
        root='/shared/perception/datasets/imu_data/final_data_per_sequence',
        seed=42,
        overfit=False,
        random_cut=False,
        selected_dataset=selected_dataset,
        return_path_only=True,
        shift=2,
        dynamic_object=False,
        IMUSEQMAXLEN=50,
    )

    print(len(dataset))

    imu_data_all = []

    import tqdm
    for i in tqdm.tqdm(range(len(dataset))):
        sample_path = dataset[i]
        sample = dataset.__getitem__(i)

        # import pdb; pdb.set_trace()

        imu_data = sample['imu_data']
        imu_data_all.append(imu_data)

        # print(sample_path)
        print(f"IMU data shape: {sample['imu_data'].shape}")
        print(f"Orientation shape: {sample['orient'].shape}")
        print(f"Translation shape: {sample['transl'].shape}")

        if i > 10:
            break
    imu_data_all = np.concatenate(imu_data_all, axis=0)
    for i in range(15):
        v = imu_data_all[..., i]
        print(f"IMU data {i} shape: {v.shape}")
        print(f"IMU data {i} mean: {v.mean()}")
        print(f"IMU data {i} std: {v.std()}")
        print(f"IMU data {i} min: {v.min()}")
        print(f"IMU data {i} max: {v.max()}")
        print(f"IMU data {i} median: {np.median(v)}")

    import pdb; pdb.set_trace()
    # imu_std = np.std(imu_data_all, axis=0, keepdims=True)
    # np.save(f'/scratch/bfyo/tcheng1/imu_std.npy', imu_std)
    #     # gt_text[i] = sample_path
    #     print(f"Description: {sample['description']}")
    #     # gt_text[i] = sample['description']
    #     # print(f"IMU data shape: {sample['imu_data'].shape}, dtype: {sample['imu_data'].dtype}, device: {sample['imu_data'].device}")
    #     # print(f"Orientation shape: {sample['orient'].shape}, dtype: {sample['orient'].dtype}, device: {sample['orient'].device}")
    #     # print(f"Translation shape: {sample['transl'].shape}, dtype: {sample['transl'].dtype}, device: {sample['transl'].device}")
    #     # print(f"Pose shape: {sample['pose'].shape}, dtype: {sample['pose'].dtype}, device: {sample['pose'].device}")
    #     # print(f"Objects: {sample['objects'].keys()}")

    #     # for obj_key in sample['objects'].keys():
    #     #     print(obj_key)
    #         # print(f"Object {obj_key}: rot dtype: {sample['objects'][obj_key]['rot'].dtype}, transl dtype: {sample['objects'][obj_key]['transl'].dtype}, bbox dtype: {sample['objects'][obj_key]['bbox'].dtype}")
    #         # print(f"Object {obj_key}: rot device: {sample['objects'][obj_key]['rot'].device}, transl device: {sample['objects'][obj_key]['transl'].device}, bbox device: {sample['objects'][obj_key]['bbox'].device}")
    #         # break

    # # # save the gt_text to a pickle file
    # # with open(f'/scratch/benk/tcheng1/gt_text/{selected_dataset}_gt_text.pkl', 'wb') as f:
    # #     pickle.dump(gt_text, f)
    # # print(f"Saved gt_text to /scratch/benk/tcheng1/gt_text/{selected_dataset}_gt_text.pkl")
