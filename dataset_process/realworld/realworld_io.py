"""Convert the legacy IMUPoser / DIP-IMU pickles to the shared real-IMU sample schema.

Both real-world datasets ship *measured* accelerometer / orientation readings,
not the virtual-sensor trajectories (``imu_traj``) that ``process_imu_data``
turns into synthetic readings.  They therefore get their own sample schema and
their own loader branch (``training.imu_dataset.process_real_imu_data``):

    motion_data_smpl85   float32 [T, 85]   SMPL85 layout (hand pose = 0)
    imu_acc              float32 [T, 6, 3] pre-crop acceleration, model sensor order
    imu_ori              float32 [T, 6, 3, 3] pre-crop orientation, model sensor order
    imu_acc_smoothed     bool             True when the 3-frame moving average was
                                          already applied at conversion time
    texts                list[str]        empty (no captions in either release)
    objects              {}               loader supplies a floor (estimated for DIP)
    imu_sensor_names / imu_source_devices / fps / motion_id / actor_id / source / id

The arrays are stored exactly in the representation the historical map-style
loader (``process_imuposer_data`` / ``process_dipimu_data``) built *before* its
random crop, so cropping, first-frame alignment, angular velocity and the
gravity offset happen at load time and the numerics are unchanged:

* IMUPoser: raw world-frame readings of the five phones/watch, re-indexed to
  the six model slots.  Smoothing is applied on the crop at load time.
* DIP-IMU: sensor-frame acceleration (gravity removed) integrated through the
  same relative-rotation chain the legacy loader used (``R_sim`` starts at the
  identity for every sensor), smoothed on the whole sequence as before.

Model sensor slots are ``left_hip, right_hip, left_ear, right_ear, left_elbow,
right_elbow``.  Slot 3 (``right_ear``) duplicates the single head device of
both releases; evaluation masks it (``active_imu_id`` without index 3).
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

SCHEMA_VERSION = 1
FPS = 30.0
SOURCES = ("imuposer", "dipimu", "ncsa")

IMU_SENSOR_NAMES = (
    "left_hip", "right_hip", "left_ear", "right_ear", "left_elbow", "right_elbow",
)
# Model slot -> legacy device index (``permute_idx`` of the legacy loader).
LEGACY_TO_MODEL_SLOTS = (2, 3, 4, 4, 0, 1)
IMUPOSER_DEVICES = (
    "left_wrist", "right_wrist", "left_front_pocket", "right_front_pocket", "head",
)
DIPIMU_DEVICES = (
    "left_wrist", "right_wrist", "left_thigh", "right_thigh", "head", "pelvis",
)
DUPLICATED_SENSOR_SLOT = 3

# Subject-level protocol of the legacy loader (train = first eight subjects,
# val == test = the last two).
IMUPOSER_SPLIT = {
    "train": tuple(f"P{i}" for i in range(1, 9)),
    "val": ("P9", "P10"),
}
DIPIMU_SPLIT = {
    "train": tuple(f"s_{i:02d}" for i in range(1, 9)),
    "val": ("s_09", "s_10"),
}

DIP_GRAVITY = torch.tensor([0.0, -9.798, 0.0])
DIP_SOURCE_FPS = 60.0


def smplx_params_to_smpl85(smplx_params: dict) -> np.ndarray:
    """Pack ``global_orient / body_pose / transl / betas`` into the SMPL85 layout.

    ``transl`` in both releases is already the global pelvis position, which is
    what ``motion_data_smpl85[:, 72:75]`` holds for every other IMU4D dataset.
    Hand pose (66:72) is zero; ``betas`` (75:85) are broadcast when given once.
    """
    orient = np.asarray(smplx_params["global_orient"], dtype=np.float32)
    body = np.asarray(smplx_params["body_pose"], dtype=np.float32)
    transl = np.asarray(smplx_params["transl"], dtype=np.float32)
    if orient.shape[1:] != (3,) or body.shape[1:] != (63,) or transl.shape[1:] != (3,):
        raise ValueError(
            "Unexpected SMPL-X parameter shapes: "
            f"{orient.shape}, {body.shape}, {transl.shape}"
        )
    n = min(len(orient), len(body), len(transl))
    betas = np.asarray(smplx_params.get("betas", np.zeros(10)), dtype=np.float32)
    if betas.ndim == 1:
        betas = np.broadcast_to(betas[None], (n, betas.shape[0]))
    betas = betas[:n]
    if betas.shape[1] != 10:
        raise ValueError(f"Expected 10 shape coefficients, got {betas.shape}")
    motion = np.zeros((n, 85), dtype=np.float32)
    motion[:, 0:3] = orient[:n]
    motion[:, 3:66] = body[:n]
    motion[:, 72:75] = transl[:n]
    motion[:, 75:85] = betas
    return motion


def _base_sample(source: str, motion_id: str, actor_id: str, devices: tuple) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "id": f"{source}/{actor_id}/{motion_id}",
        "motion_id": motion_id,
        "actor_id": actor_id,
        "fps": FPS,
        "texts": [],
        "objects": {},
        "imu_sensor_names": list(IMU_SENSOR_NAMES),
        "imu_source_devices": [devices[i] for i in LEGACY_TO_MODEL_SLOTS],
        "imu_duplicated_sensor_slot": DUPLICATED_SENSOR_SLOT,
    }


# IMUPoser readings live in Apple's z-up world; GT / synthetic data are y-up.
# 180 deg about (0,1,1)/sqrt2: y <-> z, x -> -x.  See README "IMUPoser real-vs-synthetic diagnosis".
WORLD_ZUP_TO_YUP = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def apply_world_zup_to_yup(sample: dict) -> dict:
    """Rotate ``imu_ori`` / ``imu_acc`` from the z-up Apple world into the y-up SMPL world (in place)."""
    w = WORLD_ZUP_TO_YUP.T  # world_yup = W0^T world_zup
    sample["imu_ori"] = np.ascontiguousarray(np.einsum("ij,tnjk->tnik", w, np.asarray(sample["imu_ori"], dtype=np.float32)))
    sample["imu_acc"] = np.ascontiguousarray(np.einsum("ij,tnj->tni", w, np.asarray(sample["imu_acc"], dtype=np.float32)))
    corr = dict(sample.get("imu_corrections") or {})
    corr["world_zup_to_yup"] = True
    sample["imu_corrections"] = corr
    return sample


def set_loader_gravity(sample: dict, add_gravity: bool) -> dict:
    """``imu_acc_add_gravity=False`` makes ``process_real_imu_data`` skip the (0,-9.8,0) offset."""
    sample["imu_acc_add_gravity"] = bool(add_gravity)
    corr = dict(sample.get("imu_corrections") or {})
    corr["loader_gravity"] = bool(add_gravity)
    sample["imu_corrections"] = corr
    return sample


def imuposer_legacy_to_sample(data: dict, *, motion_id: str, actor_id: str) -> dict:
    """Legacy IMUPoser pickle (``smplx_params`` + ``imu_data`` [T, 60]) -> sample."""
    motion = smplx_params_to_smpl85(data["smplx_params"])
    imu = np.asarray(data["imu_data"], dtype=np.float32)
    if imu.ndim != 2 or imu.shape[1] != 60:
        raise ValueError(f"IMUPoser imu_data must be [T, 60], got {imu.shape}")
    n = min(len(motion), len(imu))
    motion, imu = motion[:n], imu[:n]
    acc = imu[:, :15].reshape(n, 5, 3)
    ori = imu[:, 15:].reshape(n, 5, 3, 3)
    slots = list(LEGACY_TO_MODEL_SLOTS)
    sample = _base_sample("imuposer", motion_id, actor_id, IMUPOSER_DEVICES)
    sample.update(
        motion_data_smpl85=motion,
        imu_acc=np.ascontiguousarray(acc[:, slots]),
        imu_ori=np.ascontiguousarray(ori[:, slots]),
        imu_acc_smoothed=False,
    )
    return sample


def dipimu_legacy_to_sample(data: dict, *, motion_id: str, actor_id: str, direct_readings: bool = False) -> dict:
    """Legacy DIP-IMU pickle (``smplx_params`` + ``imu_acc`` / ``imu_ori``) -> sample.

    Reproduces the pre-crop part of the legacy ``process_dipimu_data``: gravity
    is removed in the sensor frame, the acceleration is smoothed over the whole
    sequence, and the orientation chain is re-integrated from the identity via
    axis-angle increments (the historical "approximate ESKF").  The increments
    are zero-padded at the end exactly as before, so the chain keeps its
    historical one-frame lag (``R_sim[t] = ori[1]^T ori[t+1]``); models were
    fine-tuned with this convention and the numerics are preserved bit-for-bit.
    """
    from training.imu_dataset import (
        axis_angle_to_rotation_matrix,
        rotation_matrix_to_axis_angle,
        smooth_avg,
    )

    motion = smplx_params_to_smpl85(data["smplx_params"])
    acc_raw = np.asarray(data["imu_acc"], dtype=np.float32)
    ori_raw = np.asarray(data["imu_ori"], dtype=np.float32)
    if acc_raw.shape[1:] != (6, 3) or ori_raw.shape[1:] != (6, 3, 3):
        raise ValueError(
            f"DIP-IMU imu_acc/imu_ori must be [T,6,3]/[T,6,3,3], got {acc_raw.shape}, {ori_raw.shape}"
        )
    n = min(len(motion), len(acc_raw), len(ori_raw))
    motion = motion[:n]
    ori = torch.tensor(ori_raw[:n]).float()
    acc = torch.tensor(acc_raw[:n]).float()

    if direct_readings:
        # The legacy arrays are already y-up world orientations (bone convention, ~5 deg
        # from the GT bone rotation on the thighs) and gravity-free world accelerations
        # (mean ~0). Use them as they are instead of re-integrating the orientation from
        # the identity, which expresses everything in each sensor's first-frame frame
        # (see README "DIP-IMU real-vs-synthetic diagnosis").
        slots = list(LEGACY_TO_MODEL_SLOTS)
        sample = _base_sample("dipimu", motion_id, actor_id, DIPIMU_DEVICES)
        sample.update(
            motion_data_smpl85=motion,
            imu_acc=np.ascontiguousarray(acc.numpy()[:, slots]),
            imu_ori=np.ascontiguousarray(ori.numpy()[:, slots]),
            imu_acc_smoothed=False,
            imu_corrections={"direct_readings": True},
        )
        return sample

    w = rotation_matrix_to_axis_angle(ori[:-1].transpose(2, 3).matmul(ori[1:])).view(-1, ori.shape[1], 3) * DIP_SOURCE_FPS
    w = torch.cat((w, torch.zeros_like(w[:1])))
    a = ori.transpose(2, 3).matmul((acc - DIP_GRAVITY).unsqueeze(-1)).squeeze(-1)

    a_s = torch.tensor(smooth_avg(a, s=3)).float()
    r_sim = torch.empty(n, 6, 3, 3)
    r_sim[0] = torch.eye(3).float()
    d_r = axis_angle_to_rotation_matrix(w / DIP_SOURCE_FPS).view(-1, 6, 3, 3).cpu()
    for i in range(1, n):
        r_sim[i] = r_sim[i - 1].matmul(d_r[i])
    a_sim = r_sim.matmul(a_s.unsqueeze(-1)).squeeze(-1)

    slots = list(LEGACY_TO_MODEL_SLOTS)
    sample = _base_sample("dipimu", motion_id, actor_id, DIPIMU_DEVICES)
    sample.update(
        motion_data_smpl85=motion,
        imu_acc=np.ascontiguousarray(a_sim.numpy()[:, slots]),
        imu_ori=np.ascontiguousarray(r_sim.numpy()[:, slots]),
        imu_acc_smoothed=True,
    )
    return sample


LEGACY_CONVERTERS = {
    "imuposer": imuposer_legacy_to_sample,
    "dipimu": dipimu_legacy_to_sample,
}
SUBJECT_SPLITS = {"imuposer": IMUPOSER_SPLIT, "dipimu": DIPIMU_SPLIT}


def validate_sample(sample: dict, *, source: str | None = None) -> dict:
    """Raise on schema violations; return a short summary record."""
    if sample.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"invalid schema_version {sample.get('schema_version')!r}")
    if sample.get("source") not in SOURCES or (source is not None and sample["source"] != source):
        raise ValueError(f"invalid source {sample.get('source')!r}")
    motion = np.asarray(sample["motion_data_smpl85"])
    acc = np.asarray(sample["imu_acc"])
    ori = np.asarray(sample["imu_ori"])
    if motion.ndim != 2 or motion.shape[1] != 85 or motion.dtype != np.float32:
        raise ValueError(f"expected float32 motion [T,85], got {motion.dtype} {motion.shape}")
    t = len(motion)
    if acc.shape != (t, 6, 3) or ori.shape != (t, 6, 3, 3):
        raise ValueError(f"expected imu_acc [T,6,3] / imu_ori [T,6,3,3] with T={t}, got {acc.shape}, {ori.shape}")
    if acc.dtype != np.float32 or ori.dtype != np.float32:
        raise ValueError("imu arrays must be float32")
    if not (np.isfinite(motion).all() and np.isfinite(acc).all() and np.isfinite(ori).all()):
        raise ValueError("non-finite human/IMU data")
    if not np.allclose(ori @ np.swapaxes(ori, -1, -2), np.eye(3), atol=1e-3):
        raise ValueError("imu_ori is not a rotation matrix")
    if not isinstance(sample.get("imu_acc_smoothed"), (bool, np.bool_)):
        raise ValueError("imu_acc_smoothed must be a bool")
    if list(sample.get("imu_sensor_names", ())) != list(IMU_SENSOR_NAMES):
        raise ValueError("unexpected imu_sensor_names")
    if not isinstance(sample.get("texts"), list) or sample.get("objects") != {}:
        raise ValueError("texts must be a list and objects must be empty")
    expected_id = f"{sample['source']}/{sample['actor_id']}/{sample['motion_id']}"
    if sample.get("id") != expected_id:
        raise ValueError(f"id {sample.get('id')!r} != {expected_id!r}")
    return {
        "id": sample["id"],
        "motion_id": sample["motion_id"],
        "actor_id": sample["actor_id"],
        "frames": t,
    }


def atomic_write_pickle(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
