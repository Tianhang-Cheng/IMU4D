"""Ground annotations in the *stored* motion frame, before any random crop.

MotionMillion and DIP-IMU do not preserve a floor-at-zero origin. Their ground
is a pseudo-label inferred from neutral-body FK, matching the body used for IMU
synthesis and motion evaluation. Explicit scene floors always take precedence.
No body/IMU translation is changed here. A joint contact plane is an estimate,
not a measured sole surface; clips without stable foot support are not used for
anchor supervision.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_process.custom_path import smplx_model_path


# Sources whose stored frame has no floor-at-zero guarantee, so the floor must
# be estimated from the motion itself.  NCSA joined on 2026-09-21: the
# meeting-room capture lives in the calibration world whose origin is cam_01,
# about 1.5 m ABOVE the floor, so the "source_floor" default put the ground
# plane ~0.79 m above the pelvis.  IMUPoser / HiPHI / OMOMO are left out on
# purpose -- their storage frame already has the floor at y = 0.
ESTIMATED_GROUND_SOURCES = frozenset({"motionmillion", "dipimu", "ncsa"})
GROUND_VERSION = 1
GROUND_DIRNAME = "ground_v1"
FOOT_JOINTS = (7, 8, 10, 11)
_ESTIMATES: OrderedDict = OrderedDict()


def ground_object_height(objects: dict, default: float = 0.0) -> float:
    """Read the first floor height from either a static [3] or temporal [T,3] pose."""
    floor = objects.get("ground")
    if floor is None:
        return float(default)
    translation = floor["transl"]
    if hasattr(translation, "detach"):
        translation = translation.detach().float().cpu().numpy()
    translation = np.asarray(translation)
    if translation.ndim not in (1, 2) or translation.shape[-1] != 3 or not translation.size:
        raise ValueError(f"Invalid ground translation shape {translation.shape}")
    return float(translation.reshape(-1, 3)[0, 1])


def motion_fingerprint(motion: np.ndarray) -> str:
    value = np.ascontiguousarray(motion)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str((value.shape, value.dtype.str)).encode())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


@lru_cache(maxsize=2)
def neutral_skeleton(model_path: str = smplx_model_path) -> tuple[np.ndarray, np.ndarray]:
    """Load only neutral rest joints; FK needs no GPU or skinned mesh."""
    with np.load(model_path, allow_pickle=True) as model:
        joints = (model["J_regressor"] @ model["v_template"])[:22].astype(np.float64)
        parents = model["kintree_table"][0, :22].astype(int)
    parents[0] = -1
    return joints, parents


def world_joints(motion: np.ndarray, skeleton=None) -> np.ndarray:
    """SMPL85 (or legacy 69) -> neutral joints, with joint 0 at stored pelvis."""
    motion = np.asarray(motion)
    if motion.ndim != 2 or motion.shape[1] not in (69, 75, 85) or not len(motion):
        raise ValueError(f"Expected nonempty SMPL motion [T,69/75/85], got {motion.shape}")
    rest, parents = neutral_skeleton() if skeleton is None else skeleton
    local = Rotation.from_rotvec(motion[:, :66].reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
    rotations = local.copy()
    joints = np.empty((len(motion), 22, 3), dtype=np.float64)
    joints[:, 0] = motion[:, 66:69] if motion.shape[1] == 69 else motion[:, 72:75]
    for joint in range(1, 22):
        parent = parents[joint]
        joints[:, joint] = joints[:, parent] + np.einsum(
            "tij,j->ti", rotations[:, parent], rest[joint] - rest[parent]
        )
        rotations[:, joint] = rotations[:, parent] @ local[:, joint]
    return joints


def estimate_ground_plane(motion: np.ndarray, fps: float = 30.0, *, skeleton=None) -> dict:
    """Estimate over the whole clip, never over the randomly selected window.

    Uniform sampling caps FK work at 256 frames. Contact checks use the actual
    time gaps, reject obvious airborne/non-foot support and retain uncertainty
    explicitly. They cannot establish the existence of a physical floor from
    motion alone (e.g. feet resting on a raised platform).
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    motion = np.asarray(motion)
    if not len(motion):
        raise ValueError("Cannot estimate ground from an empty motion")
    index = np.unique(np.linspace(0, len(motion) - 1, min(len(motion), 256), dtype=int))
    record = {
        "version": GROUND_VERSION,
        "method": "neutral_smplx_joint_contact_q02",
        "num_frames": len(motion),
        "motion_hash": motion_fingerprint(motion),
        "fps": float(fps),
        "height_y": 0.0,
        "valid": False,
        "reasons": [],
    }
    if not np.isfinite(motion[:, :75]).all():
        record["reasons"] = ["nonfinite_motion"]
        return record
    joints = world_joints(motion[index], skeleton=skeleton)
    feet = joints[:, FOOT_JOINTS]
    foot_low = feet[..., 1].min(axis=1)
    foot_floor = float(np.quantile(foot_low, 0.02))
    body_floor = float(np.quantile(joints[..., 1].min(axis=1), 0.02))
    # A lying/hand-supported clip still gets a finite display estimate, but no
    # invented ground-pose target. Foot joints alone can be far above its floor.
    record["height_y"] = min(foot_floor, body_floor)
    reasons = []
    if body_floor < foot_floor - 0.12:
        reasons.append("non_foot_support")
    if len(index) < 3:
        reasons.append("too_few_frames")
        stable = np.zeros(len(index), dtype=bool)
    else:
        velocity = np.gradient(feet, index / fps, axis=0)
        near_floor = np.abs(feet[..., 1] - foot_floor) <= 0.05
        stable = (near_floor & (np.linalg.norm(velocity, axis=-1) <= 0.30)).any(axis=1)
        if stable.sum() < max(3, int(np.ceil(0.02 * len(index)))):
            reasons.append("no_stable_foot_contact")
    if float(np.quantile(foot_low, 0.10)) - foot_floor > 0.12:
        reasons.append("unstable_lower_envelope")
    record["stable_contact_fraction"] = float(stable.mean())
    record["reasons"] = reasons
    record["valid"] = not reasons
    return record


def sidecar_path(shard: str | Path) -> Path:
    """<root>/wds/<split>/<shard>.tar -> <root>/ground_v1/<split>/<shard>.jsonl."""
    shard = Path(shard)
    return shard.parents[2] / GROUND_DIRNAME / shard.parent.name / f"{shard.stem}.jsonl"


@lru_cache(maxsize=8)
def _read_sidecar(path: str, mtime_ns: int, size: int) -> dict:
    records = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record["id"]
            if key in records:
                raise ValueError(f"Duplicate ground record {key!r} in {path}")
            records[key] = record
    return records


def resolve_ground_plane(sample: dict, motion: np.ndarray, source: str, fps: float) -> dict:
    """Explicit annotation, then sidecar, then cached on-demand FK.

    On-demand FK makes correctness independent of finishing a corpus-wide
    sidecar job. Sidecars and the small per-worker cache store only scalars,
    never the motion arrays. Stale sidecars are rejected by a content hash.
    """
    explicit = sample.get("ground_plane")
    if explicit is not None:
        height = float(explicit["height_y"])
        if not np.isfinite(height):
            raise ValueError("ground_plane.height_y must be finite")
        return {**explicit, "height_y": height, "valid": bool(explicit.get("valid", True))}
    if source not in ESTIMATED_GROUND_SOURCES:
        return {"height_y": 0.0, "valid": True, "method": "source_floor"}

    fingerprint = motion_fingerprint(motion)
    key = (source, fingerprint, float(fps))
    if key in _ESTIMATES:
        _ESTIMATES.move_to_end(key)
        return dict(_ESTIMATES[key])
    record = None
    shard = sample.get("__url__")
    if shard and "://" not in str(shard):
        path = sidecar_path(shard)
        if path.is_file():
            stat = path.stat()
            candidate = _read_sidecar(str(path), stat.st_mtime_ns, stat.st_size).get(sample.get("id"))
            if candidate is not None:
                if (candidate.get("version") != GROUND_VERSION
                        or candidate.get("motion_hash") != fingerprint
                        or candidate.get("fps") != float(fps)):
                    raise ValueError(f"Stale ground sidecar for {sample.get('id')!r}: {path}")
                if not np.isfinite(candidate["height_y"]) or not isinstance(candidate.get("valid"), bool):
                    raise ValueError(f"Invalid ground record for {sample.get('id')!r}: {path}")
                record = candidate
    if record is None:
        try:
            record = estimate_ground_plane(motion, fps)
        except FileNotFoundError:
            # Test/minimal deployments may omit the licensed body model. Never
            # silently turn their old pelvis-height floor into supervision.
            record = {"height_y": 0.0, "valid": False, "method": "unavailable",
                      "reasons": ["missing_smplx_model"]}
    _ESTIMATES[key] = record
    if len(_ESTIMATES) > 4096:
        _ESTIMATES.popitem(last=False)
    return dict(record)
