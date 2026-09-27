"""Load-time chair objects for the NCSA meeting-room clips.

The ``ncsa/objects`` release (HF ``TianhangCheng7/IMU4DData``, built by
``dataset_process/ncsa/export_chair_objects.py``) holds one chair mesh and static
per-session poses in the **calibration / cam_01 world**, plus ``sample_index.json``
binding packed clip ids to sessions.  It deliberately does not touch the WDS
shards, so this module converts and attaches the poses at load time -- the same
pattern as the ``ground_v1`` sidecars.

Two conversions, both verified against the packed data (2026-09-22):

* **World frame.**  ``convert_meeting_room_mv._motion_in_imu_world`` packs the
  pelvis as ``bridge @ transl_cam + REST_PELVIS`` although the true pelvis is
  ``transl_cam + REST_PELVIS`` in camera world.  Relative to the packed human the
  camera world therefore maps as ``p -> bridge @ p + (REST_PELVIS - bridge @
  REST_PELVIS)`` (~0.7 m in Y, since ``bridge`` is close to ``diag(1, -1, -1)``).
  With that offset the lowest chair point lands 0-5 cm from the session's
  ``ground_v1`` floor on all 15 instances; a pure rotation misses it by 0.7 m.
* **Local frame.**  ``assets/chair.obj`` is Z-up with the seat facing +Y (the
  backrest sits at -Y).  The loader's object convention
  (``asset_canonical_frames.json``) is +Y up, front +Z, so the rotation is
  re-expressed with ``_MESH_FROM_CANONICAL`` and the extent permuted to match.
  The mesh is bbox-centred, so the translation is unchanged.

Opt-in: nothing is attached unless ``ENV_ROOT`` names a release directory.
``run.py`` sets it from ``dataset.params.ncsa_object_root`` before the loaders
fork their workers.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_process.humoto.humoto_io import REST_PELVIS

ENV_ROOT = "IMU4D_NCSA_OBJECT_ROOT"

# Columns are the canonical axes (x, y=up, z=front) written in mesh coordinates:
# v_mesh = _MESH_FROM_CANONICAL @ v_canonical.
_MESH_FROM_CANONICAL = np.array(
    [[-1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]]
)


def object_root() -> Path | None:
    root = os.environ.get(ENV_ROOT, "").strip()
    return Path(root) if root else None


@lru_cache(maxsize=None)
def _load_release(root: str) -> dict[str, dict]:
    """sample id -> {"frames": int, "objects": {key: camera-world record}}."""
    root_path = Path(root)
    index_path = root_path / "sample_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"{ENV_ROOT}={root}: no sample_index.json")
    sessions: dict[str, dict] = {}
    bindings: dict[str, dict] = {}
    for entry in json.loads(index_path.read_text()):
        path = entry["annotation_path"]
        if path not in sessions:
            sessions[path] = json.loads((root_path / path).read_text())["objects"]
        bindings[entry["id"]] = {
            "frames": int(entry["frames"]),
            "objects": {key: sessions[path][key] for key in entry["object_keys"]},
        }
    return bindings


def chair_track_in_imu_world(record: dict, bridge: np.ndarray) -> np.ndarray:
    """Camera-world release record -> ``[qw, qx, qy, qz, t, extent]`` in packed IMU world."""
    bridge = np.asarray(bridge, dtype=np.float64)
    mesh_to_world = np.asarray(record["mesh_to_world"], dtype=np.float64)
    rotation = bridge @ mesh_to_world[:3, :3] @ _MESH_FROM_CANONICAL
    rest = REST_PELVIS.astype(np.float64)
    translation = bridge @ mesh_to_world[:3, 3] + (rest - bridge @ rest)
    extent = np.abs(_MESH_FROM_CANONICAL.T) @ np.asarray(record["extent_xyz"], dtype=np.float64)
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    quat_wxyz = np.roll(quat_xyzw, 1)
    return np.concatenate([quat_wxyz, translation, extent]).astype(np.float32)


def attach_ncsa_objects(model_sample: dict, payload: dict) -> None:
    """Add the bound chairs to ``model_sample["objects"]`` (no-op when disabled or unbound)."""
    root = object_root()
    if root is None:
        return
    binding = _load_release(str(root)).get(model_sample.get("id"))
    if binding is None:
        return
    n_frames = len(payload["motion_data_smpl85"])
    if n_frames != binding["frames"]:
        raise ValueError(
            f"{model_sample['id']}: release binds {binding['frames']} frames, sample has {n_frames}"
        )
    bridge = payload["motion_provenance"]["multiview_to_imu_world_rotation"]
    objects = dict(model_sample.get("objects") or {})
    metadata = dict(model_sample.get("object_metadata") or {})
    for key, record in binding["objects"].items():
        if key in objects:
            raise ValueError(f"{model_sample['id']}: object {key!r} already present")
        objects[key] = chair_track_in_imu_world(record, bridge)
        metadata[key] = {"asset_id": record["asset_id"], "category": record["category"]}
    model_sample["objects"] = objects
    model_sample["object_metadata"] = metadata
