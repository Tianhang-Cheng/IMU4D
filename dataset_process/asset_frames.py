"""Per-asset canonical (upright) frames produced by canonicalize_asset_frames.py.

Applied at load time so the processed WDS shards stay untouched:
  * training/imu_dataset.py re-expresses GT tracks (rotation, AABB centre, extent);
  * visualize/eval_rerun.py renders the canonical OBJ for those assets;
  * run.py uses the canonical extent for predicted bounding boxes.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_PATH = Path(__file__).resolve().parent / "asset_canonical_frames.json"
REPO_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def load_canonical_frames(path: str | None = None) -> dict[str, dict[str, Any]]:
    file = Path(path) if path else DEFAULT_PATH
    if not file.is_file():
        return {}
    payload = json.loads(file.read_text(encoding="utf-8"))
    return {row["asset_id"]: row for row in payload["assets"] if row.get("applied")}


def asset_key(source: str, name: str, metadata: Mapping[str, Any] | None) -> str:
    """Stable per-mesh key. OMOMO part tracks (part == 'bottom', e.g. mop_bottom) get their own key so
    the root track is never re-expressed with the part mesh (2026-09-15 fix)."""
    metadata = metadata or {}
    if metadata.get("asset_id"):
        return str(metadata["asset_id"])
    base = metadata.get("mesh_id") or metadata.get("source_category") or name
    part = metadata.get("part")
    if part and part not in ("whole", "rigid", "top"):
        base = f"{base}_{part}"
    return f"{source}/{base}"


def canonical_frame(asset_id: str) -> dict[str, Any] | None:
    return load_canonical_frames().get(asset_id)


def canonicalize_track(track: np.ndarray, asset_id: str) -> np.ndarray:
    """Re-express a ``[T, 10]`` / ``[10]`` wxyz_xyz_bbox track in the asset's canonical frame.

    R'_t = R_t R_c^T ; ext'_t = s_t * canonical_extent ; c'_t = AABB centre of the canonical mesh in world.
    s_t = extent_t[0] / local_extent[0] (uniform per-frame scale) unless the frame carries ``fixed_scale``
    (HUMOTO tracks store a unit placeholder extent; meshes are in metres).  ``track_translation_ref`` says
    what the stored translation is: the original mesh AABB centre (OMOMO) or the mesh origin (HiPHI, HUMOTO).
    World geometry is unchanged.
    """
    frame = canonical_frame(asset_id)
    if frame is None:
        return track
    track = np.asarray(track, dtype=np.float32)
    squeeze = track.ndim == 1
    if squeeze:
        track = track[None]
    if track.shape[1] < 10:
        raise ValueError(f"Canonical frame for {asset_id!r} needs a bbox extent, got layout {track.shape}")
    w, x, y, z = frame["canonical_rotation_wxyz"]
    R_c = Rotation.from_quat([x, y, z, w]).as_matrix()
    R = Rotation.from_quat(track[:, [1, 2, 3, 0]].astype(np.float64)).as_matrix()
    R_new = R @ R_c.T
    if frame.get("fixed_scale") is not None:
        scale = np.full((len(track), 1), float(frame["fixed_scale"]))
    else:
        scale = track[:, 7:8].astype(np.float64) / float(frame["local_extent"][0])
    center = track[:, 4:7].astype(np.float64)
    if frame.get("track_translation_ref", "aabb_centre") == "origin":
        # stored translation is the mesh origin: move to the original AABB centre first
        center = center + np.einsum("tij,tj->ti", R, scale * np.asarray(frame["local_center"], dtype=np.float64)[None])
    center = center + np.einsum("tij,tj->ti", R_new, scale * np.asarray(frame["canonical_center"], dtype=np.float64)[None])
    extent = scale * np.asarray(frame["canonical_extent"], dtype=np.float64)[None]
    quat = Rotation.from_matrix(R_new).as_quat()[:, [3, 0, 1, 2]]
    out = np.concatenate([quat, center, extent], axis=-1).astype(np.float32)
    return out[0] if squeeze else out


def canonical_mesh_path(asset_id: str) -> Path | None:
    frame = canonical_frame(asset_id)
    if frame is None or "canonical_mesh" not in frame:
        return None
    path = Path(frame["canonical_mesh"])
    return path if path.is_absolute() else REPO_ROOT / path


def canonical_extent_m(asset: Mapping[str, Any]) -> list[float] | None:
    """Catalog ``extent_m`` re-expressed in the canonical frame (metres_per_mesh_unit * canonical_extent)."""
    frame = canonical_frame(str(asset.get("asset_id", "")))
    extent = asset.get("extent_m")
    if frame is None or "canonical_extent" not in frame:
        return None if extent is None else [float(v) for v in extent]
    unit = frame.get("metres_per_mesh_unit", asset.get("reference_scale"))
    if unit is None:
        return None if extent is None else [float(v) for v in extent]
    return [float(v) * float(unit) for v in frame["canonical_extent"]]
