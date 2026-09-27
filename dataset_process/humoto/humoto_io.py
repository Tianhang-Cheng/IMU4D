"""IO and schema conversion helpers for the legacy HuMoTo release."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_process.object_taxonomy import normalize_object_annotations


FPS = 30.0
IMU_SENSOR_NAMES = (
    "left_hip",
    "right_hip",
    "left_ear",
    "right_ear",
    "left_elbow",
    "right_elbow",
)

# HuMOTO stores the SMPL-X translation parameter in motion_smpl. Adding the
# model's rest-pelvis location converts that parameter to the global pelvis
# position used by the shared schema. The legacy virtual-IMU generator treated
# that SMPL translation as a pelvis position, leaving its sensor positions at
# -REST_PELVIS relative to FK. Correct those sensors separately; object positions
# are already world-space and must not receive any offset.
REST_PELVIS = np.array(
    [0.00312326, -0.35140744, 0.01203655], dtype=np.float32
)


def load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_pickle(sample: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".pkl",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            pickle.dump(dict(sample), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def motion_to_smpl85(value: np.ndarray) -> np.ndarray:
    """Convert HuMoTo's root+body+hands+pelvis layout to SMPL85."""

    motion = np.asarray(value, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] not in (75, 85):
        raise ValueError(f"Expected HuMoTo motion [T,75] or [T,85], got {motion.shape}")
    motion = motion.copy()
    motion[:, 72:75] += REST_PELVIS
    if motion.shape[1] == 75:
        motion = np.concatenate(
            [motion, np.zeros((len(motion), 10), dtype=np.float32)], axis=-1
        )
    return motion


def world_imu_positions(positions: np.ndarray, sample: Mapping) -> np.ndarray:
    """Upgrade uncorrected sensor positions; both loader and viewer use this."""
    already_corrected = bool(sample.get("imu_position_rest_pelvis_correction_applied", False))
    # The older converter that shifted all scene geometry also shifted IMUs.
    already_corrected |= (sample.get("legacy_rest_pelvis_correction") is not None
                          and sample.get("world_geometry_rest_pelvis_correction_applied") is not False)
    return np.asarray(positions) if already_corrected else np.asarray(positions, dtype=np.float64) + REST_PELVIS


def corrected_imu(value: np.ndarray, frame_count: int) -> np.ndarray:
    """Undo the legacy virtual-sensor generator's missing rest-pelvis offset."""

    imu = np.asarray(value, dtype=np.float32)
    if imu.shape != (frame_count, 6, 6):
        raise ValueError(f"Expected HuMoTo imu_traj [{frame_count},6,6], got {imu.shape}")
    imu = imu.copy()
    imu[:, :, 3:6] += REST_PELVIS
    return imu


def _catalog_entry(catalog: Mapping[str, object], name: str) -> dict:
    value = catalog.get(name, {})
    if isinstance(value, (list, tuple)):
        return {"extent": value}
    if not isinstance(value, dict):
        raise TypeError(f"Object catalog entry {name!r} must be an object or [x,y,z]")
    return dict(value)


def _extent(entry: Mapping[str, object], name: str, require_extent: bool) -> tuple[np.ndarray, str]:
    value = entry.get("extent")
    if value is None:
        if require_extent and name != "ground":
            raise ValueError(
                f"No bbox extent for {name!r}; add it to --object-catalog or "
                "omit --require-object-extents to use the legacy unit placeholder"
            )
        return np.ones(3, dtype=np.float32), "legacy_unit_placeholder"
    extent = np.asarray(value, dtype=np.float32)
    if extent.shape != (3,) or not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError(f"Invalid extent for {name!r}: {value!r}")
    return extent, str(entry.get("bbox_source", "object_catalog"))


def object_motion_mask(
    track: np.ndarray,
    fps: float = FPS,
    translation_threshold_mps: float = 0.03,
    rotation_threshold_dps: float = 10.0,
) -> np.ndarray:
    """Classify actual movement independently of trajectory availability."""

    moving = np.zeros(len(track), dtype=bool)
    if len(track) < 2:
        return moving
    translation_speed = np.zeros(len(track), dtype=np.float32)
    translation_speed[1:] = np.linalg.norm(np.diff(track[:, 4:7], axis=0), axis=-1) * fps
    rotations = Rotation.from_quat(track[:, [1, 2, 3, 0]])
    rotation_speed = np.zeros(len(track), dtype=np.float32)
    rotation_speed[1:] = np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude()) * fps
    moving = (translation_speed > translation_threshold_mps) | (
        rotation_speed > rotation_threshold_dps
    )
    if len(moving) >= 5:
        votes = np.convolve(moving.astype(np.int16), np.ones(5, dtype=np.int16), mode="same")
        moving = votes >= 3
    return moving.astype(bool)


def make_object_tracks(
    source_objects: Mapping[str, np.ndarray],
    frame_count: int,
    catalog: Mapping[str, object],
    require_extent: bool,
    translation_threshold_mps: float,
    rotation_threshold_dps: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict], dict[str, np.ndarray]]:
    tracks: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    moving_masks: dict[str, np.ndarray] = {}
    for source_name, value in source_objects.items():
        pose = np.asarray(value, dtype=np.float32)
        has_temporal_track = pose.ndim == 2
        if pose.ndim == 2:
            if pose.shape[0] != frame_count or pose.shape[1] not in (7, 10):
                raise ValueError(
                    f"Object {source_name!r} must have shape [{frame_count},7/10], got {pose.shape}"
                )
        elif pose.ndim != 1 or pose.shape[0] not in (7, 10):
            raise ValueError(f"Object {source_name!r} must have shape [7] or [10], got {pose.shape}")

        entry = _catalog_entry(catalog, source_name)
        category = str(entry.get("category", source_name))
        if category in tracks:
            raise ValueError(
                f"Object category map merges multiple instances into {category!r}; "
                "the shared dict requires unique instance keys"
            )
        source_track = pose if has_temporal_track else np.broadcast_to(pose, (frame_count, len(pose)))
        quaternion = source_track[:, :4].copy()
        norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
        if not np.isfinite(norm).all() or np.any(norm < 1e-8):
            raise ValueError(f"Object {source_name!r} has an invalid quaternion")
        quaternion /= norm
        translation = source_track[:, 4:7].copy()
        # Object poses are already in the same world frame as the converted
        # pelvis position and IMU positions. REST_PELVIS is only a semantic
        # SMPL-translation-to-pelvis conversion, not a global scene offset.
        if source_track.shape[1] == 10:
            extent = source_track[:, 7:10].copy()
            bbox_source = "legacy_sample"
        else:
            one_extent, bbox_source = _extent(entry, source_name, require_extent)
            extent = np.broadcast_to(one_extent, (frame_count, 3)).copy()
        track = np.concatenate([quaternion, translation, extent], axis=-1).astype(np.float32)
        tracks[category] = track
        temporal_mode = (
            "dynamic"
            if has_temporal_track and not np.allclose(track[:, :7], track[:1, :7], atol=1e-6)
            else "static"
        )
        moving_masks[category] = (
            object_motion_mask(
                track,
                translation_threshold_mps=translation_threshold_mps,
                rotation_threshold_dps=rotation_threshold_dps,
            )
            if temporal_mode == "dynamic"
            else np.zeros(frame_count, dtype=bool)
        )

        object_metadata = {
            "category": category,
            "source_category": source_name,
            "object_id": str(entry.get("object_id", source_name)),
            "asset_id": str(entry.get("asset_id", f"humoto/{source_name}")),
            "pose_layout": "wxyz_xyz_bbox_xyz",
            "bbox_convention": "local_extent_centered_at_track_translation",
            "bbox_source": bbox_source,
            "temporal_mode": temporal_mode,
        }
        if temporal_mode == "dynamic":
            object_metadata["motion_thresholds"] = {
                "translation_mps": translation_threshold_mps,
                "rotation_dps": rotation_threshold_dps,
                "majority_window_frames": 5,
            }
        if entry.get("mesh_path") is not None:
            object_metadata["mesh_path"] = str(entry["mesh_path"])
        metadata[category] = object_metadata
    tracks, metadata, _, moving_masks = normalize_object_annotations(
        tracks,
        metadata,
        motion_masks=moving_masks,
        source="humoto",
    )
    return tracks, metadata, moving_masks


def convert_legacy_sample(
    legacy: Mapping[str, object],
    source_index: int,
    catalog: Mapping[str, object] | None = None,
    require_extent: bool = False,
    translation_threshold_mps: float = 0.03,
    rotation_threshold_dps: float = 10.0,
) -> dict:
    """Convert one legacy ``all/*.pkl`` record to shared schema version 1."""

    motion_source = legacy.get("motion_smpl", legacy.get("motion_data_smpl85"))
    if motion_source is None:
        raise KeyError("HuMoTo sample has neither motion_smpl nor motion_data_smpl85")
    motion = motion_to_smpl85(np.asarray(motion_source))
    imu = corrected_imu(np.asarray(legacy["imu_traj"]), len(motion))
    tracks, metadata, moving_masks = make_object_tracks(
        legacy.get("objects", {}),
        len(motion),
        catalog or {},
        require_extent,
        translation_threshold_mps,
        rotation_threshold_dps,
    )
    valid_masks = {name: np.ones(len(motion), dtype=bool) for name in tracks}
    temporal_modes = {value["temporal_mode"] for value in metadata.values()}
    task_mode = (
        "mixed"
        if temporal_modes == {"static", "dynamic"}
        else "dynamic_hoi"
        if temporal_modes == {"dynamic"}
        else "static_scene"
    )
    descriptions = legacy.get("texts", legacy.get("text", legacy.get("description", [])))
    if descriptions is None:
        descriptions = []
    elif isinstance(descriptions, str):
        descriptions = [descriptions]
    else:
        descriptions = [str(value) for value in descriptions]
    motion_id = f"{source_index:07d}"
    return {
        "schema_version": 1,
        "source": "humoto",
        "id": f"HUMOTO/{motion_id}",
        "motion_id": motion_id,
        "actor_id": "unknown",
        "actor_id_source": "not_provided_by_release",
        "source_index": int(source_index),
        "fps": FPS,
        "motion_data_smpl85": motion,
        "imu_traj": imu,
        "imu_sensor_names": list(IMU_SENSOR_NAMES),
        "objects": tracks,
        "object_metadata": metadata,
        "object_valid_mask": valid_masks,
        "object_motion_mask": moving_masks,
        "task_mode": task_mode,
        "annotation_scope": "full_scene",
        "description": descriptions,
        "texts": descriptions,
        "train_ready": True,
        "coordinate_frame": "right_handed_Y_up_metres",
        "smpl_translation_to_pelvis_offset": REST_PELVIS.copy(),
        "world_geometry_rest_pelvis_correction_applied": False,
        "imu_position_rest_pelvis_correction_applied": True,
    }
