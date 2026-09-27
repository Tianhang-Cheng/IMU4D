"""Shared IO, geometry, and coordinate helpers for OMOMO conversion."""

from __future__ import annotations

import json
import pickle
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import trimesh
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

from dataset_process.object_taxonomy import canonical_category


FPS = 30.0
IMU_SENSOR_NAMES = (
    "left_hip",
    "right_hip",
    "left_ear",
    "right_ear",
    "left_elbow",
    "right_elbow",
)
IMU_VERTEX_IDS = (4133, 6877, 229, 940, 4576, 7313)
IMU_JOINT_IDS = (1, 2, 15, 15, 18, 19)

# OMOMO is right-handed Z-up. IMU4D uses right-handed Y-up.
# [x, y, z]_omomo -> [x, z, -y]_imu4d is a -90 degree X rotation.
OMOMO_TO_IMU4D = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)

CANONICAL_OBJECT_NAMES = {
    "clothesstand": "clothes_rack",
    "floorlamp": "floor_lamp",
    "largebox": "box",
    "largetable": "table",
    "monitor": "monitor",
    "mop": "mop",
    "plasticbox": "box",
    "smallbox": "box",
    "smalltable": "side_table",
    "suitcase": "suitcase",
    "trashcan": "trash_can",
    "tripod": "tripod",
    "vacuum": "vacuum",
    "whitechair": "chair",
    "woodchair": "chair",
}


@dataclass(frozen=True)
class SequenceRecord:
    motion_id: str
    actor_id: str
    object_name: str
    source_split: str
    source_key: int
    frames: int
    description: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "motion_id": self.motion_id,
            "actor_id": self.actor_id,
            "object_name": self.object_name,
            "canonical_object_name": canonical_object_name(self.object_name),
            "source_split": self.source_split,
            "source_key": self.source_key,
            "frames": self.frames,
            "duration_sec": self.frames / FPS,
            "description": list(self.description),
        }


def parse_motion_id(motion_id: str) -> tuple[str, str]:
    parts = motion_id.split("_")
    if len(parts) != 3 or not parts[0].startswith("sub"):
        raise ValueError(f"Unexpected OMOMO sequence name: {motion_id!r}")
    return parts[0], parts[1]


def canonical_object_name(name: str) -> str:
    try:
        return canonical_category(CANONICAL_OBJECT_NAMES[name])
    except KeyError as error:
        raise ValueError(f"Unknown OMOMO object category: {name!r}") from error


def load_text_annotations(path: Path) -> dict[str, list[str]]:
    annotations: dict[str, list[str]] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".json"):
                continue
            payload = json.loads(archive.read(member))
            for motion_id, text in payload.items():
                annotations.setdefault(motion_id, []).append(str(text))
    return annotations


def sequence_file(data_root: Path, source_split: str) -> Path:
    return data_root / f"{source_split}_diffusion_manip_seq_joints24.p"


def load_sequences(data_root: Path, source_split: str) -> dict[int, dict]:
    path = sequence_file(data_root, source_split)
    if not path.is_file():
        raise FileNotFoundError(f"Missing OMOMO sequence file: {path}")
    data = joblib.load(path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def index_release(data_root: Path, annotation_zip: Path) -> list[SequenceRecord]:
    texts = load_text_annotations(annotation_zip)
    records: list[SequenceRecord] = []
    seen: set[str] = set()
    for source_split in ("train", "test"):
        for key, sequence in load_sequences(data_root, source_split).items():
            motion_id = str(sequence["seq_name"])
            if motion_id in seen:
                raise ValueError(f"Duplicate OMOMO motion_id: {motion_id}")
            seen.add(motion_id)
            actor_id, object_name = parse_motion_id(motion_id)
            frames = len(sequence["trans"])
            records.append(
                SequenceRecord(
                    motion_id=motion_id,
                    actor_id=actor_id,
                    object_name=object_name,
                    source_split=source_split,
                    source_key=int(key),
                    frames=frames,
                    description=tuple(texts.get(motion_id, ())),
                )
            )
    return sorted(records, key=lambda record: record.motion_id)


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_to_target_vectors(vectors: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", OMOMO_TO_IMU4D, vectors)


def source_to_target_rotations(rotations: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...jk->...ik", OMOMO_TO_IMU4D, rotations)


def axis_angle_to_target(axis_angle: np.ndarray) -> np.ndarray:
    source = Rotation.from_rotvec(np.asarray(axis_angle).reshape(-1, 3)).as_matrix()
    target = source_to_target_rotations(source)
    return Rotation.from_matrix(target).as_rotvec().reshape(np.asarray(axis_angle).shape)


def motion_data_smpl85(sequence: dict) -> np.ndarray:
    root = axis_angle_to_target(np.asarray(sequence["root_orient"], dtype=np.float32))
    body = np.asarray(sequence["pose_body"], dtype=np.float32).reshape(-1, 63)
    # OMOMO's global root joint is `trans - trans2joint`; its trainer performs
    # the inverse conversion when reconstructing the SMPL translation parameter.
    pelvis_source = np.asarray(sequence["trans"], dtype=np.float32) - np.asarray(
        sequence["trans2joint"], dtype=np.float32
    )[None]
    pelvis = source_to_target_vectors(pelvis_source).astype(np.float32)
    betas = np.asarray(sequence["betas"], dtype=np.float32).reshape(-1)[:10]
    betas = np.repeat(betas[None], len(root), axis=0)
    result = np.concatenate(
        [
            root,
            body,
            np.zeros((len(root), 6), dtype=np.float32),
            pelvis,
            betas,
        ],
        axis=-1,
    ).astype(np.float32)
    if result.shape != (len(root), 85):
        raise RuntimeError(f"Internal SMPL85 shape error: {result.shape}")
    return result


def mesh_bbox(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    return (minimum + maximum) / 2.0, maximum - minimum


def make_object_track(
    rotation_source: np.ndarray,
    translation_source: np.ndarray,
    scale: np.ndarray,
    local_bbox_center: np.ndarray,
    local_bbox_extent: np.ndarray,
) -> np.ndarray:
    rotation = source_to_target_rotations(
        np.asarray(rotation_source, dtype=np.float32)
    ).astype(np.float32)
    translation = np.asarray(translation_source, dtype=np.float32).reshape(-1, 3)
    scale = np.asarray(scale, dtype=np.float32).reshape(-1, 1)
    center_source = translation + np.einsum(
        "tij,tj->ti", rotation_source, scale * local_bbox_center[None]
    )
    center = source_to_target_vectors(center_source).astype(np.float32)
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat().astype(np.float32)
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
    extent = scale * local_bbox_extent[None]
    return np.concatenate([quaternion_wxyz, center, extent], axis=-1).astype(
        np.float32
    )


def _sanitize_object_scale(scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a positive scale plus the source annotation-validity mask.

    A small number of OMOMO records use a zero object scale for frames where
    that object's pose annotation is absent.  Keep those frames in the
    temporal record, but mask them from supervision and borrow the nearest
    valid scale so the stored bbox remains a valid geometric primitive.
    """
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    valid = np.isfinite(scale) & (scale > 1e-8)
    if not np.any(valid):
        raise ValueError("Object track has no positive source scale")
    safe = scale.copy()
    valid_indices = np.flatnonzero(valid)
    missing_indices = np.flatnonzero(~valid)
    nearest = valid_indices[
        np.abs(valid_indices[None, :] - missing_indices[:, None]).argmin(axis=1)
    ]
    safe[missing_indices] = safe[nearest]
    return safe, valid


def object_tracks(sequence: dict, mesh_root: Path) -> tuple[dict, dict, dict]:
    _, object_name = parse_motion_id(str(sequence["seq_name"]))
    canonical = canonical_object_name(object_name)
    tracks: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, object]] = {}
    valid_masks: dict[str, np.ndarray] = {}
    # The first whole-object release predicts no articulation. For mop and
    # vacuum, obj_* is the top/root trajectory; the whole composite mesh gives
    # the bbox while obj_bottom_* is intentionally excluded from supervision.
    mesh_path = mesh_root / f"{object_name}_cleaned_simplified.obj"
    local_center, local_extent = mesh_bbox(mesh_path)
    safe_scale, valid_mask = _sanitize_object_scale(sequence["obj_scale"])
    track = make_object_track(
        sequence["obj_rot"],
        sequence["obj_trans"],
        safe_scale,
        local_center,
        local_extent,
    )
    tracks[canonical] = track
    metadata[canonical] = {
        "category": canonical,
        "source_category": object_name,
        "asset_id": f"omomo/{object_name}",
        "part": "whole",
        "temporal_mode": "dynamic",
        "mesh_file": mesh_path.name,
        "bbox_convention": "oriented_local_extent_centered_at_track_translation",
    }
    valid_masks[canonical] = valid_mask
    return tracks, metadata, valid_masks


def motion_mask(
    track: np.ndarray,
    fps: float = FPS,
    translation_threshold_mps: float = 0.03,
    rotation_threshold_dps: float = 10.0,
) -> np.ndarray:
    translation_speed = np.r_[
        0.0, np.linalg.norm(np.diff(track[:, 4:7], axis=0), axis=-1) * fps
    ]
    quaternion_xyzw = track[:, [1, 2, 3, 0]]
    relative = Rotation.from_quat(quaternion_xyzw[1:]) * Rotation.from_quat(
        quaternion_xyzw[:-1]
    ).inv()
    angular_speed = np.r_[0.0, np.rad2deg(relative.magnitude()) * fps]
    moving = (translation_speed > translation_threshold_mps) | (
        angular_speed > rotation_threshold_dps
    )
    return (median_filter(moving.astype(np.uint8), size=5, mode="nearest") > 0)


def atomic_write_pickle(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
