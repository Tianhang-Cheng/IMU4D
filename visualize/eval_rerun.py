"""Write paired GT/pred Rerun recordings from one evaluation rollout.

This module is imported lazily by the training loop so Rerun and SMPL-X are
only loaded when a configured full-eval sample is actually encountered.
"""

from __future__ import annotations

import json
import os
from itertools import cycle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rerun as rr
import torch

from utils.rotation2 import convert_rotation
from dataset_process.asset_frames import asset_key, canonical_mesh_path
from visualize.viz_hoi_rrd import (
    HUMAN_COLOR,
    IMU_COLOR,
    OBJECT_COLORS,
    HumanMeshGenerator,
    _blueprint,
    _load_mesh,
    _log_ground,
    _vertex_normals,
)


PRED_HUMAN_COLOR = [129, 140, 248, 235]
GT_TRAJECTORY_COLOR = [245, 158, 11, 180]
PRED_TRAJECTORY_COLOR = [99, 102, 241, 200]
COMPARE_FILE = "compare.rrd"
ROLLOUT_FILE = "rollout.pkl"


def _to_plain(value: Any) -> Any:
    """Recursively convert tensors to numpy so the rollout pickles without torch."""

    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def dump_rollout(rollout: Mapping[str, Any], path: Path) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pkl")
    with open(temporary, "wb") as handle:
        pickle.dump(_to_plain(rollout), handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def pred_object_color(color: list[int]) -> list[int]:
    """Blend a GT object color toward the prediction hue so pred geometry reads
    as one family while keeping enough of the per-object hue to match pairs."""

    blended = [
        int(round(0.45 * float(channel) + 0.55 * float(target)))
        for channel, target in zip(color[:3], PRED_HUMAN_COLOR[:3])
    ]
    return blended + [color[3] if len(color) > 3 else 230]


def _array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def normalized_sample_id(value: Any) -> str:
    """Normalize the batch-size-one sample id emitted by eval_model."""

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one sample id, got {value!r}")
        value = value[0]
    return str(value)


def safe_motion_id(sample_id: str) -> str:
    """Return a filesystem-safe sequence name without its dataset prefix."""

    motion_id = sample_id.rsplit("/", 1)[-1]
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in motion_id)
    if not safe or safe in {".", ".."}:
        raise ValueError(f"Invalid motion id in {sample_id!r}")
    return safe


def _recording_id(rollout: Mapping[str, Any], sample_id: str, role: str) -> str:
    metadata = rollout.get("metadata", {})
    return f"{metadata.get('dataset', 'eval')}_{safe_motion_id(sample_id)}_step{metadata.get('step', 0)}_{role}"


def motion_to_smpl85(sample: Mapping[str, Any]) -> np.ndarray:
    """Convert an eval ``gt`` or ``pred`` motion dictionary to SMPL85."""

    transl = _array(sample["transl"]).reshape(-1, 3)
    orient = _array(sample["orient"]).reshape(-1, 3)
    pose = _array(sample["pose"]).reshape(-1, 63)
    length = min(len(transl), len(orient), len(pose))
    motion = np.zeros((length, 85), dtype=np.float32)
    motion[:, :3] = orient[:length]
    motion[:, 3:66] = pose[:length]
    motion[:, 72:75] = transl[:length]
    return motion


def select_frames(length: int, source_fps: float, record_fps: float, max_seconds: float) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot record an empty rollout")
    if source_fps <= 0 or record_fps <= 0 or max_seconds <= 0:
        raise ValueError("FPS and max_seconds must be positive")
    end = min(length, max(1, round(source_fps * max_seconds)))
    stride = max(1, round(source_fps / record_fps))
    frames = np.arange(0, end, stride, dtype=np.int32)
    if frames[-1] != end - 1:
        frames = np.r_[frames, end - 1].astype(np.int32)
    return frames


def _object_track(value: Any, length: int) -> np.ndarray:
    array = _array(value)
    if array.ndim == 1:
        return np.broadcast_to(array, (length, len(array)))
    if array.ndim == 2:
        if len(array) < length:
            raise ValueError(f"Object track is shorter than motion: {len(array)} < {length}")
        return array[:length]
    raise ValueError(f"Expected a static or temporal object field, got {array.shape}")


def _object_pose_track(value: Mapping[str, Any], length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotations = _object_track(value["rot"], length)
    translations = _object_track(value["transl"], length)
    bbox = _object_track(value.get("bbox", np.ones(3, dtype=np.float32)), length)
    matrices = convert_rotation(
        torch.from_numpy(np.array(rotations, copy=True)).float(), "6d", "mat"
    )
    quaternions = convert_rotation(matrices, "mat", "quat").cpu().numpy()
    return quaternions, translations, bbox


def resolve_object_mesh(
    name: str,
    object_value: Mapping[str, Any],
    object_metadata: Mapping[str, Mapping[str, Any]],
    asset_roots: Mapping[str, Path],
) -> tuple[Path, Mapping[str, Any]]:
    """Resolve either GT sample metadata or predicted catalog metadata."""

    metadata = object_metadata.get(name, {})
    asset_id = str(object_value.get("asset_id") or metadata.get("asset_id") or "")
    mesh_path = object_value.get("mesh_path") or metadata.get("mesh_path")
    source = asset_id.split("/", 1)[0] if "/" in asset_id else ""
    if not asset_id and "mesh_file" in metadata:
        # GT OMOMO metadata carries no asset_id; use the loader's key (part tracks get their own).
        asset_id = asset_key("omomo", name, metadata)
    canonical = canonical_mesh_path(asset_id)
    if canonical is not None:
        # Asset with a canonical upright frame: tracks were re-expressed at load
        # time, so render the matching canonical OBJ.
        return canonical, metadata

    if mesh_path:
        if not source:
            source = "hiphi"
        return asset_roots[source] / str(mesh_path), metadata
    if "mesh_file" in metadata:
        return asset_roots["omomo"] / "captured_objects" / str(metadata["mesh_file"]), metadata
    if source == "humoto":
        # Unified multi-part HuMOTO objects use the canonical asset-id suffix
        # (for example soap_dispenser), while source_category can name only one
        # legacy part (soap_dispenser_body).
        source_name = asset_id.split("/", 1)[1]
        return (
            asset_roots["humoto"]
            / "humoto_objects_0805"
            / source_name
            / f"{source_name}.obj",
            metadata,
        )
    raise ValueError(f"No mesh identity is available for predicted object {name!r}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _record_variant(
    *,
    role: str,
    rollout: Mapping[str, Any],
    motion: np.ndarray,
    human_vertices: np.ndarray,
    frames: np.ndarray,
    human_faces: np.ndarray,
    output_path: Path,
    source_fps: float,
    record_fps: float,
    asset_roots: Mapping[str, Path],
) -> None:
    sample_id = normalized_sample_id(rollout["sample_idx"])
    sample = rollout[role]
    objects = dict(sample.get("objects", {}))
    gt_objects = rollout["gt"].get("objects", {})
    # A ground plane is reference geometry, not a predicted semantic object.
    # Keep it visible even when object decoding did not emit it.
    if "ground" not in objects and "ground" in gt_objects:
        objects["ground"] = gt_objects["ground"]
    object_metadata = rollout.get("metadata", {}).get("object_metadata", {})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.rrd")
    temporary.unlink(missing_ok=True)
    # Distinct ids per dataset / checkpoint step: the Rerun viewer merges files
    # that share a recording id, which mixes e.g. zero-shot and fine-tuned bodies.
    recording = rr.RecordingStream(
        application_id=f"IMU4D full eval {role}",
        recording_id=_recording_id(rollout, sample_id, role),
    )
    recording.save(temporary, default_blueprint=_blueprint(record_fps))
    recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    description = sample.get("description", "")
    if isinstance(description, (list, tuple)):
        description = description[0] if description else ""
    metadata = rollout.get("metadata", {})
    summary = (
        f"# {metadata.get('dataset', 'HOI')} — {sample_id}\n\n"
        f"- role: `{role}`\n"
        f"- checkpoint step: `{metadata.get('step')}`\n"
        f"- prediction IMU layout: `{metadata.get('layout')}`\n"
        f"- source FPS: `{source_fps:g}`\n"
        f"- MPJPE: `{metadata.get('mpjpe_mm')}` mm\n"
        f"- active IMUs: `{', '.join(rollout['input'].get('active_imu_sensor_names', []))}`\n\n"
        f"**Description:** {description or '(none)'}\n"
    )
    recording.log("description", rr.TextDocument(summary, media_type="text/markdown"), static=True)
    recording.log(
        "world/trajectories/pelvis",
        rr.LineStrips3D([motion[frames, 72:75]], colors=[[245, 158, 11, 180]], radii=0.007),
        static=True,
    )

    prepared_objects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for color, (name, value) in zip(cycle(OBJECT_COLORS), objects.items()):
        quaternions, translations, bbox = _object_pose_track(value, len(motion))
        prepared_objects[name] = (quaternions, translations, bbox)
        if name == "ground":
            _log_ground(recording, float(translations[0, 1]))
            continue
        mesh_path, source_metadata = resolve_object_mesh(
            name, value, object_metadata, asset_roots
        )
        vertices, faces, normals, box_edges, mesh_extent = _load_mesh(mesh_path)
        recording.log(
            f"world/objects/{name}/mesh",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_normals=normals,
                albedo_factor=color,
            ),
            static=True,
        )
        recording.log(
            f"world/objects/{name}/bbox",
            rr.LineStrips3D(box_edges, colors=[color], radii=0.008),
            static=True,
        )
        if source_metadata.get("bbox_source") == "legacy_unit_placeholder":
            prepared_objects[name] = (quaternions, translations, np.broadcast_to(mesh_extent, bbox.shape))
        recording.log(
            f"world/trajectories/{name}",
            rr.LineStrips3D([translations[frames]], colors=[color], radii=0.009),
            static=True,
        )

    imu_positions_value = rollout["input"].get("imu_positions")
    imu_positions = None if imu_positions_value is None else _array(imu_positions_value)
    active_names = rollout["input"].get("active_imu_sensor_names", [])
    active_ids = [
        index
        for index in range(imu_positions.shape[1] if imu_positions is not None else 0)
        if index not in rollout["input"].get("invalid_imu_id_list", [])
    ]
    human_color = HUMAN_COLOR if role == "gt" else PRED_HUMAN_COLOR
    for logged_index, source_frame in enumerate(frames):
        recording.set_time("time", duration=float(source_frame) / source_fps)
        vertices = human_vertices[logged_index]
        recording.log(
            "world/human",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=human_faces,
                vertex_normals=_vertex_normals(vertices, human_faces),
                albedo_factor=human_color,
            ),
        )
        if imu_positions is not None and source_frame < len(imu_positions):
            recording.log(
                "world/virtual_imus",
                rr.Points3D(
                    positions=imu_positions[source_frame, active_ids],
                    colors=[IMU_COLOR],
                    radii=0.025,
                    labels=active_names,
                    show_labels=False,
                ),
            )
        for name, (quaternions, translations, bbox) in prepared_objects.items():
            if name == "ground":
                continue
            quaternion = quaternions[source_frame]
            recording.log(
                f"world/objects/{name}",
                rr.Transform3D(
                    translation=translations[source_frame],
                    quaternion=rr.Quaternion(xyzw=quaternion[[1, 2, 3, 0]]),
                    scale=bbox[source_frame],
                ),
            )
    recording.flush()
    recording.disconnect()
    os.replace(temporary, output_path)


def _log_role_scene(
    recording: rr.RecordingStream,
    *,
    role: str,
    prefix: str,
    rollout: Mapping[str, Any],
    motion: np.ndarray,
    frames: np.ndarray,
    asset_roots: Mapping[str, Path],
    trajectory_color: list[int],
    object_color_fn=None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Log the static part (trajectories, object meshes) of one role under
    ``prefix`` and return the prepared object pose tracks for the time loop.
    The ground plane is shared reference geometry and is not logged here."""

    sample = rollout[role]
    objects = dict(sample.get("objects", {}))
    object_metadata = rollout.get("metadata", {}).get("object_metadata", {})
    recording.log(
        f"{prefix}/trajectories/pelvis",
        rr.LineStrips3D([motion[frames, 72:75]], colors=[trajectory_color], radii=0.007),
        static=True,
    )
    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for color, (name, value) in zip(cycle(OBJECT_COLORS), objects.items()):
        if name == "ground":
            continue
        if object_color_fn is not None:
            color = object_color_fn(color)
        quaternions, translations, bbox = _object_pose_track(value, len(motion))
        prepared[name] = (quaternions, translations, bbox)
        mesh_path, source_metadata = resolve_object_mesh(name, value, object_metadata, asset_roots)
        vertices, faces, normals, box_edges, mesh_extent = _load_mesh(mesh_path)
        recording.log(
            f"{prefix}/objects/{name}/mesh",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_normals=normals,
                albedo_factor=color,
            ),
            static=True,
        )
        recording.log(
            f"{prefix}/objects/{name}/bbox",
            rr.LineStrips3D(box_edges, colors=[color], radii=0.008),
            static=True,
        )
        if source_metadata.get("bbox_source") == "legacy_unit_placeholder":
            prepared[name] = (quaternions, translations, np.broadcast_to(mesh_extent, bbox.shape))
        recording.log(
            f"{prefix}/trajectories/{name}",
            rr.LineStrips3D([translations[frames]], colors=[color], radii=0.009),
            static=True,
        )
    return prepared


def _log_object_poses(
    recording: rr.RecordingStream,
    prefix: str,
    prepared: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    source_frame: int,
) -> None:
    for name, (quaternions, translations, bbox) in prepared.items():
        quaternion = quaternions[source_frame]
        recording.log(
            f"{prefix}/objects/{name}",
            rr.Transform3D(
                translation=translations[source_frame],
                quaternion=rr.Quaternion(xyzw=quaternion[[1, 2, 3, 0]]),
                scale=bbox[source_frame],
            ),
        )


def _record_compare(
    *,
    rollout: Mapping[str, Any],
    gt_motion: np.ndarray,
    pred_motion: np.ndarray,
    gt_vertices: np.ndarray,
    pred_vertices: np.ndarray,
    frames: np.ndarray,
    human_faces: np.ndarray,
    output_path: Path,
    source_fps: float,
    record_fps: float,
    asset_roots: Mapping[str, Path],
) -> None:
    """Write one recording with GT (orange) and prediction (indigo) overlaid.

    GT lives under ``world/gt`` and the prediction under ``world/pred`` so either
    side can be toggled in the viewer's entity tree. Ground and virtual IMUs are
    shared input/reference geometry and are logged once.
    """

    sample_id = normalized_sample_id(rollout["sample_idx"])
    metadata = rollout.get("metadata", {})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.rrd")
    temporary.unlink(missing_ok=True)
    recording = rr.RecordingStream(
        application_id="IMU4D full eval compare",
        recording_id=_recording_id(rollout, sample_id, "compare"),
    )
    recording.save(temporary, default_blueprint=_blueprint(record_fps))
    recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    gt_sample = rollout["gt"]
    pred_sample = rollout["pred"]
    gt_description = gt_sample.get("description", "")
    if isinstance(gt_description, (list, tuple)):
        gt_description = gt_description[0] if gt_description else ""
    pred_description = pred_sample.get("description", "")
    if isinstance(pred_description, (list, tuple)):
        pred_description = pred_description[0] if pred_description else ""
    gt_object_names = [name for name in gt_sample.get("objects", {}) if name != "ground"]
    pred_object_names = [name for name in pred_sample.get("objects", {}) if name != "ground"]
    summary = (
        f"# {metadata.get('dataset', 'HOI')} — {sample_id}\n\n"
        f"**GT** = orange human / saturated objects (`world/gt`)  \n"
        f"**Pred** = indigo human / indigo-tinted objects (`world/pred`)\n\n"
        f"- checkpoint step: `{metadata.get('step')}`\n"
        f"- prediction IMU layout: `{metadata.get('layout')}`\n"
        f"- source FPS: `{source_fps:g}`\n"
        f"- MPJPE: `{metadata.get('mpjpe_mm')}` mm\n"
        f"- object id accuracy: `{metadata.get('object_id_accuracy')}`\n"
        f"- object rot / transl L1: `{metadata.get('object_rotation_l1')}` / `{metadata.get('object_translation_l1')}`\n"
        f"- active IMUs: `{', '.join(rollout['input'].get('active_imu_sensor_names', []))}`\n"
        f"- GT objects: `{', '.join(gt_object_names) or '(none)'}`\n"
        f"- Pred objects: `{', '.join(pred_object_names) or '(none)'}`\n\n"
        f"**GT text:** {gt_description or '(none)'}\n\n"
        f"**Pred text:** {pred_description or '(none)'}\n"
    )
    recording.log("description", rr.TextDocument(summary, media_type="text/markdown"), static=True)

    gt_objects = gt_sample.get("objects", {})
    if "ground" in gt_objects:
        _, ground_translations, _ = _object_pose_track(gt_objects["ground"], len(gt_motion))
        _log_ground(recording, float(ground_translations[0, 1]))

    gt_prepared = _log_role_scene(
        recording, role="gt", prefix="world/gt", rollout=rollout, motion=gt_motion,
        frames=frames, asset_roots=asset_roots, trajectory_color=GT_TRAJECTORY_COLOR,
    )
    pred_prepared = _log_role_scene(
        recording, role="pred", prefix="world/pred", rollout=rollout, motion=pred_motion,
        frames=frames, asset_roots=asset_roots, trajectory_color=PRED_TRAJECTORY_COLOR,
        object_color_fn=pred_object_color,
    )

    imu_positions_value = rollout["input"].get("imu_positions")
    imu_positions = None if imu_positions_value is None else _array(imu_positions_value)
    active_names = rollout["input"].get("active_imu_sensor_names", [])
    active_ids = [
        index
        for index in range(imu_positions.shape[1] if imu_positions is not None else 0)
        if index not in rollout["input"].get("invalid_imu_id_list", [])
    ]
    for logged_index, source_frame in enumerate(frames):
        recording.set_time("time", duration=float(source_frame) / source_fps)
        for prefix, vertices, color in (
            ("world/gt", gt_vertices[logged_index], HUMAN_COLOR),
            ("world/pred", pred_vertices[logged_index], PRED_HUMAN_COLOR),
        ):
            recording.log(
                f"{prefix}/human",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=human_faces,
                    vertex_normals=_vertex_normals(vertices, human_faces),
                    albedo_factor=color,
                ),
            )
        if imu_positions is not None and source_frame < len(imu_positions):
            recording.log(
                "world/virtual_imus",
                rr.Points3D(
                    positions=imu_positions[source_frame, active_ids],
                    colors=[IMU_COLOR],
                    radii=0.025,
                    labels=active_names,
                    show_labels=False,
                ),
            )
        _log_object_poses(recording, "world/gt", gt_prepared, source_frame)
        _log_object_poses(recording, "world/pred", pred_prepared, source_frame)
    recording.flush()
    recording.disconnect()
    os.replace(temporary, output_path)


def export_eval_pair(
    rollout: Mapping[str, Any],
    output_dir: Path | str,
    *,
    smplx_model: Path | str,
    asset_roots: Mapping[str, Path | str],
    record_fps: float = 10.0,
    max_seconds: float = 20.0,
    device: str = "cpu",
    chunk_size: int = 64,
) -> dict[str, Any]:
    """Export ``gt.rrd``, ``pred.rrd`` and the overlaid ``compare.rrd`` for one
    eval rollout. Existing predictions are refreshed when pose filtering changes."""

    output_dir = Path(output_dir)
    metadata_path = output_dir / "metadata.json"
    # Raw 30 fps rollout (motion + per-frame object tracks) for offline
    # studies such as trajectory filtering; the .rrd files are subsampled.
    dump_rollout(rollout, output_dir / ROLLOUT_FILE)
    cached_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file() else {}
    )
    refresh_pose = cached_metadata.get("pose_filter") != rollout.get("metadata", {}).get("pose_filter")
    if (
        (output_dir / "gt.rrd").is_file()
        and (output_dir / "pred.rrd").is_file()
        and (output_dir / COMPARE_FILE).is_file()
        and metadata_path.is_file()
        and not refresh_pose
    ):
        (output_dir / "error.json").unlink(missing_ok=True)
        return cached_metadata
    roots = {name: Path(path).resolve() for name, path in asset_roots.items()}
    required_roots = {"omomo", "hiphi", "humoto"}
    if set(roots) != required_roots:
        raise ValueError(f"asset_roots must contain exactly {sorted(required_roots)}")
    sample_id = normalized_sample_id(rollout["sample_idx"])
    source_fps = float(rollout.get("metadata", {}).get("fps", 30.0))
    gt_motion = motion_to_smpl85(rollout["gt"])
    pred_motion = motion_to_smpl85(rollout["pred"])
    length = min(len(gt_motion), len(pred_motion))
    frames = select_frames(length, source_fps, record_fps, max_seconds)
    body = HumanMeshGenerator(Path(smplx_model).resolve(), torch.device(device), chunk_size)
    human_faces = body.faces
    gt_motion = gt_motion[:length]
    pred_motion = pred_motion[:length]
    vertices_by_role: dict[str, np.ndarray] = {}
    compare_path = output_dir / COMPARE_FILE
    for role, motion in (("gt", gt_motion), ("pred", pred_motion)):
        output_path = output_dir / f"{role}.rrd"
        if output_path.is_file() and compare_path.is_file() and not refresh_pose:
            continue
        vertices = body(motion, frames)
        vertices_by_role[role] = vertices
        if output_path.is_file() and not (refresh_pose and role == "pred"):
            continue
        _record_variant(
            role=role,
            rollout=rollout,
            motion=motion,
            human_vertices=vertices,
            frames=frames,
            human_faces=human_faces,
            output_path=output_path,
            source_fps=source_fps,
            record_fps=record_fps,
            asset_roots=roots,
        )
    if not compare_path.is_file() or refresh_pose:
        _record_compare(
            rollout=rollout,
            gt_motion=gt_motion,
            pred_motion=pred_motion,
            gt_vertices=vertices_by_role["gt"],
            pred_vertices=vertices_by_role["pred"],
            frames=frames,
            human_faces=human_faces,
            output_path=compare_path,
            source_fps=source_fps,
            record_fps=record_fps,
            asset_roots=roots,
        )
    result = {
        "status": "complete",
        "sample_id": sample_id,
        "motion_id": safe_motion_id(sample_id),
        "dataset": rollout.get("metadata", {}).get("dataset"),
        "step": rollout.get("metadata", {}).get("step"),
        "prediction_layout": rollout.get("metadata", {}).get("layout"),
        "eval_frames": rollout.get("metadata", {}).get("eval_frames"),
        "source_fps": source_fps,
        "record_fps": float(record_fps),
        "max_seconds": float(max_seconds),
        "recorded_frames": int(len(frames)),
        "mpjpe_mm": rollout.get("metadata", {}).get("mpjpe_mm"),
        "pose_filter": rollout.get("metadata", {}).get("pose_filter"),
        "object_id_accuracy": rollout.get("metadata", {}).get("object_id_accuracy"),
        "files": {"gt": "gt.rrd", "pred": "pred.rrd", "compare": COMPARE_FILE},
    }
    _atomic_json(metadata_path, result)
    (output_dir / "error.json").unlink(missing_ok=True)
    return result
