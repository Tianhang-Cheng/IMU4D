"""Create a headless Rerun recording for an OMOMO, HiPHI, or HuMOTO sample.

The script never opens a window or starts a server. It writes a portable
``.rrd`` file that can be downloaded and opened with the Rerun viewer locally.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import smplx
import torch
import trimesh

# Keep direct ``python visualize/viz_hoi_rrd.py`` invocation working.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_process.ground_plane import resolve_ground_plane
from dataset_process.humoto.humoto_io import world_imu_positions

HUMAN_COLOR = [247, 176, 67, 235]
IMU_COLOR = [56, 189, 248, 255]
OBJECT_COLORS = (
    [236, 72, 153, 230],
    [20, 184, 166, 230],
    [139, 92, 246, 230],
    [234, 88, 12, 230],
    [14, 165, 233, 230],
    [132, 204, 22, 230],
    [239, 68, 68, 230],
    [168, 85, 247, 230],
    [6, 182, 212, 230],
    [245, 158, 11, 230],
    [16, 185, 129, 230],
    [99, 102, 241, 230],
)
GROUND_COLOR = [100, 116, 139, 80]
BOX_EDGES = tuple(
    (left, right)
    for left in range(8)
    for right in range(left + 1, 8)
    if bin(left ^ right).count("1") == 1
)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute smooth per-vertex normals for one triangle mesh."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def _load_mesh(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Object mesh not found: {path}")
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    vertices = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.uint32)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if np.any(extent <= 0):
        raise ValueError(f"Degenerate object mesh: {path}")
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    # A unit local AABB lets each track's bbox extent act directly as the
    # Transform3D scale, independent of whether the source OBJ uses cm or m.
    normalized = (vertices - center) / extent
    normalized = normalized.astype(np.float32)

    # Build a tight display OBB in the original mesh frame, then express its
    # edges in the same normalized local coordinates as the rendered mesh.
    # Line strips preserve the box exactly even under the parent's non-uniform
    # scale; a nested Boxes3D primitive could be sheared by that transform.
    to_obb, obb_extent = trimesh.bounds.oriented_bounds(loaded)
    from_obb = np.linalg.inv(to_obb)
    signs = np.asarray(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=np.float32,
    )
    obb_corners = signs * np.asarray(obb_extent, dtype=np.float32) / 2
    obb_corners = (
        obb_corners @ from_obb[:3, :3].T + from_obb[:3, 3]
    )
    normalized_corners = (obb_corners - center) / extent
    box_edges = np.asarray(
        [normalized_corners[[left, right]] for left, right in BOX_EDGES],
        dtype=np.float32,
    )
    return normalized, faces, _vertex_normals(normalized, faces), box_edges, extent


def _object_mesh_path(
    sample: dict,
    name: str,
    omomo_mesh_root: Path,
    hiphi_root: Path,
    humoto_mesh_root: Path,
) -> Path:
    metadata = sample["object_metadata"][name]
    if "mesh_file" in metadata:
        return omomo_mesh_root / metadata["mesh_file"]
    if "mesh_path" in metadata:
        return hiphi_root / metadata["mesh_path"]
    if str(metadata.get("asset_id", "")).startswith("humoto/"):
        source_name = metadata.get("source_category", name)
        return humoto_mesh_root / source_name / f"{source_name}.obj"
    raise ValueError(f"No mesh reference for object {name!r}")


def _sample_source(sample: dict) -> str:
    source = sample.get("source")
    if source == "omomo":
        return "OMOMO"
    if isinstance(source, dict) or "frame_lu" in sample:
        return "HiPHI"
    if source == "humoto":
        return "HuMOTO"
    return str(source or "unknown")


def _ensure_ground(sample: dict) -> bool:
    """Use the loader's stored-frame floor estimate when a release omits it."""

    if "ground" in sample["objects"]:
        ground_metadata = sample.get("object_metadata", {}).get("ground", {})
        return bool(ground_metadata.get("synthetic"))
    frame_count = len(sample["motion_data_smpl85"])
    source = sample.get("source", "unknown")
    source = source if isinstance(source, str) else "hiphi"
    ground = resolve_ground_plane(sample, sample["motion_data_smpl85"], source, float(sample.get("fps", 30)))
    pose = np.asarray(
        [1.0, 0.0, 0.0, 0.0, 0.0, ground["height_y"], 0.0, 1.0, 1.0, 1.0],
        dtype=np.float32,
    )
    sample["objects"]["ground"] = np.broadcast_to(
        pose, (frame_count, len(pose))
    ).copy()
    sample.setdefault("object_metadata", {})["ground"] = {
        "category": "ground",
        "source_category": "ground",
        "object_id": "ground",
        "asset_id": "synthetic/ground",
        "pose_layout": "wxyz_xyz_bbox_xyz",
        "bbox_source": "visualization_synthetic_plane",
        "temporal_mode": "static",
        "synthetic": True,
    }
    sample.setdefault("object_valid_mask", {})["ground"] = np.ones(
        frame_count, dtype=bool
    )
    sample.setdefault("object_motion_mask", {})["ground"] = np.zeros(
        frame_count, dtype=bool
    )
    sample["ground_plane"] = ground
    sample.setdefault("object_anchor_valid", {})["ground"] = ground["valid"]
    return True


def _align_humoto_objects_to_rendered_human(sample: dict) -> float:
    """Upgrade stale HuMOTO samples that baked REST_PELVIS into world geometry."""

    if sample.get("source") != "humoto":
        return 0.0
    existing_offset = sample.get("visualization_object_y_offset_m")
    if existing_offset is not None:
        return float(existing_offset)
    if (sample.get("world_geometry_rest_pelvis_correction_applied") is False
            or (sample.get("world_geometry_rest_pelvis_correction_applied") is None
                and sample.get("legacy_rest_pelvis_correction") is None)):
        sample["visualization_object_y_offset_m"] = 0.0
        return 0.0
    correction = np.asarray(
        sample.get("legacy_rest_pelvis_correction"), dtype=np.float32
    )
    if correction.shape != (3,):
        raise ValueError(
            "HuMOTO sample lacks its [3] legacy_rest_pelvis_correction"
        )
    y_offset = -float(correction[1])
    for name, track_value in sample["objects"].items():
        track = np.asarray(track_value, dtype=np.float32).copy()
        if track.ndim != 2 or track.shape[1] < 7:
            raise ValueError(f"Object {name!r} has invalid track shape {track.shape}")
        track[:, 4:7] -= correction
        sample["objects"][name] = track
    sample["visualization_object_y_offset_m"] = y_offset
    return y_offset


def _log_ground(recording: rr.RecordingStream, height: float) -> None:
    """Render a finite floor slab and reference grid for the ground object."""

    radius = 3.0
    recording.log(
        "world/objects/ground",
        rr.Boxes3D(
            centers=[[0.0, height - 0.01, 0.0]],
            half_sizes=[[radius, 0.01, radius]],
            colors=[GROUND_COLOR],
            labels=["ground"],
        ),
        static=True,
    )
    coordinates = np.arange(-radius, radius + 0.001, 0.5, dtype=np.float32)
    lines = []
    for coordinate in coordinates:
        lines.append([[-radius, height, coordinate], [radius, height, coordinate]])
        lines.append([[coordinate, height, -radius], [coordinate, height, radius]])
    recording.log(
        "world/objects/ground/grid",
        rr.LineStrips3D(lines, colors=[[71, 85, 105, 120]], radii=0.003),
        static=True,
    )


def _select_frames(
    sample: dict,
    record_fps: float,
    max_seconds: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    source_fps = float(sample.get("fps", 30.0))
    length = len(sample["motion_data_smpl85"])
    clip_frames = min(length, max(1, round(max_seconds * source_fps)))
    if clip_frames < length:
        moving = np.zeros(length, dtype=bool)
        for mask in sample.get("object_motion_mask", {}).values():
            moving |= np.asarray(mask, dtype=bool)
        moving_indices = np.flatnonzero(moving)
        center = int(np.median(moving_indices)) if len(moving_indices) else length // 2
        start = max(0, min(center - clip_frames // 2, length - clip_frames))
    else:
        start = 0
    end = start + clip_frames
    stride = max(1, round(source_fps / record_fps))
    frames = np.arange(start, end, stride, dtype=np.int32)
    if frames[-1] != end - 1:
        frames = np.r_[frames, end - 1].astype(np.int32)
    return frames, (start, end)


class HumanMeshGenerator:
    def __init__(self, model_path: Path, device: torch.device, chunk_size: int):
        self.device = device
        self.chunk_size = chunk_size
        self.model = smplx.SMPLX(
            str(model_path),
            gender="neutral",
            ext=model_path.suffix.lstrip("."),
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
        ).to(device)
        self.model.eval()
        self.faces = np.asarray(self.model.faces, dtype=np.uint32)

    @torch.no_grad()
    def __call__(self, motion: np.ndarray, frames: np.ndarray) -> np.ndarray:
        selected = np.asarray(motion[frames], dtype=np.float32)
        meshes: list[np.ndarray] = []
        for start in range(0, len(selected), self.chunk_size):
            batch_motion = selected[start : start + self.chunk_size]
            batch = len(batch_motion)
            tensor = torch.from_numpy(batch_motion).to(self.device)
            zeros3 = torch.zeros((batch, 3), device=self.device)
            zeros45 = torch.zeros((batch, 45), device=self.device)
            expression = torch.zeros(
                (batch, self.model.num_expression_coeffs), device=self.device
            )
            output = self.model(
                global_orient=tensor[:, :3],
                body_pose=tensor[:, 3:66],
                betas=tensor[:, 75:85],
                transl=zeros3,
                left_hand_pose=zeros45,
                right_hand_pose=zeros45,
                jaw_pose=zeros3,
                leye_pose=zeros3,
                reye_pose=zeros3,
                expression=expression,
                return_verts=True,
            )
            pelvis_delta = tensor[:, 72:75] - output.joints[:, 0]
            vertices = output.vertices + pelvis_delta[:, None]
            meshes.append(vertices.float().cpu().numpy())
        return np.concatenate(meshes, axis=0).astype(np.float32)


def _blueprint(record_fps: float) -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                name="Human + dynamic object",
                background=[248, 250, 252],
                # The viewer's built-in grid sits on the y=0 plane, i.e. at the
                # first-frame pelvis (the eval frame puts the pelvis at the origin),
                # which reads as a floor through the hips/chest. The real floor is
                # the ground slab + grid logged by _log_ground at the ground height.
                line_grid=rrb.archetypes.LineGrid3D(visible=False),
            ),
            rrb.TextDocumentView(origin="description", name="Sample metadata"),
            column_shares=[4, 1],
        ),
        rrb.TimePanel(timeline="time", fps=record_fps, playback_speed=1.0),
        collapse_panels=False,
    )


def record_sample(
    sample_path: Path,
    output_path: Path,
    smplx_model: Path,
    omomo_mesh_root: Path,
    hiphi_root: Path,
    humoto_mesh_root: Path,
    device: torch.device,
    record_fps: float,
    max_seconds: float,
    chunk_size: int,
) -> dict[str, object]:
    with sample_path.open("rb") as handle:
        sample = pickle.load(handle)
    if not sample.get("train_ready", False):
        raise ValueError(f"Sample is not train-ready: {sample_path}")
    synthetic_ground = _ensure_ground(sample)
    humoto_y_offset = _align_humoto_objects_to_rendered_human(sample)
    motion = np.asarray(sample["motion_data_smpl85"], dtype=np.float32)
    imu = np.asarray(sample["imu_traj"], dtype=np.float32)
    if sample.get("source") == "humoto":
        imu = imu.copy()
        imu[:, :, 3:6] = world_imu_positions(imu[:, :, 3:6], sample)
    frames, (clip_start, clip_end) = _select_frames(sample, record_fps, max_seconds)

    body = HumanMeshGenerator(smplx_model, device, chunk_size)
    human_vertices = body(motion, frames)
    source = _sample_source(sample)
    description = sample.get("description", [])
    description_text = description[0] if description else "(no released text annotation)"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream(
        application_id=f"IMU4D {source} HOI",
        recording_id=f"{source.lower()}_{sample['motion_id']}",
    )
    recording.save(output_path, default_blueprint=_blueprint(record_fps))
    recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    metadata_text = (
        f"# {source}: {sample['motion_id']}\n\n"
        f"- actor: `{sample['actor_id']}`\n"
        f"- source FPS: `{sample['fps']}`\n"
        f"- recorded source frames: `[{clip_start}, {clip_end})`\n"
        f"- Rerun frames: `{len(frames)}` at about `{record_fps:g} FPS`\n"
        f"- objects: `{', '.join(sample['objects'])}`\n"
        f"- ground: `{'synthetic plane' if synthetic_ground else 'dataset track'}`\n\n"
        f"- HuMOTO visualization Y correction: `{humoto_y_offset:g} m`\n\n"
        f"**Description:** {description_text}\n"
    )
    recording.log("description", rr.TextDocument(metadata_text, media_type="text/markdown"), static=True)

    # Static paths make it easy to read global motion while scrubbing.
    recording.log(
        "world/trajectories/pelvis",
        rr.LineStrips3D(
            [motion[frames, 72:75]], colors=[[245, 158, 11, 180]], radii=0.007
        ),
        static=True,
    )
    display_extents: dict[str, np.ndarray | None] = {}
    for color, (name, track_value) in zip(
        cycle(OBJECT_COLORS), sample["objects"].items()
    ):
        if name == "ground":
            height = float(np.asarray(track_value, dtype=np.float32)[0, 5])
            _log_ground(recording, height)
            display_extents[name] = None
            continue
        mesh_path = _object_mesh_path(
            sample, name, omomo_mesh_root, hiphi_root, humoto_mesh_root
        )
        vertices, faces, normals, box_edges, mesh_extent = _load_mesh(mesh_path)
        metadata = sample["object_metadata"][name]
        display_extents[name] = (
            mesh_extent
            if metadata.get("bbox_source") == "legacy_unit_placeholder"
            else None
        )
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
            rr.LineStrips3D(
                box_edges,
                colors=[color],
                radii=0.008,
            ),
            static=True,
        )
        track = np.asarray(track_value, dtype=np.float32)
        recording.log(
            f"world/trajectories/{name}",
            rr.LineStrips3D(
                [track[frames, 4:7]], colors=[color], radii=0.009
            ),
            static=True,
        )

    for logged_index, source_frame in enumerate(frames):
        recording.set_time(
            "time", duration=float(source_frame - clip_start) / float(sample["fps"])
        )
        recording.log(
            "world/human",
            rr.Mesh3D(
                vertex_positions=human_vertices[logged_index],
                triangle_indices=body.faces,
                vertex_normals=_vertex_normals(
                    human_vertices[logged_index], body.faces
                ),
                albedo_factor=HUMAN_COLOR,
            ),
        )
        recording.log(
            "world/virtual_imus",
            rr.Points3D(
                positions=imu[source_frame, :, 3:6],
                colors=[IMU_COLOR],
                radii=0.025,
                labels=sample.get("imu_sensor_names"),
                show_labels=False,
            ),
        )
        for color, (name, track_value) in zip(
            cycle(OBJECT_COLORS), sample["objects"].items()
        ):
            if name == "ground":
                continue
            track = np.asarray(track_value, dtype=np.float32)[source_frame]
            quaternion_xyzw = track[[1, 2, 3, 0]]
            scale = display_extents[name]
            if scale is None:
                scale = track[7:10]
            recording.log(
                f"world/objects/{name}",
                rr.Transform3D(
                    translation=track[4:7],
                    quaternion=rr.Quaternion(xyzw=quaternion_xyzw),
                    scale=scale,
                ),
            )

    recording.flush()
    recording.disconnect()
    return {
        "source": source,
        "motion_id": sample["motion_id"],
        "source_frames": len(motion),
        "clip": [clip_start, clip_end],
        "logged_frames": len(frames),
        "record_fps": record_fps,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .rrd (default: <dataset>/rerun/<sample-name>.rrd).",
    )
    parser.add_argument(
        "--smplx-model",
        type=Path,
        default=Path("data/models/smplx/SMPLX_NEUTRAL.npz"),
    )
    parser.add_argument(
        "--omomo-mesh-root",
        type=Path,
        default=Path("data/raw/omomo/release_v1/data/captured_objects"),
    )
    parser.add_argument(
        "--hiphi-root",
        type=Path,
        default=Path("data/raw/hiphi/release_v1"),
    )
    parser.add_argument(
        "--humoto-mesh-root",
        type=Path,
        default=Path("data/raw/humoto/v1/humoto_objects_0805"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--record-fps", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    if args.record_fps <= 0 or args.max_seconds <= 0:
        raise ValueError("--record-fps and --max-seconds must be positive")
    sample_path = args.sample.resolve()
    output_path = args.output
    if output_path is None:
        # Expected layout: <dataset>/samples/<split>/<sample-name>.pkl
        if sample_path.parent.parent.name != "samples":
            raise ValueError("--output is required outside a samples/<split> directory")
        output_path = sample_path.parents[2] / "rerun" / f"{sample_path.stem}.rrd"
    result = record_sample(
        sample_path,
        output_path.resolve(),
        args.smplx_model.resolve(),
        args.omomo_mesh_root.resolve(),
        args.hiphi_root.resolve(),
        args.humoto_mesh_root.resolve(),
        torch.device(args.device),
        args.record_fps,
        args.max_seconds,
        args.chunk_size,
    )
    import json

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
