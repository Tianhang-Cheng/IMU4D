"""Score the object/scene side of a full evaluation from its raw records.

Input is what ``run.py`` dumps next to every full-eval summary
(``records-*.jsonl`` + ``tracks/*.npz``), so this never re-runs the model and
every threshold stays changeable after the fact.

What it reports, per ``frames-NNN/datasets/<source>/metrics/<layout>`` pass:

* Category identification: precision / recall / F1, micro over all objects and
  macro over categories, plus a per-category breakdown. Objects are matched
  one-to-one within a category by maximum IoU (``metric.scene.humoto``), so a
  match is by construction a correct category.
* Asset identification: the same P/R/F1 with the match constrained to equal
  ``asset_id`` (e.g. ``humoto/dining_chair``, not just ``chair``), plus the
  asset accuracy among category-matched pairs.
* Geometry: mean oriented 3D IoU over matched pairs and P/R/F1 at every
  requested IoU threshold. ``3D_IoU_pct`` is conditional on a match, so it
  ignores missed targets and extra (e.g. duplicate) predictions; a method that
  sprays several boxes per category gets a best-of-k pick and scores high on
  few matches. Two unconditional variants sit next to it: ``3D_IoU_gt_pct``
  (sum of matched IoU / #GT, a miss counts as 0) and ``PQ_IoU_pct``
  (2 * sum IoU / (#pred + #GT), panoptic-quality style, which also penalises
  unmatched predictions).

The ground plane is excluded from every count above.
* Dynamic objects (OMOMO and HiPHI only): dynamic-track recall, moving/static
  ADD-S AUC up to 10% of object diameter, and moving mesh-velocity error.

Bounding boxes come from ``catalog_extent_m`` (mesh-derived, from
``dataset_process/object_identity_catalog.json``) rather than the extent stored
in the sample: HUMOTO ships a unit placeholder there. Both are in the record, so
``--extent-source sample`` can reproduce the old behaviour.

Usage::

    python -m evaluation.score_scene_metrics exp/showo_pretrain_full/evaluation/full/step-054500
    python -m evaluation.score_scene_metrics exp/<run>/evaluation/full/step-N \\
        --iou-thresholds 0.25 0.5 0.75
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_process.asset_frames import canonical_extent_m, canonical_mesh_path  # noqa: E402
from dataset_process.identity_catalog import IdentityCatalog  # noqa: E402
from dataset_process.object_taxonomy import is_static_ground  # noqa: E402
from metric.core.rotation import rotation_matrix  # noqa: E402
from metric.scene.hiphi import add_s_per_frame, mesh_velocity_error_per_frame  # noqa: E402
from metric.scene.humoto import match_objects  # noqa: E402

SCHEMA_VERSION = 3  # 3: asset-level identification
DEFAULT_IOU_THRESHOLDS = (0.25, 0.5, 0.75)
# These fallback thresholds match the OMOMO and HiPHI converters. New full-eval
# artifacts carry the exact shard masks, so the fallback is only for old tracks.
DEFAULT_MOVING_SPEED_MPS = 0.03
DEFAULT_MOVING_ANGULAR_DPS = 10.0
DEFAULT_MOVING_FILTER = 5
DEFAULT_MOVING_MIN_RUN = 1
DEFAULT_MESH_POINTS = 512
DEFAULT_AUC_DIAMETER_FRACTION = 0.1
DYNAMIC_DATASETS = frozenset({"omomo", "hiphi"})

# Raw asset roots, for the ADD-S mesh of assets without a canonical mesh.
MESH_ROOTS = {
    "humoto": Path("data/raw/humoto/v1"),
    "hiphi": Path("data/raw/hiphi/release_v1"),
    "omomo": Path("data/raw/omomo/release_v1/data"),
}


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall)


def _object_box(entry: Mapping[str, Any], extent_source: str) -> dict[str, Any] | None:
    """One first-frame oriented box, or None when no usable extent exists."""

    extent = entry.get("catalog_extent_m") if extent_source == "catalog" else None
    if extent is None:
        extent = entry.get("sample_extent_m")
    if extent is None:
        return None
    extent = [float(value) for value in extent]
    if len(extent) != 3 or min(extent) <= 0:
        return None
    return {
        "category": str(entry["category"]),
        "asset_id": entry.get("asset_id"),
        "rot": list(entry["rot6d_first"]),
        "transl": list(entry["transl_first"]),
        "size": extent,
        "rotation_representation": "6d",
    }


def _scene_objects(
    entries: Mapping[str, Mapping[str, Any]], extent_source: str
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Boxes, their instance names, and the count dropped for a missing extent."""

    boxes: list[dict[str, Any]] = []
    names: list[str] = []
    dropped = 0
    for name, entry in entries.items():
        if is_static_ground(str(entry.get("category", name))):
            continue
        box = _object_box(entry, extent_source)
        if box is None:
            dropped += 1
            continue
        boxes.append(box)
        names.append(name)
    return boxes, names, dropped


def _moving_mask(
    rotations: np.ndarray,
    translations: np.ndarray,
    fps: float,
    speed_threshold: float,
    angular_threshold_dps: float,
    filter_frames: int,
    min_run: int,
    filter_pad_mode: str = "edge",
) -> np.ndarray:
    """Per-frame moving flag, mirroring the trunk's GT motion-state labels."""

    frames = len(translations)
    if frames < 2:
        return np.zeros(frames, dtype=bool)
    linear = np.linalg.norm(np.diff(translations, axis=0), axis=-1) * fps
    relative = np.einsum("tij,tkj->tik", rotations[1:], rotations[:-1])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angular = np.degrees(np.arccos(cosine)) * fps
    moving = np.zeros(frames, dtype=bool)
    moving[1:] = (linear > speed_threshold) | (angular > angular_threshold_dps)

    if filter_frames > 1:
        padded = np.pad(
            moving.astype(np.float64), filter_frames // 2, mode=filter_pad_mode
        )
        window = np.ones(filter_frames) / filter_frames
        moving = np.convolve(padded, window, mode="valid")[:frames] > 0.5

    if min_run > 1:  # flip runs too short to be real motion
        start = 0
        for index in range(1, frames + 1):
            if index == frames or moving[index] != moving[start]:
                if index - start < min_run:
                    moving[start:index] = not moving[start]
                start = index
    return moving


class MeshPointCache:
    """Sampled surface points per asset, for ADD-S."""

    def __init__(self, catalog: IdentityCatalog, points: int, repo_root: Path) -> None:
        self._catalog = catalog
        self._points = points
        self._repo_root = repo_root
        self._cache: dict[str, np.ndarray | None] = {}
        self._diameters: dict[str, float | None] = {}

    def _mesh_file(self, asset_id: str) -> Path | None:
        canonical = canonical_mesh_path(asset_id)
        if canonical is not None and canonical.is_file():
            return canonical
        index = self._catalog.asset_id_to_index.get(asset_id)
        if index is None:
            return None
        asset = self._catalog.assets[index]
        mesh_path = asset.get("mesh_path")
        root = MESH_ROOTS.get(str(asset.get("source", "")))
        if mesh_path is None or root is None:
            return None
        candidate = self._repo_root / root / str(mesh_path)
        return candidate if candidate.is_file() else None

    def get(self, asset_id: str | None) -> np.ndarray | None:
        if asset_id is None:
            return None
        if asset_id not in self._cache:
            self._cache[asset_id] = self._load(asset_id)
        return self._cache[asset_id]

    def diameter(self, asset_id: str | None) -> float | None:
        """Approximate canonical mesh diameter from the deterministic samples."""

        if asset_id is None:
            return None
        if asset_id not in self._diameters:
            points = self.get(asset_id)
            if points is None:
                self._diameters[asset_id] = None
            else:
                maximum = 0.0
                for start in range(0, len(points), 128):
                    distances = np.linalg.norm(
                        points[start : start + 128, None] - points[None], axis=-1
                    )
                    maximum = max(maximum, float(distances.max(initial=0.0)))
                self._diameters[asset_id] = maximum if maximum > 0 else None
        return self._diameters[asset_id]

    def _load(self, asset_id: str) -> np.ndarray | None:
        """Surface points in metres, centred the way the tracks assume.

        Source meshes are not in a common unit -- HiPHI ships centimetres and
        every OMOMO asset has its own arbitrary scale -- and the box convention
        is ``extent centered at track translation``. Both are fixed by mapping
        the mesh AABB onto the catalog's canonical extent in metres.
        """

        mesh_file = self._mesh_file(asset_id)
        index = self._catalog.asset_id_to_index.get(asset_id)
        if mesh_file is None or index is None:
            return None
        extent = canonical_extent_m(self._catalog.assets[index])
        if extent is None:
            return None
        try:
            import trimesh

            mesh = trimesh.load(mesh_file, force="mesh", process=False)
            seed = int.from_bytes(
                hashlib.sha256(asset_id.encode("utf-8")).digest()[:8], "little"
            )
            samples, _ = trimesh.sample.sample_surface(mesh, self._points, seed=seed)
            samples = np.asarray(samples, dtype=np.float64)
        except Exception:  # a missing or unreadable mesh only costs ADD-S
            return None
        lower, upper = samples.min(axis=0), samples.max(axis=0)
        size = np.maximum(upper - lower, 1e-9)
        return (samples - (lower + upper) / 2.0) * (np.asarray(extent, dtype=np.float64) / size)


def _load_records(pass_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(pass_dir.glob("records-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def track_file_name(sample_id: object) -> str:
    """Stem run.py gives a sample's ``tracks/<id>.npz`` file."""

    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(sample_id)
    )


def _track(arrays: Mapping[str, np.ndarray], side: str, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    rot_key, transl_key = f"{side}|{name}|rot", f"{side}|{name}|transl"
    if rot_key not in arrays or transl_key not in arrays:
        return None
    rotations = rotation_matrix(np.asarray(arrays[rot_key], dtype=np.float64), "6d")
    translations = np.asarray(arrays[transl_key], dtype=np.float64).reshape(-1, 3)
    if len(rotations) != len(translations) or len(rotations) < 2:
        return None
    return rotations, translations


def _track_masks(
    arrays: Mapping[str, np.ndarray],
    name: str,
    frames: int,
    rotations: np.ndarray,
    translations: np.ndarray,
    fps: float,
    moving_speed: float,
    moving_angular: float,
    moving_filter: int,
    moving_min_run: int,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return annotation-valid and GT-motion masks for one saved GT track."""

    valid_key = f"gt|{name}|valid_mask"
    motion_key = f"gt|{name}|motion_mask"
    valid = (
        np.asarray(arrays[valid_key], dtype=bool).reshape(-1)[:frames]
        if valid_key in arrays
        else np.ones(frames, dtype=bool)
    )
    if len(valid) != frames:
        return np.zeros(frames, dtype=bool), np.zeros(frames, dtype=bool), "invalid"
    if motion_key in arrays:
        moving = np.asarray(arrays[motion_key], dtype=bool).reshape(-1)[:frames]
        if len(moving) != frames:
            return np.zeros(frames, dtype=bool), np.zeros(frames, dtype=bool), "invalid"
        source = "shard"
    else:
        moving = _moving_mask(
            rotations[:frames],
            translations[:frames],
            fps,
            moving_speed,
            moving_angular,
            moving_filter,
            moving_min_run,
            "constant" if dataset == "hiphi" else "edge",
        )
        source = "recomputed"
    return valid, moving & valid, source


def _add_s_auc_pct(
    errors_m: np.ndarray, diameter_m: float, max_diameter_fraction: float
) -> float:
    """Normalized area under the ADD-S success curve from 0 to fraction*d."""

    threshold = diameter_m * max_diameter_fraction
    if threshold <= 0:
        raise ValueError("ADD-S AUC threshold must be positive")
    errors = np.asarray(errors_m, dtype=np.float64)
    if errors.size == 0:
        raise ValueError("ADD-S AUC needs at least one frame")
    return 100.0 * float(np.maximum(0.0, 1.0 - errors / threshold).mean())


def _mesh_velocity_error_mm_s(
    pred_rotations: np.ndarray,
    pred_translations: np.ndarray,
    gt_rotations: np.ndarray,
    gt_translations: np.ndarray,
    mesh_points: np.ndarray,
    moving: np.ndarray,
    fps: float,
) -> float | None:
    valid_pairs = moving[:-1] & moving[1:]
    if not np.any(valid_pairs):
        return None
    discrepancy = mesh_velocity_error_per_frame(
        pred_rotations,
        pred_translations,
        gt_rotations,
        gt_translations,
        mesh_points,
        fps,
    )
    return float(discrepancy[valid_pairs].mean() * 1000.0)


def score_pass(
    pass_dir: Path,
    iou_thresholds: Sequence[float],
    extent_source: str,
    moving_speed: float,
    moving_angular: float,
    moving_filter: int,
    moving_min_run: int,
    fps: float,
    mesh_points: MeshPointCache | None,
    auc_diameter_fraction: float = DEFAULT_AUC_DIAMETER_FRACTION,
) -> dict[str, Any] | None:
    records = _load_records(pass_dir)
    if not records:
        return None

    total_pred = total_gt = 0
    matched_ious: list[float] = []
    per_category: dict[str, dict[str, float]] = defaultdict(
        lambda: {"tp": 0, "pred": 0, "gt": 0}
    )
    geometry_tp = {threshold: 0 for threshold in iou_thresholds}
    asset_tp = 0
    asset_correct_given_category = 0
    dropped_pred = dropped_gt = 0
    dataset = pass_dir.parents[1].name.lower()
    dynamic_enabled = mesh_points is not None and dataset in DYNAMIC_DATASETS
    dynamic_records: list[dict[str, Any]] = []
    dynamic_target_tracks = 0
    dynamic_matched_tracks = 0
    non_dynamic_target_tracks = 0
    unavailable_target_tracks = 0
    mask_sources: dict[str, int] = defaultdict(int)
    tracks_dir = pass_dir / "tracks"

    for record in records:
        objects = record.get("objects", {})
        pred_boxes, pred_names, pred_dropped = _scene_objects(
            objects.get("pred", {}), extent_source
        )
        gt_boxes, gt_names, gt_dropped = _scene_objects(
            objects.get("gt", {}), extent_source
        )
        dropped_pred += pred_dropped
        dropped_gt += gt_dropped
        total_pred += len(pred_boxes)
        total_gt += len(gt_boxes)
        for box in pred_boxes:
            per_category[box["category"]]["pred"] += 1
        for box in gt_boxes:
            per_category[box["category"]]["gt"] += 1
        matches = (
            match_objects(pred_boxes, gt_boxes, "6d")
            if pred_boxes and gt_boxes
            else []
        )
        # Asset level: the same matcher keyed on asset_id. An object without an
        # asset_id can never match at this level.
        def _by_asset(boxes, side):
            return [
                {**box, "category": box["asset_id"] or f"__no_asset_{side}_{i}"}
                for i, box in enumerate(boxes)
            ]
        if pred_boxes and gt_boxes:
            asset_tp += len(match_objects(_by_asset(pred_boxes, "pred"), _by_asset(gt_boxes, "gt"), "6d"))
        for pred_index, gt_index, _ in matches:
            pred_asset = pred_boxes[pred_index]["asset_id"]
            asset_correct_given_category += int(
                pred_asset is not None and pred_asset == gt_boxes[gt_index]["asset_id"]
            )
        for pred_index, gt_index, iou in matches:
            matched_ious.append(iou)
            per_category[gt_boxes[gt_index]["category"]]["tp"] += 1
            for threshold in iou_thresholds:
                if iou >= threshold:
                    geometry_tp[threshold] += 1

        if not dynamic_enabled:
            continue
        track_file = tracks_dir / (track_file_name(record["sample_id"]) + ".npz")
        if not track_file.is_file():
            unavailable_target_tracks += len(gt_names)
            continue
        with np.load(track_file) as arrays:
            arrays = {key: arrays[key] for key in arrays.files}
        gt_dynamic: dict[int, dict[str, Any]] = {}
        for gt_index, gt_name in enumerate(gt_names):
            gt_track = _track(arrays, "gt", gt_name)
            if gt_track is None:
                unavailable_target_tracks += 1
                continue
            gt_rotations, gt_translations = gt_track
            frames = len(gt_rotations)
            valid, moving, mask_source = _track_masks(
                arrays,
                gt_name,
                frames,
                gt_rotations,
                gt_translations,
                fps,
                moving_speed,
                moving_angular,
                moving_filter,
                moving_min_run,
                dataset,
            )
            mask_sources[mask_source] += 1
            if mask_source == "invalid":
                unavailable_target_tracks += 1
                continue
            if not moving.any():
                non_dynamic_target_tracks += 1
                continue
            dynamic_target_tracks += 1
            gt_dynamic[gt_index] = {
                "track": gt_track,
                "valid": valid,
                "moving": moving,
            }

        for pred_index, gt_index, _ in matches:
            if gt_index not in gt_dynamic:
                continue
            dynamic_matched_tracks += 1
            gt_name, pred_name = gt_names[gt_index], pred_names[pred_index]
            pred_track = _track(arrays, "pred", pred_name)
            if pred_track is None:
                continue
            gt_rotations, gt_translations = gt_dynamic[gt_index]["track"]
            pred_rotations, pred_translations = pred_track
            frames = min(len(gt_rotations), len(pred_rotations))
            valid = gt_dynamic[gt_index]["valid"][:frames]
            moving = gt_dynamic[gt_index]["moving"][:frames]
            asset_id = objects.get("gt", {}).get(gt_name, {}).get("asset_id")
            dynamic_records.append(
                {
                    "id": f"{record['sample_id']}/{gt_name}",
                    "category": gt_boxes[gt_index]["category"],
                    "pred_rotations": pred_rotations[:frames],
                    "pred_translations": pred_translations[:frames],
                    "gt_rotations": gt_rotations[:frames],
                    "gt_translations": gt_translations[:frames],
                    "moving_mask": moving,
                    "static_mask": valid & ~moving,
                    "fps": fps,
                    "mesh_points": mesh_points.get(asset_id),
                    "diameter_m": mesh_points.diameter(asset_id),
                }
            )

    if total_gt == 0 and total_pred == 0:
        # A source with no objects at all (LINGO carries only the excluded
        # ground plane). Reporting zeros here would read as "scored 0%".
        return {
            "schema_version": SCHEMA_VERSION,
            "task": "scene_objects",
            "protocol": {"extent_source": extent_source, "ground_excluded": True},
            "counts": {"samples": len(records), "predicted_objects": 0, "target_objects": 0},
            "summary": {},
            "dynamic": {},
            "per_category": {},
            "note": "no non-ground objects in this source; nothing to score",
        }

    precision = total_pred and sum(v["tp"] for v in per_category.values()) / total_pred
    recall = total_gt and sum(v["tp"] for v in per_category.values()) / total_gt
    summary: dict[str, Any] = {
        "category_precision_micro_pct": 100.0 * float(precision or 0.0),
        "category_recall_micro_pct": 100.0 * float(recall or 0.0),
        "category_f1_micro_pct": 100.0 * _f1(float(precision or 0.0), float(recall or 0.0)),
    }
    asset_precision = asset_tp / total_pred if total_pred else 0.0
    asset_recall = asset_tp / total_gt if total_gt else 0.0
    summary["asset_precision_micro_pct"] = 100.0 * asset_precision
    summary["asset_recall_micro_pct"] = 100.0 * asset_recall
    summary["asset_f1_micro_pct"] = 100.0 * _f1(asset_precision, asset_recall)
    category_tp = sum(v["tp"] for v in per_category.values())
    summary["asset_accuracy_given_category_pct"] = (
        100.0 * asset_correct_given_category / category_tp if category_tp else 0.0
    )

    category_rows: dict[str, dict[str, float]] = {}
    macro_precision, macro_recall, macro_f1 = [], [], []
    for category, counts in sorted(per_category.items()):
        category_precision = counts["tp"] / counts["pred"] if counts["pred"] else 0.0
        category_recall = counts["tp"] / counts["gt"] if counts["gt"] else 0.0
        category_f1 = _f1(category_precision, category_recall)
        category_rows[category] = {
            "precision_pct": 100.0 * category_precision,
            "recall_pct": 100.0 * category_recall,
            "f1_pct": 100.0 * category_f1,
            "predictions": counts["pred"],
            "targets": counts["gt"],
            "matched": counts["tp"],
        }
        if counts["gt"]:  # macro average over the categories the GT contains
            macro_precision.append(category_precision)
            macro_recall.append(category_recall)
            macro_f1.append(category_f1)
    summary["category_precision_macro_pct"] = 100.0 * float(np.mean(macro_precision or [0.0]))
    summary["category_recall_macro_pct"] = 100.0 * float(np.mean(macro_recall or [0.0]))
    summary["category_f1_macro_pct"] = 100.0 * float(np.mean(macro_f1 or [0.0]))
    summary["categories_in_gt"] = len(macro_f1)

    summary["3D_IoU_pct"] = 100.0 * float(np.mean(matched_ious)) if matched_ious else 0.0
    iou_sum = float(np.sum(matched_ious)) if matched_ious else 0.0
    summary["3D_IoU_gt_pct"] = 100.0 * iou_sum / total_gt if total_gt else 0.0
    summary["PQ_IoU_pct"] = 200.0 * iou_sum / (total_pred + total_gt)
    for threshold in iou_thresholds:
        geometry_precision = geometry_tp[threshold] / total_pred if total_pred else 0.0
        geometry_recall = geometry_tp[threshold] / total_gt if total_gt else 0.0
        key = f"{threshold:g}"
        summary[f"P@{key}_pct"] = 100.0 * geometry_precision
        summary[f"R@{key}_pct"] = 100.0 * geometry_recall
        summary[f"F1@{key}_pct"] = 100.0 * _f1(geometry_precision, geometry_recall)

    dynamic_summary: dict[str, Any] = {}
    if dynamic_enabled:
        moving_auc_rows: list[float] = []
        static_auc_rows: list[float] = []
        velocity_rows: list[float] = []
        tracks_with_mesh = 0
        for record in dynamic_records:
            points = record["mesh_points"]
            diameter = record["diameter_m"]
            if points is None or diameter is None:
                continue
            tracks_with_mesh += 1
            add_s = add_s_per_frame(
                record["pred_rotations"],
                record["pred_translations"],
                record["gt_rotations"],
                record["gt_translations"],
                points,
            )
            moving = record["moving_mask"]
            static = record["static_mask"]
            if np.any(moving):
                moving_auc_rows.append(
                    _add_s_auc_pct(add_s[moving], diameter, auc_diameter_fraction)
                )
            if np.any(static):
                static_auc_rows.append(
                    _add_s_auc_pct(add_s[static], diameter, auc_diameter_fraction)
                )
            velocity = _mesh_velocity_error_mm_s(
                record["pred_rotations"],
                record["pred_translations"],
                record["gt_rotations"],
                record["gt_translations"],
                points,
                moving,
                record["fps"],
            )
            if velocity is not None and np.isfinite(velocity):
                velocity_rows.append(velocity)

        dynamic_summary = {
            "target_tracks": dynamic_target_tracks,
            "matched_tracks": dynamic_matched_tracks,
            "pose_tracks": len(dynamic_records),
            "tracks_with_mesh": tracks_with_mesh,
            "tracks_without_mesh": len(dynamic_records) - tracks_with_mesh,
            "moving_auc_tracks": len(moving_auc_rows),
            "static_auc_tracks": len(static_auc_rows),
            "velocity_tracks": len(velocity_rows),
            "non_dynamic_target_tracks": non_dynamic_target_tracks,
            "unavailable_target_tracks": unavailable_target_tracks,
        }
        if dynamic_target_tracks:
            dynamic_summary["track_recall_pct"] = (
                100.0 * dynamic_matched_tracks / dynamic_target_tracks
            )
        if moving_auc_rows:
            dynamic_summary["ADD_S_AUC_0.1d_moving_pct"] = float(
                np.mean(moving_auc_rows)
            )
        if static_auc_rows:
            dynamic_summary["ADD_S_AUC_0.1d_static_pct"] = float(
                np.mean(static_auc_rows)
            )
        if velocity_rows:
            dynamic_summary["mesh_velocity_error_moving_mm_s"] = float(
                np.mean(velocity_rows)
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "task": "scene_objects",
        "protocol": {
            "matching": "one-to-one maximum-IoU within a category (metric.scene.humoto)",
            "extent_source": extent_source,
            "iou_thresholds": [float(value) for value in iou_thresholds],
            "moving_definition": {
                "speed_mps": moving_speed,
                "angular_dps": moving_angular,
                "majority_window_frames": moving_filter,
                "min_run_frames": moving_min_run,
                "fps": fps,
                "mask_sources": dict(sorted(mask_sources.items())),
            },
            "dynamic_evaluation": {
                "enabled": dynamic_enabled,
                "supported_datasets": sorted(DYNAMIC_DATASETS),
                "dataset": dataset,
                "auc_max_diameter_fraction": auc_diameter_fraction,
                "aggregation": "frame metric per track, then macro mean over tracks",
                "static_phase": "valid non-moving frames of GT tracks that move at least once",
                "unmatched_dynamic_targets": "counted by track recall; excluded from conditional pose metrics",
            },
            "ground_excluded": True,
        },
        "counts": {
            "samples": len(records),
            "predicted_objects": total_pred,
            "target_objects": total_gt,
            "matched_objects": len(matched_ious),
            "asset_matched_objects": asset_tp,
            "predictions_without_extent": dropped_pred,
            "targets_without_extent": dropped_gt,
        },
        "summary": summary,
        "dynamic": dynamic_summary,
        "per_category": category_rows,
    }


def iter_pass_dirs(root: Path) -> Iterable[Path]:
    """Every metrics/<layout> directory that carries raw records."""

    if (root / "records-00.jsonl").is_file() or any(root.glob("records-*.jsonl")):
        yield root
        return
    for path in sorted(root.rglob("records-*.jsonl")):
        yield path.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="Step directory, run directory, or a single pass directory")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=list(DEFAULT_IOU_THRESHOLDS))
    parser.add_argument("--extent-source", choices=["catalog", "sample"], default="catalog")
    parser.add_argument("--moving-speed", type=float, default=DEFAULT_MOVING_SPEED_MPS)
    parser.add_argument("--moving-angular", type=float, default=DEFAULT_MOVING_ANGULAR_DPS)
    parser.add_argument("--moving-filter", type=int, default=DEFAULT_MOVING_FILTER)
    parser.add_argument("--moving-min-run", type=int, default=DEFAULT_MOVING_MIN_RUN)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--mesh-points", type=int, default=DEFAULT_MESH_POINTS)
    parser.add_argument(
        "--auc-diameter-fraction",
        type=float,
        default=DEFAULT_AUC_DIAMETER_FRACTION,
        help="Upper ADD-S AUC threshold as a fraction of canonical mesh diameter",
    )
    parser.add_argument("--no-dynamic", action="store_true", help="Skip the track metrics (no mesh loading)")
    parser.add_argument("--output-name", default="scene_metrics.json")
    args = parser.parse_args()

    if not 0 < args.auc_diameter_fraction <= 1:
        parser.error("--auc-diameter-fraction must be in (0, 1]")

    catalog = IdentityCatalog.load(REPO_ROOT / "dataset_process/object_identity_catalog.json")
    mesh_points = None if args.no_dynamic else MeshPointCache(catalog, args.mesh_points, REPO_ROOT)

    written = 0
    root = args.root.resolve()
    for pass_dir in iter_pass_dirs(root):
        result = score_pass(
            pass_dir,
            args.iou_thresholds,
            args.extent_source,
            args.moving_speed,
            args.moving_angular,
            args.moving_filter,
            args.moving_min_run,
            args.fps,
            mesh_points,
            args.auc_diameter_fraction,
        )
        if result is None:
            continue
        output_path = pass_dir / args.output_name
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written += 1
        summary = result["summary"]
        label = pass_dir.relative_to(root) if pass_dir != root else pass_dir.name
        if not summary:
            print(f"{label}: {result['note']}")
            continue
        print(
            f"{label}: "
            f"ID-F1 {summary['category_f1_micro_pct']:.1f}%  "
            f"asset-F1 {summary['asset_f1_micro_pct']:.1f}%  "
            f"IoU|match {summary['3D_IoU_pct']:.1f}%  "
            f"IoU|GT {summary['3D_IoU_gt_pct']:.1f}%  "
            f"PQ-IoU {summary['PQ_IoU_pct']:.1f}%  "
            f"F1@0.5 {summary.get('F1@0.5_pct', float('nan')):.1f}%  "
            f"({result['counts']['matched_objects']}/{result['counts']['target_objects']} objects)"
        )
        dynamic = result["dynamic"]
        if dynamic:
            print(
                f"{label}: dynamic recall "
                f"{dynamic.get('track_recall_pct', float('nan')):.1f}%  "
                f"ADD-S AUC moving "
                f"{dynamic.get('ADD_S_AUC_0.1d_moving_pct', float('nan')):.1f}%  "
                f"static {dynamic.get('ADD_S_AUC_0.1d_static_pct', float('nan')):.1f}%  "
                f"mesh velocity "
                f"{dynamic.get('mesh_velocity_error_moving_mm_s', float('nan')):.1f} mm/s"
            )
    print(f"Wrote {written} {args.output_name} file(s).")


if __name__ == "__main__":
    main()
