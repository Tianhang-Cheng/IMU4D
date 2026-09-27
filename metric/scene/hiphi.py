"""HiPHI dynamic interacted-object trajectory metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.distance import cdist

from ..core.common import as_float_array, load_jsonl, result_document, summarize_records, write_json
from ..core.rotation import geodesic_distance_deg, rotation_matrix, transform_points


def _track_arrays(
    track: Mapping[str, Any], default_rotation_representation: str
) -> tuple[np.ndarray, np.ndarray]:
    rotations = track.get("rotation", track.get("rot"))
    translations = track.get("translation", track.get("transl"))
    if rotations is None or translations is None:
        raise ValueError("Each track needs rotation/rot and translation/transl")
    representation = str(track.get("rotation_representation", default_rotation_representation))
    matrices = rotation_matrix(rotations, representation)
    translations = as_float_array(translations, "object translations")
    if matrices.ndim != 3 or matrices.shape[-2:] != (3, 3):
        raise ValueError(f"Track rotations must convert to [T,3,3], got {matrices.shape}")
    if translations.shape != (len(matrices), 3):
        raise ValueError(f"Track translations must be [T,3], got {translations.shape}")
    return matrices, translations


def _nearest_neighbor_mean(first: np.ndarray, second: np.ndarray, chunk_size: int = 1024) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(first), chunk_size):
        distances = cdist(first[start : start + chunk_size], second)
        total += float(np.min(distances, axis=1).sum())
        count += len(distances)
    return total / max(count, 1)


def add_s_per_frame(
    pred_rotations: np.ndarray,
    pred_translations: np.ndarray,
    gt_rotations: np.ndarray,
    gt_translations: np.ndarray,
    mesh_points: np.ndarray,
) -> np.ndarray:
    """Symmetry-aware average closest-point distance for every frame."""

    pred_points = transform_points(mesh_points, pred_rotations, pred_translations)
    gt_points = transform_points(mesh_points, gt_rotations, gt_translations)
    return np.asarray(
        [_nearest_neighbor_mean(pred, gt) for pred, gt in zip(pred_points, gt_points)],
        dtype=np.float64,
    )


def mesh_velocity_error_per_frame(
    pred_rotations: np.ndarray,
    pred_translations: np.ndarray,
    gt_rotations: np.ndarray,
    gt_translations: np.ndarray,
    mesh_points: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Corresponding posed-mesh velocity error for each adjacent frame pair."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    pred_world = transform_points(mesh_points, pred_rotations, pred_translations)
    gt_world = transform_points(mesh_points, gt_rotations, gt_translations)
    if pred_world.shape != gt_world.shape:
        raise ValueError(
            f"Predicted and GT posed meshes must match, got "
            f"{pred_world.shape} and {gt_world.shape}"
        )
    return np.linalg.norm(
        np.diff(pred_world, axis=0) * fps - np.diff(gt_world, axis=0) * fps,
        axis=-1,
    ).mean(axis=-1)


def evaluate_track(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    valid_mask: Sequence[bool] | None,
    fps: float,
    default_rotation_representation: str = "quaternion",
    mesh_points: Any | None = None,
    scale_to_mm: float = 1000.0,
) -> dict[str, float]:
    pred_rotation, pred_translation = _track_arrays(prediction, default_rotation_representation)
    gt_rotation, gt_translation = _track_arrays(target, default_rotation_representation)
    if pred_rotation.shape != gt_rotation.shape or pred_translation.shape != gt_translation.shape:
        raise ValueError("Predicted and GT tracks must have the same length")
    length = len(pred_rotation)
    if valid_mask is None:
        valid = np.ones(length, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != (length,):
            raise ValueError(f"valid_mask must have shape [{length}], got {valid.shape}")
    if not np.any(valid):
        raise ValueError("Track has no valid frames")
    if fps <= 0:
        raise ValueError("fps must be positive")

    translation_distance = np.linalg.norm(pred_translation - gt_translation, axis=-1)
    rotation_distance = geodesic_distance_deg(pred_rotation, gt_rotation)
    metrics = {
        "translation_error_mm": float(translation_distance[valid].mean() * scale_to_mm),
        "rotation_error_deg": float(rotation_distance[valid].mean()),
    }

    valid_pairs = valid[:-1] & valid[1:]
    if np.any(valid_pairs):
        pred_linear_velocity = np.diff(pred_translation, axis=0) * fps
        gt_linear_velocity = np.diff(gt_translation, axis=0) * fps
        metrics["translation_velocity_error_mm_s"] = float(
            np.linalg.norm(pred_linear_velocity - gt_linear_velocity, axis=-1)[valid_pairs].mean()
            * scale_to_mm
        )
        pred_delta_rotation = pred_rotation[1:] @ np.swapaxes(pred_rotation[:-1], -1, -2)
        gt_delta_rotation = gt_rotation[1:] @ np.swapaxes(gt_rotation[:-1], -1, -2)
        metrics["angular_velocity_error_deg_s"] = float(
            geodesic_distance_deg(pred_delta_rotation, gt_delta_rotation)[valid_pairs].mean() * fps
        )

    if mesh_points is not None:
        points = as_float_array(mesh_points, "mesh_points")
        if points.ndim != 2 or points.shape[-1] != 3 or len(points) == 0:
            raise ValueError(f"mesh_points must be a non-empty [P,3] array, got {points.shape}")
        add_s = add_s_per_frame(
            pred_rotation, pred_translation, gt_rotation, gt_translation, points
        )
        metrics["ADD_S_mm"] = float(add_s[valid].mean() * scale_to_mm)
        if np.any(valid_pairs):
            point_velocity_error = mesh_velocity_error_per_frame(
                pred_rotation,
                pred_translation,
                gt_rotation,
                gt_translation,
                points,
                fps,
            )
            metrics["temporal_velocity_error_mm_s"] = float(
                point_velocity_error[valid_pairs].mean() * scale_to_mm
            )
    return metrics


def evaluate_hiphi_records(
    records: Sequence[Mapping[str, Any]],
    default_rotation_representation: str = "quaternion",
    position_unit: str = "m",
) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    by_category: dict[str, list[dict[str, float]]] = defaultdict(list)
    scale_to_mm = 1000.0 if position_unit == "m" else 1.0
    for index, record in enumerate(records):
        prediction = record.get("prediction", record.get("pred"))
        target = record.get("target", record.get("gt"))
        if not isinstance(prediction, Mapping) or not isinstance(target, Mapping):
            raise ValueError("HiPHI records need prediction/pred and target/gt tracks")
        metrics = evaluate_track(
            prediction,
            target,
            record.get("valid_mask"),
            float(record.get("fps", 30.0)),
            default_rotation_representation,
            record.get("mesh_points"),
            scale_to_mm,
        )
        predicted_category = str(record.get("predicted_category", prediction.get("category", "")))
        target_category = str(record.get("category", record.get("target_category", target.get("category", "unknown"))))
        if predicted_category:
            metrics["identity_accuracy_pct"] = 100.0 * float(predicted_category == target_category)
        metric_rows.append(metrics)
        by_category[target_category].append(metrics)
        per_sample.append(
            {"id": str(record.get("id", index)), "category": target_category, "metrics": metrics}
        )
    result = result_document(
        "hiphi_dynamic_object_trajectory",
        per_sample,
        metric_rows,
        {
            "frames": "full valid object track",
            "position_input_unit": position_unit,
            "default_rotation_representation": default_rotation_representation,
            "ADD_S": "one-way closest-point distance from predicted posed mesh to GT posed mesh",
            "temporal_velocity": "mean corresponding-mesh-point velocity discrepancy",
            "aggregation": "track mean and population std; per-category track means",
        },
    )
    result["per_category"] = {
        category: summarize_records(rows) for category, rows in sorted(by_category.items())
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HiPHI dynamic object tracks")
    parser.add_argument("--input", required=True, help="JSONL file in the format documented in metric/README.md")
    parser.add_argument("--output", help="Output JSON path; prints JSON when omitted")
    parser.add_argument("--rotation-representation", default="quaternion", choices=["matrix", "quaternion", "axis_angle", "6d"])
    parser.add_argument("--position-unit", default="m", choices=["m", "mm"])
    args = parser.parse_args()
    result = evaluate_hiphi_records(
        load_jsonl(args.input), args.rotation_representation, args.position_unit
    )
    write_json(result, args.output)


if __name__ == "__main__":
    main()
