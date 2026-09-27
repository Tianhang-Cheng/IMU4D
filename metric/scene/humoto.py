"""HUMOTO first-frame multi-object configuration metrics."""

from __future__ import annotations

import argparse
import itertools
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError

from ..core.common import as_float_array, load_jsonl, write_json
from ..core.rotation import rotation_matrix


def _object_value(obj: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    raise ValueError(f"Object is missing one of {keys}: {obj.keys()}")


def _object_box(obj: Mapping[str, Any], default_rotation_representation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = as_float_array(_object_value(obj, "translation", "transl", "center"), "box center")
    size = as_float_array(_object_value(obj, "size", "bbox", "extent"), "box size")
    representation = str(obj.get("rotation_representation", default_rotation_representation))
    rotation = rotation_matrix(_object_value(obj, "rotation", "rot"), representation)
    if center.shape != (3,) or size.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("Each box must contain one rotation, XYZ center, and XYZ full size")
    if np.any(size <= 0):
        raise ValueError(f"Box size must be positive, got {size}")
    return rotation, center, size


def _halfspaces(rotation: np.ndarray, center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = rotation.T
    normals = np.concatenate([axes, -axes], axis=0)
    offsets = np.concatenate(
        [axes @ center + size / 2.0, -(axes @ center) + size / 2.0], axis=0
    )
    return normals, offsets


def _intersection_volume(
    first: tuple[np.ndarray, np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray, np.ndarray],
    tolerance: float = 1e-8,
) -> float:
    first_a, first_b = _halfspaces(*first)
    second_a, second_b = _halfspaces(*second)
    inequalities = np.concatenate([first_a, second_a], axis=0)
    offsets = np.concatenate([first_b, second_b], axis=0)
    vertices: list[np.ndarray] = []
    for indices in itertools.combinations(range(len(inequalities)), 3):
        matrix = inequalities[list(indices)]
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        point = np.linalg.solve(matrix, offsets[list(indices)])
        if np.all(inequalities @ point <= offsets + tolerance):
            vertices.append(point)
    if len(vertices) < 4:
        return 0.0
    unique = np.unique(np.round(np.asarray(vertices), decimals=10), axis=0)
    if len(unique) < 4:
        return 0.0
    try:
        return float(ConvexHull(unique).volume)
    except QhullError:
        return 0.0


def oriented_box_iou_3d(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    rotation_representation: str = "quaternion",
) -> float:
    pred_box = _object_box(prediction, rotation_representation)
    gt_box = _object_box(target, rotation_representation)
    intersection = _intersection_volume(pred_box, gt_box)
    pred_volume = float(np.prod(pred_box[2]))
    gt_volume = float(np.prod(gt_box[2]))
    union = pred_volume + gt_volume - intersection
    return float(np.clip(intersection / union, 0.0, 1.0)) if union > 0 else 0.0


def _category(obj: Mapping[str, Any]) -> str:
    value = obj.get("category", obj.get("identity", obj.get("name")))
    if value is None:
        raise ValueError("Every object needs category, identity, or name")
    return str(value)


def match_objects(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    rotation_representation: str = "quaternion",
) -> list[tuple[int, int, float]]:
    """Maximum-IoU one-to-one matching, constrained to equal categories."""

    matches: list[tuple[int, int, float]] = []
    categories = sorted({_category(obj) for obj in [*predictions, *targets]})
    for category in categories:
        pred_indices = [i for i, obj in enumerate(predictions) if _category(obj) == category]
        gt_indices = [i for i, obj in enumerate(targets) if _category(obj) == category]
        if not pred_indices or not gt_indices:
            continue
        ious = np.asarray(
            [
                [
                    oriented_box_iou_3d(predictions[pred], targets[gt], rotation_representation)
                    for gt in gt_indices
                ]
                for pred in pred_indices
            ]
        )
        rows, columns = linear_sum_assignment(-ious)
        matches.extend(
            (pred_indices[row], gt_indices[column], float(ious[row, column]))
            for row, column in zip(rows, columns)
        )
    return matches


def evaluate_humoto_records(
    records: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
    rotation_representation: str = "quaternion",
) -> dict[str, Any]:
    total_predictions = 0
    total_targets = 0
    identity_tp = 0
    geometry_tp = 0
    iou_sum = 0.0
    per_sample: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        predictions = record.get("predictions", record.get("pred", []))
        targets = record.get("targets", record.get("gt", []))
        if not isinstance(predictions, list) or not isinstance(targets, list):
            raise ValueError("HUMOTO records need prediction and target object lists")
        matches = match_objects(predictions, targets, rotation_representation)
        sample_geometry_tp = sum(iou >= iou_threshold for _, _, iou in matches)
        sample_iou_sum = sum(iou for _, _, iou in matches)
        total_predictions += len(predictions)
        total_targets += len(targets)
        identity_tp += len(matches)
        geometry_tp += sample_geometry_tp
        iou_sum += sample_iou_sum
        per_sample.append(
            {
                "id": str(record.get("id", index)),
                "prediction_count": len(predictions),
                "target_count": len(targets),
                "identity_matches": len(matches),
                "geometry_matches": sample_geometry_tp,
                "iou_sum": sample_iou_sum,
            }
        )
    precision_denominator = max(total_predictions, 1)
    recall_denominator = max(total_targets, 1)
    summary = {
        "3D_IoU_pct": 100.0 * iou_sum / recall_denominator,
        f"P@{iou_threshold:g}_pct": 100.0 * geometry_tp / precision_denominator,
        f"R@{iou_threshold:g}_pct": 100.0 * geometry_tp / recall_denominator,
        "ID_P_pct": 100.0 * identity_tp / precision_denominator,
        "ID_R_pct": 100.0 * identity_tp / recall_denominator,
    }
    return {
        "schema_version": 1,
        "task": "humoto_first_frame_object_configuration",
        "protocol": {
            "frame": 0,
            "iou_threshold": iou_threshold,
            "matching": "category-constrained one-to-one Hungarian matching maximizing oriented 3D IoU",
            "aggregation": "micro precision/recall; IoU sum divided by GT object count (unmatched GT contributes zero)",
            "rotation_representation": rotation_representation,
        },
        "counts": {
            "samples": len(records),
            "predictions": total_predictions,
            "targets": total_targets,
            "identity_true_positive": identity_tp,
            "geometry_true_positive": geometry_tp,
        },
        "summary": summary,
        "per_sample": per_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HUMOTO first-frame object configurations")
    parser.add_argument("--input", required=True, help="JSONL file in the format documented in metric/README.md")
    parser.add_argument("--output", help="Output JSON path; prints JSON when omitted")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--rotation-representation", default="quaternion", choices=["matrix", "quaternion", "axis_angle", "6d"])
    args = parser.parse_args()
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise SystemExit("--iou-threshold must be in [0,1]")
    result = evaluate_humoto_records(
        load_jsonl(args.input), args.iou_threshold, args.rotation_representation
    )
    write_json(result, args.output)


if __name__ == "__main__":
    main()
