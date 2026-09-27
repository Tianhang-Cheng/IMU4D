"""Model-independent motion metrics used by all quantitative tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ..core.common import as_float_array, result_document, write_json
from ..core.rotation import geodesic_distance_deg, rotation_matrix


def _frame_mask(length: int, valid_mask: np.ndarray | None) -> np.ndarray:
    if valid_mask is None:
        return np.ones(length, dtype=bool)
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != (length,):
        raise ValueError(f"valid_mask must have shape [{length}], got {mask.shape}")
    if not np.any(mask):
        raise ValueError("valid_mask contains no valid frames")
    return mask


def mpjpe_mm(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    root_align: bool = True,
    root_index: int = 0,
    scale_to_mm: float = 1000.0,
) -> float:
    prediction = as_float_array(prediction, "predicted joints")
    target = as_float_array(target, "ground-truth joints")
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError(f"joints must be equal [T,J,3] arrays, got {prediction.shape} and {target.shape}")
    mask = _frame_mask(len(prediction), valid_mask)
    if root_align:
        prediction = prediction - prediction[:, root_index : root_index + 1]
        target = target - target[:, root_index : root_index + 1]
    return float(np.linalg.norm(prediction[mask] - target[mask], axis=-1).mean() * scale_to_mm)


def _similarity_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = np.sum(source_centered**2)
    if variance < 1e-12:
        return np.broadcast_to(target_mean, source.shape).copy()
    covariance = source_centered.T @ target_centered
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(right_t.T @ left.T))
    rotation = right_t.T @ correction @ left.T
    scale = np.trace(rotation @ covariance) / variance
    return scale * (source @ rotation.T) + target_mean - scale * (source_mean @ rotation.T)


def pa_mpjpe_mm(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    scale_to_mm: float = 1000.0,
) -> float:
    prediction = as_float_array(prediction, "predicted joints")
    target = as_float_array(target, "ground-truth joints")
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError(f"joints must be equal [T,J,3] arrays, got {prediction.shape} and {target.shape}")
    mask = _frame_mask(len(prediction), valid_mask)
    errors = []
    for pred_frame, gt_frame in zip(prediction[mask], target[mask]):
        aligned = _similarity_align(pred_frame, gt_frame)
        errors.append(np.linalg.norm(aligned - gt_frame, axis=-1).mean())
    return float(np.mean(errors) * scale_to_mm)


def mpjre_deg(
    prediction: np.ndarray,
    target: np.ndarray,
    representation: str = "matrix",
    valid_mask: np.ndarray | None = None,
) -> float:
    pred_matrix = rotation_matrix(prediction, representation)
    gt_matrix = rotation_matrix(target, representation)
    if pred_matrix.shape != gt_matrix.shape or pred_matrix.ndim != 4:
        raise ValueError(
            f"joint rotations must convert to equal [T,J,3,3] arrays, got {pred_matrix.shape} and {gt_matrix.shape}"
        )
    mask = _frame_mask(len(pred_matrix), valid_mask)
    return float(geodesic_distance_deg(pred_matrix[mask], gt_matrix[mask]).mean())


def mpjve_mm(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    root_translation_prediction: np.ndarray | None = None,
    root_translation_target: np.ndarray | None = None,
    scale_to_mm: float = 1000.0,
) -> float:
    """Mean per-vertex error on root-aligned meshes.

    If vertices are already generated with zero root translation, omit the two
    root-translation arrays. Otherwise both must be supplied and are subtracted.
    """

    prediction = as_float_array(prediction, "predicted vertices")
    target = as_float_array(target, "ground-truth vertices")
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError(f"vertices must be equal [T,V,3] arrays, got {prediction.shape} and {target.shape}")
    if (root_translation_prediction is None) != (root_translation_target is None):
        raise ValueError("Supply both predicted and target root translations, or neither")
    if root_translation_prediction is not None:
        pred_root = as_float_array(root_translation_prediction, "predicted root translation")
        gt_root = as_float_array(root_translation_target, "ground-truth root translation")
        if pred_root.shape != (len(prediction), 3) or gt_root.shape != pred_root.shape:
            raise ValueError("root translations must have shape [T,3]")
        prediction = prediction - pred_root[:, None]
        target = target - gt_root[:, None]
    mask = _frame_mask(len(prediction), valid_mask)
    return float(np.linalg.norm(prediction[mask] - target[mask], axis=-1).mean() * scale_to_mm)


def mte_mm(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    scale_to_mm: float = 1000.0,
) -> float:
    """Root-trajectory absolute translation RMSE (evo APE convention)."""

    prediction = as_float_array(prediction, "predicted root trajectory")
    target = as_float_array(target, "ground-truth root trajectory")
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[-1] != 3:
        raise ValueError(f"root trajectories must be equal [T,3] arrays, got {prediction.shape} and {target.shape}")
    mask = _frame_mask(len(prediction), valid_mask)
    squared_distance = np.sum((prediction[mask] - target[mask]) ** 2, axis=-1)
    return float(np.sqrt(squared_distance.mean()) * scale_to_mm)


def evaluate_motion_sample(
    sample: dict[str, Any],
    rotation_representation: str = "matrix",
    scale_to_mm: float = 1000.0,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    mask = sample.get("valid_mask")
    if "pred_joints" in sample and "gt_joints" in sample:
        metrics["MPJPE_mm"] = mpjpe_mm(
            sample["pred_joints"], sample["gt_joints"], mask, scale_to_mm=scale_to_mm
        )
        metrics["PA_MPJPE_mm"] = pa_mpjpe_mm(
            sample["pred_joints"], sample["gt_joints"], mask, scale_to_mm=scale_to_mm
        )
    if "pred_rotations" in sample and "gt_rotations" in sample:
        metrics["MPJRE_deg"] = mpjre_deg(
            sample["pred_rotations"], sample["gt_rotations"], rotation_representation, mask
        )
    if "pred_vertices" in sample and "gt_vertices" in sample:
        metrics["MPJVE_mm"] = mpjve_mm(
            sample["pred_vertices"],
            sample["gt_vertices"],
            mask,
            sample.get("pred_root_translation_for_vertices"),
            sample.get("gt_root_translation_for_vertices"),
            scale_to_mm,
        )
    if "pred_root_translation" in sample and "gt_root_translation" in sample:
        metrics["MTE_mm"] = mte_mm(
            sample["pred_root_translation"],
            sample["gt_root_translation"],
            mask,
            scale_to_mm,
        )
    if not metrics:
        raise ValueError("Motion sample contains none of the supported pred_*/gt_* pairs")
    return metrics


def _load_np_sample(path: Path, max_frames: int | None) -> tuple[str, dict[str, Any]]:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        sample = {key: loaded[key] for key in loaded.files}
        loaded.close()
    else:
        value = loaded.item() if loaded.shape == () else loaded
        if not isinstance(value, dict):
            raise ValueError(f"{path} must contain a dict or an NPZ mapping")
        sample = value
    if max_frames is not None:
        for key, value in list(sample.items()):
            if key.startswith(("pred_", "gt_")) or key == "valid_mask":
                if isinstance(value, np.ndarray) and value.ndim >= 1:
                    sample[key] = value[:max_frames]
    return path.stem, sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model-independent IMU4D motion exports")
    parser.add_argument("--input-dir", required=True, help="Directory containing one .npz/.npy per sample")
    parser.add_argument("--output", help="Output JSON path; prints JSON when omitted")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--rotation-representation", default="matrix", choices=["matrix", "quaternion", "axis_angle", "6d"])
    parser.add_argument("--position-unit", default="m", choices=["m", "mm"])
    args = parser.parse_args()

    paths = sorted([*Path(args.input_dir).glob("*.npz"), *Path(args.input_dir).glob("*.npy")])
    if not paths:
        raise SystemExit(f"No .npz/.npy files found in {args.input_dir}")
    rows: list[dict[str, float]] = []
    per_sample: list[dict[str, Any]] = []
    for path in paths:
        sample_id, sample = _load_np_sample(path, args.max_frames)
        metrics = evaluate_motion_sample(
            sample,
            rotation_representation=args.rotation_representation,
            scale_to_mm=1000.0 if args.position_unit == "m" else 1.0,
        )
        rows.append(metrics)
        per_sample.append({"id": sample_id, "metrics": metrics})
    document = result_document(
        "motion",
        per_sample,
        rows,
        {
            "max_frames": args.max_frames,
            "position_input_unit": args.position_unit,
            "rotation_representation": args.rotation_representation,
            "aggregation": "sample mean and population std",
        },
    )
    write_json(document, args.output)


if __name__ == "__main__":
    main()
