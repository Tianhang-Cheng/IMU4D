"""Adapt IMU4D saved samples to the canonical :mod:`metric.motion` contract.

Numerical metric definitions intentionally live in ``metric.motion``.  This
module only knows the repository-specific saved ``pred``/``gt`` dictionary and
how to decode its SMPL-X pose into joints and vertices.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Iterable

import numpy as np
from tqdm import tqdm

from metric.core.common import result_document
from metric.motion import evaluate_motion_sample


MOTION_RESULT_SCHEMA_VERSION = 2
MOTION_IMPLEMENTATION = "metric.motion via evaluation.motion_adapter"


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _sample_order(path: Path) -> tuple[int, int | str]:
    match = re.search(r"(?:^|/)id_(\d+)(?:_|$)", str(path))
    return (0, int(match.group(1))) if match else (1, path.name)


def saved_motion_paths(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("*.npy"), key=_sample_order)


def adapt_saved_motion_sample(
    sample: dict[str, Any],
    max_frames: int | None,
    geometry_decoder: Callable[..., tuple[Any, Any, Any, Any]] | None = None,
) -> dict[str, np.ndarray]:
    """Return one saved IMU4D sample in ``metric.motion``'s array schema."""

    if geometry_decoder is None:
        from utils.metrics import decode_smplx_motion_geometry

        geometry_decoder = decode_smplx_motion_geometry
    gt_joints, pred_joints, gt_vertices, pred_vertices = geometry_decoder(
        sample, max_frame_length=max_frames
    )

    def frames(value: Any, width: int) -> np.ndarray:
        array = _numpy(value).reshape(-1, width)
        return array if max_frames is None else array[:max_frames]

    return {
        "pred_joints": _numpy(pred_joints),
        "gt_joints": _numpy(gt_joints),
        "pred_rotations": frames(sample["pred"]["pose"], 63).reshape(-1, 21, 3),
        "gt_rotations": frames(sample["gt"]["pose"], 63).reshape(-1, 21, 3),
        "pred_vertices": _numpy(pred_vertices),
        "gt_vertices": _numpy(gt_vertices),
        "pred_root_translation": frames(sample["pred"]["transl"], 3),
        "gt_root_translation": frames(sample["gt"]["transl"], 3),
    }


def evaluate_saved_motion_sample(
    sample: dict[str, Any],
    max_frames: int | None,
    geometry_decoder: Callable[..., tuple[Any, Any, Any, Any]] | None = None,
) -> dict[str, float]:
    canonical = adapt_saved_motion_sample(sample, max_frames, geometry_decoder)
    return evaluate_motion_sample(canonical, rotation_representation="axis_angle")


def evaluate_saved_motion_files(
    paths: Iterable[Path],
    max_frames: int | None,
    *,
    verbose: bool = True,
    sample_transform: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate saved samples and return the canonical result document."""

    paths = list(paths)
    if not paths:
        raise ValueError("No saved motion .npy files were provided")
    rows: list[dict[str, float]] = []
    per_sample: list[dict[str, Any]] = []
    iterator = tqdm(paths, disable=not verbose, desc="motion metrics")
    for path in iterator:
        sample = np.load(path, allow_pickle=True).item()
        if not isinstance(sample, dict):
            raise ValueError(f"{path} does not contain a saved sample dictionary")
        if sample_transform is not None:
            sample = sample_transform(path, sample)
        metrics = evaluate_saved_motion_sample(sample, max_frames)
        rows.append(metrics)
        per_sample.append({"id": path.stem, "metrics": metrics})

    document = result_document(
        "motion",
        per_sample,
        rows,
        {
            "max_frames": max_frames,
            "position_input_unit": "m",
            "rotation_representation": "axis_angle",
            "sample_aggregation": "mean over samples; population std",
            "MPJPE": "root-aligned joints, millimetres",
            "PA-MPJPE": "prediction aligned to ground truth per frame by a similarity transform, millimetres",
            "MPJRE": "SO(3) geodesic joint rotation error, degrees",
            "MPJVE": "root-local SMPL-X vertex error, millimetres",
            "MTE": "unaligned root-translation ATE RMSE, millimetres",
            "implementation": MOTION_IMPLEMENTATION,
        },
    )
    document["schema_version"] = MOTION_RESULT_SCHEMA_VERSION
    return document


def evaluate_saved_motion_directory(
    root: str | Path, max_frames: int | None, *, verbose: bool = True
) -> dict[str, Any]:
    paths = saved_motion_paths(root)
    if not paths:
        raise ValueError(f"No .npy motion samples found in {root}")
    return evaluate_saved_motion_files(paths, max_frames, verbose=verbose)


def motion_result_is_current(path: str | Path) -> bool:
    """Whether a result file uses the canonical v2 implementation."""

    import json

    path = Path(path)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        payload.get("schema_version") == MOTION_RESULT_SCHEMA_VERSION
        and payload.get("protocol", {}).get("implementation") == MOTION_IMPLEMENTATION
    )
