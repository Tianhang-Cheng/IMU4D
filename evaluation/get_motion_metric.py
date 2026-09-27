"""Compatibility CLI for motion metrics on saved IMU4D ``*.npy`` outputs.

The metric formulas live exclusively in :mod:`metric.motion`. This module
keeps the historical command and ``mean/std/raw`` NPY output format, plus the
optional shifted-window adapter used by older experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.motion_adapter import evaluate_saved_motion_files, saved_motion_paths


LEGACY_NAMES = {
    "MPJPE": "MPJPE_mm",
    "PA MPJPE": "PA_MPJPE_mm",
    "MJPRE": "MPJRE_deg",
    "MPJVE": "MPJVE_mm",
    "MTE": "MTE_mm",
}


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _shifted_window_average(
    path: Path, sample: dict[str, Any], model: str
) -> dict[str, Any]:
    """Apply the historical shifted-2 overlap average before evaluation."""

    if model not in {"Ours", "GPT2"}:
        return sample
    shifted_path = Path(str(path).replace(
        "viz_test_generate_number_shifted_0",
        "viz_test_generate_number_shifted_2",
    ))
    if shifted_path == path or not shifted_path.is_file():
        return sample

    from dataset_process.motionmillion_and_lingo.convert_lingo_dataset import (
        align_poses_to_frame,
        transform_aligned_to_reference,
    )

    shifted = np.load(shifted_path, allow_pickle=True).item()
    prediction = sample["pred"]
    shifted_prediction = shifted["pred"]
    cut_len = min(
        len(_numpy(prediction["pose"])) - 2,
        len(_numpy(shifted_prediction["pose"])),
    )
    if cut_len <= 0:
        return sample
    orient, translation, pose, _, _ = align_poses_to_frame(
        _numpy(shifted_prediction["orient"])[:cut_len],
        _numpy(shifted_prediction["transl"])[:cut_len],
        _numpy(shifted_prediction["pose"])[:cut_len],
        ref_frame=0,
    )
    orient, translation, pose = transform_aligned_to_reference(
        orient,
        translation,
        pose,
        _numpy(prediction["orient"])[2],
        _numpy(prediction["transl"])[2],
    )
    # Loading produced a private sample dictionary, so in-place replacement is
    # safe and avoids a deep copy of tokenizer/debug fields.
    prediction["orient"] = _numpy(prediction["orient"]).copy()
    prediction["transl"] = _numpy(prediction["transl"]).copy()
    prediction["pose"] = _numpy(prediction["pose"]).copy()
    prediction["orient"][2 : 2 + cut_len] = (
        prediction["orient"][2 : 2 + cut_len] + orient
    ) / 2.0
    prediction["transl"][2 : 2 + cut_len] = (
        prediction["transl"][2 : 2 + cut_len] + translation
    ) / 2.0
    prediction["pose"][2 : 2 + cut_len] = (
        prediction["pose"][2 : 2 + cut_len] + pose
    ) / 2.0
    return sample


def _legacy_result(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary = document["summary"]
    return {
        "mean": {
            legacy: summary[canonical]["mean"]
            for legacy, canonical in LEGACY_NAMES.items()
        },
        "std": {
            legacy: summary[canonical]["std"]
            for legacy, canonical in LEGACY_NAMES.items()
        },
        "raw": {
            legacy: summary[canonical]["raw"]
            for legacy, canonical in LEGACY_NAMES.items()
        },
    }


def get_number(
    model: str,
    dataset: str,
    result_folder: str,
    log_dir: str,
    apply_shited_window_avg: bool = False,
    eval_frame_length: int = 60,
    verbose: bool = True,
) -> dict[str, dict[str, Any]] | None:
    """Evaluate a directory while preserving the legacy Python API."""

    paths = saved_motion_paths(result_folder)
    if not paths:
        return None
    transform = None
    if apply_shited_window_avg:
        transform = lambda path, sample: _shifted_window_average(path, sample, model)
    document = evaluate_saved_motion_files(
        paths,
        eval_frame_length,
        verbose=verbose,
        sample_transform=transform,
    )
    metrics = _legacy_result(document)

    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "avg" if apply_shited_window_avg else "raw"
    stem = f"{model}_{dataset}_{eval_frame_length}frame_{suffix}_motion_metric"
    text_path = output_dir / f"{stem}.txt"
    npy_path = output_dir / f"{stem}.npy"
    lines = [
        f"Apply shifted window average: {apply_shited_window_avg}",
        f"Eval frame length: {eval_frame_length}",
        "# Mean metrics:",
        *(f"{name}: {metrics['mean'][name]:.2f}" for name in LEGACY_NAMES),
        "# Std metrics:",
        *(f"{name}: {metrics['std'][name]:.2f}" for name in LEGACY_NAMES),
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.save(npy_path, metrics)

    print("########################################################")
    print(
        f"{model} {dataset} motion metric on {len(paths)} samples "
        f"with frame length {eval_frame_length}:"
    )
    for name in LEGACY_NAMES:
        unit = " degrees" if name == "MJPRE" else " mm"
        print(f"{name}: {metrics['mean'][name]:.2f}{unit}")
    print(f"Saved metrics to {text_path} and {npy_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate canonical motion metrics from saved IMU4D NPY outputs."
    )
    parser.add_argument("--result-folder", required=True)
    parser.add_argument("--log-dir")
    parser.add_argument("--model", default="Ours")
    parser.add_argument("--dataset", default="LINGO")
    parser.add_argument("--apply-shifted-window-avg", action="store_true")
    parser.add_argument("--eval-frame-length", type=int, default=60)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result_folder = Path(args.result_folder)
    if not result_folder.is_dir():
        parser.error(f"Result folder does not exist: {result_folder}")
    log_dir = Path(args.log_dir) if args.log_dir else result_folder / "evaluation"
    result = get_number(
        args.model,
        args.dataset,
        str(result_folder),
        str(log_dir),
        apply_shited_window_avg=args.apply_shifted_window_avg,
        eval_frame_length=args.eval_frame_length,
        verbose=not args.quiet,
    )
    if result is None:
        parser.error(f"No .npy files found in {result_folder}")


if __name__ == "__main__":
    main()
