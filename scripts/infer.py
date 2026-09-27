#!/usr/bin/env python3
"""Run a released IMU4D model on one sample or a WebDataset test split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "pretrain": ("showo_pretrain_full", "configs/showo_pretrain_full.yaml", "LINGO"),
    "noise": ("showo_pretrain_full_noise", "configs/showo_pretrain_full_noise.yaml", "LINGO"),
    "imuposer": ("showo_finetune_imuposer", "configs/showo_finetune_imuposer.yaml", "imuposer"),
    "dipimu": ("showo_finetune_dipimu", "configs/showo_finetune_dipimu.yaml", "dipimu"),
    "ncsa": ("showo_finetune_ncsa_fix", "configs/showo_finetune_ncsa.yaml", "ncsa"),
    "hiphi": ("showo_finetune_hiphi", "configs/showo_finetune_hiphi.yaml", "HiPHI"),
    "omomo": ("showo_finetune_omomo", "configs/showo_finetune_omomo.yaml", "OMOMO"),
    "humoto": ("showo_finetune_humoto", "configs/showo_finetune_humoto.yaml", "HUMOTO"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="pretrain")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Packed IMU .pkl sample")
    source.add_argument("--dataset-root", type=Path, help="Processed dataset root with wds/test shards")
    parser.add_argument("--weights", type=Path, help="Override the downloaded pytorch_model.bin")
    parser.add_argument("--output", type=Path, help="Output directory (default: exp/inference/<model>)")
    parser.add_argument("--dataset", help="Input dataset label, when different from the model's default")
    parser.add_argument("--frames", type=int, default=60, help="Maximum generated frames")
    parser.add_argument("--max-samples", type=int, default=50, help="Dataset test samples to evaluate")
    parser.add_argument("--invalid-imu-ids", help="0-based sensor slots to mask; model-specific default")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()

    run_name, config, default_dataset = MODELS[args.model]
    sample = args.input.expanduser().resolve() if args.input else None
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else None
    weights = (args.weights or ROOT / "checkpoints" / run_name / "pytorch_model.bin").expanduser().resolve()
    output = (args.output or ROOT / "exp" / "inference" / args.model).expanduser().resolve()
    if sample is not None and not sample.is_file():
        parser.error(f"Input sample does not exist: {sample}")
    if dataset_root is not None and not (dataset_root / "wds" / "manifest.json").is_file():
        parser.error(f"WebDataset manifest not found under: {dataset_root}")
    if not weights.is_file():
        parser.error(f"Weights not found: {weights}; download the selected model first")
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if output.exists():
        parser.error(f"Output directory already exists: {output}; choose a new --output")
    default_invalid = "[0,3,5]" if args.model == "ncsa" else (
        "[]" if args.model in {"imuposer", "dipimu"} else "[3]"
    )
    invalid_imu_ids = args.invalid_imu_ids or default_invalid

    # run.py accepts checkpoint-*/unwrapped_model/pytorch_model.bin. A symlink
    # exposes the flat release file without duplicating a multi-GB tensor.
    with tempfile.TemporaryDirectory(prefix="imu4d-infer-") as temporary:
        unwrapped = Path(temporary) / "checkpoint-0" / "unwrapped_model"
        unwrapped.mkdir(parents=True)
        (unwrapped / "pytorch_model.bin").symlink_to(weights)
        command = [
            sys.executable, "run.py", f"config={config}",
            f"experiment.ckpt_dir={temporary}",
            "experiment.resume_from_checkpoint=True",
            "experiment.strict_resume=True",
            "experiment.mode=test",
            f"experiment.name=inference_{args.model}",
            f"experiment.output_dir={output}",
            f"experiment.eval_selected_dataset={args.dataset or default_dataset}",
            f"experiment.max_eval_imu_len={args.frames}",
            f"experiment.eval_invalid_imu_id={invalid_imu_ids}",
            "experiment.eval_imu_configs=null",
            f"experiment.save_test_sample={sample is not None}",
            "experiment.save_ckpt_when_eval=False",
            "experiment.full_eval_rerun.enabled=False",
            "dataset.params.num_workers=0",
            "training.lora.enabled=False",
        ]
        if sample is not None:
            command.append(f"experiment.eval_selected_imu_seq={sample}")
        else:
            command.extend([
                "experiment.eval_selected_imu_seq=null",
                f"dataset.params.imu_path_or_url={dataset_root}",
                f"experiment.max_eval_sample_num={args.max_samples}",
            ])
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
        if args.model in {"hiphi", "omomo", "humoto"}:
            environment["OBJ_GEOM_FEATURES"] = "null"
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print(f"Inference output: {output}")


if __name__ == "__main__":
    main()
