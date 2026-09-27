#!/usr/bin/env python3
"""Unified launcher for version-controlled pretraining YAML profiles."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.utils import load_config_file


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _latest_checkpoint(root: Path) -> tuple[Path, int]:
    candidates = []
    for path in root.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and (path / "unwrapped_model" / "pytorch_model.bin").is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No complete checkpoint-* found under {root}")
    step, checkpoint = max(candidates)
    return checkpoint, step


def _weights_only_resume() -> bool:
    """EVAL_WEIGHTS_ONLY=1 resumes from the weights alone.

    An evaluation-only checkpoint copy carries unwrapped_model/pytorch_model.bin
    and scheduler.bin but not the 24 GB optimizer.bin, so the full accelerate
    state resume the training lifecycles validate is impossible (see
    exp/ablation_cascade/m2t/README.md). This relaxes that validation to the
    weights-only path run.py already implements; the caller pairs it with
    experiment.full_eval_only=True, so no training step is ever taken from the
    missing optimizer state.
    """
    value = os.environ.get("EVAL_WEIGHTS_ONLY", "0")
    if value not in {"0", "1"}:
        raise ValueError("EVAL_WEIGHTS_ONLY must be 0 or 1")
    return value == "1"


def _require_resume_state(checkpoint: Path) -> bool:
    """Check the resume state files, and report whether weights-only is in force."""
    weights_only = _weights_only_resume()
    required = ("scheduler.bin",) if weights_only else ("optimizer.bin", "scheduler.bin")
    for state_name in required:
        if not (checkpoint / state_name).is_file():
            raise FileNotFoundError(f"Missing resume state: {checkpoint / state_name}")
    return weights_only


def _checkpoint_step(checkpoint: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
    if match is None:
        raise ValueError(
            f"Cannot read a step number from {checkpoint.name}; "
            "a step-continuing warm start needs an exact checkpoint-<step> directory"
        )
    return int(match.group(1))


def _validate_inputs(config) -> None:
    roots = [str(config.dataset.params.imu_path_or_url)]
    roots.extend(str(item) for item in config.dataset.params.train_imu_paths_or_urls)
    full_eval_datasets = config.experiment.get("full_eval_datasets", None)
    if full_eval_datasets is not None:
        roots.extend(
            str(entry.get("root"))
            for entry in full_eval_datasets
            if entry.get("root") is not None
        )
    for value in dict.fromkeys(roots):
        manifest = _path(value) / "wds" / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Missing WebDataset manifest: {manifest}\n"
                "Prepare the dataset before starting training."
            )

    launcher = config.get("launcher", {})
    for value in launcher.get("required_files", []):
        required = _path(str(value))
        if not required.is_file():
            raise FileNotFoundError(f"Required file does not exist: {required}")
    for pattern in launcher.get("required_globs", []):
        resolved = str(_path(str(pattern)))
        if not glob.glob(resolved):
            raise FileNotFoundError(f"Required files do not match: {resolved}")


def _weights_init_overrides(
    checkpoint: Path,
) -> tuple[list[str], tempfile.TemporaryDirectory]:
    """Weights-only start from an exact checkpoint.

    run.py resolves a resume source out of experiment.ckpt_dir by picking its
    highest checkpoint-*, so the source is pinned behind a temporary directory
    holding a single symlink.  Checkpoints are still written to output_dir.
    """
    model_path = checkpoint / "unwrapped_model" / "pytorch_model.bin"
    if not model_path.is_file():
        raise FileNotFoundError(f"Initial checkpoint weights do not exist: {model_path}")
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
    checkpoint_name = checkpoint.name if match else "checkpoint-0"
    temporary_root = tempfile.TemporaryDirectory(prefix="imu4d-pretrain-init-")
    Path(temporary_root.name, checkpoint_name).symlink_to(checkpoint.resolve())
    return [
        "experiment.ckpt_dir=" + temporary_root.name,
        "experiment.resume_from_checkpoint=True",
        "experiment.load_without_optimizer=True",
        "experiment.reset_step_on_weight_load=True",
    ], temporary_root


def _lifecycle_overrides(config) -> tuple[list[str], tempfile.TemporaryDirectory | None]:
    launcher = config.get("launcher", {})
    lifecycle = str(launcher.get("lifecycle", "new"))
    output_dir = _path(str(config.experiment.output_dir))
    temporary_root = None

    # WARM_START_FROM overrides the profile's own lifecycle with a weights-only
    # start from an exact checkpoint.  Use it when an architecture change makes
    # a strict resume impossible: tensors the checkpoint and the model share are
    # copied (experiment.strict_resume must be false), modules the checkpoint
    # predates keep their fresh initialization, and the optimizer, scheduler and
    # step counter all start over.  The run needs its own FULL_DIR so the source
    # checkpoints stay untouched and the restarted step numbering cannot collide
    # with them.
    warm_start = os.environ.get("WARM_START_FROM", "")
    if warm_start:
        if output_dir.exists():
            raise FileExistsError(
                f"Output directory already exists: {output_dir}; "
                "WARM_START_FROM needs a fresh FULL_DIR"
            )
        if config.experiment.get("strict_resume", True):
            raise ValueError(
                "WARM_START_FROM requires experiment.strict_resume=false; "
                "a strict load rejects any checkpoint predating a module"
            )
        checkpoint = _path(warm_start)
        keep_step = os.environ.get("WARM_START_KEEP_STEP", "0")
        if keep_step not in {"0", "1"}:
            raise ValueError("WARM_START_KEEP_STEP must be 0 or 1")
        overrides, temporary_root = _weights_init_overrides(checkpoint)
        if keep_step == "1":
            # Continue the source checkpoint's step numbering instead of
            # restarting at 0.  run.py then advances the rebuilt scheduler to
            # that step, so max_train_steps stays the run's absolute end.
            step = _checkpoint_step(checkpoint)
            target_steps = int(config.training.max_train_steps)
            if target_steps <= step:
                raise ValueError(
                    f"training.max_train_steps ({target_steps}) must exceed "
                    f"warm-started step {step}"
                )
            overrides = [
                "experiment.reset_step_on_weight_load=False"
                if item.startswith("experiment.reset_step_on_weight_load=")
                else item
                for item in overrides
            ]
            print(
                f"Warm-starting {output_dir.name} from weights of {checkpoint}, "
                f"continuing at step {step} through {target_steps}"
            )
        else:
            print(f"Warm-starting {output_dir.name} from weights of {checkpoint}")
        return overrides, temporary_root

    if lifecycle == "new":
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        return [
            "experiment.ckpt_dir=" + str(output_dir),
            "experiment.resume_from_checkpoint=False",
            "experiment.load_without_optimizer=False",
            "experiment.reset_step_on_weight_load=False",
        ], None

    if lifecycle == "new_or_resume":
        resume = os.environ.get("RESUME_FULL", "0")
        if resume not in {"0", "1"}:
            raise ValueError("RESUME_FULL must be 0 or 1")
        if resume == "0":
            if output_dir.exists():
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}; "
                    "set RESUME_FULL=1 to resume"
                )
            return [
                "experiment.ckpt_dir=" + str(output_dir),
                "experiment.resume_from_checkpoint=False",
                "experiment.load_without_optimizer=False",
                "experiment.reset_step_on_weight_load=False",
            ], None

        checkpoint, step = _latest_checkpoint(output_dir)
        weights_only = _require_resume_state(checkpoint)
        target_steps = int(config.training.max_train_steps)
        if target_steps <= step:
            raise ValueError(
                f"training.max_train_steps ({target_steps}) must exceed resumed step {step}"
            )
        print(
            f"Resuming {output_dir.name} from checkpoint-{step}"
            + (" (weights only)" if weights_only else "")
        )
        return [
            "experiment.ckpt_dir=" + str(output_dir),
            "experiment.resume_from_checkpoint=True",
            "experiment.load_without_optimizer=" + str(weights_only),
            "experiment.reset_step_on_weight_load=False",
        ], None

    if lifecycle == "init_or_resume":
        resume = os.environ.get("RESUME_FULL", "0")
        if resume not in {"0", "1"}:
            raise ValueError("RESUME_FULL must be 0 or 1")
        if resume == "1":
            checkpoint, step = _latest_checkpoint(output_dir)
            weights_only = _require_resume_state(checkpoint)
            target_steps = int(config.training.max_train_steps)
            if target_steps <= step:
                raise ValueError(
                    f"training.max_train_steps ({target_steps}) must exceed resumed step {step}"
                )
            print(
                f"Resuming {output_dir.name} from checkpoint-{step}"
                + (" (weights only)" if weights_only else "")
            )
            return [
                "experiment.ckpt_dir=" + str(output_dir),
                "experiment.resume_from_checkpoint=True",
                "experiment.load_without_optimizer=" + str(weights_only),
                "experiment.reset_step_on_weight_load=False",
            ], None

        if output_dir.exists():
            raise FileExistsError(
                f"Output directory already exists: {output_dir}; set RESUME_FULL=1 to resume"
            )

        # Pin a new training family to an exact source checkpoint when one is
        # configured.  A temporary checkpoint root keeps run.py's existing
        # resume interface while preventing a later checkpoint in the source
        # directory from silently changing initialization.
        if launcher.get("init_checkpoint") is not None:
            checkpoint = _path(str(launcher.init_checkpoint))
            model_path = checkpoint / "unwrapped_model" / "pytorch_model.bin"
            if not model_path.is_file() and checkpoint.is_dir() and not re.fullmatch(
                r"checkpoint-(\d+)", checkpoint.name
            ):
                # A run directory: pin its latest complete checkpoint.
                checkpoint, _ = _latest_checkpoint(checkpoint)
                model_path = checkpoint / "unwrapped_model" / "pytorch_model.bin"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Initial checkpoint weights do not exist: {model_path}"
                )
            match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
            checkpoint_name = checkpoint.name if match else "checkpoint-0"
            temporary_root = tempfile.TemporaryDirectory(
                prefix="imu4d-pretrain-init-"
            )
            Path(temporary_root.name, checkpoint_name).symlink_to(
                checkpoint.resolve()
            )
            init_root = Path(temporary_root.name)
            print(f"Initializing from exact checkpoint: {model_path}")
        else:
            init_root = _path(str(launcher.init_ckpt_dir))
            _latest_checkpoint(init_root)

        return [
            "experiment.ckpt_dir=" + str(init_root),
            "experiment.resume_from_checkpoint=True",
            "experiment.load_without_optimizer=True",
            "experiment.reset_step_on_weight_load=True",
        ], temporary_root

    if lifecycle == "weights_init":
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        return _weights_init_overrides(_path(str(launcher.init_checkpoint)))

    raise ValueError(f"Unknown launcher.lifecycle: {lifecycle}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print without launching")
    parser.add_argument("config", help="YAML profile under configs/")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="OmegaConf CLI overrides")
    args = parser.parse_args()

    config_path = _path(args.config).resolve()
    config = load_config_file(config_path)
    OmegaConf.resolve(config)
    _validate_inputs(config)
    if "FULL_STEPS" in os.environ:
        config.training.max_train_steps = int(os.environ["FULL_STEPS"])
    lifecycle_overrides, temporary_root = _lifecycle_overrides(config)

    gpu_id = os.environ.get("TRAIN_GPU_ID", "0")
    accelerate_config = os.environ.get("ACCELERATE_CONFIG", "configs/accelerate/1_gpus.yaml")
    port = os.environ.get("MAIN_PROCESS_PORT", "1644")
    extra = shlex.split(os.environ.get("EXTRA_OVERRIDES", ""))
    explicit = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    compatibility_overrides = []
    if "FULL_STEPS" in os.environ:
        compatibility_overrides.append(
            "training.max_train_steps=" + os.environ["FULL_STEPS"]
        )
    command = [
        "accelerate",
        "launch",
        "--config_file",
        accelerate_config,
        "--main_process_port",
        port,
        "run.py",
        "config=" + str(config_path),
        *lifecycle_overrides,
        *compatibility_overrides,
        *extra,
        *explicit,
    ]
    env = os.environ.copy()
    env["TRAIN_GPU_ID"] = gpu_id
    print("Launching:", shlex.join(command))
    if args.dry_run:
        if temporary_root is not None:
            temporary_root.cleanup()
        return
    try:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    finally:
        if temporary_root is not None:
            temporary_root.cleanup()


if __name__ == "__main__":
    main()
