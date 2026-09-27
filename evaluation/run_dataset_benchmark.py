"""Run the canonical 3/5-point x 60/480-frame benchmark for one dataset.

Each frame length gets an independent job directory and can run on a separate
GPU.  Checkpoint weights are symlinked, never copied or modified.  The model
pass retains per-sample motion arrays and caption records, then
``evaluation.score_saved_outputs`` computes all standard motion/text metrics
offline from those artifacts.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

# Mirrors run.py's NUM_IMU_SENSORS (len(IMU_SENSOR_NAMES)); duplicated here so
# the runner does not import the training entry point just to size a layout.
NUM_IMU_SENSORS = 6

# Synthetic-mocap default: every slot carries a simulated reading, so the
# canonical rows may place sensors anywhere in the six-slot layout.
DEFAULT_LAYOUTS = [
    {"name": "3pt_lhip_lear_lelbow", "active_imu_id": [0, 2, 4]},
    {"name": "5pt_full", "active_imu_id": [0, 1, 2, 4, 5]},
]

# Real-IMU captures only measure the slots their hardware actually occupies.
# The NCSA phone / earbud / watch capture fills 1 (phone, right thigh), 2 and 3
# (earbud, copied into both ear slots) and 4 (watch, left wrist); slots 0 and 5
# hold zero acceleration and identity orientation, so a layout that activates
# them scores the model against sensors that were never worn.  Hence these
# datasets carry their own layout list instead of DEFAULT_LAYOUTS.
NCSA_3PT_LAYOUTS = [
    {"name": "3pt_rhip_lear_lelbow", "active_imu_id": [1, 2, 4]},
    {"name": "3pt_rhip_rear_lelbow", "active_imu_id": [1, 3, 4]},
    {"name": "2pt_rhip_lelbow", "active_imu_id": [1, 4]},
]
NCSA_2PT_LAYOUTS = [
    {"name": "2pt_rhip_lelbow", "active_imu_id": [1, 4]},
]
# IMUPoser and DIP-IMU provide all six model slots, except that slot 3 is a
# duplicate of the single head sensor. Match showo_base.yaml's evaluation rows
# and never score that duplicated slot.
REAL_IMU_LAYOUTS = [
    {"name": "3pt_lhip_lear_lelbow", "active_imu_id": [0, 2, 4]},
    {"name": "3pt_rhip_lear_relbow", "active_imu_id": [1, 2, 5]},
    {"name": "5pt_full", "active_imu_id": [0, 1, 2, 4, 5]},
]

BUILTIN_DATASETS: dict[str, dict[str, Any]] = {
    "humanml": {
        "root": "data/processed/motionmillion/v1",
        "sources": ["MotionUnion"],
        "sample_id_regex": r"^MotionUnion/humanml/",
        "sample_ids_file": "evaluation/subsets/humanml_nomirror_test_seed42_n2000.txt",
    },
    "lingo": {
        "root": "data/processed/motionmillion/v1",
        "sources": ["LINGO"],
        "sample_id_regex": r"^LINGO/",
    },
    "imuposer": {
        "root": "data/processed/imuposer/v2",
        "sources": ["imuposer"],
        "config": "configs/showo_finetune_imuposer.yaml",
        "run_dir": "exp/showo_finetune_imuposer",
        "layouts": REAL_IMU_LAYOUTS,
        "contiguous_windows": True,
        "skip_text": True,
    },
    "dipimu": {
        "root": "data/processed/dipimu/v2",
        "sources": ["dipimu"],
        "config": "configs/showo_finetune_dipimu.yaml",
        "run_dir": "exp/showo_finetune_dipimu",
        "layouts": REAL_IMU_LAYOUTS,
        "contiguous_windows": True,
        "skip_text": True,
    },
    # The two calibrated multi-view meeting-room captures.  Their test split is
    # a temporal hold-out of the same sessions (2 s guard band), packed as clips
    # of 240-499 frames -- shorter than the 480 row -- so they are windowed
    # contiguously rather than scored once per clip.  No captions ship with
    # either variant, so text metrics are skipped by default.
    "ncsa_meeting_3pt": {
        "root": "data/processed/ncsa/meeting_room_3pt_mv",
        "sources": ["ncsa"],
        "config": "configs/showo_finetune_ncsa.yaml",
        "run_dir": "exp/showo_finetune_ncsa",
        "layouts": NCSA_3PT_LAYOUTS,
        "contiguous_windows": True,
        "skip_text": True,
    },
    "ncsa_meeting_2pt": {
        "root": "data/processed/ncsa/meeting_room_2pt_mv",
        "sources": ["ncsa"],
        "config": "configs/showo_finetune_ncsa.yaml",
        "run_dir": "exp/showo_finetune_ncsa",
        "layouts": NCSA_2PT_LAYOUTS,
        "contiguous_windows": True,
        "skip_text": True,
    },
}


def _parse_layout(value: str) -> dict[str, Any]:
    """Parse a ``--layout name=0,2,4`` argument into an eval_imu_configs entry.

    ``name=none`` activates no sensor at all, so every IMU window stays the mask
    embedding.  The motion-input ablations (m2t / m2s / m2ts) read clean GT
    motion and never consume an inertial signal -- their profiles set
    ``eval_imu_configs: null`` for exactly this reason -- so a 3- or 5-point row
    would only feed them sensor tokens they never saw in training.  run.py takes
    ``invalid_imu_id`` as the complement of ``active_imu_id``, and names this
    layout ``default`` when the profile leaves eval_imu_configs unset, hence the
    conventional spelling ``--layout default=none``.
    """
    name, separator, ids = value.partition("=")
    name = name.strip()
    if not separator or not name:
        raise ValueError(f"--layout must be NAME=ID[,ID...] or NAME=none, got {value!r}")
    if ids.strip().lower() == "none":
        return {"name": name, "invalid_imu_id": list(range(NUM_IMU_SENSORS))}
    try:
        active = [int(part) for part in ids.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError(f"--layout {value!r} has a non-integer sensor id") from error
    if not active or len(set(active)) != len(active) or any(x < 0 for x in active):
        raise ValueError(f"--layout {value!r} needs distinct non-negative sensor ids")
    return {"name": name, "active_imu_id": active}


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _latest_checkpoint(run_dir: Path) -> tuple[Path, int]:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and (path / "unwrapped_model" / "pytorch_model.bin").is_file():
            candidates.append((int(match.group(1)), path.resolve()))
    if not candidates:
        raise FileNotFoundError(f"No complete checkpoint-* found below {run_dir}")
    step, checkpoint = max(candidates)
    return checkpoint, step


def _checkpoint(path: str | None, run_dir: Path) -> tuple[Path, int]:
    if path is None:
        return _latest_checkpoint(run_dir)
    checkpoint = _absolute(path)
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
    if match is None or not (checkpoint / "unwrapped_model" / "pytorch_model.bin").is_file():
        raise ValueError(f"Not a complete checkpoint-<step>: {checkpoint}")
    return checkpoint, int(match.group(1))


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not safe:
        raise ValueError(f"Dataset name has no safe path characters: {value!r}")
    return safe.lower()


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_sample_ids(path: Path) -> list[str]:
    sample_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not sample_ids:
        raise ValueError(f"Sample-id file is empty: {path}")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"Sample-id file contains duplicate ids: {path}")
    return sample_ids


def _check_gpus(gpus: list[str], allow_busy: bool) -> None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("warning: nvidia-smi unavailable; GPU occupancy was not checked", flush=True)
        return
    used = {}
    for line in output.splitlines():
        index, memory = (part.strip() for part in line.split(",", 1))
        used[index] = int(memory)
    unknown = [gpu for gpu in gpus if gpu not in used]
    if unknown:
        raise ValueError(f"Unknown GPU ids: {unknown}; available ids are {sorted(used)}")
    busy = {gpu: used[gpu] for gpu in gpus if used[gpu] > 1024}
    if busy and not allow_busy:
        details = ", ".join(f"GPU {gpu}: {memory} MiB" for gpu, memory in busy.items())
        raise RuntimeError(
            f"Requested GPU(s) are already in use ({details}). Choose free devices or "
            "pass --allow-busy-gpus to share them explicitly."
        )
    print(
        "GPU memory before launch: "
        + ", ".join(f"{gpu}={used[gpu]} MiB" for gpu in gpus),
        flush=True,
    )


def _prepare_checkpoint_link(job_dir: Path, checkpoint: Path) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    link = job_dir / checkpoint.name
    if link.is_symlink():
        if link.resolve() != checkpoint:
            raise RuntimeError(f"Existing checkpoint link points elsewhere: {link}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    link.symlink_to(checkpoint)


def _pass_dirs(
    job_dir: Path, step: int, frame: int, dataset_slug: str, layouts: list[dict[str, Any]]
) -> list[Path]:
    root = (
        job_dir
        / "evaluation/full"
        / f"step-{step:06d}"
        / f"frames-{frame:03d}"
        / "datasets"
        / dataset_slug
        / "metrics"
    )
    return [root / layout["name"] for layout in layouts]


def _inference_complete(
    job_dir: Path, step: int, frame: int, dataset_slug: str, layouts: list[dict[str, Any]]
) -> bool:
    for pass_dir in _pass_dirs(job_dir, step, frame, dataset_slug, layouts):
        summary = pass_dir / "summary.json"
        if not summary.is_file():
            return False
        payload = json.loads(summary.read_text(encoding="utf-8"))
        count = int(payload.get("evaluated_samples", 0))
        if count <= 0:
            return False
        # A text-only pass (model.supervise == [text]) decodes no motion and no
        # objects, so it writes the readable per-sample dumps but never a .npy;
        # requiring one here would make every rerun redo the model pass.
        if not payload.get("text_only", False):
            if len(list((pass_dir / "per_sequence").glob("*.npy"))) != count:
                return False
        record_count = 0
        for path in pass_dir.glob("records-*.jsonl"):
            with path.open(encoding="utf-8") as handle:
                record_count += sum(1 for line in handle if line.strip())
        if record_count != count:
            return False
    return True


def _run_logged(command: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} ===\n")
        log.write(shlex.join(command) + "\n")
        log.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _run_frame(
    *,
    frame: int,
    gpu: str,
    port: int,
    job_dir: Path,
    checkpoint: Path,
    step: int,
    config: Path,
    dataset_entry: dict[str, Any],
    dataset_slug: str,
    layouts: list[dict[str, Any]],
    min_clip_frames: int,
    lexical_workers: int,
    skip_text: bool,
    skip_bert: bool,
    force_postprocess: bool,
    overrides: list[str],
) -> Path:
    _prepare_checkpoint_link(job_dir, checkpoint)
    env = os.environ.copy()
    env.update(
        {
            "TRAIN_GPU_ID": str(gpu),
            "MAIN_PROCESS_PORT": str(port),
            "FULL_DIR": str(job_dir),
            "RESUME_FULL": "1",
            # launch_pretrain validates the resume lifecycle before it reads
            # command-line overrides.  Let a completed final checkpoint pass
            # that validation for this evaluation-only child process.
            "FULL_STEPS": str(step + 1),
        }
    )
    step_dir = job_dir / "evaluation/full" / f"step-{step:06d}"
    if not _inference_complete(job_dir, step, frame, dataset_slug, layouts):
        command = [
            sys.executable,
            "scripts/launch_pretrain.py",
            str(config),
            "--",
            "experiment.full_eval_only=True",
            "experiment.full_eval_every=1",
            # The unified launcher validates a normal resume by requiring the
            # configured training horizon to be strictly after the checkpoint.
            # A benchmark is evaluation-only, so the final checkpoint is a
            # valid (and common) target; raise the temporary horizon by one
            # solely to satisfy that lifecycle check.  run.py exits after the
            # full evaluation and never takes this extra optimizer step.
            f"training.max_train_steps={step + 1}",
            f"experiment.full_eval_frames=[{frame}]",
            "experiment.full_eval_datasets=" + json.dumps([dataset_entry], separators=(",", ":")),
            "experiment.eval_imu_configs=" + json.dumps(layouts, separators=(",", ":")),
            f"experiment.full_eval_min_clip_frames={min_clip_frames}",
            "experiment.full_eval_dump_records=True",
            "experiment.full_eval_save_samples=True",
            "experiment.full_eval_rerun.enabled=False",
            # Profile-specific overrides come last so they win over the
            # benchmark defaults above (an eval-only checkpoint copied without
            # optimizer.bin needs experiment.load_without_optimizer=True; a
            # cascade pass needs experiment.full_eval_motion_source).
            *overrides,
        ]
        print(f"[{dataset_entry['name']} f{frame:03d}] inference on GPU {gpu}", flush=True)
        _run_logged(command, env, job_dir / "inference.log")
    else:
        print(f"[{dataset_entry['name']} f{frame:03d}] reusing complete inference", flush=True)

    command = [
        sys.executable,
        "-m",
        "evaluation.score_saved_outputs",
        str(step_dir),
        "--layouts",
        *(layout["name"] for layout in layouts),
        "--lexical-workers",
        str(lexical_workers),
        "--bert-device",
        "cuda:0",
    ]
    if skip_text:
        command.append("--skip-text")
    else:
        # Text scoring uses rewrite sidecars. Motion-only real-IMU datasets
        # deliberately ship none, so do not attempt to resolve them when text
        # metrics have been explicitly disabled.
        command.extend([
            "--dataset-root",
            str(dataset_entry["root"]),
            "--rewrite-split",
            "test",
        ])
    if skip_bert:
        command.append("--skip-bert")
    if force_postprocess:
        command.append("--force")
    post_env = env.copy()
    post_env.pop("TRAIN_GPU_ID", None)
    post_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[{dataset_entry['name']} f{frame:03d}] offline metrics on GPU {gpu}", flush=True)
    _run_logged(command, post_env, job_dir / "postprocess.log")
    return step_dir


def _collect_results(
    output_dir: Path,
    jobs: list[tuple[int, Path]],
    dataset_name: str,
    step: int,
    dataset_slug: str,
    layouts: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for frame, job_dir in sorted(jobs):
        for pass_dir in _pass_dirs(job_dir, step, frame, dataset_slug, layouts):
            base = json.loads((pass_dir / "summary.json").read_text(encoding="utf-8"))
            row: dict[str, Any] = {
                "dataset": dataset_name,
                "step": step,
                "frames": frame,
                "layout": pass_dir.name,
                "imu_points": base.get("imu_point_count"),
                "samples": base.get("evaluated_samples"),
            }
            motion_path = pass_dir / "canonical_motion_metrics.json"
            if motion_path.is_file():
                motion = json.loads(motion_path.read_text(encoding="utf-8"))["summary"]
                row.update({name: value["mean"] for name, value in motion.items()})
            text_path = pass_dir / "canonical_text_metrics.json"
            if text_path.is_file():
                text = json.loads(text_path.read_text(encoding="utf-8"))["summary"]
                row.update({name: value["mean"] for name, value in text.items()})
            rows.append(row)
    columns = [
        "dataset", "step", "frames", "layout", "imu_points", "samples",
        "MPJPE_mm", "PA_MPJPE_mm", "MPJRE_deg", "MPJVE_mm", "MTE_mm",
        "BLEU_1_pct", "BLEU_4_pct", "ROUGE_L_pct", "CIDEr_pct",
        "BERTScore_F1_pct",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_json_atomic(output_dir / "results.json", rows)


def _write_evaluated_sample_manifest(
    output_dir: Path,
    jobs: list[tuple[int, Path]],
    dataset_name: str,
    step: int,
    dataset_slug: str,
    layouts: list[dict[str, Any]],
) -> None:
    """Persist the exact stream consumed by every completed benchmark pass.

    The packed test stream is deterministic, but recording its ids makes a
    later model comparison independent of shard ordering or future dataset
    repacks. Layouts at the same frame count must contain the same ordered ids;
    contiguous windowing deliberately yields a different stream for 60 vs 480.
    """

    per_frame: dict[str, dict[str, Any]] = {}
    for frame, job_dir in sorted(jobs):
        reference_ids: list[str] | None = None
        for pass_dir in _pass_dirs(job_dir, step, frame, dataset_slug, layouts):
            summary = json.loads((pass_dir / "summary.json").read_text(encoding="utf-8"))
            sample_ids: list[str] = []
            for record_path in sorted(pass_dir.glob("records-*.jsonl")):
                with record_path.open(encoding="utf-8") as handle:
                    sample_ids.extend(
                        str(json.loads(line)["sample_id"])
                        for line in handle
                        if line.strip()
                    )
            if len(sample_ids) != int(summary["evaluated_samples"]):
                raise RuntimeError(
                    f"Cannot create sample manifest: {pass_dir} has "
                    f"{len(sample_ids)} records for {summary['evaluated_samples']} samples"
                )
            if len(set(sample_ids)) != len(sample_ids):
                raise RuntimeError(f"Cannot create sample manifest: duplicate ids in {pass_dir}")
            if reference_ids is None:
                reference_ids = sample_ids
            elif sample_ids != reference_ids:
                raise RuntimeError(
                    "Layouts selected different sample streams; refusing to "
                    f"publish a reproducibility manifest ({pass_dir})"
                )
        if not reference_ids:
            raise RuntimeError(
                f"Cannot create sample manifest from empty {frame}-frame benchmark"
            )
        ids_path = output_dir / f"evaluated_sample_ids_frames-{frame:03d}.txt"
        temporary = ids_path.with_suffix(ids_path.suffix + ".tmp")
        temporary.write_text("\n".join(reference_ids) + "\n", encoding="utf-8")
        temporary.replace(ids_path)
        per_frame[str(frame)] = {
            "count": len(reference_ids),
            "ordered_ids_file": ids_path.name,
            "sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
        }
    if not per_frame:
        raise RuntimeError("Cannot create sample manifest from an empty benchmark")
    _write_json_atomic(
        output_dir / "evaluated_sample_manifest.json",
        {
            "schema_version": 2,
            "dataset": dataset_name,
            "step": step,
            "frames": per_frame,
            "selection": (
                "Deterministic test-stream order after source/regex/rewrite/length filters. "
                "For contiguous-window evaluation, each frame length has its own "
                "non-overlapping window list; reuse matching-frame lists for comparisons."
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Display name, e.g. HumanML")
    parser.add_argument("--root", help="Packed dataset root containing wds/manifest.json")
    parser.add_argument("--source", dest="sources", action="append", help="Allowed top-level id prefix; repeatable")
    parser.add_argument("--sample-id-regex", help="Regex searched against the complete sample id")
    parser.add_argument(
        "--sample-ids-file",
        help="Exact reusable sample-id allow-list (one full id per line; # comments allowed)",
    )
    parser.add_argument("--sample-num", type=int, help="Maximum matching test samples; default is all")
    parser.add_argument(
        "--run-dir",
        help="Run directory holding checkpoint-*; default is the dataset profile's, "
             "else exp/showo_pretrain_full",
    )
    parser.add_argument("--checkpoint", help="Exact checkpoint-<step>; default latest in --run-dir")
    parser.add_argument(
        "--config",
        help="Profile YAML; default is the dataset profile's, else "
             "configs/showo_pretrain_full.yaml",
    )
    parser.add_argument(
        "--layout",
        dest="layouts",
        action="append",
        metavar="NAME=ID[,ID...]",
        help="Replace the dataset's layout list; repeatable, e.g. 2pt_rhip_lelbow=1,4. "
             "NAME=none activates no sensor (every IMU window is the mask embedding), "
             "which is the only meaningful row for the GT-motion-input ablations: "
             "--layout default=none",
    )
    parser.add_argument("--output-dir", help="Default: <run>/evaluation/benchmarks/<dataset>/step-N")
    parser.add_argument("--frames", nargs="+", type=int, default=[60, 480])
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument(
        "--allow-busy-gpus",
        action="store_true",
        help="Allow launch when a requested GPU already uses more than 1 GiB",
    )
    parser.add_argument("--port-base", type=int, default=1930)
    parser.add_argument("--min-clip-frames", type=int, default=1)
    parser.add_argument(
        "--contiguous-windows",
        action="store_true",
        help=(
            "Split each selected test sequence into adjacent non-overlapping windows "
            "for every --frames length."
        ),
    )
    parser.add_argument(
        "--contiguous-window-min-frames",
        type=int,
        default=24,
        help="Drop a final contiguous tail shorter than this many frames (default: 24).",
    )
    parser.add_argument("--lexical-workers", type=int, default=8)
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--force-postprocess", action="store_true")
    parser.add_argument(
        "--override",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "Extra run.py config override appended after the benchmark's own; "
            "repeatable. Used for eval-only checkpoints copied without "
            "optimizer.bin (experiment.load_without_optimizer=True) and for "
            "cascade passes (experiment.full_eval_motion_source=<cache.npz>)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.frames or len(set(args.frames)) != len(args.frames) or any(frame <= 0 for frame in args.frames):
        parser.error("--frames must contain distinct positive integers")
    if not args.gpus:
        parser.error("--gpus cannot be empty")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must list distinct device ids")
    if not 0 < args.min_clip_frames <= min(args.frames):
        parser.error("--min-clip-frames must be in [1, min(--frames)]")
    if args.contiguous_window_min_frames <= 0:
        parser.error("--contiguous-window-min-frames must be positive")
    if args.sample_num is not None and args.sample_num <= 0:
        parser.error("--sample-num must be positive")
    if args.lexical_workers < 1:
        parser.error("--lexical-workers must be at least 1")

    profile = BUILTIN_DATASETS.get(args.dataset.lower(), {})
    root_value = args.root or profile.get("root")
    if root_value is None:
        parser.error("--root is required for datasets without a built-in profile")
    dataset_root = _absolute(root_value)
    if not (dataset_root / "wds/manifest.json").is_file():
        parser.error(f"No wds/manifest.json under {dataset_root}")
    if args.layouts:
        try:
            layouts = [_parse_layout(value) for value in args.layouts]
        except ValueError as error:
            parser.error(str(error))
        names = [layout["name"] for layout in layouts]
        if len(set(names)) != len(names):
            parser.error(f"--layout names must be distinct, got {names}")
    else:
        layouts = profile.get("layouts") or DEFAULT_LAYOUTS
    # A real-IMU capture leaves its unworn slots at zero acceleration and
    # identity orientation, so a layout over them is not a harder row, it is a
    # meaningless one.  The profile carries the physically valid list; only an
    # explicit --layout may depart from it.
    sources = args.sources or profile.get("sources") or [args.dataset]
    contiguous_windows = bool(args.contiguous_windows or profile.get("contiguous_windows", False))
    skip_text = bool(args.skip_text or profile.get("skip_text", False))
    sample_id_regex = args.sample_id_regex or profile.get("sample_id_regex")
    if sample_id_regex:
        try:
            re.compile(sample_id_regex)
        except re.error as error:
            parser.error(f"Invalid --sample-id-regex: {error}")
    sample_ids_value = args.sample_ids_file or profile.get("sample_ids_file")
    sample_ids_path = _absolute(sample_ids_value) if sample_ids_value else None
    sample_ids = None
    if sample_ids_path is not None:
        if not sample_ids_path.is_file():
            parser.error(f"Sample-id file does not exist: {sample_ids_path}")
        try:
            sample_ids = _read_sample_ids(sample_ids_path)
        except ValueError as error:
            parser.error(str(error))
        if args.sample_num is not None and args.sample_num != len(sample_ids):
            parser.error(
                f"--sample-num ({args.sample_num}) must match the "
                f"{len(sample_ids)} ids in --sample-ids-file"
            )
        bad_sources = [
            sample_id for sample_id in sample_ids
            if sample_id.split("/", 1)[0] not in sources
        ]
        if bad_sources:
            parser.error(
                f"Sample-id file contains ids outside --source {sources}: "
                f"{bad_sources[:3]}"
            )
        if sample_id_regex:
            pattern = re.compile(sample_id_regex)
            bad_regex_ids = [sample_id for sample_id in sample_ids if pattern.search(sample_id) is None]
            if bad_regex_ids:
                parser.error(
                    f"Sample-id file contains ids outside --sample-id-regex: "
                    f"{bad_regex_ids[:3]}"
                )

    run_dir = _absolute(args.run_dir or profile.get("run_dir") or "exp/showo_pretrain_full")
    checkpoint, step = _checkpoint(args.checkpoint, run_dir)
    config = _absolute(
        args.config or profile.get("config") or "configs/showo_pretrain_full.yaml"
    )
    if not config.is_file():
        parser.error(f"Config does not exist: {config}")
    dataset_slug = _safe_name(profile.get("output_slug", args.dataset))
    output_dir = (
        _absolute(args.output_dir)
        if args.output_dir
        else run_dir / "evaluation/benchmarks" / dataset_slug / f"step-{step:06d}"
    )
    dataset_entry: dict[str, Any] = {
        "name": args.dataset,
        "root": str(dataset_root),
        "sample_num": len(sample_ids) if sample_ids is not None else args.sample_num,
        "sources": list(sources),
        "contiguous_windows": contiguous_windows,
        "contiguous_window_min_frames": int(args.contiguous_window_min_frames),
    }
    if sample_id_regex:
        dataset_entry["sample_id_regex"] = sample_id_regex
    if sample_ids_path is not None:
        dataset_entry["sample_ids_file"] = str(sample_ids_path)
        dataset_entry["sample_ids_sha256"] = hashlib.sha256(
            sample_ids_path.read_bytes()
        ).hexdigest()
    benchmark_config = {
        "schema_version": 1,
        "dataset": dataset_entry,
        "checkpoint": str(checkpoint),
        "step": step,
        "config": str(config),
        "frames": args.frames,
        "min_clip_frames": args.min_clip_frames,
        "contiguous_windows": contiguous_windows,
        "contiguous_window_min_frames": int(args.contiguous_window_min_frames),
        "layouts": layouts,
        "save_samples": True,
        "dump_records": True,
        "text_references": "qwen3_0.6B_rewrite_v2 (raw WDS captions forbidden)",
    }

    _check_gpus(args.gpus, args.allow_busy_gpus)

    jobs = [
        (frame, output_dir / "jobs" / f"frames-{frame:03d}")
        for frame in args.frames
    ]
    if args.dry_run:
        print(json.dumps(benchmark_config, indent=2))
        for index, (frame, job_dir) in enumerate(jobs):
            print(f"frame={frame} gpu={args.gpus[index % len(args.gpus)]} job_dir={job_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "benchmark_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != benchmark_config:
            raise RuntimeError(
                f"Existing benchmark parameters differ at {config_path}; "
                "choose a new --output-dir instead of mixing results"
            )
    else:
        _write_json_atomic(config_path, benchmark_config)

    queues: list[list[tuple[int, Path]]] = [[] for _ in args.gpus]
    for index, job in enumerate(jobs):
        queues[index % len(args.gpus)].append(job)

    def run_queue(index: int) -> None:
        gpu = args.gpus[index]
        for offset, (frame, job_dir) in enumerate(queues[index]):
            _run_frame(
                frame=frame,
                gpu=gpu,
                port=args.port_base + index * 20 + offset,
                job_dir=job_dir,
                checkpoint=checkpoint,
                step=step,
                config=config,
                dataset_entry=dataset_entry,
                dataset_slug=dataset_slug,
                layouts=layouts,
                min_clip_frames=args.min_clip_frames,
                lexical_workers=args.lexical_workers,
                skip_text=skip_text,
                skip_bert=args.skip_bert,
                force_postprocess=args.force_postprocess,
                overrides=list(args.overrides or []),
            )

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(run_queue, index) for index in range(len(args.gpus)) if queues[index]]
        for future in futures:
            future.result()
    _collect_results(output_dir, jobs, args.dataset, step, dataset_slug, layouts)
    _write_evaluated_sample_manifest(output_dir, jobs, args.dataset, step, dataset_slug, layouts)
    print(f"complete: {output_dir}")
    print(f"summary:  {output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
