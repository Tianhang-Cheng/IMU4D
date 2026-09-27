"""Compute canonical motion and caption metrics from saved full-eval outputs.

The expensive model pass must have been run with
``experiment.full_eval_save_samples=True`` and
``experiment.full_eval_dump_records=True``.  This command never runs the model;
it consumes ``per_sequence/*.npy`` and ``records-*.jsonl`` in-place and writes
reusable metric artifacts next to them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from dataset_process.wds_pipeline.wds_loader import load_rewrite_texts
from evaluation.motion_adapter import (
    evaluate_saved_motion_directory,
    motion_result_is_current,
)
from metric.core.common import write_json
from metric.text.evaluator import evaluate_text_records


def _pass_dirs(root: Path, layouts: set[str] | None) -> Iterable[Path]:
    candidates = [root] if (root / "summary.json").is_file() else [
        path.parent for path in root.rglob("summary.json")
    ]
    for path in sorted(set(candidates)):
        if layouts is None or path.name in layouts:
            yield path


def _load_records(pass_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(pass_dir.glob("records-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def _write_text_input(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def score_pass(
    pass_dir: Path,
    *,
    lexical_workers: int,
    bert_device: str,
    include_bert: bool,
    include_text: bool,
    include_motion: bool,
    force: bool,
    rewritten_texts: dict[str, list[str]] | None,
    text_output_name: str = "canonical_text_metrics.json",
    reference_source: str | None = None,
) -> None:
    summary = json.loads((pass_dir / "summary.json").read_text(encoding="utf-8"))
    frames = int(summary["eval_frames"])
    sample_count = int(summary["evaluated_samples"])
    motion_output = pass_dir / "canonical_motion_metrics.json"
    per_sequence = pass_dir / "per_sequence"
    npy_count = len(list(per_sequence.glob("*.npy"))) if per_sequence.is_dir() else 0
    if include_motion and npy_count:
        if npy_count != sample_count:
            raise RuntimeError(
                f"{pass_dir}: summary has {sample_count} samples but per_sequence "
                f"contains {npy_count} .npy files"
            )
        if force or not motion_result_is_current(motion_output):
            payload = evaluate_saved_motion_directory(
                per_sequence, frames, verbose=False
            )
            write_json(payload, motion_output)
            print(f"motion: {motion_output}")
    elif include_motion:
        print(f"motion: skipped {pass_dir} (no per_sequence/*.npy)")

    if not include_text:
        return
    text_output = pass_dir / text_output_name
    records = _load_records(pass_dir)
    if not records:
        print(f"text: skipped {pass_dir} (no records-*.jsonl)")
        return
    if len(records) != sample_count:
        raise RuntimeError(
            f"{pass_dir}: summary has {sample_count} samples but records contain "
            f"{len(records)} rows"
        )
    text_rows = []
    missing_rewrite_ids = []
    for index, record in enumerate(records):
        prediction = record.get("pred_text")
        sample_id = str(record.get("sample_id", index))
        references = (
            rewritten_texts.get(sample_id)
            if rewritten_texts is not None
            else record.get("gt_texts")
        )
        if rewritten_texts is not None and not references:
            missing_rewrite_ids.append(sample_id)
            continue
        if not isinstance(prediction, str):
            raise ValueError(f"{pass_dir}: record {index} has no pred_text string")
        if isinstance(references, str):
            references = [references]
        if not isinstance(references, list) or not references:
            raise ValueError(f"{pass_dir}: record {index} has no gt_texts")
        text_rows.append(
            {
                "id": sample_id,
                "prediction": prediction,
                "references": [str(reference) for reference in references],
            }
        )
    if rewritten_texts is not None:
        print(
            f"text references: {len(text_rows)} rewrite(s), "
            f"{len(missing_rewrite_ids)} missing; raw record captions ignored"
        )
    if not text_rows:
        print(f"text: skipped {pass_dir} (no rewritten references)")
        return
    _write_text_input(pass_dir / "text_eval_input.jsonl", text_rows)
    needs_text = force or not text_output.is_file()
    if text_output.is_file() and reference_source is not None:
        existing_text = json.loads(text_output.read_text(encoding="utf-8"))
        needs_text |= existing_text.get("protocol", {}).get("reference_source") != reference_source
    if include_bert and text_output.is_file() and not needs_text:
        existing_text = json.loads(text_output.read_text(encoding="utf-8"))
        needs_text = "BERTScore_F1_pct" not in existing_text.get("summary", {})
    if needs_text:
        result = evaluate_text_records(
            text_rows,
            include_bert=include_bert,
            bert_device=bert_device,
            lexical_workers=lexical_workers,
        )
        result["protocol"]["reference_source"] = reference_source
        write_json(result, text_output)
        print(f"text: {text_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Step, frame, dataset, or metric-pass directory")
    parser.add_argument("--layouts", nargs="+", help="Only score these layout directory names")
    parser.add_argument("--lexical-workers", type=int, default=8)
    parser.add_argument("--bert-device", default="cuda:0")
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument(
        "--skip-motion",
        action="store_true",
        help="Only score saved text records; do not recompute per-sequence motion metrics",
    )
    parser.add_argument("--force", action="store_true", help="Recompute existing canonical metrics")
    parser.add_argument(
        "--dataset-root",
        help="Packed dataset root; loads rewrite sidecars instead of record GT captions",
    )
    parser.add_argument("--rewrite-split", default="test")
    parser.add_argument(
        "--rewrite-root",
        help="Rewrite sidecar root to score against (e.g. data/processed/hiphi/"
        "qwen3_0.6B_rewrite_v2); default is the sibling qwen3_0.6B_rewrite_v2",
    )
    parser.add_argument(
        "--text-output-name",
        default="canonical_text_metrics.json",
        help="Text metrics filename inside each pass; use another name to keep "
        "the canonical metrics file when scoring against other references",
    )
    args = parser.parse_args()

    if args.lexical_workers < 1:
        parser.error("--lexical-workers must be at least 1")
    pass_dirs = list(_pass_dirs(args.root, set(args.layouts) if args.layouts else None))
    if not pass_dirs:
        parser.error(f"No summary.json metric passes found below {args.root}")
    rewritten_texts = None
    if args.dataset_root:
        rewritten_texts = load_rewrite_texts(
            [args.dataset_root], args.rewrite_split, enabled=True,
            rewrite_roots=[args.rewrite_root] if args.rewrite_root else None,
        )
        if not rewritten_texts:
            parser.error(
                f"No valid rewrite captions found for {args.dataset_root} "
                f"split {args.rewrite_split}"
            )
        print(
            f"loaded {len(rewritten_texts)} rewrite captions from "
            f"{args.rewrite_root or args.dataset_root} ({args.rewrite_split})"
        )
    reference_cache = {}
    for pass_dir in pass_dirs:
        pass_references = rewritten_texts
        reference_source = str(Path(args.rewrite_root or (
            Path(args.dataset_root).parent / "qwen3_0.6B_rewrite_v2"
        )).resolve()) if args.dataset_root else None
        if not args.skip_text and not args.dataset_root:
            dataset_dir = next((p for p in pass_dir.resolve().parents if p.parent.name == "datasets"), None)
            if dataset_dir is None:
                parser.error("Cannot infer dataset; provide --dataset-root to load v2 evaluation references")
            dataset_root = Path(__file__).resolve().parents[1] / "data/processed" / dataset_dir.name / "v1"
            rewrite_root = Path(args.rewrite_root) if args.rewrite_root else dataset_root.parent / "qwen3_0.6B_rewrite_v2"
            reference_source = str(rewrite_root.resolve())
            if reference_source not in reference_cache:
                reference_cache[reference_source] = load_rewrite_texts(
                    [str(dataset_root)], args.rewrite_split, enabled=True,
                    rewrite_roots=[str(rewrite_root)],
                )
            pass_references = reference_cache[reference_source]
            if not pass_references:
                parser.error(f"No valid evaluation rewrite captions found at {rewrite_root}; provide --dataset-root")
        score_pass(
            pass_dir,
            lexical_workers=args.lexical_workers,
            bert_device=args.bert_device,
            include_bert=not args.skip_bert,
            include_text=not args.skip_text,
            include_motion=not args.skip_motion,
            force=args.force,
            rewritten_texts=pass_references,
            text_output_name=args.text_output_name,
            reference_source=reference_source,
        )


if __name__ == "__main__":
    main()
