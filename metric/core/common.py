"""Shared I/O, validation, and aggregation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def as_float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object on {path}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def summarize_records(
    records: Iterable[Mapping[str, float]],
) -> dict[str, dict[str, Any]]:
    """Return mean/std/count/raw for every finite scalar metric.

    Missing metrics are allowed. This is useful for motion samples for which a
    baseline exports joints but not SMPL-X vertices, for example.
    """

    rows = list(records)
    keys = sorted({key for row in rows for key in row})
    summary: dict[str, dict[str, Any]] = {}
    for key in keys:
        values = np.asarray(
            [float(row[key]) for row in rows if key in row and np.isfinite(row[key])],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "count": int(values.size),
            "raw": values.tolist(),
        }
    return summary


def write_json(value: Any, path: str | Path | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    if path is None:
        print(text)
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"Saved metrics to {output}")


def result_document(
    task: str,
    per_sample: list[dict[str, Any]],
    metric_rows: list[Mapping[str, float]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": task,
        "protocol": dict(protocol),
        "summary": summarize_records(metric_rows),
        "per_sample": per_sample,
    }
