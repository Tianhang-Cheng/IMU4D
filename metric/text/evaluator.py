"""MotionGPT3-compatible caption metrics for IMU-to-MotionText."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.common import load_jsonl, result_document, write_json


def _nlg_evaluator() -> Any:
    try:
        # datasets==2.9.0 (required by nlg_metricverse==0.9.9) still imports
        # PyExtensionType, which newer PyArrow releases expose as ExtensionType.
        import pyarrow as pa

        if not hasattr(pa, "PyExtensionType"):
            pa.PyExtensionType = pa.ExtensionType
        from nlgmetricverse import NLGMetricverse, load_metric
    except ImportError as exc:
        raise RuntimeError(
            "Text metrics require nlg_metricverse==0.9.9. "
            "Install metric/text/requirements.txt first."
        ) from exc
    metrics = [
        load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
        load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
        load_metric("rouge"),
        load_metric("cider"),
    ]
    return NLGMetricverse(metrics)


def _lexical_pair_scores(evaluator: Any, prediction: str, reference: str) -> dict[str, float]:
    scores = evaluator(predictions=[prediction], references=[reference])
    return {
        "ROUGE_L_pct": 100.0 * float(scores["rouge"]["rougeL"]),
        "CIDEr_pct": 100.0 * float(scores["cider"]["score"]),
        "BLEU_1_pct": 100.0 * float(scores["bleu_1"]["score"]),
        "BLEU_4_pct": 100.0 * float(scores["bleu_4"]["score"]),
    }


_WORKER_EVALUATOR: Any | None = None


def _init_lexical_worker() -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = _nlg_evaluator()


def _lexical_pair_worker(pair: tuple[str, str]) -> dict[str, float]:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("Lexical metric worker was not initialized")
    return _lexical_pair_scores(_WORKER_EVALUATOR, *pair)


def _bert_pair_scores(
    predictions: list[str], references: list[str], device: str
) -> np.ndarray:
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore requires bert_score==0.3.13. "
            "Install metric/text/requirements.txt or pass --skip-bert."
        ) from exc
    _, _, f1 = bert_score(
        predictions,
        references,
        lang="en",
        rescale_with_baseline=True,
        idf=True,
        device=device,
        verbose=False,
    )
    return f1.detach().cpu().numpy().astype(np.float64) * 100.0


def evaluate_text_records(
    records: Sequence[Mapping[str, Any]],
    include_bert: bool = True,
    bert_device: str = "cpu",
    lexical_workers: int = 1,
) -> dict[str, Any]:
    """Evaluate captions, taking each metric's maximum over GT references.

    This follows the explicit multi-reference rule in ``latex/sec/X_supp.tex``.
    The lexical backend and BERTScore arguments match MotionGPT3's public
    implementation.
    """

    if lexical_workers < 1:
        raise ValueError("lexical_workers must be at least 1")
    normalized: list[tuple[str, str, list[str]]] = []
    flattened_predictions: list[str] = []
    flattened_references: list[str] = []
    flattened_owners: list[int] = []
    for index, record in enumerate(records):
        prediction = record.get("prediction", record.get("pred"))
        references = record.get("references", record.get("gt"))
        if not isinstance(prediction, str):
            raise ValueError(f"Text record {index} needs a string prediction")
        if isinstance(references, str):
            references = [references]
        if not isinstance(references, list) or not references or not all(
            isinstance(reference, str) for reference in references
        ):
            raise ValueError(f"Text record {index} needs one or more string references")
        normalized.append((str(record.get("id", index)), prediction, references))
        for reference in references:
            flattened_predictions.append(prediction)
            flattened_references.append(reference)
            flattened_owners.append(index)

    pairs = list(zip(flattened_predictions, flattened_references))
    if lexical_workers == 1:
        evaluator = _nlg_evaluator()
        flattened_scores = [
            _lexical_pair_scores(evaluator, prediction, reference)
            for prediction, reference in pairs
        ]
    else:
        chunksize = max(1, len(pairs) // (lexical_workers * 8))
        with ProcessPoolExecutor(
            max_workers=lexical_workers,
            initializer=_init_lexical_worker,
        ) as executor:
            flattened_scores = list(
                executor.map(_lexical_pair_worker, pairs, chunksize=chunksize)
            )

    scores_by_owner: dict[int, list[dict[str, float]]] = defaultdict(list)
    for owner, scores in zip(flattened_owners, flattened_scores):
        scores_by_owner[owner].append(scores)

    per_sample: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    for index, (sample_id, _prediction, references) in enumerate(normalized):
        scores_by_reference = scores_by_owner[index]
        metrics = {
            key: max(score[key] for score in scores_by_reference)
            for key in scores_by_reference[0]
        }
        metric_rows.append(metrics)
        per_sample.append(
            {
                "id": sample_id,
                "reference_count": len(references),
                "metrics": metrics,
            }
        )

    if include_bert:
        bert_values = _bert_pair_scores(
            flattened_predictions, flattened_references, bert_device
        )
        by_owner: dict[int, list[float]] = defaultdict(list)
        for owner, value in zip(flattened_owners, bert_values):
            by_owner[owner].append(float(value))
        for index, metrics in enumerate(metric_rows):
            metrics["BERTScore_F1_pct"] = max(by_owner[index])

    return result_document(
        "text_caption",
        per_sample,
        metric_rows,
        {
            "reference_policy": "compute against every reference and take the per-metric maximum",
            "scale": "all reported values multiplied by 100 to match paper tables",
            "lexical_backend": "nlg_metricverse==0.9.9 (MotionGPT3 settings)",
            "BERTScore": (
                "bert_score==0.3.13; lang=en; rescale_with_baseline=True; idf=True"
                if include_bert
                else "skipped"
            ),
            "aggregation": "sample mean and population std",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IMU4D text captions")
    parser.add_argument("--input", required=True, help="JSONL file with prediction and references")
    parser.add_argument("--output", help="Output JSON path; prints JSON when omitted")
    parser.add_argument("--skip-bert", action="store_true", help="Compute only BLEU/ROUGE-L/CIDEr")
    parser.add_argument("--bert-device", default="cpu", help="BERTScore device, e.g. cpu or cuda:0")
    parser.add_argument(
        "--lexical-workers",
        type=int,
        default=1,
        help="Worker processes for BLEU/ROUGE-L/CIDEr pair scoring",
    )
    args = parser.parse_args()
    result = evaluate_text_records(
        load_jsonl(args.input),
        include_bert=not args.skip_bert,
        bert_device=args.bert_device,
        lexical_workers=args.lexical_workers,
    )
    write_json(result, args.output)


if __name__ == "__main__":
    main()
