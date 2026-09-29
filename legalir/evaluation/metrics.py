"""Metric Recall@5/AllGold@5/MRR@5 và lưu dự đoán eval.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

from typing import Any, Sequence

from legalir.core.config import Paths
from legalir.core.utils import atomic_json, log, write_jsonl


def task_metrics(rankings: Sequence[Sequence[str]], records: Sequence[dict[str, Any]], k: int = 5) -> dict[str, float]:
    if len(rankings) != len(records):
        raise ValueError(
            f"Metric input mismatch: {len(rankings):,} rankings for {len(records):,} records"
        )
    recalls, precisions, reciprocal = [], [], []; exact = 0
    for ranking, record in zip(rankings, records):
        predicted = list(dict.fromkeys(map(str, ranking[:k]))); gold = set(map(str, record["answers"]))
        if not gold:
            raise ValueError(f'Query {record.get("qid", "<unknown>")} has no gold answer')
        hits = len(gold & set(predicted))
        recalls.append(hits / len(gold)); precisions.append(hits / len(predicted) if predicted else 0.0)
        ranks = [index + 1 for index, cid in enumerate(predicted) if cid in gold]
        reciprocal.append(1.0 / min(ranks) if ranks else 0.0); exact += int(gold.issubset(predicted))
    count = max(1, len(records))
    return {f"Recall@{k}": sum(recalls) / count, f"Precision@{k}": sum(precisions) / count,
            f"MRR@{k}": sum(reciprocal) / count, f"AllGold@{k}": exact / count}


def save_eval_prediction_artifacts(
    paths: Paths, records: Sequence[dict[str, Any]], rankings: Sequence[Sequence[str]],
    valid_cids: set[str],
) -> int:
    """Persist selected validation predictions and actionable top-5 errors."""
    if len(rankings) != len(records):
        raise ValueError(
            f"Eval prediction count mismatch: {len(rankings):,} rankings for {len(records):,} records"
        )
    submission: dict[str, dict[str, list[str]]] = {}
    wrong: list[dict[str, Any]] = []
    for record, ranking in zip(records, rankings):
        qid = str(record["qid"])
        full_ranking = list(dict.fromkeys(map(str, ranking)))
        predicted = full_ranking[:5]
        gold_answers = list(dict.fromkeys(map(str, record["answers"])))
        gold = set(gold_answers)
        predicted_set = set(predicted)
        missing_gold = [cid for cid in gold_answers if cid not in predicted_set]
        submission[qid] = {"answer": predicted}
        if not missing_gold:
            continue
        rank_by_cid = {cid: index + 1 for index, cid in enumerate(full_ranking)}
        hits = [cid for cid in predicted if cid in gold]
        wrong.append({
            "qid": qid,
            "question": str(record["question"]),
            "answer": predicted,
            "gold_answers": gold_answers,
            "missing_gold": missing_gold,
            "retrieved_gold": hits,
            "non_gold_top5": [cid for cid in predicted if cid not in gold],
            "gold_candidate_ranks": {cid: rank_by_cid.get(cid) for cid in gold_answers},
            "recall_at_5": len(hits) / max(1, len(gold)),
        })
    validate_submission(submission, records, valid_cids)
    atomic_json(paths.eval_submission_json, submission)
    write_jsonl(paths.eval_wrong_predictions, wrong)
    log(
        f"Saved eval submission ({len(records):,} queries) -> {paths.eval_submission_json}; "
        f"top-5 errors={len(wrong):,} -> {paths.eval_wrong_predictions}"
    )
    return len(wrong)


def validate_submission(submission: Any, test: Sequence[dict[str, Any]], valid_cids: set[str]) -> None:
    if not isinstance(submission, dict): raise TypeError("submission.json must be a JSON object")
    expected = {row["qid"] for row in test}
    if set(submission) != expected:
        missing, extra = expected - set(submission), set(submission) - expected
        raise ValueError(f"Submission query IDs mismatch: missing={len(missing)}, extra={len(extra)}")
    for qid, value in submission.items():
        answers = value.get("answer") if isinstance(value, dict) else None
        if not isinstance(answers, list) or not 1 <= len(answers) <= 5: raise ValueError(f"{qid}: answer must contain 1..5 IDs")
        if any(not isinstance(cid, str) for cid in answers): raise TypeError(f"{qid}: document IDs must be strings")
        if len(answers) != len(set(answers)): raise ValueError(f"{qid}: duplicate document IDs")
        unknown = set(answers) - valid_cids
        if unknown: raise ValueError(f"{qid}: unknown document IDs {sorted(unknown)}")
