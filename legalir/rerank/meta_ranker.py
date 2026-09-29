"""Meta-ranker listwise trên đặc trưng hạng của các tín hiệu.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import math
from typing import Any, Sequence

from legalir.core.constants import META_FEATURE_VERSION
from legalir.core.config import Config
from legalir.core.utils import minmax
from legalir.evaluation.metrics import task_metrics
from legalir.evaluation.dense_eval import subset_values
from legalir.rerank.repair_fusion import REPAIR_SIGNALS, repair_signal_scores


META_SIGNALS = (*REPAIR_SIGNALS, "bm25")


def meta_ranker_feature_matrix(
    rows: Sequence[dict[str, Any]], evidence_k: int,
) -> tuple[Any, list[str]]:
    """Build query-relative candidate features for supervised top-5 repair.

    RRF only sees rank and forces every signal to have a non-negative global
    weight.  These features retain rank, normalized score, top-k agreement and
    cross-signal interactions.  BM25 is merely one learnable feature; the
    optimizer can give it zero or negative influence when lexical overlap is a
    distractor, which is common in this legal corpus.
    """
    import numpy as np

    count = len(rows)
    if count == 0:
        return np.empty((0, 0), dtype="float64"), []

    raw_by_signal: dict[str, list[float]] = {}
    for signal in REPAIR_SIGNALS:
        raw_by_signal[signal] = repair_signal_scores(rows, signal, evidence_k)
    raw_by_signal["bm25"] = [float(row.get("bm25", 0.0)) for row in rows]

    rank_scores: dict[str, list[float]] = {}
    normalized_scores: dict[str, list[float]] = {}
    reciprocal_scores: dict[str, list[float]] = {}
    ranks: dict[str, list[int]] = {}
    for signal in META_SIGNALS:
        raw = raw_by_signal[signal]
        finite = [value for value in raw if math.isfinite(value)]
        floor = (min(finite) - 1.0) if finite else -1.0
        safe = [value if math.isfinite(value) else floor for value in raw]
        normalized_scores[signal] = minmax(safe)
        order = sorted(range(count), key=lambda index: (-safe[index], str(rows[index]["cid"])))
        signal_ranks = [0] * count
        for rank, index in enumerate(order, start=1):
            signal_ranks[index] = rank
        ranks[signal] = signal_ranks
        rank_scores[signal] = [
            (count - rank) / max(1, count - 1) for rank in signal_ranks
        ]
        reciprocal_scores[signal] = [1.0 / rank for rank in signal_ranks]

    names: list[str] = []
    columns: list[list[float]] = []
    for signal in META_SIGNALS:
        for suffix, values in (
            ("rank", rank_scores[signal]),
            ("rr", reciprocal_scores[signal]),
            ("score", normalized_scores[signal]),
            ("top5", [float(rank <= 5) for rank in ranks[signal]]),
            ("top10", [float(rank <= 10) for rank in ranks[signal]]),
        ):
            names.append(f"{signal}_{suffix}")
            columns.append(values)

    semantic_rank = np.asarray([rank_scores[signal] for signal in REPAIR_SIGNALS], dtype="float64")
    dense_rank = semantic_rank[:3]
    rerank_rank = semantic_rank[3:]
    consensus_columns = {
        "semantic_rank_mean": semantic_rank.mean(axis=0),
        "semantic_rank_min": semantic_rank.min(axis=0),
        "semantic_rank_max": semantic_rank.max(axis=0),
        "dense_rank_mean": dense_rank.mean(axis=0),
        "rerank_rank_mean": rerank_rank.mean(axis=0),
        "semantic_top5_votes": np.asarray([
            sum(ranks[signal][index] <= 5 for signal in REPAIR_SIGNALS) / len(REPAIR_SIGNALS)
            for index in range(count)
        ]),
        "semantic_top10_votes": np.asarray([
            sum(ranks[signal][index] <= 10 for signal in REPAIR_SIGNALS) / len(REPAIR_SIGNALS)
            for index in range(count)
        ]),
    }
    for name, values in consensus_columns.items():
        names.append(name); columns.append(values.tolist())

    # Pairwise agreement lets a linear listwise model express gates such as
    # "promote only when both rerankers agree" without a fragile hand-tuned
    # threshold.  Include BM25 interactions separately so lexical evidence can
    # help only when a semantic voter supports it.
    for left in range(len(REPAIR_SIGNALS)):
        for right in range(left, len(REPAIR_SIGNALS)):
            names.append(f"agree_{REPAIR_SIGNALS[left]}_{REPAIR_SIGNALS[right]}")
            columns.append((semantic_rank[left] * semantic_rank[right]).tolist())
    bm25_rank = np.asarray(rank_scores["bm25"], dtype="float64")
    for name, values in (
        ("bm25_x_dense", bm25_rank * dense_rank.mean(axis=0)),
        ("bm25_x_rerank", bm25_rank * rerank_rank.mean(axis=0)),
        ("bm25_x_consensus", bm25_rank * semantic_rank.mean(axis=0)),
    ):
        names.append(name); columns.append(values.tolist())

    matrix = np.asarray(columns, dtype="float64").T
    return matrix, names


def fit_listwise_meta_ranker(
    rows_by_query: Sequence[Sequence[dict[str, Any]]],
    records: Sequence[dict[str, Any]], evidence_k: int,
    l2: float, seed: int,
) -> dict[str, Any]:
    """Fit a deterministic linear ListNet-style ranker with query softmax."""
    import numpy as np

    matrices: list[Any] = []
    targets: list[Any] = []
    feature_names: list[str] | None = None
    for rows, record in zip(rows_by_query, records):
        matrix, names = meta_ranker_feature_matrix(rows, evidence_k)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise AssertionError("Meta-ranker feature order changed between queries")
        gold = set(map(str, record["answers"]))
        positive = np.asarray([str(row["cid"]) in gold for row in rows], dtype="float64")
        if positive.sum() <= 0.0:
            continue
        positive /= positive.sum()
        matrices.append(matrix); targets.append(positive)
    if not matrices or feature_names is None:
        raise ValueError("No validation query has a gold document in the candidate pool")

    maximum = max(matrix.shape[0] for matrix in matrices)
    feature_count = matrices[0].shape[1]
    tensor = np.zeros((len(matrices), maximum, feature_count), dtype="float64")
    target = np.zeros((len(matrices), maximum), dtype="float64")
    mask = np.zeros((len(matrices), maximum), dtype=bool)
    for index, (matrix, labels) in enumerate(zip(matrices, targets)):
        size = matrix.shape[0]
        tensor[index, :size] = matrix; target[index, :size] = labels; mask[index, :size] = True

    flat = tensor[mask]
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale < 1e-6] = 1.0
    tensor = (tensor - mean) / scale
    tensor[~mask] = 0.0

    def objective(weights: Any) -> tuple[float, Any]:
        scores = np.einsum("qnf,f->qn", tensor, weights)
        scores[~mask] = -1e9
        peak = scores.max(axis=1, keepdims=True)
        exponential = np.exp(np.clip(scores - peak, -60.0, 0.0)) * mask
        probability = exponential / np.maximum(exponential.sum(axis=1, keepdims=True), 1e-12)
        loss = -float((target * np.log(np.maximum(probability, 1e-12))).sum()) / len(matrices)
        loss += 0.5 * l2 * float(np.dot(weights, weights))
        difference = (probability - target) * mask
        gradient = np.einsum("qnf,qn->f", tensor, difference) / len(matrices) + l2 * weights
        return loss, gradient

    initial = np.zeros(feature_count, dtype="float64")
    try:
        from scipy.optimize import minimize
        optimized = minimize(
            lambda value: objective(value), initial, method="L-BFGS-B", jac=True,
            options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-7, "maxls": 30},
        )
        weights = np.asarray(optimized.x, dtype="float64")
        optimizer = {"name": "L-BFGS-B", "success": bool(optimized.success), "iterations": int(optimized.nit)}
    except (ImportError, ValueError) as exc:
        # Numpy-only Adam fallback keeps Kaggle offline execution independent
        # of scipy packaging details.
        weights = initial.copy(); first = np.zeros_like(weights); second = np.zeros_like(weights)
        for step in range(1, 801):
            _, gradient = objective(weights)
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            corrected_first = first / (1.0 - 0.9 ** step)
            corrected_second = second / (1.0 - 0.999 ** step)
            weights -= 0.03 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        optimizer = {"name": "numpy_adam", "success": True, "fallback_reason": str(exc)}

    return {
        "feature_version": META_FEATURE_VERSION,
        "feature_names": feature_names,
        "evidence_k": evidence_k,
        "l2": l2,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "training_queries_with_pool_gold": len(matrices),
        "optimizer": optimizer,
        "seed": seed,
    }


def meta_ranker_rankings(
    rows_by_query: Sequence[Sequence[dict[str, Any]]], model: dict[str, Any],
) -> list[list[str]]:
    import numpy as np

    if model.get("feature_version") != META_FEATURE_VERSION:
        raise ValueError("Meta-ranker feature version is stale; rerun --stage eval")
    mean = np.asarray(model["mean"], dtype="float64")
    scale = np.asarray(model["scale"], dtype="float64")
    weights = np.asarray(model["weights"], dtype="float64")
    outputs: list[list[str]] = []
    for rows in rows_by_query:
        matrix, names = meta_ranker_feature_matrix(rows, int(model["evidence_k"]))
        if names != list(model["feature_names"]):
            raise AssertionError("Meta-ranker feature identity mismatch")
        scores = ((matrix - mean) / scale) @ weights
        order = sorted(range(len(rows)), key=lambda index: (-float(scores[index]), str(rows[index]["cid"])))
        outputs.append([str(rows[index]["cid"]) for index in order])
    return outputs


def crossfit_listwise_meta_ranker(
    cfg: Config, rows: Sequence[Sequence[dict[str, Any]]], records: Sequence[dict[str, Any]],
    calibration_indices: Sequence[int], locked_indices: Sequence[int],
) -> tuple[dict[str, Any], list[list[str]], dict[str, Any]]:
    """Choose regularization using honest two-fold out-of-fold rankings."""
    candidates: list[tuple[tuple[float, ...], float, list[list[str]], dict[str, Any]]] = []
    for l2 in (0.001, 0.003, 0.01, 0.03, 0.1):
        combined: list[list[str] | None] = [None] * len(records)
        fold_report = []
        for train_indices, test_indices, name in (
            (calibration_indices, locked_indices, "train_calibration_predict_locked"),
            (locked_indices, calibration_indices, "train_locked_predict_calibration"),
        ):
            train_rows = subset_values(rows, train_indices)
            train_records = subset_values(records, train_indices)
            model = fit_listwise_meta_ranker(
                train_rows, train_records, cfg.chunks_per_doc, l2, cfg.seed
            )
            predicted = meta_ranker_rankings(subset_values(rows, test_indices), model)
            for index, ranking in zip(test_indices, predicted):
                combined[index] = ranking
            fold_report.append({"name": name, "trained_queries": model["training_queries_with_pool_gold"]})
        if any(ranking is None for ranking in combined):
            raise AssertionError("Meta-ranker OOF prediction is incomplete")
        rankings = [ranking for ranking in combined if ranking is not None]
        metrics = task_metrics(rankings, records, 5)
        key = (
            metrics["Recall@5"], metrics["Precision@5"], metrics["AllGold@5"], metrics["MRR@5"]
        )
        candidates.append((key, l2, rankings, {"l2": l2, "metrics": metrics, "folds": fold_report}))
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_l2, best_rankings, best_report = candidates[0]
    final_model = fit_listwise_meta_ranker(rows, records, cfg.chunks_per_doc, best_l2, cfg.seed)
    report = {
        "selection": "two-fold out-of-fold Recall@5 then Precision/AllGold/MRR",
        "best_l2": best_l2,
        "oof_metrics": best_report["metrics"],
        "candidates": [item[3] for item in candidates],
    }
    return final_model, best_rankings, report
