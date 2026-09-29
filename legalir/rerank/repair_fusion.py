"""Fusion RRF nhiều tín hiệu của pipeline gốc (lưới trọng số).

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import math
from typing import Any, Iterator, Sequence


# Two-tier ensemble: tier 1 is the dense A/B/C vote (dense_a/dense_b/dense_c
# fields, already fused via fixed_b_pool_with_dense_votes), tier 2 is the
# R1/R2 reranker vote (rerank_r1_chunks/rerank_r2_chunks). Both tiers are
# combined here into one flat weighted RRF fusion; there is no
# separate anchor/gate/rescue layer -- the calibrated weight vector alone
# decides the final ranking.
REPAIR_SIGNALS = ("a", "b", "c", "r1", "r1_top2", "r2", "r2_top2")


def repair_weight_grid(units: int = 20) -> list[dict[str, float]]:
    """Simplex grid over REPAIR_SIGNALS.

    Generic over the number of signals via recursive composition enumeration
    (stars-and-bars), so it does not need hand-unrolled nested loops per
    signal count.
    """
    if units < 1:
        raise ValueError("repair weight units must be positive")

    def compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in compositions(total - first, parts - 1):
                yield (first, *rest)

    return [
        {signal: value / units for signal, value in zip(REPAIR_SIGNALS, values)}
        for values in compositions(units, len(REPAIR_SIGNALS))
    ]


def repair_signal_scores(
    rows: Sequence[dict[str, Any]], signal: str, evidence_k: int,
) -> list[float]:
    if signal == "b":
        return [float(row.get("dense_b", row["dense"])) for row in rows]
    if signal == "a":
        return [float(row["dense_a"]) for row in rows]
    if signal == "c":
        return [float(row["dense_c"]) for row in rows]
    if signal in {"r1", "r1_top2", "r2", "r2_top2"}:
        reranker = signal.split("_", 1)[0]
        key = f"rerank_{reranker}_chunks"
        outputs = []
        for row in rows:
            chunks = list(map(float, row.get(key, [])))[:evidence_k]
            if not chunks:
                outputs.append(-math.inf)
                continue
            ranked = sorted(chunks, reverse=True)
            outputs.append(
                sum(ranked[:2]) / min(2, len(ranked))
                if signal.endswith("_top2") else ranked[0]
            )
        return outputs
    raise ValueError(f"Unknown repair signal: {signal}")


def precompute_repair_features(
    rows_by_query: Sequence[Sequence[dict[str, Any]]], evidence_k: int,
    rrf_constant: float,
) -> list[dict[str, Any]]:
    """Cache normalized RRF ballots for dense and reranker evidence signals."""
    if evidence_k < 1 or rrf_constant <= 0.0:
        raise ValueError("repair evidence and RRF constant must be positive")
    outputs = []
    for rows in rows_by_query:
        count = len(rows)
        features = [[0.0] * len(REPAIR_SIGNALS) for _ in rows]
        for signal_index, signal in enumerate(REPAIR_SIGNALS):
            values = repair_signal_scores(rows, signal, evidence_k)
            order = sorted(
                range(count), key=lambda index: (-values[index], str(rows[index]["cid"]))
            )
            best_rrf = 1.0 / (rrf_constant + 1.0)
            worst_rrf = 1.0 / (rrf_constant + max(1, count))
            scale = max(1e-12, best_rrf - worst_rrf)
            for rank, index in enumerate(order, start=1):
                features[index][signal_index] = (
                    1.0 / (rrf_constant + rank) - worst_rrf
                ) / scale if count > 1 else 1.0
        outputs.append({"features": features})
    return outputs


def fused_repair_rankings_from_features(
    rows_by_query: Sequence[Sequence[dict[str, Any]]],
    features_by_query: Sequence[dict[str, Any]], config: dict[str, Any],
) -> list[list[str]]:
    """Flat weighted-RRF fusion over all configured REPAIR_SIGNALS.

    Tier 1 (a, b, c dense votes) and tier 2 (r1, r2 reranker votes) are
    combined in one pass: the calibrated weight vector alone decides the
    final ranking, with no anchor/promotion-limit/gate/rescue logic on top.
    """
    weights = config["weights"]
    weight_vector = [float(weights.get(signal, 0.0)) for signal in REPAIR_SIGNALS]
    total = sum(weight_vector)
    if total <= 0.0:
        raise ValueError("At least one repair weight must be positive")
    weight_vector = [weight / total for weight in weight_vector]
    outputs = []
    for rows, cached in zip(rows_by_query, features_by_query):
        scores = [
            sum(weight * feature for weight, feature in zip(weight_vector, row_features))
            for row_features in cached["features"]
        ]
        order = sorted(
            range(len(rows)), key=lambda index: (-scores[index], str(rows[index]["cid"]))
        )
        outputs.append([str(rows[index]["cid"]) for index in order])
    return outputs


def fused_repair_rankings(
    rows_by_query: Sequence[Sequence[dict[str, Any]]], config: dict[str, Any],
) -> list[list[str]]:
    features = precompute_repair_features(
        rows_by_query, int(config["evidence_k"]), float(config["rrf_constant"])
    )
    return fused_repair_rankings_from_features(rows_by_query, features, config)


def fused_repair_ranking(
    rows: Sequence[dict[str, Any]], config: dict[str, Any],
) -> list[str]:
    return fused_repair_rankings([rows], config)[0]
