"""Thứ tự ứng viên sau fusion dense + reranker (dùng khi mine).

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import math
from typing import Any, Sequence

from legalir.core.utils import minmax


def fused_ranking(rows: Sequence[dict[str, Any]], reranker_weight: float,
                  pool_k: int, chunk_max_weight: float,
                  fusion_method: str, r2_weight: float = 0.0,
                  lexical_weight: float = 0.0, evidence_k: int = 4) -> list[str]:
    """Legacy dense+cross-encoder RRF/score fusion, kept only for
    current_fused_candidate_order (used by R2 mining to identify R1's actual
    top-5 failure modes). Not used by evaluate()/predict_submission()
    anymore, which use fused_repair_ranking() instead.
    """
    eligible = [
        row for row in rows[:pool_k]
        if math.isfinite(float(row.get("rerank_max", row.get("rerank", -math.inf))))
    ]
    if not eligible: eligible = list(rows[:pool_k])
    cross_raw = []
    for row in eligible:
        r1_chunks = list(map(float, row.get("rerank_r1_chunks", [])))[:evidence_k]
        r2_chunks = list(map(float, row.get("rerank_r2_chunks", r1_chunks)))[:evidence_k]
        if r1_chunks:
            length = min(len(r1_chunks), len(r2_chunks))
            ensemble_chunks = [
                (1.0 - r2_weight) * r1_chunks[index] + r2_weight * r2_chunks[index]
                for index in range(length)
            ]
            cross_raw.append(max(ensemble_chunks))
        else:
            cross_raw.append(
                chunk_max_weight * float(row.get("rerank_max", row.get("rerank", 0.0)))
                + (1.0 - chunk_max_weight) * float(row.get("rerank_mean", row.get("rerank", 0.0)))
            )
    dense_normalized = minmax([float(row["dense"]) for row in eligible])
    lexical_normalized = minmax([float(row.get("bm25", 0.0)) for row in eligible])
    retrieval_raw = [
        (1.0 - lexical_weight) * dense + lexical_weight * lexical
        for dense, lexical in zip(dense_normalized, lexical_normalized)
    ]
    if fusion_method == "score":
        rerank = minmax(cross_raw)
        scores = [
            reranker_weight * cross + (1.0 - reranker_weight) * base
            for base, cross in zip(retrieval_raw, rerank)
        ]
    elif fusion_method == "rrf":
        dense_order = sorted(range(len(eligible)), key=lambda i: (-retrieval_raw[i], eligible[i]["cid"]))
        cross_order = sorted(range(len(eligible)), key=lambda i: (-cross_raw[i], eligible[i]["cid"]))
        dense_rank = {index: rank + 1 for rank, index in enumerate(dense_order)}
        cross_rank = {index: rank + 1 for rank, index in enumerate(cross_order)}
        rrf_constant = 10.0
        scores = [
            (1.0 - reranker_weight) / (rrf_constant + dense_rank[index])
            + reranker_weight / (rrf_constant + cross_rank[index])
            for index in range(len(eligible))
        ]
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    return [row["cid"] for _, row in sorted(zip(scores, eligible), key=lambda item: (-item[0], item[1]["cid"]))]


def current_fused_candidate_order(
    candidates: Sequence[dict[str, Any]], teacher_scores: Sequence[float],
    lexical_scores: Sequence[float], pool_k: int,
) -> list[int]:
    """Reproduce the safe R1 RRF baseline to identify actual final top-5 errors."""
    scored: list[dict[str, Any]] = []
    for row, teacher, lexical in zip(candidates, teacher_scores, lexical_scores):
        item = dict(row)
        item.update({
            "rerank_max": float(teacher), "rerank_mean": float(teacher),
            "rerank_r1_chunks": [float(teacher)], "rerank_r2_chunks": [float(teacher)],
            "bm25": float(lexical),
        })
        scored.append(item)
    ordered_cids = fused_ranking(
        scored, 0.4, min(pool_k, len(scored)), 1.0, "rrf", 0.0, 0.0, 1
    )
    index_by_cid = {str(row["cid"]): index for index, row in enumerate(candidates)}
    return [index_by_cid[cid] for cid in ordered_cids]


def missing_gold_from_top_k(
    candidates: Sequence[dict[str, Any]], positives: set[str], k: int = 5,
) -> set[str]:
    return positives - {str(row["cid"]) for row in candidates[:k]}
