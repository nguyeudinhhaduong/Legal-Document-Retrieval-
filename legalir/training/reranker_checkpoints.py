"""Chấm checkpoint reranker bằng Recall@5 trên pool cố định.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import json
import math
from typing import Any, Sequence

from legalir.core.config import Config, Paths
from legalir.core.utils import atomic_json, log, progress, read_jsonl, stable_key, write_jsonl
from legalir.core.artifacts import artifact_identity, chunking_fingerprint
from legalir.data.chunk_store import load_chunk_arrays, load_chunk_cids_only
from legalir.retrieval.dense_fusion import DENSE_FUSION_VERSION, dense_ensemble_fingerprint, ensemble_dense_candidates, resolved_dense_member_quota
from legalir.models.reranker import BGEReranker
from legalir.evaluation.metrics import task_metrics


def reranker_checkpoint_pool_identity(
    cfg: Config, paths: Paths, records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": 1,
        "qids": [str(row["qid"]) for row in records],
        "chunking": chunking_fingerprint(paths.chunk_meta),
        "dense_ensemble": dense_ensemble_fingerprint(paths),
        "dense_fusion_version": DENSE_FUSION_VERSION,
        "candidate_docs": cfg.rerank_top_k,
        "dense_chunk_k": cfg.dense_chunk_k,
        "chunks_per_doc": cfg.chunks_per_doc,
        "dense_member_quota": resolved_dense_member_quota(cfg, paths),
    }


def load_or_build_reranker_checkpoint_pool(
    cfg: Config, paths: Paths,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[str]]:
    """Build one immutable validation pool shared by every R1/R2 epoch.

    Selecting checkpoints on mined one-positive groups is not the competition
    objective: a query may have multiple relevant CIDs and the final boundary
    is document rank five in the 60-document union. This cache makes the more
    expensive but correctly aligned check deterministic and reusable by R2.
    """
    all_records = read_jsonl(paths.validation)
    records = sorted(
        all_records, key=lambda row: stable_key(cfg.seed + 89, str(row["qid"]))
    )[:cfg.rerank_checkpoint_eval_queries]
    identity = reranker_checkpoint_pool_identity(cfg, paths, records)
    cached_rows: dict[str, list[dict[str, Any]]] = {}
    try:
        saved_identity = json.loads(
            paths.reranker_checkpoint_pool_meta.read_text(encoding="utf-8")
        )
        if saved_identity == identity and paths.reranker_checkpoint_pool.is_file():
            for item in read_jsonl(paths.reranker_checkpoint_pool):
                cached_rows[str(item["qid"])] = list(item["candidates"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        cached_rows = {}
    qids = [str(row["qid"]) for row in records]
    if all(qid in cached_rows for qid in qids):
        candidates = [cached_rows[qid] for qid in qids]
        log(f"Reusing true-R@5 checkpoint pool: {len(records):,} validation queries")
    else:
        chunk_cids = load_chunk_cids_only(paths.chunks)
        candidates = ensemble_dense_candidates(
            cfg, paths, [str(row["question"]) for row in records], chunk_cids,
            limit=cfg.rerank_top_k,
        )
        compact = []
        for record, rows in zip(records, candidates):
            compact_rows = [
                {"cid": str(row["cid"]), "chunks": list(map(int, row["chunks"]))}
                for row in rows[:cfg.rerank_top_k]
            ]
            compact.append(compact_rows)
        candidates = compact
        write_jsonl(paths.reranker_checkpoint_pool, (
            {"qid": record["qid"], "candidates": rows}
            for record, rows in zip(records, candidates)
        ))
        atomic_json(paths.reranker_checkpoint_pool_meta, identity)
        log(f"Built true-R@5 checkpoint pool: {len(records):,} validation queries")
    chunk_texts, _, _ = load_chunk_arrays(paths.chunks)
    return records, candidates, chunk_texts


def score_reranker_checkpoint_pool(
    scorer: BGEReranker, records: Sequence[dict[str, Any]],
    candidates: Sequence[Sequence[dict[str, Any]]], chunk_texts: Sequence[str],
    description: str,
) -> list[list[float]]:
    """Return document-max logits on the exact fixed validation pool."""
    was_training = scorer.model.training
    scorer.model.eval()
    outputs: list[list[float]] = []
    for record, rows in progress(
        zip(records, candidates), total=len(records), desc=description,
    ):
        documents: list[str] = []
        owners: list[int] = []
        for owner, row in enumerate(rows):
            for chunk in row.get("chunks", [])[:scorer.cfg.chunks_per_doc]:
                documents.append(chunk_texts[int(chunk)])
                owners.append(owner)
        scores = [-math.inf] * len(rows)
        batch_size = scorer.cfg.rerank_checkpoint_batch_size
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            values = scorer.scores([str(record["question"])] * len(batch), batch)
            for owner, value in zip(owners[start:start + len(batch)], values):
                scores[owner] = max(scores[owner], float(value))
        outputs.append(scores)
    if was_training:
        scorer.model.train()
    return outputs


def reranker_pool_rankings(
    candidates: Sequence[Sequence[dict[str, Any]]],
    scores: Sequence[Sequence[float]],
) -> list[list[str]]:
    return [
        [str(rows[index]["cid"]) for index in sorted(
            range(len(rows)), key=lambda index: (-float(values[index]), str(rows[index]["cid"]))
        )]
        for rows, values in zip(candidates, scores)
    ]


def reranker_pool_metrics(
    records: Sequence[dict[str, Any]], candidates: Sequence[Sequence[dict[str, Any]]],
    scores: Sequence[Sequence[float]],
) -> dict[str, float]:
    return task_metrics(reranker_pool_rankings(candidates, scores), records, 5)


def reranker_checkpoint_r1_score_identity(
    cfg: Config, paths: Paths, records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": 1,
        "pool": reranker_checkpoint_pool_identity(cfg, paths, records),
        "r1_adapter": list(artifact_identity(
            paths.rerank_adapter / "adapter_model.safetensors"
        )[1:]),
        "model": cfg.reranker_model,
        "max_length": cfg.reranker_max_length,
        "evidence_chunks": cfg.chunks_per_doc,
    }


def save_reranker_checkpoint_r1_scores(
    cfg: Config, paths: Paths, records: Sequence[dict[str, Any]],
    scores: Sequence[Sequence[float]],
) -> None:
    write_jsonl(paths.reranker_checkpoint_r1_scores, (
        {"qid": record["qid"], "scores": list(map(float, values))}
        for record, values in zip(records, scores)
    ))
    atomic_json(
        paths.reranker_checkpoint_r1_scores_meta,
        reranker_checkpoint_r1_score_identity(cfg, paths, records),
    )


def load_reranker_checkpoint_r1_scores(
    cfg: Config, paths: Paths, records: Sequence[dict[str, Any]],
    candidates: Sequence[Sequence[dict[str, Any]]],
) -> list[list[float]] | None:
    try:
        saved_identity = json.loads(
            paths.reranker_checkpoint_r1_scores_meta.read_text(encoding="utf-8")
        )
        if saved_identity != reranker_checkpoint_r1_score_identity(cfg, paths, records):
            return None
        by_qid = {
            str(item["qid"]): list(map(float, item["scores"]))
            for item in read_jsonl(paths.reranker_checkpoint_r1_scores)
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    outputs: list[list[float]] = []
    for record, rows in zip(records, candidates):
        values = by_qid.get(str(record["qid"]))
        if values is None or len(values) != len(rows):
            return None
        outputs.append(values)
    log(f"Reusing selected R1 checkpoint scores: {len(outputs):,} validation queries")
    return outputs


def paired_reranker_pool_metrics(
    records: Sequence[dict[str, Any]], candidates: Sequence[Sequence[dict[str, Any]]],
    r1_scores: Sequence[Sequence[float]], r2_scores: Sequence[Sequence[float]],
) -> tuple[dict[str, float], float]:
    """Best small RRF blend used only to select a complementary R2 epoch."""
    best_metrics: dict[str, float] | None = None
    best_weight = 0.0
    for step in range(11):
        r2_weight = step / 10.0
        rankings: list[list[str]] = []
        for rows, first, second in zip(candidates, r1_scores, r2_scores):
            count = len(rows)
            rank1 = [0] * count
            rank2 = [0] * count
            for rank, index in enumerate(sorted(
                range(count), key=lambda index: (-float(first[index]), str(rows[index]["cid"]))
            ), start=1):
                rank1[index] = rank
            for rank, index in enumerate(sorted(
                range(count), key=lambda index: (-float(second[index]), str(rows[index]["cid"]))
            ), start=1):
                rank2[index] = rank
            order = sorted(
                range(count),
                key=lambda index: (
                    -(
                        (1.0 - r2_weight) / (10.0 + rank1[index])
                        + r2_weight / (10.0 + rank2[index])
                    ),
                    str(rows[index]["cid"]),
                ),
            )
            rankings.append([str(rows[index]["cid"]) for index in order])
        metrics = task_metrics(rankings, records, 5)
        if best_metrics is None or reranker_checkpoint_metric_key(metrics) > reranker_checkpoint_metric_key(best_metrics):
            best_metrics = metrics
            best_weight = r2_weight
    if best_metrics is None:
        raise RuntimeError("R1/R2 checkpoint evaluation produced no weight candidate")
    return best_metrics, best_weight


def reranker_checkpoint_metric_key(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(metrics["Recall@5"]), float(metrics["AllGold@5"]),
        float(metrics["MRR@5"]), float(metrics["Precision@5"]),
    )
