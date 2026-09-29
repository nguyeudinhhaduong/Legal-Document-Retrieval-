"""Cache điểm rerank theo fingerprint của model/index/chunk.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from legalir.core.config import Config, Paths
from legalir.core.utils import log
from legalir.core.artifacts import artifact_identity, chunking_fingerprint
from legalir.retrieval.dense_fusion import DENSE_FUSION_VERSION, dense_ensemble_fingerprint, resolved_dense_member_quota


def rerank_cache_path(cfg: Config, paths: Paths, records: Sequence[dict[str, Any]]) -> Path:
    payload = {
        "schema": 12, "qids": [row["qid"] for row in records],
        "candidate_retriever": "quota_protected_union_a_b_c",
        "embedding_voters": dense_ensemble_fingerprint(paths),
        "r1_adapter": artifact_identity(paths.rerank_adapter / "adapter_model.safetensors"),
        "r2_adapter": artifact_identity(paths.second_rerank_adapter / "adapter_model.safetensors"),
        "r2_scored_even_when_training_guardrail_rejected": True,
        "model": cfg.reranker_model,
        "architecture": "AutoModelForSequenceClassification",
        "chunking": chunking_fingerprint(paths.chunk_meta),
        "candidate_docs": cfg.candidate_docs, "rerank_top_k": cfg.rerank_top_k,
        "dense_chunk_k": cfg.dense_chunk_k,
        "retrieval_fusion": DENSE_FUSION_VERSION,
        "dense_member_quota": resolved_dense_member_quota(cfg, paths),
        "chunk_aggregation": "store_per_chunk_r1_r2",
        "ensemble_candidate_depth_multiplier": 3,
        "chunks_per_doc": cfg.chunks_per_doc, "max_length": cfg.reranker_max_length,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return paths.root / f"reranked_{len(records)}q_{digest}.jsonl"


def load_complete_rerank_cache(
    cache_path: Path, records: Sequence[dict[str, Any]], rebuild: bool,
) -> list[list[dict[str, Any]]] | None:
    """Return a complete cache before loading any of the five voting models."""
    if rebuild or not cache_path.exists():
        return None
    cached: dict[str, list[dict[str, Any]]] = {}
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                cached[str(row["qid"])] = row["candidates"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    qids = [str(row["qid"]) for row in records]
    if all(qid in cached for qid in qids):
        log(f"Complete B/A/C/R1/R2 cache reused: {len(qids):,}/{len(qids):,} queries")
        return [cached[qid] for qid in qids]
    return None


def dense_calibration_selected_b_only(paths: Paths) -> bool:
    if not paths.dense_calibration.is_file():
        return False
    try:
        report = json.loads(paths.dense_calibration.read_text(encoding="utf-8"))
        weights = list(map(float, report["best_weights"]))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return (
        len(weights) >= 2
        and math.isclose(weights[0], 0.0, abs_tol=1e-8)
        and math.isclose(weights[1], 1.0, abs_tol=1e-8)
        and all(math.isclose(weight, 0.0, abs_tol=1e-8) for weight in weights[2:])
    )


def load_legacy_b_reranker_cache(
    cfg: Config, paths: Paths, records: Sequence[dict[str, Any]], current_path: Path,
) -> tuple[list[list[dict[str, Any]]], bool] | None:
    """Migrate a complete old B/R1/R2 cache instead of rerunning models.

    The previous run selected dense weights [0, 1, 0], so its cached pool is
    exactly B's ranking even though the numeric `dense` field is an RRF score.
    A cache already containing dense_a/dense_c can be reused with no inference;
    otherwise only those two missing votes are computed from existing indexes.
    """
    if cfg.rebuild:
        return None
    qids = [str(row["qid"]) for row in records]
    # A candidate cache written before the current prepared/chunks.jsonl was
    # last (re)built cannot be trusted: its dense_b/rerank_r1_chunks/
    # rerank_r2_chunks scores were computed against whatever chunk
    # boundaries existed at the time, which -- after chunk_document's
    # Điều/Khoản-atomic rewrite (2026-08-19) -- may no longer match. Only
    # chunks_mtime or newer candidates are eligible for migration.
    chunks_mtime = paths.chunks.stat().st_mtime_ns if paths.chunks.exists() else 0
    candidates = sorted(
        (
            path for path in paths.root.glob(f"reranked_{len(records)}q_*.jsonl")
            if path.resolve() != current_path.resolve()
            and path.stat().st_mtime_ns >= chunks_mtime
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    b_only = dense_calibration_selected_b_only(paths)
    for candidate_path in candidates:
        cached: dict[str, list[dict[str, Any]]] = {}
        with candidate_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                    cached[str(item["qid"])] = item["candidates"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        if not all(qid in cached for qid in qids):
            continue
        outputs: list[list[dict[str, Any]]] = []
        has_dense_votes = True
        has_distinct_r2 = False
        valid = True
        for qid in qids:
            rows = list(cached[qid])
            if len(rows) < cfg.candidate_docs:
                valid = False; break
            rows.sort(key=lambda row: (-float(row.get("dense", -math.inf)), str(row["cid"])))
            selected = rows[:cfg.candidate_docs]
            if len({str(row["cid"]) for row in selected}) != len(selected):
                valid = False; break
            for rank, row in enumerate(selected, start=1):
                r1 = list(map(float, row.get("rerank_r1_chunks", [])))
                r2 = list(map(float, row.get("rerank_r2_chunks", [])))
                if not r1 or not r2:
                    valid = False; break
                has_distinct_r2 = has_distinct_r2 or r1 != r2
                has_dense_votes = has_dense_votes and "dense_a" in row and "dense_c" in row
                # Preserve B's rank while removing the old ensemble score scale.
                row["dense_b"] = 1.0 / (60.0 + rank)
                row["dense"] = row["dense_b"]
            if not valid:
                break
            outputs.append(selected)
        if not valid or not has_distinct_r2:
            continue
        if not has_dense_votes and not b_only:
            continue
        log(
            f"Migrating cached B pool + R1/R2 from {candidate_path.name}; "
            + ("A/C votes also cached" if has_dense_votes else "only A/C votes need rescoring")
        )
        return outputs, has_dense_votes
    return None


def write_complete_rerank_cache(
    cache_path: Path, records: Sequence[dict[str, Any]], rows_by_query: Sequence[Sequence[dict[str, Any]]],
) -> None:
    temp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record, rows in zip(records, rows_by_query):
            handle.write(json.dumps(
                {"qid": record["qid"], "candidates": rows}, ensure_ascii=False
            ) + "\n")
    os.replace(temp, cache_path)
