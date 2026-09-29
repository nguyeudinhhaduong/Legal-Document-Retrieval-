"""Truy hồi + chấm điểm reranker cho toàn bộ candidate pool.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from legalir.core.constants import META_FEATURE_VERSION
from legalir.core.config import Config, Paths
from legalir.core.utils import log, progress
from legalir.core.artifacts import artifact_identity, chunking_fingerprint, valid_lora_adapter
from legalir.data.chunk_store import load_chunk_arrays
from legalir.retrieval.dense_fusion import DENSE_FUSION_VERSION, dense_ensemble_fingerprint, require_ensemble_voters, union_abc_pool_with_dense_votes
from legalir.training.negatives import bm25_candidate_scores
from legalir.models.reranker import BGEReranker
from legalir.rerank.cache import load_complete_rerank_cache, rerank_cache_path


def rerank_candidates(cfg: Config, paths: Paths, records: Sequence[dict[str, Any]],
                      candidates: Sequence[list[dict[str, Any]]], chunk_texts: Sequence[str],
                      cache_path: Path) -> list[list[dict[str, Any]]]:
    r1_adapter = paths.rerank_adapter if valid_lora_adapter(paths.rerank_adapter) else None
    # Always score an available R2.  Its training-time guardrail no longer
    # decides usage: validation voting can assign it zero weight when harmful.
    use_r2 = valid_lora_adapter(paths.second_rerank_adapter)
    cached: dict[str, list[dict[str, Any]]] = {}
    if cache_path.exists() and not cfg.rebuild:
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line); cached[str(row["qid"])] = row["candidates"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        log(f"Rerank cache: {len(cached):,}/{len(records):,} queries")
    scorers: list[BGEReranker] = []; outputs = []
    mode = "w" if cfg.rebuild else "a"
    with cache_path.open(mode, encoding="utf-8") as cache:
        for record, rows in progress(zip(records, candidates), total=len(records), desc="BGE rerank"):
            if record["qid"] in cached:
                outputs.append(cached[record["qid"]]); continue
            if not scorers:
                scorers.append(BGEReranker(cfg, training=False, adapter=r1_adapter))
                if use_r2:
                    scorers.append(BGEReranker(cfg, training=False, adapter=paths.second_rerank_adapter))
            rows.sort(key=lambda row: (-float(row["dense"]), row["cid"]))
            selected = rows[:cfg.rerank_top_k]; queries, documents, owners = [], [], []
            for index, row in enumerate(selected):
                for chunk in row["chunks"]:
                    queries.append(record["question"]); documents.append(chunk_texts[chunk]); owners.append(index)
            per_model_doc_scores: list[list[list[float]]] = []
            for scorer in scorers:
                per_doc_scores: list[list[float]] = [[] for _ in selected]
                for start in range(0, len(queries), cfg.rerank_batch_size):
                    scores = scorer.scores(queries[start:start + cfg.rerank_batch_size], documents[start:start + cfg.rerank_batch_size])
                    for owner, score in zip(owners[start:start + cfg.rerank_batch_size], scores):
                        per_doc_scores[owner].append(float(score))
                per_model_doc_scores.append(per_doc_scores)
            lexical_scores, _ = bm25_candidate_scores(
                record["question"], selected, chunk_texts, cfg.chunks_per_doc
            ) if cfg.bm25_enabled else ([0.0] * len(selected), [0] * len(selected))
            for index, row in enumerate(selected):
                r1_scores = per_model_doc_scores[0][index]
                r2_scores = per_model_doc_scores[1][index] if len(per_model_doc_scores) > 1 else list(r1_scores)
                row["rerank_r1_chunks"] = r1_scores; row["rerank_r2_chunks"] = r2_scores
                row["rerank_r1_max"] = max(r1_scores) if r1_scores else -math.inf
                row["rerank_r2_max"] = max(r2_scores) if r2_scores else -math.inf
                row["rerank_max"] = row["rerank_r1_max"]
                row["rerank_mean"] = sum(r1_scores) / len(r1_scores) if r1_scores else -math.inf
                row["rerank"] = row["rerank_max"]
                row["bm25"] = float(lexical_scores[index])
            for row in rows[len(selected):]:
                row["rerank_max"] = row["rerank_mean"] = row["rerank"] = -math.inf
                row["rerank_r1_chunks"] = row["rerank_r2_chunks"] = []
                row["rerank_r1_max"] = row["rerank_r2_max"] = -math.inf; row["bm25"] = 0.0
            outputs.append(rows)
            cache.write(json.dumps({"qid": record["qid"], "candidates": rows}, ensure_ascii=False) + "\n")
            cache.flush()
    for scorer in scorers: scorer.close()
    return outputs


def retrieve_and_rerank(cfg: Config, paths: Paths, records: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    require_ensemble_voters(paths)
    cache_path = rerank_cache_path(cfg, paths, records)
    cached = load_complete_rerank_cache(cache_path, records, cfg.rebuild)
    if cached is not None:
        return cached
    chunk_texts, chunk_cids, _ = load_chunk_arrays(paths.chunks)
    candidates = union_abc_pool_with_dense_votes(cfg, paths, records, chunk_cids)
    return rerank_candidates(cfg, paths, records, candidates, chunk_texts, cache_path)


def ranking_pipeline_fingerprint(cfg: Config, paths: Paths) -> dict[str, Any]:
    """Identity of every artifact/recipe that determines final top-five ranks."""
    return {
        "schema": 1,
        "dense_fusion_version": DENSE_FUSION_VERSION,
        "meta_feature_version": META_FEATURE_VERSION,
        "chunking": chunking_fingerprint(paths.chunk_meta),
        "split_seed": cfg.seed,
        "eval_size": cfg.eval_size,
        "embedding_voters": dense_ensemble_fingerprint(paths),
        "dense_calibration": list(artifact_identity(paths.dense_calibration)[1:]),
        "r1_adapter": list(artifact_identity(paths.rerank_adapter / "adapter_model.safetensors")[1:]),
        "r2_adapter": list(artifact_identity(paths.second_rerank_adapter / "adapter_model.safetensors")[1:]),
        "candidate_docs": cfg.candidate_docs,
        "dense_chunk_k": cfg.dense_chunk_k,
        "chunks_per_doc": cfg.chunks_per_doc,
        "rerank_top_k": cfg.rerank_top_k,
        "reranker_max_length": cfg.reranker_max_length,
        "bm25_enabled": cfg.bm25_enabled,
    }
