"""Xây FAISS index và truy hồi ứng viên dense.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from legalir._deps import faiss
from legalir.core.config import Config, Paths
from legalir.core.utils import atomic_json, log, progress, read_jsonl
from legalir.core.artifacts import artifact_identity, chunking_fingerprint, dense_index_meta_path
from legalir.models.dense_encoder import DenseEncoder, is_bge_m3_model
from legalir.data.chunk_store import load_chunk_arrays


def base_index_path(cfg: Config, paths: Paths) -> Path:
    return Path(cfg.base_index) if cfg.base_index else paths.base_index


def train_groups_path(cfg: Config, paths: Paths) -> Path:
    return Path(cfg.train_groups_file) if cfg.train_groups_file else paths.groups


def dense_index_recipe(
    cfg: Config, paths: Paths, base: bool, stage: int,
) -> dict[str, Any]:
    adapters = {
        1: paths.embed_adapter, 2: paths.second_embed_adapter,
        3: paths.third_embed_adapter,
    }
    adapter_identity = None if base else list(
        artifact_identity(adapters[stage] / "adapter_model.safetensors")[1:]
    )
    return {
        "schema": 1,
        "base_model": cfg.embedding_model,
        "adapter_identity": adapter_identity,
        "stage": 0 if base else stage,
        "embedding_dim": cfg.embedding_dim,
        "embedding_max_length": cfg.embedding_max_length,
        "pooling": "cls" if is_bge_m3_model(cfg.embedding_model) else "last_token",
        "query_instruction": not is_bge_m3_model(cfg.embedding_model),
        "chunking": chunking_fingerprint(paths.chunk_meta),
    }


def groups_in_current_train_split(
    groups: Sequence[dict[str, Any]], paths: Paths
) -> list[dict[str, Any]]:
    """Prevent supplied groups from leaking into a changed validation split."""
    allowed = {str(row["qid"]) for row in read_jsonl(paths.train)}
    return [
        row for row in groups
        if str(row.get("source_qid", row["qid"])) in allowed
    ]


def build_dense_index(cfg: Config, paths: Paths, base: bool = False, stage: int = 1) -> Path:
    if stage not in {1, 2, 3}:
        raise ValueError("Dense fine-tuning stage must be 1, 2 or 3")
    if base and cfg.base_index:
        target = base_index_path(cfg, paths)
        if not target.is_file():
            raise FileNotFoundError(f"Supplied base FAISS index does not exist: {target}")
        log(f"Using supplied base dense index (no rebuild): {target}")
        return target
    stage_indexes = {1: paths.final_index, 2: paths.second_index, 3: paths.third_index}
    stage_adapters = {1: paths.embed_adapter, 2: paths.second_embed_adapter, 3: paths.third_embed_adapter}
    target = paths.base_index if base else stage_indexes[stage]
    meta_path = dense_index_meta_path(target)
    expected_recipe = dense_index_recipe(cfg, paths, base, stage)
    chunk_count = sum(1 for _ in paths.chunks.open(encoding="utf-8")) if paths.chunks.exists() else 0
    if target.exists() and not cfg.rebuild:
        # A restored/symlinked index from a previous run (possibly built
        # under old chunk boundaries -- see chunk_document's Điều/Khoản
        # rewrite) must not be silently reused once chunks.jsonl has been
        # regenerated with a different chunk count; a stale index would
        # otherwise pass this existence check while mine_groups/dense_
        # candidates later fail on a chunk_id/vector-count mismatch.
        existing = faiss.read_index(str(target))
        try:
            saved_recipe = json.loads(meta_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            saved_recipe = None
        if (
            existing.ntotal == chunk_count
            and existing.d == cfg.embedding_dim
            and saved_recipe == expected_recipe
        ):
            log(f"Reusing dense index: {target}"); return target
        log(
            f"Dense index stale: {target}; vectors={existing.ntotal:,}/{chunk_count:,}, "
            f"dimension={existing.d}/{cfg.embedding_dim}, recipe_match="
            f"{saved_recipe == expected_recipe}; rebuilding"
        )
    texts, _, _ = load_chunk_arrays(paths.chunks)
    adapter = None if base else stage_adapters[stage]
    if not base and not (adapter / "adapter_config.json").exists():
        adapter = None; log("Embedding adapter absent; final index uses the strong base model")
    encoder = DenseEncoder(cfg, adapter=adapter); index: Any | None = None
    for start in progress(range(0, len(texts), cfg.embed_batch_size), desc="Dense index"):
        matrix = encoder.tensor(texts[start:start + cfg.embed_batch_size], False).cpu().numpy()
        matrix = matrix[:, :cfg.embedding_dim].astype("float32"); faiss.normalize_L2(matrix)
        if index is None: index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
    if index is None: raise ValueError("No chunks to index")
    temp = target.with_suffix(".faiss.tmp"); faiss.write_index(index, str(temp)); os.replace(temp, target)
    atomic_json(meta_path, expected_recipe)
    encoder.close(); log(f"Saved FAISS index ({index.ntotal:,} vectors): {target}"); return target


def dense_candidates(
    cfg: Config, encoder: DenseEncoder, index: Any, questions: Sequence[str],
    chunk_cids: Sequence[str], limit: int | None = None,
) -> list[list[dict[str, Any]]]:
    matrix = encoder.encode(questions, True, cfg.query_batch_size, "Encode queries")
    matrix = matrix[:, :cfg.embedding_dim].astype("float32"); faiss.normalize_L2(matrix)
    if index.ntotal != len(chunk_cids) or index.ntotal == 0:
        raise ValueError("Dense retrieval requires a nonempty index aligned with chunk CIDs")
    search_k = min(cfg.dense_chunk_k, index.ntotal)
    scores, indices = index.search(matrix, search_k)
    results: list[list[dict[str, Any]]] = []; doc_limit = limit or cfg.candidate_docs
    minimum_docs = min(doc_limit, cfg.candidate_docs, len(set(map(str, chunk_cids))))
    for query_i, (row_scores, row_indices) in enumerate(zip(scores, indices)):
        docs: dict[str, list[tuple[float, int]]] = defaultdict(list)
        query_k = search_k
        while True:
            docs.clear()
            for score, chunk_id in zip(row_scores, row_indices):
                if chunk_id >= 0: docs[str(chunk_cids[int(chunk_id)])].append((float(score), int(chunk_id)))
            if len(docs) >= minimum_docs or query_k >= index.ntotal:
                break
            # FAISS ranks chunks, so a long document can occupy the entire
            # initial search. Expand only this query, reusing its embedding.
            query_k = min(index.ntotal, max(query_k + 1, query_k * 2))
            expanded_scores, expanded_indices = index.search(matrix[query_i:query_i + 1], query_k)
            row_scores, row_indices = expanded_scores[0], expanded_indices[0]
        if query_k != search_k:
            log(f"Dense query {query_i}: expanded chunk search {search_k}->{query_k}; unique docs={len(docs)}")
        ranked = sorted(docs.items(), key=lambda item: (-item[1][0][0], item[0]))[:doc_limit]
        results.append([{
            "cid": cid, "dense": values[0][0],
            "chunks": [chunk for _, chunk in values[:cfg.chunks_per_doc]],
        } for cid, values in ranked])
    return results
