"""Mine nhóm sửa lỗi cho C (dense stage 3) và R2.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import gc
from collections import defaultdict
from typing import Any

from legalir._deps import faiss
from legalir.core.config import Config, Paths
from legalir.core.utils import log, progress, read_jsonl, stable_key, write_jsonl
from legalir.core.artifacts import artifact_identity, chunking_fingerprint, valid_lora_adapter
from legalir.data.text import clean_legal_text
from legalir.models.dense_encoder import DenseEncoder
from legalir.data.chunk_store import load_chunk_arrays
from legalir.retrieval.dense_index import dense_candidates
from legalir.retrieval.dense_fusion import DENSE_FUSION_VERSION, dense_ensemble_fingerprint, ensemble_dense_candidates, fuse_dense_ensemble_members
from legalir.training.negatives import best_positive_chunk, bm25_candidate_scores, select_stratified_negatives, teacher_candidate_scores, teacher_hard_negative_rows
from legalir.retrieval.fused_order import current_fused_candidate_order, missing_gold_from_top_k
from legalir.models.reranker import BGEReranker


def mine_embedding_stage3_groups(cfg: Config, paths: Paths) -> list[dict[str, Any]]:
    """Mine an A/B boundary specialist plus broad-query rehearsal for C."""
    required = (
        (paths.embed_adapter, paths.final_index),
        (paths.second_embed_adapter, paths.second_index),
    )
    if any(not valid_lora_adapter(adapter) or not index.is_file() for adapter, index in required):
        raise FileNotFoundError("LoRA A and B adapters/indexes are required to mine C")
    teacher_ab = {
        name: {
            "adapter": list(artifact_identity(adapter / "adapter_model.safetensors")[1:]),
            "index": list(artifact_identity(index)[1:]),
        }
        for name, (adapter, index) in zip(("a", "b"), required)
    }
    recipe = {
        "version": "embedding_stage3_ab_boundary_rehearsal_v5",
        "dense_fusion_version": DENSE_FUSION_VERSION,
        "hard_query_rule": "a_or_b_missing_any_gold_at_5",
        "positive_rule": "gold_missing_from_either_a_or_b_top5",
        "rehearsal_ratio": cfg.embed_stage3_rehearsal_ratio,
        "negatives": cfg.embed_stage3_hard_negatives,
        "pool": cfg.candidate_docs, "chunks_per_doc": cfg.chunks_per_doc,
        "dense_chunk_k": cfg.dense_chunk_k,
        "teacher_ab": teacher_ab,
        "chunking": chunking_fingerprint(paths.chunk_meta),
        "split_seed": cfg.seed,
        "eval_size": cfg.eval_size,
    }
    if paths.third_stage_groups.exists() and not cfg.rebuild:
        cached = read_jsonl(paths.third_stage_groups)
        if cached and all(row.get("mining_recipe") == recipe for row in cached):
            log(f"Reusing {len(cached):,} embedding LoRA C groups: {paths.third_stage_groups}")
            return cached
    records = read_jsonl(paths.train)
    if cfg.train_groups:
        records = sorted(records, key=lambda row: stable_key(cfg.seed, row["qid"]))[:cfg.train_groups]
    chunk_texts, chunk_cids, _ = load_chunk_arrays(paths.chunks)
    doc_to_chunks: dict[str, list[int]] = defaultdict(list)
    for index, cid in enumerate(chunk_cids): doc_to_chunks[cid].append(index)
    questions = [row["question"] for row in records]
    members: list[list[list[dict[str, Any]]]] = []
    for adapter_path, index_path in required:
        encoder = DenseEncoder(cfg, adapter=adapter_path)
        index = faiss.read_index(str(index_path))
        try:
            members.append(dense_candidates(
                cfg, encoder, index, questions, chunk_cids,
                limit=max(cfg.candidate_docs, 60),
            ))
        finally:
            encoder.close(); del index; gc.collect()
    candidates_by_query = fuse_dense_ensemble_members(
        members, (0.5, 0.5), max(cfg.candidate_docs, 60),
        cfg.chunks_per_doc, cfg.dense_member_quota,
    )
    groups: list[dict[str, Any]] = []; hard_queries = 0; rehearsal_queries = 0
    iterator = zip(records, candidates_by_query, members[0], members[1])
    for record, candidates, a_rows, b_rows in progress(
        iterator, total=len(records), desc="Mine embedding C"
    ):
        positives = set(map(str, record["answers"]))
        missing_a = missing_gold_from_top_k(a_rows, positives, 5)
        missing_b = missing_gold_from_top_k(b_rows, positives, 5)
        target_positives = missing_a | missing_b
        is_hard = bool(target_positives)
        rehearsal_value = int.from_bytes(
            stable_key(cfg.seed + 3031, record["qid"])[:8], "big"
        ) / float(2**64 - 1)
        is_rehearsal = not is_hard and rehearsal_value < cfg.embed_stage3_rehearsal_ratio
        if not is_hard and not is_rehearsal:
            continue
        if is_hard:
            hard_queries += 1
        else:
            rehearsal_queries += 1
            target_positives = positives
        teacher_scores = [float(row["dense"]) for row in candidates]
        teacher_chunks = [int(row["chunks"][0]) for row in candidates]
        negatives = teacher_hard_negative_rows(
            candidates, positives, teacher_scores, teacher_chunks,
            cfg.embed_stage3_hard_negatives,
            int.from_bytes(stable_key(cfg.seed + 2027, record["qid"])[:8], "big"),
        )
        if len(negatives) < cfg.embed_stage3_hard_negatives: continue
        for positive_cid in sorted(target_positives):
            positive_index = next((i for i, row in enumerate(candidates) if row["cid"] == positive_cid), None)
            pos_chunk = (
                teacher_chunks[positive_index] if positive_index is not None
                else best_positive_chunk(record["question"], {positive_cid}, candidates, chunk_texts, doc_to_chunks)
            )
            groups.append({
                "qid": f'{record["qid"]}:{positive_cid}', "source_qid": record["qid"],
                "query": record["question"], "positive_cid": positive_cid,
                "positive": chunk_texts[pos_chunk],
                "negatives": [chunk_texts[int(row["teacher_chunk"])] for row in negatives],
                "negative_cids": [row["cid"] for row in negatives],
                "curriculum": "ab_top5_boundary" if is_hard else "broad_rehearsal",
                "mining_recipe": recipe,
            })
    write_jsonl(paths.third_stage_groups, groups)
    log(
        f"Mined {len(groups):,} independent LoRA C groups from {hard_queries:,} "
        f"A/B top-5 boundary failures + {rehearsal_queries:,} rehearsal queries "
        f"({cfg.embed_stage3_hard_negatives} union-ranked negatives each)"
    )
    return groups


def mine_reranker_stage2_groups(cfg: Config, paths: Paths) -> list[dict[str, Any]]:
    recipe = {
        "version": "reranker_stage2_document_top5_repair_v4",
        "dense_fusion_version": DENSE_FUSION_VERSION,
        "negatives": cfg.rerank_stage2_hard_negatives,
        "pool": cfg.rerank_mining_docs, "chunks_per_doc": cfg.chunks_per_doc,
        "train_chunks_per_doc": cfg.rerank_train_chunks_per_doc,
        "hard_query_rule": "r1_dense_fusion_missing_gold_at_5",
        "positive_rule": "missing_gold_only_for_hard_queries",
        "rehearsal_ratio": cfg.rerank_stage2_rehearsal_ratio,
        "dense_chunk_k": cfg.dense_chunk_k,
        "dense": dense_ensemble_fingerprint(paths),
        "parent_r1": list(artifact_identity(paths.rerank_adapter / "adapter_model.safetensors")[1:]),
        "chunking": chunking_fingerprint(paths.chunk_meta),
        "split_seed": cfg.seed,
        "eval_size": cfg.eval_size,
    }
    if paths.second_reranker_groups.exists() and not cfg.rebuild:
        cached = read_jsonl(paths.second_reranker_groups)
        if cached and all(row.get("mining_recipe") == recipe for row in cached):
            log(f"Reusing {len(cached):,} R2 groups: {paths.second_reranker_groups}")
            return cached
    records = read_jsonl(paths.train)
    chunk_texts, chunk_cids, _ = load_chunk_arrays(paths.chunks)
    doc_to_chunks: dict[str, list[int]] = defaultdict(list)
    for index, cid in enumerate(chunk_cids): doc_to_chunks[cid].append(index)
    candidates_by_query = ensemble_dense_candidates(
        cfg, paths, [row["question"] for row in records], chunk_cids,
        limit=cfg.rerank_mining_docs,
    )
    scorer = BGEReranker(cfg, training=False, adapter=paths.rerank_adapter)
    groups: list[dict[str, Any]] = []
    hard_queries = 0; rehearsal_queries = 0
    for record, candidates in progress(zip(records, candidates_by_query), total=len(records), desc="Mine reranker R2"):
        positives = set(map(str, record["answers"]))
        scores, chunks, ranked_chunks = teacher_candidate_scores(
            scorer, record["question"], candidates, chunk_texts
        )
        lexical_scores, lexical_chunks = bm25_candidate_scores(record["question"], candidates, chunk_texts, cfg.chunks_per_doc)
        r1_order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        fused_order = current_fused_candidate_order(
            candidates, scores, lexical_scores, cfg.rerank_top_k
        )
        fused_top5 = {str(candidates[index]["cid"]) for index in fused_order[:5]}
        missing_positives = positives - fused_top5
        is_hard = bool(missing_positives)
        rehearsal_value = int.from_bytes(
            stable_key(cfg.seed + 4079, str(record["qid"]))[:8], "big"
        ) / float(2**64 - 1)
        is_rehearsal = (
            not is_hard and rehearsal_value < cfg.rerank_stage2_rehearsal_ratio
        )
        if not is_hard and not is_rehearsal:
            continue
        target_positives = missing_positives if is_hard else positives
        if is_hard: hard_queries += 1
        else: rehearsal_queries += 1
        positive_texts = {
            clean_legal_text(chunk_texts[index]).casefold()
            for cid in positives for index in doc_to_chunks.get(cid, [])
        }
        negatives = select_stratified_negatives(
            candidates, positives, scores, chunks, lexical_scores, lexical_chunks,
            (("fused_top5_error", 5, fused_order[:5]),
             ("r1_hard", 5, r1_order),
             ("dense_boundary", 3, list(range(5, min(20, len(candidates))))),
             ("dense_tail", 2, list(range(20, min(50, len(candidates)))))),
            cfg.rerank_stage2_hard_negatives, chunk_texts, positive_texts,
        )
        if len(negatives) < cfg.rerank_stage2_hard_negatives: continue
        candidate_index_by_cid = {
            str(row["cid"]): index for index, row in enumerate(candidates)
        }
        for positive_cid in sorted(target_positives):
            positive_index = next((i for i, row in enumerate(candidates) if row["cid"] == positive_cid), None)
            pos_chunk = (
                chunks[positive_index] if positive_index is not None
                else best_positive_chunk(record["question"], {positive_cid}, candidates, chunk_texts, doc_to_chunks)
            )
            positive_chunk_ids = (
                ranked_chunks[positive_index][:cfg.rerank_train_chunks_per_doc]
                if positive_index is not None else [pos_chunk]
            ) or [pos_chunk]
            negative_chunk_ids = []
            for row in negatives:
                candidate_index = candidate_index_by_cid.get(str(row["cid"]))
                values = (
                    ranked_chunks[candidate_index][:cfg.rerank_train_chunks_per_doc]
                    if candidate_index is not None else [int(row["teacher_chunk"])]
                )
                negative_chunk_ids.append(values or [int(row["teacher_chunk"])])
            groups.append({
                "qid": f'{record["qid"]}:{positive_cid}', "source_qid": record["qid"],
                "query": record["question"], "positive_cid": positive_cid,
                "positive": chunk_texts[pos_chunk],
                "negatives": [chunk_texts[values[0]] for values in negative_chunk_ids],
                "positive_chunks": [chunk_texts[index] for index in positive_chunk_ids],
                "negative_chunk_groups": [
                    [chunk_texts[index] for index in values]
                    for values in negative_chunk_ids
                ],
                "negative_cids": [row["cid"] for row in negatives],
                "negative_sources": [row["negative_source"] for row in negatives],
                "curriculum": "r1_top5_failure" if is_hard else "rehearsal",
                "mining_recipe": recipe,
            })
    scorer.close(); write_jsonl(paths.second_reranker_groups, groups)
    log(
        f"Mined {len(groups):,} document-level R2 groups from {hard_queries:,} "
        f"R1 top-5 failures + {rehearsal_queries:,} rehearsal queries with "
        f"{cfg.rerank_stage2_hard_negatives} stratified negatives"
    )
    return groups
