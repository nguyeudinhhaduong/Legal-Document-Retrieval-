"""Mine nhóm huấn luyện cho dense A/B và reranker R1.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from legalir._deps import faiss
from legalir.core.config import Config, Paths
from legalir.core.utils import log, progress, read_jsonl, stable_key, write_jsonl
from legalir.core.artifacts import chunking_fingerprint
from legalir.data.text import informative_tokens
from legalir.models.dense_encoder import DenseEncoder
from legalir.data.chunk_store import load_chunk_arrays
from legalir.retrieval.dense_index import base_index_path, dense_candidates, groups_in_current_train_split, train_groups_path
from legalir.retrieval.dense_fusion import ensemble_dense_candidates, resolved_dense_ensemble_weight, resolved_dense_ensemble_weights, resolved_dense_member_quota
from legalir.training.negatives import best_positive_chunk, hardest_row_chunk, teacher_candidate_scores, teacher_hard_negative_rows
from legalir.models.reranker import BGEReranker


def mine_groups(
    cfg: Config, paths: Paths, force_final: bool = False, force_ensemble: bool = False
) -> list[dict[str, Any]]:
    if force_final and force_ensemble:
        raise ValueError("Choose either stage-1 or ensemble hard-negative mining")
    cache_path = paths.reranker_groups if force_ensemble else paths.second_stage_groups
    if (force_final or force_ensemble) and cache_path.exists() and not cfg.rebuild:
        groups = read_jsonl(cache_path)
        expected_negatives = (
            cfg.rerank_hard_negatives if force_ensemble
            else cfg.embed_stage2_hard_negatives
        )
        expected_pool = cfg.rerank_mining_docs if force_ensemble else cfg.candidate_docs
        expected_lexical_ratio = cfg.embed_stage2_lexical_negative_ratio if force_final else None
        expected_chunking = chunking_fingerprint(paths.chunk_meta)
        recipe_matches = bool(groups) and all(
            row.get("mining_negatives") == expected_negatives
            and row.get("mining_pool") == expected_pool
            and row.get("mining_chunk_k") == cfg.dense_chunk_k
            and row.get("mining_method") == (
                "base_reranker_teacher_document_evidence_v4" if force_ensemble
                else "dense_teacher_distill_lexical_stratified_v4"
            )
            and row.get("mining_chunking") == expected_chunking
            and row.get("mining_split_seed") == cfg.seed
            and row.get("mining_eval_size") == cfg.eval_size
            and (
                not force_final
                or float(row.get("mining_lexical_ratio", -1.0)) == expected_lexical_ratio
            )
            and (
                not force_ensemble
                or (
                    row.get("ensemble_weights") == list(resolved_dense_ensemble_weights(cfg, paths))
                    and row.get("dense_member_quota") == resolved_dense_member_quota(cfg, paths)
                )
            )
            for row in groups
        )
        if recipe_matches:
            source = "hard ensemble" if force_ensemble else "BGE LoRA A"
            log(f"Reusing {len(groups):,} {source}-mined groups: {cache_path}")
            return groups
        log(f"Mining recipe changed; rebuilding groups: {cache_path}")
    if not force_final and not force_ensemble and cfg.train_groups_file:
        supplied = train_groups_path(cfg, paths)
        if not supplied.is_file():
            raise FileNotFoundError(f"Supplied training-groups file does not exist: {supplied}")
        raw_groups = read_jsonl(supplied)
        groups = groups_in_current_train_split(raw_groups, paths)
        if not groups:
            raise ValueError(f"Supplied training-groups file is empty: {supplied}")
        removed = len(raw_groups) - len(groups)
        log(
            f"Using {len(groups):,} supplied training groups (no mining); "
            f"filtered {removed:,} groups reserved by the current validation split: {supplied}"
        )
        return groups
    if not force_final and not force_ensemble and paths.groups.exists() and not cfg.rebuild:
        groups = read_jsonl(paths.groups)
        expected_chunking = chunking_fingerprint(paths.chunk_meta)
        if groups and all(
            row.get("mining_chunking") == expected_chunking
            and row.get("mining_chunk_k") == cfg.dense_chunk_k
            and row.get("mining_split_seed") == cfg.seed
            and row.get("mining_eval_size") == cfg.eval_size
            for row in groups
        ):
            log(f"Reusing {len(groups):,} mined training groups: {paths.groups}")
            return groups
        log(f"Chunking changed since {paths.groups} was mined; rebuilding")
    index_path = paths.final_index if force_final else base_index_path(cfg, paths)
    records = read_jsonl(paths.train)
    if cfg.train_groups: records = sorted(records, key=lambda row: stable_key(cfg.seed, row["qid"]))[:cfg.train_groups]
    chunk_texts, chunk_cids, _ = load_chunk_arrays(paths.chunks)
    doc_to_chunks: dict[str, list[int]] = defaultdict(list)
    for index, cid in enumerate(chunk_cids): doc_to_chunks[cid].append(index)
    questions = [row["question"] for row in records]
    if force_ensemble:
        retrieved = ensemble_dense_candidates(
            cfg, paths, questions, chunk_cids, limit=cfg.rerank_mining_docs
        )
    else:
        adapter = paths.embed_adapter if force_final and (paths.embed_adapter / "adapter_config.json").exists() else None
        dense_index = faiss.read_index(str(index_path))
        if dense_index.ntotal != len(chunk_cids):
            raise ValueError(
                f"FAISS/chunk mismatch: index has {dense_index.ntotal:,} vectors but "
                f"{paths.chunks} has {len(chunk_cids):,} chunks"
            )
        if dense_index.d != cfg.embedding_dim:
            raise ValueError(
                f"FAISS dimension mismatch: index dimension is {dense_index.d}, "
                f"but --embedding-dim is {cfg.embedding_dim}"
            )
        encoder = DenseEncoder(cfg, adapter=adapter)
        retrieved = dense_candidates(cfg, encoder, dense_index, questions, chunk_cids)
        encoder.close()
    teacher_scorer = None
    if force_final or force_ensemble:
        purpose = "reranker" if force_ensemble else "embedding stage 2"
        log(f"Mining {purpose} false positives with the base BGE reranker")
        teacher_scorer = BGEReranker(cfg, training=False)
    groups = []
    mining_iterator = progress(
        zip(records, retrieved), total=len(records),
        desc=(
            "Teacher hard-negative mining"
            if teacher_scorer is not None else "Dense hard-negative mining"
        ),
    )
    for record, candidates in mining_iterator:
        all_positives = set(record["answers"])
        negative_count = (
            cfg.rerank_hard_negatives if force_ensemble
            else cfg.embed_stage2_hard_negatives if force_final
            else cfg.hard_negatives
        )
        mining_pool = cfg.rerank_mining_docs if force_ensemble else cfg.candidate_docs
        ranked_teacher_chunks: list[list[int]] | None = None
        if teacher_scorer is not None:
            teacher_scores, teacher_chunks, ranked_teacher_chunks = teacher_candidate_scores(
                teacher_scorer, record["question"], candidates, chunk_texts
            )
        else:
            teacher_scores = [float(row["dense"]) for row in candidates]
            teacher_chunks = [int(row["chunks"][0]) for row in candidates]
        lexical_scores = None
        if force_final:
            query_terms = informative_tokens(record["question"])
            lexical_scores = [
                len(query_terms & informative_tokens(chunk_texts[chunk])) / max(1, len(query_terms))
                for chunk in teacher_chunks
            ]
        negatives = (
            teacher_hard_negative_rows(
                candidates, all_positives, teacher_scores, teacher_chunks,
                negative_count,
                int.from_bytes(stable_key(cfg.seed, record["qid"])[:8], "big"),
                lexical_scores=lexical_scores,
                lexical_ratio=cfg.embed_stage2_lexical_negative_ratio if force_final else 0.0,
            )
            if teacher_scorer is not None
            else [row for row in candidates if row["cid"] not in all_positives][:negative_count]
        )
        if len(negatives) < negative_count: continue
        candidate_index_by_cid = {
            str(row["cid"]): index for index, row in enumerate(candidates)
        }
        # A multi-answer query creates one group per relevant document, so dense
        # and cross-encoder training do not silently ignore secondary gold IDs.
        for positive_cid in sorted(all_positives):
            positive_candidate_index = next(
                (index for index, row in enumerate(candidates) if row["cid"] == positive_cid),
                None,
            )
            pos_chunk = (
                teacher_chunks[positive_candidate_index]
                if teacher_scorer is not None and positive_candidate_index is not None
                else best_positive_chunk(
                    record["question"], {positive_cid}, candidates, chunk_texts, doc_to_chunks
                )
            )
            positive_teacher_score = None
            if teacher_scorer is not None:
                positive_teacher_score = (
                    float(teacher_scores[positive_candidate_index])
                    if positive_candidate_index is not None
                    else float(teacher_scorer.scores(
                        [record["question"]], [chunk_texts[pos_chunk]]
                    )[0])
                )
            negative_teacher_scores = [
                float(row.get("teacher_score", row["dense"])) for row in negatives
            ]
            # Gold labels remain authoritative when the base teacher makes a
            # mistake; distillation still transfers its ordering among negatives.
            if positive_teacher_score is not None and negative_teacher_scores:
                positive_teacher_score = max(
                    positive_teacher_score, max(negative_teacher_scores) + 1.0
                )
            positive_chunk_ids = [pos_chunk]
            negative_chunk_ids = [
                [
                    int(row["teacher_chunk"])
                    if teacher_scorer is not None
                    else hardest_row_chunk(record["question"], row, chunk_texts)
                ]
                for row in negatives
            ]
            if force_ensemble and ranked_teacher_chunks is not None:
                positive_chunk_ids = (
                    ranked_teacher_chunks[positive_candidate_index][
                        :cfg.rerank_train_chunks_per_doc
                    ]
                    if positive_candidate_index is not None else [pos_chunk]
                ) or [pos_chunk]
                negative_chunk_ids = []
                for row in negatives:
                    candidate_index = candidate_index_by_cid.get(str(row["cid"]))
                    values = (
                        ranked_teacher_chunks[candidate_index][
                            :cfg.rerank_train_chunks_per_doc
                        ]
                        if candidate_index is not None else [int(row["teacher_chunk"])]
                    )
                    negative_chunk_ids.append(values or [int(row["teacher_chunk"])])
            groups.append({
                "qid": f'{record["qid"]}:{positive_cid}', "source_qid": record["qid"],
                "query": record["question"], "positive_cid": positive_cid,
                "positive": chunk_texts[pos_chunk],
                "negatives": [chunk_texts[values[0]] for values in negative_chunk_ids],
                **({
                    "positive_chunks": [chunk_texts[index] for index in positive_chunk_ids],
                    "negative_chunk_groups": [
                        [chunk_texts[index] for index in values]
                        for values in negative_chunk_ids
                    ],
                } if force_ensemble else {}),
                "negative_cids": [row["cid"] for row in negatives],
                "negative_teacher_scores": negative_teacher_scores,
                "positive_teacher_score": positive_teacher_score,
                "mining_negatives": negative_count,
                "mining_pool": mining_pool,
                "mining_chunk_k": cfg.dense_chunk_k,
                "mining_method": (
                    "base_reranker_teacher_document_evidence_v4" if force_ensemble
                    else "dense_teacher_distill_lexical_stratified_v4" if force_final
                    else "dense_rank_v1"
                ),
                "mining_chunking": chunking_fingerprint(paths.chunk_meta),
                "mining_split_seed": cfg.seed,
                "mining_eval_size": cfg.eval_size,
                "mining_lexical_ratio": (
                    cfg.embed_stage2_lexical_negative_ratio if force_final else None
                ),
                "ensemble_stage1_weight": (
                    resolved_dense_ensemble_weight(cfg, paths) if force_ensemble else None
                ),
                "ensemble_weights": (
                    list(resolved_dense_ensemble_weights(cfg, paths)) if force_ensemble else None
                ),
                "dense_member_quota": (
                    resolved_dense_member_quota(cfg, paths) if force_ensemble else None
                ),
            })
    if teacher_scorer is not None:
        teacher_scorer.close()
    output_path = (
        paths.reranker_groups if force_ensemble
        else paths.second_stage_groups if force_final
        else paths.groups
    )
    write_jsonl(output_path, groups)
    negative_count = (
        cfg.rerank_hard_negatives if force_ensemble
        else cfg.embed_stage2_hard_negatives if force_final
        else cfg.hard_negatives
    )
    log(f"Mined {len(groups):,} training groups with {negative_count} hard negatives each")
    return groups
