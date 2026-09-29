"""Hợp nhất nhiều dense retriever A/B/C thành candidate pool.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from legalir._deps import faiss
from legalir.core.config import Config, Paths
from legalir.core.utils import progress
from legalir.core.artifacts import artifact_identity, embedding_ensemble_identity, valid_lora_adapter
from legalir.models.dense_encoder import DenseEncoder
from legalir.retrieval.dense_index import dense_candidates


DENSE_FUSION_VERSION = "weighted_rrf_union_round_robin_evidence_v3"


def dense_ensemble_members(
    cfg: Config, paths: Paths, questions: Sequence[str], chunk_cids: Sequence[str],
    model_limit: int,
) -> list[list[list[dict[str, Any]]]]:
    """Retrieve each LoRA once so validation can calibrate fusion cheaply."""
    specifications = [
        (paths.embed_adapter, paths.final_index),
        (paths.second_embed_adapter, paths.second_index),
    ]
    if valid_lora_adapter(paths.third_embed_adapter) and paths.third_index.is_file():
        specifications.append((paths.third_embed_adapter, paths.third_index))
    per_model: list[list[list[dict[str, Any]]]] = []
    # Retrieve deeper from each encoder before RRF truncation. Taking only N
    # from each model to build a final top-N pool makes the union needlessly
    # brittle and discards complementary candidates ranked just below N.
    per_model_limit = min(cfg.dense_chunk_k, max(model_limit, model_limit * 3))
    for adapter, index_path in specifications:
        if not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing BGE-M3 LoRA adapter for ensemble: {adapter}")
        if not index_path.exists():
            raise FileNotFoundError(f"Missing BGE-M3 FAISS index for ensemble: {index_path}")
        encoder = DenseEncoder(cfg, adapter=adapter)
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(chunk_cids) or index.d != cfg.embedding_dim:
            encoder.close()
            raise ValueError(
                f"Ensemble FAISS mismatch for {index_path}: vectors={index.ntotal}, "
                f"dimension={index.d}, expected vectors={len(chunk_cids)}, "
                f"dimension={cfg.embedding_dim}"
            )
        try:
            per_model.append(
                dense_candidates(
                    cfg, encoder, index, questions, chunk_cids, limit=per_model_limit
                )
            )
        finally:
            encoder.close()
    return per_model


def fuse_dense_ensemble_members(
    per_model: Sequence[Sequence[Sequence[dict[str, Any]]]],
    stage1_weight: float | Sequence[float], model_limit: int, chunks_per_doc: int,
    member_quota: int = 0,
) -> list[list[dict[str, Any]]]:
    if len(per_model) not in {2, 3}:
        raise ValueError("Dense ensemble requires two or three member rankings")
    if isinstance(stage1_weight, (float, int)):
        specifications = (float(stage1_weight), 1.0 - float(stage1_weight))
    else:
        specifications = tuple(map(float, stage1_weight))
    if len(specifications) != len(per_model):
        raise ValueError("Dense ensemble weights must align with member rankings")
    if any(weight < 0.0 for weight in specifications) or not math.isclose(sum(specifications), 1.0, abs_tol=1e-6):
        raise ValueError("Dense ensemble weights must be non-negative and sum to one")

    outputs: list[list[dict[str, Any]]] = []
    rrf_constant = 60.0
    for query_index in range(len(per_model[0])):
        required_by_member = [
            {
                str(row["cid"])
                for row in rows_by_query[query_index][:member_quota]
            }
            if member_quota > 0 else set()
            for rows_by_query in per_model
        ]
        merged: dict[str, dict[str, Any]] = {}
        for model_index, rows_by_query in enumerate(per_model):
            weight = specifications[model_index]
            for rank, row in enumerate(rows_by_query[query_index], start=1):
                item = merged.setdefault(row["cid"], {
                    "cid": row["cid"], "dense": 0.0,
                    "_chunks_by_member": [[] for _ in per_model],
                })
                item["dense"] += weight / (rrf_constant + rank)
                item["_chunks_by_member"][model_index].extend(
                    chunk for chunk in row["chunks"]
                    if chunk not in item["_chunks_by_member"][model_index]
                )
        ranked = sorted(merged.values(), key=lambda row: (-float(row["dense"]), row["cid"]))
        if member_quota > 0:
            # Reserve complementary evidence without changing the RRF order
            # inside the selected pool. A member with calibration weight zero
            # can still contribute a few genuinely novel candidates.
            required = set().union(*required_by_member)
            selected_cids = set(required)
            for row in ranked:
                if len(selected_cids) >= model_limit:
                    break
                selected_cids.add(str(row["cid"]))
            ranked = [row for row in ranked if str(row["cid"]) in selected_cids]
        # Do not concatenate A->B->C and truncate: that silently gave all
        # evidence slots to A whenever A already had chunks_per_doc chunks.
        # Interleave members with positive fusion weight plus any member whose
        # quota introduced this CID. This preserves B-only behavior when B is
        # the sole voter while exposing complementary C evidence to R1/R2.
        member_order = sorted(
            range(len(specifications)), key=lambda index: (-specifications[index], index)
        )
        for row in ranked:
            cid = str(row["cid"])
            chunks_by_member = row.pop("_chunks_by_member")
            active_members = [
                index for index in member_order
                if chunks_by_member[index]
                and (specifications[index] > 0.0 or cid in required_by_member[index])
            ]
            if not active_members:
                active_members = [index for index in member_order if chunks_by_member[index]]
            selected_chunks: list[int] = []
            maximum_depth = max((len(chunks_by_member[index]) for index in active_members), default=0)
            for depth in range(maximum_depth):
                for model_index in active_members:
                    values = chunks_by_member[model_index]
                    if depth < len(values) and values[depth] not in selected_chunks:
                        selected_chunks.append(values[depth])
                        if len(selected_chunks) >= chunks_per_doc:
                            break
                if len(selected_chunks) >= chunks_per_doc:
                    break
            row["chunks"] = selected_chunks
        outputs.append(ranked[:model_limit])
    return outputs


def dense_ensemble_fingerprint(paths: Paths) -> dict[str, list[int | None]]:
    identity = embedding_ensemble_identity(paths)
    identity.update({
        "stage3_adapter": list(artifact_identity(paths.third_embed_adapter / "adapter_model.safetensors")),
        "stage3_index": list(artifact_identity(paths.third_index)),
    })
    return {key: values[1:] for key, values in identity.items()}


def dense_calibration_is_valid(cfg: Config, paths: Paths) -> bool:
    if not paths.dense_calibration.exists(): return False
    try:
        calibration = json.loads(paths.dense_calibration.read_text(encoding="utf-8"))
        return (
            calibration.get("schema") == 4
            and calibration.get("fusion_version") == DENSE_FUSION_VERSION
            and calibration.get("candidate_docs") == cfg.candidate_docs
            and calibration.get("dense_member_quota") == cfg.dense_member_quota
            and calibration.get("dense_chunk_k") == cfg.dense_chunk_k
            and calibration.get("split_seed") == cfg.seed
            and calibration.get("eval_size") == cfg.eval_size
            and calibration.get("embedding_fingerprint") == dense_ensemble_fingerprint(paths)
            and (
                "best_weights" in calibration
                or 0.0 <= float(calibration["best_stage1_weight"]) <= 1.0
            )
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


def resolved_dense_ensemble_weights(cfg: Config, paths: Paths) -> tuple[float, ...]:
    if dense_calibration_is_valid(cfg, paths):
        try:
            calibration = json.loads(paths.dense_calibration.read_text(encoding="utf-8"))
            if "best_weights" in calibration:
                return tuple(map(float, calibration["best_weights"]))
            weight = float(calibration["best_stage1_weight"])
            return (weight, 1.0 - weight)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    if valid_lora_adapter(paths.third_embed_adapter) and paths.third_index.is_file():
        return (cfg.ensemble_stage1_weight, 1.0 - cfg.ensemble_stage1_weight, 0.0)
    return (cfg.ensemble_stage1_weight, 1.0 - cfg.ensemble_stage1_weight)


def resolved_dense_ensemble_weight(cfg: Config, paths: Paths) -> float:
    """Compatibility accessor used by old R1 fingerprints."""
    return resolved_dense_ensemble_weights(cfg, paths)[0]


def resolved_dense_member_quota(cfg: Config, paths: Paths) -> int:
    if dense_calibration_is_valid(cfg, paths):
        try:
            calibration = json.loads(paths.dense_calibration.read_text(encoding="utf-8"))
            return int(calibration.get("best_member_quota", cfg.dense_member_quota))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    return cfg.dense_member_quota


def resolved_ab_weights(cfg: Config, paths: Paths) -> tuple[float, float]:
    if dense_calibration_is_valid(cfg, paths):
        calibration = json.loads(paths.dense_calibration.read_text(encoding="utf-8"))
        reference = calibration.get("ab_reference", {}).get("weights")
        if isinstance(reference, list) and len(reference) >= 2:
            weights = tuple(map(float, reference[:2]))
            total = sum(weights)
            if total > 0.0: return weights[0] / total, weights[1] / total
    return cfg.ensemble_stage1_weight, 1.0 - cfg.ensemble_stage1_weight


def ensemble_dense_candidates(
    cfg: Config, paths: Paths, questions: Sequence[str], chunk_cids: Sequence[str],
    limit: int | None = None, stage1_weight: float | Sequence[float] | None = None,
) -> list[list[dict[str, Any]]]:
    """Fuse two independently trained BGE-M3 LoRAs with calibrated weighted RRF."""
    model_limit = limit or cfg.candidate_docs
    members = dense_ensemble_members(cfg, paths, questions, chunk_cids, model_limit)
    weight = resolved_dense_ensemble_weights(cfg, paths) if stage1_weight is None else stage1_weight
    return fuse_dense_ensemble_members(
        members, weight, model_limit, cfg.chunks_per_doc,
        resolved_dense_member_quota(cfg, paths),
    )


def union_abc_pool_with_dense_votes(
    cfg: Config, paths: Paths, records: Sequence[dict[str, Any]],
    chunk_cids: Sequence[str],
) -> list[list[dict[str, Any]]]:
    """Retrieve with A, B and C, union the pools, and retain each dense vote.

    The previous public pipeline retrieved with B alone and only rescored B's
    documents with A/C. That made B's candidate recall a hard ceiling. Here
    each model can introduce documents; calibrated RRF fills most slots and a
    small per-member quota protects complementary specialists.
    """
    questions = [str(row["question"]) for row in records]
    members = dense_ensemble_members(
        cfg, paths, questions, chunk_cids, cfg.candidate_docs
    )
    weights = resolved_dense_ensemble_weights(cfg, paths)
    candidates = fuse_dense_ensemble_members(
        members, weights, cfg.candidate_docs, cfg.chunks_per_doc,
        resolved_dense_member_quota(cfg, paths),
    )
    signal_names = ("dense_a", "dense_b", "dense_c")
    if len(members) != len(signal_names):
        raise ValueError("A/B/C union retrieval requires all three embedding members")
    for query_index, rows in enumerate(candidates):
        for signal_name, member_rows in zip(signal_names, members):
            score_by_cid = {
                str(row["cid"]): float(row["dense"])
                for row in member_rows[query_index]
            }
            floor = min(score_by_cid.values(), default=0.0) - 1e-3
            for row in rows:
                row[signal_name] = score_by_cid.get(str(row["cid"]), floor)
    return candidates


def require_ensemble_voters(paths: Paths) -> None:
    """Validate every checkpoint used by A/B/C + R1/R2 voting."""
    required = (
        ("embedding LoRA A", paths.embed_adapter, paths.final_index),
        ("embedding LoRA B", paths.second_embed_adapter, paths.second_index),
        ("embedding LoRA C", paths.third_embed_adapter, paths.third_index),
        ("reranker R1", paths.rerank_adapter, None),
        ("reranker R2", paths.second_rerank_adapter, None),
    )
    missing = []
    for name, adapter, index_path in required:
        if not valid_lora_adapter(adapter):
            missing.append(f"{name} adapter: {adapter}")
        if index_path is not None and not index_path.is_file():
            missing.append(f"{name} index: {index_path}")
    if missing:
        raise FileNotFoundError(
            "Ensemble voting requires all A/B/C/R1/R2 artifacts; missing:\n- "
            + "\n- ".join(missing)
        )


def b_fixed_pool_candidates(
    cfg: Config, paths: Paths, questions: Sequence[str], chunk_cids: Sequence[str],
) -> list[list[dict[str, Any]]]:
    """Retrieve the immutable document pool with embedding LoRA B only."""
    encoder = DenseEncoder(cfg, adapter=paths.second_embed_adapter)
    index = faiss.read_index(str(paths.second_index))
    if index.ntotal != len(chunk_cids) or index.d != cfg.embedding_dim:
        encoder.close()
        raise ValueError(
            f"LoRA B FAISS mismatch: vectors={index.ntotal}, dimension={index.d}; "
            f"expected vectors={len(chunk_cids)}, dimension={cfg.embedding_dim}"
        )
    try:
        candidates = dense_candidates(
            cfg, encoder, index, questions, chunk_cids, limit=cfg.candidate_docs
        )
    finally:
        encoder.close()
        del index
        gc.collect()
    for rows in candidates:
        for row in rows:
            row["dense_b"] = float(row["dense"])
    return candidates


def score_fixed_pool_with_dense_lora(
    cfg: Config, adapter: Path, index_path: Path, signal_name: str,
    questions: Sequence[str], candidates: Sequence[list[dict[str, Any]]],
    chunk_cids: Sequence[str],
) -> None:
    """Score B's fixed documents with another LoRA without changing the pool.

    Each document is scored by the maximum inner product over all of its chunks
    in that LoRA's own FAISS index.  This is an exact fixed-pool score: A/C do
    not need to retrieve the document in their own top-k to cast a vote.
    """
    import numpy as np

    doc_to_chunks: dict[str, list[int]] = defaultdict(list)
    for chunk_id, cid in enumerate(chunk_cids):
        doc_to_chunks[str(cid)].append(chunk_id)

    encoder = DenseEncoder(cfg, adapter=adapter)
    index = faiss.read_index(str(index_path))
    if index.ntotal != len(chunk_cids) or index.d != cfg.embedding_dim:
        encoder.close()
        raise ValueError(
            f"{signal_name} FAISS mismatch: vectors={index.ntotal}, dimension={index.d}; "
            f"expected vectors={len(chunk_cids)}, dimension={cfg.embedding_dim}"
        )
    try:
        query_matrix = encoder.encode(
            questions, True, cfg.query_batch_size, f"Encode queries ({signal_name})"
        )[:, :cfg.embedding_dim].astype("float32")
        faiss.normalize_L2(query_matrix)
        for query_index, rows in progress(
            enumerate(candidates), total=len(candidates), desc=f"Score B pool with {signal_name}"
        ):
            chunk_ids: list[int] = []
            owners: list[int] = []
            for owner, row in enumerate(rows):
                ids = doc_to_chunks.get(str(row["cid"]), [])
                chunk_ids.extend(ids)
                owners.extend([owner] * len(ids))
            best = [-math.inf] * len(rows)
            if chunk_ids:
                ids_array = np.asarray(chunk_ids, dtype="int64")
                try:
                    vectors = np.asarray(index.reconstruct_batch(ids_array), dtype="float32")
                except (AttributeError, TypeError):
                    vectors = np.stack([index.reconstruct(int(chunk_id)) for chunk_id in chunk_ids])
                scores = vectors @ query_matrix[query_index]
                for owner, score in zip(owners, scores):
                    best[owner] = max(best[owner], float(score))
            for row, score in zip(rows, best):
                row[signal_name] = score
    finally:
        encoder.close()
        del index
        gc.collect()


def fixed_b_pool_with_dense_votes(
    cfg: Config, paths: Paths, records: Sequence[dict[str, Any]], chunk_cids: Sequence[str],
) -> list[list[dict[str, Any]]]:
    """Run B retrieval once, then let A and C score exactly B's configured pool."""
    questions = [str(row["question"]) for row in records]
    candidates = b_fixed_pool_candidates(cfg, paths, questions, chunk_cids)
    score_fixed_pool_with_dense_lora(
        cfg, paths.embed_adapter, paths.final_index, "dense_a",
        questions, candidates, chunk_cids,
    )
    score_fixed_pool_with_dense_lora(
        cfg, paths.third_embed_adapter, paths.third_index, "dense_c",
        questions, candidates, chunk_cids,
    )
    return candidates
