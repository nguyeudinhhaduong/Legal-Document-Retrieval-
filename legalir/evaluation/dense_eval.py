"""Đánh giá dense retriever và hiệu chỉnh ensemble A/B/C.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import json
from typing import Any, Sequence

from legalir.core.config import Config, Paths
from legalir.core.utils import atomic_json, log, read_jsonl, stable_key
from legalir.models.dense_encoder import DenseEncoder, is_bge_m3_model
from legalir.data.chunk_store import load_chunk_arrays
from legalir.retrieval.dense_index import build_dense_index
from legalir.retrieval.dense_fusion import DENSE_FUSION_VERSION, dense_ensemble_fingerprint, dense_ensemble_members, fuse_dense_ensemble_members
from legalir.evaluation.metrics import task_metrics
from legalir.training.embedder import evaluate_pretrain_dense_validation


def evaluate_base_dense(cfg: Config, paths: Paths) -> None:
    """Build/reuse the matching base index and evaluate it without training."""
    index_path = build_dense_index(cfg, paths, base=True)
    encoder = DenseEncoder(cfg, training=False)
    try:
        metrics = evaluate_pretrain_dense_validation(cfg, paths, encoder)
    finally:
        encoder.close()
    result = {
        "model": cfg.embedding_model,
        "pooling": "cls" if is_bge_m3_model(cfg.embedding_model) else "last_token",
        "query_instruction": not is_bge_m3_model(cfg.embedding_model),
        "index": str(index_path),
        "eval_queries": len(read_jsonl(paths.validation)),
        "metrics": metrics,
    }
    atomic_json(paths.root / "base_validation.json", result)
    log(f"Saved base validation: {paths.root / 'base_validation.json'}")


def calibration_locked_indices(
    records: Sequence[dict[str, Any]], calibration_size: int, seed: int
) -> tuple[list[int], list[int]]:
    order = sorted(
        range(len(records)),
        key=lambda index: stable_key(seed + 930, str(records[index]["qid"])),
    )
    selected = set(order[:calibration_size])
    return (
        [index for index in range(len(records)) if index in selected],
        [index for index in range(len(records)) if index not in selected],
    )


def subset_values(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


def dense_weight_grid(member_count: int) -> list[tuple[float, ...]]:
    if member_count == 2:
        return [(step / 10.0, 1.0 - step / 10.0) for step in range(11)]
    if member_count == 3:
        return [
            (a / 10.0, b / 10.0, (10 - a - b) / 10.0)
            for a in range(11) for b in range(11 - a)
        ]
    raise ValueError("Dense weight grid supports two or three members")


def evaluate_dense_ensemble(cfg: Config, paths: Paths) -> dict[str, float]:
    records = read_jsonl(paths.validation)
    _, chunk_cids, _ = load_chunk_arrays(paths.chunks)
    members = dense_ensemble_members(
        cfg, paths, [row["question"] for row in records], chunk_cids,
        cfg.candidate_docs,
    )
    calibration_indices, locked_indices = calibration_locked_indices(
        records, cfg.calibration_size, cfg.seed
    )
    grid: dict[str, dict[str, Any]] = {}
    evaluated: dict[tuple[tuple[float, ...], int], tuple[list[list[dict[str, Any]]], dict[str, float], dict[str, float]]] = {}
    options: list[tuple[tuple[float, float, float, float], tuple[float, ...], int]] = []
    best_ab: tuple[tuple[float, float, float, float], tuple[float, ...], int] | None = None
    calibration_records = subset_values(records, calibration_indices)
    quota_options = sorted({0, min(3, cfg.dense_member_quota), cfg.dense_member_quota})
    for quota in quota_options:
        for weights in dense_weight_grid(len(members)):
            candidates = fuse_dense_ensemble_members(
                members, weights, cfg.candidate_docs, cfg.chunks_per_doc, quota,
            )
            rankings = [[row["cid"] for row in rows] for rows in candidates]
            calibration_rankings = subset_values(rankings, calibration_indices)
            metrics_at_5 = task_metrics(calibration_rankings, calibration_records, 5)
            metrics_at_pool = task_metrics(
                calibration_rankings, calibration_records, cfg.candidate_docs
            )
            key = (
                metrics_at_pool[f"Recall@{cfg.candidate_docs}"],
                metrics_at_5["Recall@5"], metrics_at_5["Precision@5"], metrics_at_5["MRR@5"],
            )
            name = "q" + str(quota) + "_weights_" + "_".join(f"{weight:.1f}" for weight in weights)
            grid[name] = {
                "weights": list(weights), "member_quota": quota,
                "calibration_at_5": metrics_at_5,
                "calibration_at_candidate_pool": metrics_at_pool,
            }
            evaluated[(weights, quota)] = (candidates, metrics_at_5, metrics_at_pool)
            options.append((key, weights, quota))
            is_ab_reference = len(weights) == 2 or (weights[2] == 0.0 and quota == 0)
            if is_ab_reference:
                if best_ab is None:
                    best_ab = (key, weights, quota)
                elif key > best_ab[0]:
                    best_ab = (key, weights, quota)
    if not options or best_ab is None:
        raise RuntimeError("Dense calibration produced no A/B reference configuration")

    def full_and_locked(weights: tuple[float, ...], quota: int) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        rankings = [[row["cid"] for row in rows] for rows in evaluated[(weights, quota)][0]]
        return (
            task_metrics(rankings, records, 5), task_metrics(rankings, records, cfg.candidate_docs),
            task_metrics(subset_values(rankings, locked_indices), subset_values(records, locked_indices), 5),
            task_metrics(subset_values(rankings, locked_indices), subset_values(records, locked_indices), cfg.candidate_docs),
        )

    ab_weights, ab_quota = best_ab[1], best_ab[2]
    ab_full5, ab_full_pool, ab_locked5, ab_locked_pool = full_and_locked(ab_weights, ab_quota)
    selected_weights, selected_quota = ab_weights, ab_quota
    full5, full_pool, locked5, locked_pool = ab_full5, ab_full_pool, ab_locked5, ab_locked_pool
    c_guardrail_passed = len(members) < 3
    locked_top5_tolerance = 0.0021
    if len(members) == 3:
        # Try every C-bearing configuration in calibration order. Previously
        # only rank 1 was checked; if it failed the locked guardrail, a stable
        # rank-2 C configuration was never considered and C was dropped.
        for _, candidate_weights, candidate_quota in sorted(
            options, key=lambda item: item[0], reverse=True
        ):
            if candidate_weights[2] <= 0.0 and candidate_quota <= 0:
                continue
            calibration5 = evaluated[(candidate_weights, candidate_quota)][1]
            calibration_pool = evaluated[(candidate_weights, candidate_quota)][2]
            candidate_full5, candidate_full_pool, candidate_locked5, candidate_locked_pool = (
                full_and_locked(candidate_weights, candidate_quota)
            )
            calibration_safe = (
                calibration_pool[f"Recall@{cfg.candidate_docs}"]
                >= evaluated[(ab_weights, ab_quota)][2][f"Recall@{cfg.candidate_docs}"]
                and calibration5["Recall@5"]
                >= evaluated[(ab_weights, ab_quota)][1]["Recall@5"] - locked_top5_tolerance
            )
            locked_safe = (
                candidate_locked5["Recall@5"]
                >= ab_locked5["Recall@5"] - locked_top5_tolerance
                and candidate_locked_pool[f"Recall@{cfg.candidate_docs}"]
                > ab_locked_pool[f"Recall@{cfg.candidate_docs}"]
            )
            if calibration_safe and locked_safe:
                selected_weights, selected_quota = candidate_weights, candidate_quota
                full5, full_pool = candidate_full5, candidate_full_pool
                locked5, locked_pool = candidate_locked5, candidate_locked_pool
                c_guardrail_passed = True
                break
    result = {
        "schema": 4, "fusion_version": DENSE_FUSION_VERSION,
        "eval_queries": len(records), "member_count": len(members),
        "split_seed": cfg.seed, "eval_size": cfg.eval_size,
        "candidate_docs": cfg.candidate_docs, "dense_chunk_k": cfg.dense_chunk_k,
        "dense_member_quota": cfg.dense_member_quota,
        "best_member_quota": selected_quota,
        "fusion": "weighted reciprocal-rank fusion",
        "selection_split": {"calibration": len(calibration_indices), "locked": len(locked_indices)},
        "selection_rule": (
            f"calibration Recall@{cfg.candidate_docs}, then Recall@5; choose first "
            f"C config with strict locked Recall@{cfg.candidate_docs} gain"
        ),
        "best_weights": list(selected_weights),
        "best_stage1_weight": selected_weights[0], "best_stage2_weight": selected_weights[1],
        "best_stage3_weight": selected_weights[2] if len(selected_weights) == 3 else 0.0,
        "c_guardrail_passed": c_guardrail_passed,
        "c_guardrail_locked_top5_tolerance": 0.0021,
        "metrics_at_5": full5, "metrics_at_candidate_pool": full_pool,
        "locked_metrics_at_5": locked5, "locked_metrics_at_candidate_pool": locked_pool,
        "ab_reference": {"weights": list(ab_weights), "member_quota": ab_quota, "metrics_at_5": ab_full5,
                         "metrics_at_candidate_pool": ab_full_pool,
                         "locked_at_5": ab_locked5, "locked_at_candidate_pool": ab_locked_pool},
        "embedding_fingerprint": dense_ensemble_fingerprint(paths), "grid": grid,
    }
    atomic_json(paths.dense_calibration, result)
    log(f"{len(members)}-LoRA dense ensemble calibration: " + json.dumps({
        "weights": result["best_weights"], "member_quota": selected_quota,
        "C_guardrail": c_guardrail_passed,
        "metrics_at_5": full5, "metrics_at_candidate_pool": full_pool,
        "locked_at_5": locked5,
    }, ensure_ascii=False))
    return full5
