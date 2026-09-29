"""Kiểm tra, ghi submission và file debug top-10.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import json
import math
import os
import shutil
import zipfile
from typing import Any, Sequence

from legalir.core.config import Config, Paths
from legalir.core.utils import atomic_json, log, minmax, read_jsonl
from legalir.data.chunk_store import load_chunk_arrays
from legalir.rerank.candidates import ranking_pipeline_fingerprint, retrieve_and_rerank
from legalir.evaluation.metrics import validate_submission
from legalir.rerank.repair_fusion import REPAIR_SIGNALS, fused_repair_rankings, precompute_repair_features, repair_signal_scores
from legalir.rerank.meta_ranker import META_SIGNALS, meta_ranker_feature_matrix, meta_ranker_rankings


def final_score_by_cid(
    rows: Sequence[dict[str, Any]], ranking_mode: str,
    repair_config: dict[str, Any], meta_model: dict[str, Any] | None,
) -> dict[str, float]:
    """Return the exact score used to sort each public-test candidate."""
    if ranking_mode == "listwise_meta_oof":
        if not meta_model:
            raise ValueError("Meta-ranker model is required for public debug scoring")
        import numpy as np
        matrix, names = meta_ranker_feature_matrix(rows, int(meta_model["evidence_k"]))
        if names != list(meta_model["feature_names"]):
            raise ValueError("Meta-ranker feature order changed while writing public debug")
        mean = np.asarray(meta_model["mean"], dtype="float64")
        scale = np.asarray(meta_model["scale"], dtype="float64")
        weights = np.asarray(meta_model["weights"], dtype="float64")
        scores = ((matrix - mean) / scale) @ weights
        return {str(row["cid"]): float(score) for row, score in zip(rows, scores)}
    cached = precompute_repair_features(
        [rows], int(repair_config["evidence_k"]), float(repair_config["rrf_constant"])
    )[0]
    weight_vector = [
        float(repair_config["weights"].get(signal, 0.0)) for signal in REPAIR_SIGNALS
    ]
    total = sum(weight_vector)
    if total <= 0.0:
        raise ValueError("At least one RRF weight must be positive")
    weight_vector = [weight / total for weight in weight_vector]
    scores = [
        sum(weight * feature for weight, feature in zip(weight_vector, features))
        for features in cached["features"]
    ]
    return {str(row["cid"]): float(score) for row, score in zip(rows, scores)}


def save_public_top10_debug(
    cfg: Config, paths: Paths, records: Sequence[dict[str, Any]],
    rows_by_query: Sequence[Sequence[dict[str, Any]]], rankings: Sequence[Sequence[str]],
    ranking_mode: str, repair_config: dict[str, Any], meta_model: dict[str, Any] | None,
) -> None:
    """Write downloadable JSONL/ZIP and print public-test ranks 1..10."""
    if not (len(records) == len(rows_by_query) == len(rankings)):
        raise ValueError("Public debug inputs must contain the same number of queries")
    chunk_texts, _, _ = load_chunk_arrays(paths.chunks)
    document_names = {
        str(row["cid"]): str(row.get("name", "")) for row in read_jsonl(paths.documents)
    }

    def finite(value: Any) -> float | None:
        number = float(value)
        return number if math.isfinite(number) else None

    temp = paths.public_debug_jsonl.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record, rows, ranking in zip(records, rows_by_query, rankings):
            row_by_cid = {str(row["cid"]): row for row in rows}
            final_scores = final_score_by_cid(rows, ranking_mode, repair_config, meta_model)
            final_values = [final_scores[str(row["cid"])] for row in rows]
            final_normalized = dict(zip(
                (str(row["cid"]) for row in rows), minmax(final_values)
            ))
            raw_by_signal = {
                signal: repair_signal_scores(rows, signal, int(repair_config["evidence_k"]))
                for signal in REPAIR_SIGNALS
            }
            raw_by_signal["bm25"] = [float(row.get("bm25", 0.0)) for row in rows]
            signal_debug: dict[str, dict[str, tuple[float | None, float, int]]] = {}
            for signal, raw_values in raw_by_signal.items():
                finite_values = [value for value in raw_values if math.isfinite(value)]
                floor = min(finite_values, default=0.0) - 1.0
                safe_values = [value if math.isfinite(value) else floor for value in raw_values]
                normalized = minmax(safe_values)
                order = sorted(
                    range(len(rows)),
                    key=lambda index: (-safe_values[index], str(rows[index]["cid"])),
                )
                rank_by_index = {index: rank for rank, index in enumerate(order, start=1)}
                signal_debug[signal] = {
                    str(row["cid"]): (finite(raw_values[index]), normalized[index], rank_by_index[index])
                    for index, row in enumerate(rows)
                }

            top10: list[dict[str, Any]] = []
            submission_answer = list(map(str, ranking[:5]))
            log(f'[PUBLIC TOP10] qid={record["qid"]} | {str(record["question"])[:220]}')
            for final_rank, cid in enumerate(list(map(str, ranking[:10])), start=1):
                row = row_by_cid[cid]
                r1_chunks = list(map(float, row.get("rerank_r1_chunks", [])))
                r2_chunks = list(map(float, row.get("rerank_r2_chunks", [])))
                chunks = []
                for evidence_rank, chunk_id in enumerate(row.get("chunks", []), start=1):
                    index = evidence_rank - 1
                    chunks.append({
                        "evidence_rank": evidence_rank,
                        "chunk_id": int(chunk_id),
                        "text": chunk_texts[int(chunk_id)],
                        "r1_score": finite(r1_chunks[index]) if index < len(r1_chunks) else None,
                        "r2_score": finite(r2_chunks[index]) if index < len(r2_chunks) else None,
                    })
                scores = {
                    signal: {
                        "raw": signal_debug[signal][cid][0],
                        "normalized_0_1": signal_debug[signal][cid][1],
                        "rank_in_pool": signal_debug[signal][cid][2],
                    }
                    for signal in META_SIGNALS
                }
                item = {
                    "final_rank": final_rank,
                    "selected_for_submission": final_rank <= 5,
                    "cid": cid,
                    "document_name": document_names.get(cid, ""),
                    "final_score": finite(final_scores[cid]),
                    "final_score_normalized_0_1": final_normalized[cid],
                    "dense_union_rrf": finite(row.get("dense", 0.0)),
                    "scores": scores,
                    "chunks": chunks,
                }
                top10.append(item)
                log(
                    f'  #{final_rank:02d} cid={cid} final={item["final_score"]:.6f} '
                    f'norm={item["final_score_normalized_0_1"]:.4f} '
                    f'A={scores["a"]["raw"]}(r{scores["a"]["rank_in_pool"]}) '
                    f'B={scores["b"]["raw"]}(r{scores["b"]["rank_in_pool"]}) '
                    f'C={scores["c"]["raw"]}(r{scores["c"]["rank_in_pool"]}) '
                    f'R1={scores["r1"]["raw"]}(r{scores["r1"]["rank_in_pool"]}) '
                    f'R2={scores["r2"]["raw"]}(r{scores["r2"]["rank_in_pool"]}) '
                    f'BM25={scores["bm25"]["raw"]}(r{scores["bm25"]["rank_in_pool"]})'
                )
            log(
                f'[PUBLIC SUBMISSION] qid={record["qid"]} '
                f'answer={json.dumps(submission_answer, ensure_ascii=False)}'
            )
            handle.write(json.dumps({
                "schema": 1,
                "qid": str(record["qid"]),
                "question": str(record["question"]),
                "ranking_mode": ranking_mode,
                "submission_answer": submission_answer,
                "top_k": 10,
                "top10": top10,
            }, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, paths.public_debug_jsonl)
    zip_temp = paths.public_debug_zip.with_suffix(".zip.tmp")
    with zipfile.ZipFile(zip_temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(paths.public_debug_jsonl, arcname=paths.public_debug_jsonl.name)
    os.replace(zip_temp, paths.public_debug_zip)
    log(
        f"Public top-10 debug saved: JSONL={paths.public_debug_jsonl.resolve()}, "
        f"download ZIP={paths.public_debug_zip.resolve()}"
    )


def predict_submission(cfg: Config, paths: Paths) -> None:
    test = read_jsonl(paths.test); calibration = json.loads(paths.calibration.read_text(encoding="utf-8"))
    if calibration.get("schema") != 7:
        raise ValueError(
            "Calibration predates supported ranking; run --stage eval again before prediction"
        )
    expected_fingerprint = ranking_pipeline_fingerprint(cfg, paths)
    if calibration.get("pipeline_fingerprint") != expected_fingerprint:
        raise ValueError(
            "Calibration does not match the current chunks/A/B/C/R1/R2/fusion recipe; "
            "run --stage eval again before prediction"
        )
    weights = {
        str(signal): float(weight)
        for signal, weight in calibration["voting_weights"].items()
    }
    repair_config = {
        "weights": weights,
        "evidence_k": int(calibration["voting_evidence_k"]),
        "rrf_constant": float(calibration["voting_rrf_constant"]),
    }
    top_configs = calibration.get("top_configs", [])
    if top_configs:
        log("Top calibration configs (rank: weights, Recall@5): " + "; ".join(
            f'{c["rank"]}: {c["weights"]} R@5={c["calibration_metrics"]["Recall@5"]:.4f}'
            for c in top_configs
        ))
    ranking_mode = str(calibration.get("ranking_mode", "rrf_baseline"))
    log(f"Using ranking_mode={ranking_mode}; RRF fallback weights={weights}")
    rows = retrieve_and_rerank(cfg, paths, test)
    meta_model: dict[str, Any] | None = None
    if ranking_mode == "listwise_meta_oof":
        meta = calibration.get("meta_ranker", {})
        if not meta.get("accepted") or not isinstance(meta.get("model"), dict):
            raise ValueError("Accepted meta-ranker model is missing; rerun --stage eval")
        meta_model = meta["model"]
        rankings = meta_ranker_rankings(rows, meta_model)
    else:
        rankings = fused_repair_rankings(rows, repair_config)
    save_public_top10_debug(
        cfg, paths, test, rows, rankings, ranking_mode, repair_config, meta_model,
    )
    submission = {
        record["qid"]: {"answer": ranking[:5]}
        for record, ranking in zip(test, rankings)
    }
    valid_cids = {row["cid"] for row in read_jsonl(paths.documents)}; validate_submission(submission, test, valid_cids)
    atomic_json(paths.submission_json, submission); temp = paths.submission_zip.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(paths.submission_json, arcname="submission.json")
    os.replace(temp, paths.submission_zip)
    with zipfile.ZipFile(paths.submission_zip) as archive:
        if archive.namelist() != ["submission.json"]: raise AssertionError("ZIP must contain only submission.json")
        packed = json.loads(archive.read("submission.json").decode("utf-8"))
    validate_submission(packed, test, valid_cids)
    if not calibration.get("new_pipeline_accepted", True) and not paths.baseline_submission_zip.exists():
        shutil.copy2(paths.submission_zip, paths.baseline_submission_zip)
    log(
        f"Submission validated: {len(test)} queries; "
        f"JSON={paths.submission_json.resolve()}; download ZIP={paths.submission_zip.resolve()}"
    )
