"""Nạp model đã train và dự đoán file test mới (không train).

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import numpy as np
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

from legalir.core.config import Config
from legalir.core.utils import atomic_json, log, write_jsonl
from legalir.core.artifacts import valid_lora_adapter
from legalir.data.loading import read_query_object
from legalir.data.chunk_store import load_chunk_arrays
from legalir.experiment.constants import ADAPTERS, INDEXES
from legalir.experiment.utils import adapter_identity, identity, read
from legalir.rerank.fusion import FIXED, SIGNALS, rank_features, rankings
from legalir.experiment.experiment import Experiment


def _loadmodel_require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _loadmodel_require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _resolve_loadmodel_model_root(cfg: Config) -> Path:
    """Accept either the exact run or its Kaggle notebook-output parent."""
    # An explicit --work-dir remains backward compatible and takes priority.
    requested = Path(
        cfg.work_dir
        if cfg.work_dir != Config.work_dir
        else cfg.loadmodel_model_dir
    )
    _loadmodel_require_dir(requested, "trained model/notebook output")

    candidates = [
        requested,
        requested / "legalir_brepair",
        requested / "runs" / "legalir_brepair",
    ]
    checked: list[Path] = []
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.append(candidate)
        if (
            (candidate / "c_complete.json").is_file()
            and (candidate / "r2_complete.json").is_file()
        ):
            log(f"Resolved completed C/R2 model run: {candidate}")
            return candidate
    for receipt in sorted(requested.rglob("c_complete.json")):
        candidate = receipt.parent
        if candidate in checked:
            continue
        checked.append(candidate)
        if (candidate / "r2_complete.json").is_file():
            log(f"Resolved completed C/R2 model run: {candidate}")
            return candidate
    raise FileNotFoundError(
        "Could not find a completed C/R2 run below "
        f"{requested}. Expected c_complete.json and r2_complete.json in one "
        "directory; checked: " + ", ".join(map(str, checked))
    )


def _resolve_loadmodel_test_file(cfg: Config, model_root: Path, source_root: Path) -> Path:
    """Resolve a relative private test path from useful launch locations."""
    requested = Path(cfg.test_file)
    if requested.is_absolute():
        return _loadmodel_require_file(requested, "test query file")
    script_dir = (
        Path(__file__).resolve().parents[2]
    )
    roots = [Path.cwd(), script_dir]
    data_path = Path(cfg.data)
    if data_path.is_dir():
        roots.append(data_path)
    roots.extend((model_root, model_root.parent, source_root, source_root.parent))
    candidates: list[Path] = []
    for root in roots:
        candidate = root / requested
        if candidate not in candidates:
            candidates.append(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Test query file not found: {cfg.test_file}. Checked: "
        + ", ".join(map(str, candidates))
    )


def _loadmodel_output_root(cfg: Config, model_root: Path) -> Path:
    if cfg.loadmodel_output_dir:
        output = Path(cfg.loadmodel_output_dir)
    elif Path("/kaggle/working").is_dir():
        output = Path("/kaggle/working/legalir_private")
    else:
        output = model_root.parent / "private_predictions"
    output = output.resolve()
    model = model_root.resolve()
    source = Path(cfg.cr2_source.replace("\\_", "_")).resolve()
    if output in {model, source} or model in output.parents or source in output.parents:
        raise ValueError(
            "--loadmodel-output-dir must be outside the source/model run so "
            "inference never modifies trained artifacts"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _loadmodel_validate_artifacts(source_root: Path, model_root: Path) -> None:
    for name in ("chunks.jsonl", "documents.jsonl", "chunk_meta.json"):
        _loadmodel_require_file(
            source_root / "prepared" / name, f"source prepared/{name}"
        )
    for index, name in enumerate(INDEXES):
        root = source_root if index < 2 else model_root
        _loadmodel_require_file(root / name, f"dense index {name}")
    _loadmodel_require_file(
        source_root / f"{INDEXES[0]}.meta.json", "A index metadata"
    )
    for name in ADAPTERS:
        adapter = model_root / name
        if not valid_lora_adapter(adapter):
            raise FileNotFoundError(f"Incomplete trained adapter: {adapter}")
    for receipt_name, adapter_name in (
        ("c_complete.json", ADAPTERS[2]),
        ("r2_complete.json", ADAPTERS[4]),
    ):
        receipt_path = _loadmodel_require_file(
            model_root / receipt_name, f"completion receipt {receipt_name}"
        )
        receipt = read(receipt_path)
        if receipt.get("adapter") != adapter_identity(model_root / adapter_name):
            raise ValueError(
                f"{receipt_name} does not match selected adapter {adapter_name}"
            )
    if not any(
        (model_root / name).is_file()
        for name in ("weight_search.json", "submission_experiments.json")
    ):
        raise FileNotFoundError(
            f"No saved fusion weights under {model_root}; expected "
            "weight_search.json or submission_experiments.json"
        )


def _loadmodel_config(cfg: Config, source_root: Path, model_root: Path) -> Config:
    """Restore inference-critical values while retaining explicit model paths."""
    result = copy.deepcopy(cfg)
    config_path = model_root / "run_config.json"
    if not config_path.is_file():
        config_path = source_root / "run_config.json"
    saved = read(config_path) if config_path.is_file() else {}
    for name in (
        "candidate_docs", "dense_chunk_k", "chunks_per_doc",
        "embedding_dim", "embedding_max_length", "reranker_max_length",
    ):
        if name in saved:
            setattr(result, name, saved[name])

    index_meta = read(source_root / f"{INDEXES[0]}.meta.json")
    result.embedding_dim = int(index_meta["embedding_dim"])
    result.embedding_max_length = int(index_meta["embedding_max_length"])
    if (
        cfg.embedding_model == Config.embedding_model
        and index_meta.get("base_model")
        and Path(str(index_meta["base_model"])).is_dir()
    ):
        result.embedding_model = str(index_meta["base_model"])
    saved_reranker = saved.get("reranker_model")
    if (
        cfg.reranker_model == Config.reranker_model
        and saved_reranker
        and Path(str(saved_reranker)).is_dir()
    ):
        result.reranker_model = str(saved_reranker)
    result.work_dir = str(model_root)
    result.cr2_source = str(source_root)
    result.rerank_top_k = int(result.candidate_docs)
    result.rerank_train_chunks_per_doc = int(result.chunks_per_doc)
    if (
        result.candidate_docs < 5
        or result.chunks_per_doc < 1
        or result.dense_chunk_k < result.candidate_docs
    ):
        raise ValueError(
            "Invalid saved retrieval budget: candidate_docs>=5, chunks_per_doc>=1 "
            "and dense_chunk_k>=candidate_docs are required"
        )
    return result


def _loadmodel_weights(model_root: Path) -> dict[str, tuple[float, ...]]:
    weight_path = model_root / "weight_search.json"
    if not weight_path.is_file():
        weight_path = model_root / "submission_experiments.json"
    raw = read(weight_path).get("weights")
    if not isinstance(raw, dict):
        raise ValueError(f"Saved weight file has no weight mapping: {weight_path}")
    expected_names = ("best_val", "fixed", "positive_grid")
    result: dict[str, tuple[float, ...]] = {}
    for name in expected_names:
        if name not in raw:
            continue
        values = raw[name]
        if isinstance(values, dict):
            try:
                weights = tuple(float(values[signal]) for signal in SIGNALS)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid saved weights {name}: {values}") from exc
        else:
            try:
                weights = tuple(map(float, values))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid saved weights {name}: {values}") from exc
        if (
            len(weights) != len(SIGNALS)
            or not all(math.isfinite(value) and value >= 0.0 for value in weights)
            or sum(weights) <= 0.0
        ):
            raise ValueError(f"Invalid saved weights {name}: {values}")
        result[name] = weights
    missing = [name for name in expected_names if name not in result]
    if missing:
        raise ValueError(
            f"Saved weight file must contain all three modes {expected_names}; "
            f"missing={missing}: {weight_path}"
        )
    if not np.allclose(result["fixed"], FIXED, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Saved fixed weights use {result['fixed']}, expected {FIXED} "
            f"in signal order {SIGNALS}"
        )
    if any(
        value < 0.1 or value > 1.0
        or not math.isclose(value * 10.0, round(value * 10.0), abs_tol=1e-9)
        for value in result["positive_grid"]
    ):
        raise ValueError(
            "positive_grid must use independent A/B/C/R1/R2 weights from 0.1 "
            "through 1.0 in increments of 0.1"
        )
    return result


def _loadmodel_experiment(
    cfg: Config, source_root: Path, model_root: Path, cache_root: Path,
) -> Experiment:
    """Construct inference state without the training manifest/resume path."""
    experiment = Experiment.__new__(Experiment)
    experiment.api = importlib.import_module("legalir.api")
    experiment.cfg = cfg
    experiment.source = source_root
    experiment.root = model_root
    experiment.cache = cache_root
    experiment.cache.mkdir(parents=True, exist_ok=True)
    experiment._file_hashes = {}
    experiment.texts, experiment.cids, _ = load_chunk_arrays(
        source_root / "prepared" / "chunks.jsonl"
    )
    chunk_meta = read(source_root / "prepared" / "chunk_meta.json")
    if len(experiment.cids) != int(chunk_meta["chunks"]):
        raise ValueError("Source chunks.jsonl does not match chunk_meta.json")
    experiment.doc_chunks = defaultdict(list)
    for chunk_id, cid in enumerate(experiment.cids):
        experiment.doc_chunks[str(cid)].append(chunk_id)
    experiment.signature = identity({
        "mode": "ex9488_loadmodel_test_v1",
        "source_chunk_meta": chunk_meta,
        "adapters": {
            name: adapter_identity(model_root / name) for name in ADAPTERS
        },
    })
    return experiment


def _write_loadmodel_outputs(
    output_root: Path,
    records: Sequence[dict[str, Any]],
    rows: Sequence[Sequence[dict[str, Any]]],
    features: Any,
    ids: Sequence[Sequence[str]],
    weights_by_name: dict[str, tuple[float, ...]],
    valid_cids: set[str],
    chunk_texts: Sequence[str],
) -> None:
    for name, weights in weights_by_name.items():
        predicted = rankings(features, ids, weights)
        submission = {
            str(record["qid"]): {"answer": ranking[:5]}
            for record, ranking in zip(records, predicted)
        }
        if len(submission) != len(records):
            raise ValueError(f"{name}: submission does not cover every query")
        for qid, value in submission.items():
            answers = list(map(str, value["answer"]))
            if len(answers) != 5 or len(set(answers)) != 5:
                raise ValueError(f"{name}: qid={qid} does not have five unique answers")
            unknown = set(answers) - valid_cids
            if unknown:
                raise ValueError(f"{name}: qid={qid} contains unknown IDs: {unknown}")

        json_path = output_root / f"private_submission_{name}.json"
        zip_path = output_root / f"private_submission_{name}.zip"
        atomic_json(json_path, submission)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("submission.json", json.dumps(submission, ensure_ascii=False))

        depth = 20 if name == "fixed" else 10
        debug_path = output_root / f"private_top{depth}_{name}.jsonl"
        normalized_weights = np.asarray(weights, dtype=np.float64)
        normalized_weights /= normalized_weights.sum()

        def debug_rows() -> Iterator[dict[str, Any]]:
            for query_index, (record, group, ranking) in enumerate(
                zip(records, rows, predicted)
            ):
                by_cid = {str(row["cid"]): row for row in group}
                feature_index = {
                    str(cid): index for index, cid in enumerate(ids[query_index])
                }
                top_rows = []
                for rank, cid in enumerate(ranking[:depth], start=1):
                    row = by_cid[cid]
                    index = feature_index[cid]
                    votes = features[query_index, index]
                    contributions = votes * normalized_weights
                    item = {
                        "rank": rank,
                        "cid": cid,
                        "raw_scores": {
                            signal: float(row[signal]) for signal in SIGNALS
                        },
                        "dense_present": {
                            signal: bool(row.get(f"{signal}_present", True))
                            for signal in SIGNALS[:3]
                        },
                        "rrf_votes": {
                            signal: float(votes[position])
                            for position, signal in enumerate(SIGNALS)
                        },
                        "weighted_rrf": {
                            signal: float(contributions[position])
                            for position, signal in enumerate(SIGNALS)
                        },
                        "final_rrf": float(np.round(contributions.sum(), 12)),
                        "chunks": ([{
                            "chunk_id": int(chunk_id),
                            "text": chunk_texts[int(chunk_id)],
                        } for chunk_id in row["chunks"]] if name == "fixed" else
                            list(map(int, row["chunks"]))),
                    }
                    top_rows.append(item)
                yield {
                    "qid": str(record["qid"]),
                    "question": str(record["question"]),
                    "signal_order": list(SIGNALS),
                    "weights": dict(zip(SIGNALS, weights)),
                    "submission_answer": ranking[:5],
                    f"top{depth}": top_rows,
                }

        write_jsonl(debug_path, debug_rows())
        log(f"Saved {len(records):,} test predictions: {zip_path}")
        log(f"Saved {name} top-{depth} scores/evidence: {debug_path}")


def run_loadmodel(cfg: Config) -> None:
    """Load completed A/B/C/R1/R2 artifacts and predict a raw test JSON."""
    source_root = _loadmodel_require_dir(
        Path(cfg.cr2_source.replace("\\_", "_")), "source run"
    )
    model_root = _resolve_loadmodel_model_root(cfg)
    test_file = _resolve_loadmodel_test_file(cfg, model_root, source_root)
    output_root = _loadmodel_output_root(cfg, model_root)
    _loadmodel_validate_artifacts(source_root, model_root)

    records = read_query_object(test_file, require_answers=False)
    if not records:
        raise ValueError("Test file contains no queries")
    qids = [str(record["qid"]) for record in records]
    if len(set(qids)) != len(qids):
        raise ValueError("Test file contains duplicate query IDs")
    weights = _loadmodel_weights(model_root)
    log(f"Fusion signal order: {SIGNALS}")
    for name in ("best_val", "fixed", "positive_grid"):
        log(f"Loaded fusion weights {name}: {dict(zip(SIGNALS, weights[name]))}")
    if cfg.loadmodel_weight_set != "all":
        if cfg.loadmodel_weight_set not in weights:
            raise KeyError(
                f"Weight set {cfg.loadmodel_weight_set!r} is absent; "
                f"available={sorted(weights)}"
            )
        weights = {cfg.loadmodel_weight_set: weights[cfg.loadmodel_weight_set]}
    inference_cfg = _loadmodel_config(cfg, source_root, model_root)
    log(
        f"Validated loadmodel input: queries={len(records):,}; test={test_file}; "
        f"source={source_root}; models={model_root}; weights={sorted(weights)}"
    )
    if cfg.loadmodel_dry_run:
        log("Loadmodel dry run complete; no GPU model was loaded")
        return

    experiment = _loadmodel_experiment(
        inference_cfg, source_root, model_root, output_root / "loadmodel_cache"
    )
    pool = read(model_root / "c_complete.json").get("pool")
    if not isinstance(pool, dict) or "weights" not in pool or "quota" not in pool:
        raise ValueError("c_complete.json does not contain the selected dense pool")
    log(
        f"Loaded C dense-pool config: weights(A,B,C)={pool['weights']}; "
        f"quota={pool['quota']}"
    )
    rows = experiment.scored_pool(
        records,
        experiment.members(records),
        pool,
        model_root / ADAPTERS[4],
    )
    features, ids = rank_features(rows)
    _write_loadmodel_outputs(
        output_root, records, rows, features, ids, weights,
        set(experiment.doc_chunks), experiment.texts,
    )
    atomic_json(output_root / "loadmodel_private_config.json", {
        "source_run": str(source_root.resolve()),
        "model_run": str(model_root.resolve()),
        "test_file": str(test_file.resolve()),
        "queries": len(records),
        "signal_order": list(SIGNALS),
        "weight_sets": {name: list(values) for name, values in weights.items()},
        "embedding_model": inference_cfg.embedding_model,
        "reranker_model": inference_cfg.reranker_model,
        "output_dir": str(output_root),
    })
