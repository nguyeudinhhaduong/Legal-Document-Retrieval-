"""Giao diện dòng lệnh: python -m legalir --stage ..."""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from legalir.core.config import Config, Paths
from legalir.core.utils import atomic_json, log
from legalir.core.artifacts import restore_previous_artifacts, valid_lora_adapter
from legalir.data.loading import prepare_data
from legalir.retrieval.dense_index import build_dense_index
from legalir.retrieval.dense_fusion import dense_calibration_is_valid
from legalir.training.mining import mine_groups
from legalir.training.embedder import train_embedder
from legalir.evaluation.dense_eval import evaluate_base_dense, evaluate_dense_ensemble
from legalir.training.reranker import train_reranker, train_reranker_stage2
from legalir.evaluation.evaluate import evaluate
from legalir.pipeline.submission import predict_submission
from legalir.pipeline.validation import validate_config, validate_cuda_runtime
from legalir.selftest import run_self_tests
from legalir.experiment.run import run
from legalir.pipeline.loadmodel import run_loadmodel

CLI_DESCRIPTION = """LegalIR – BGE-M3 LoRA A/B/C + BGE-reranker R1/R2, weighted RRF fusion.

Train C/R2 từ run nguồn:   python -m legalir --stage retrain-c-r2 --cr2-source <run>
Dự đoán file test mới:     python -m legalir --stage loadmodel --loadmodel-model-dir <run> --cr2-source <run> --test-file <test.json>
Self-test CPU:             python -m legalir --self-test
"""


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter, allow_abbrev=False)
    parser.add_argument("-f", dest="kernel", help=argparse.SUPPRESS)
    parser.add_argument("--data", default=Config.data); parser.add_argument("--work-dir", default=Config.work_dir)
    parser.add_argument("--base-index", default=Config.base_index)
    parser.add_argument("--train-groups-file", default=Config.train_groups_file)
    parser.add_argument("--resume-artifacts-dir", default=Config.resume_artifacts_dir)
    parser.add_argument("--cr2-source", default=Config.cr2_source)
    parser.add_argument(
        "--test-file", default=Config.test_file,
        help="Raw LegalIR test JSON used by --stage loadmodel",
    )
    parser.add_argument(
        "--loadmodel-output-dir", default=Config.loadmodel_output_dir,
        help="Output/cache directory for --stage loadmodel",
    )
    parser.add_argument(
        "--loadmodel-model-dir", default=Config.loadmodel_model_dir,
        help=(
            "Completed C/R2 run or its Kaggle notebook-output parent; an "
            "explicit --work-dir takes priority"
        ),
    )
    parser.add_argument(
        "--loadmodel-weight-set",
        choices=("all", "best_val", "fixed", "positive_grid"),
        default=Config.loadmodel_weight_set,
        help="Saved fusion configuration(s) exported by --stage loadmodel",
    )
    parser.add_argument(
        "--loadmodel-dry-run", action="store_true",
        help="Validate test/model paths and receipts without loading CUDA models",
    )
    parser.add_argument("--cr2-prepare-only", action="store_true")
    parser.add_argument("--cr2-eval-only", action="store_true")
    parser.add_argument("--cr2-remine-c", action=argparse.BooleanOptionalAction, default=Config.cr2_remine_c,
                        help="Refresh negatives using current C after each epoch; --no-cr2-remine-c disables refresh")
    parser.add_argument("--cr2-remine-r2", action=argparse.BooleanOptionalAction, default=Config.cr2_remine_r2,
                        help="Refresh upcoming queries with B/R1/current-R2; disable for static-negative control")
    parser.add_argument("--cr2-r2-boundary-weight", type=float, default=Config.cr2_r2_boundary_weight,
                        help="Sampled top-five boundary coefficient: 0 disables it, .3 restores the old term")
    parser.add_argument("--cr2-r2-refresh-steps", type=int, default=Config.cr2_r2_refresh_steps,
                        help="Refresh R2 negatives before each block of this many micro-steps")
    parser.add_argument("--cr2-r2-lr-decay", action=argparse.BooleanOptionalAction,
                        default=Config.cr2_r2_lr_decay)
    for name in ("c_epochs", "c_steps", "r2_epochs", "r2_steps", "negatives", "c_negatives", "c_positive_shortlist", "c_evidence"):
        parser.add_argument("--cr2-" + name.replace("_", "-"), type=int,
                            default=getattr(Config, "cr2_" + name))
    parser.add_argument("--cr2-learning-rate", type=float, default=Config.cr2_learning_rate)
    parser.add_argument("--cr2-c-learning-rate", type=float, default=Config.cr2_c_learning_rate)
    parser.add_argument("--cr2-c-replay-fraction", type=float, default=Config.cr2_c_replay_fraction)
    parser.add_argument("--reuse-trained-embeddings", dest="reuse_trained_embeddings", action="store_true")
    parser.add_argument("--retrain-embeddings", dest="reuse_trained_embeddings", action="store_false")
    parser.set_defaults(reuse_trained_embeddings=Config.reuse_trained_embeddings)
    parser.add_argument("--stage", choices=(
        "c-r2", "retrain-c-r2", "loadmodel", "prepare", "eval-base", "train-embedding", "train-embedding-stage3", "index",
        "train-reranker", "train-reranker-stage2", "eval", "calibrate", "predict", "all",
    ), default=Config.stage)
    parser.add_argument("--seed", type=int, default=Config.seed); parser.add_argument("--eval-size", type=int, default=Config.eval_size)
    parser.add_argument("--embedding-model", default=Config.embedding_model); parser.add_argument("--reranker-model", default=Config.reranker_model)
    parser.add_argument("--cache-dir", default=Config.cache_dir); parser.add_argument("--chunk-chars", type=int, default=Config.chunk_chars)
    parser.add_argument("--max-chunks-per-doc", type=int, default=Config.max_chunks_per_doc)
    parser.add_argument("--embedding-max-length", type=int, default=Config.embedding_max_length); parser.add_argument("--embedding-dim", type=int, default=Config.embedding_dim)
    parser.add_argument("--embed-batch-size", type=int, default=Config.embed_batch_size); parser.add_argument("--query-batch-size", type=int, default=Config.query_batch_size)
    parser.add_argument("--embed-epochs", type=int, default=Config.embed_epochs); parser.add_argument("--embed-train-batch-size", type=int, default=Config.embed_train_batch_size)
    parser.add_argument(
        "--embed-steps-per-epoch", type=int, default=Config.embed_steps_per_epoch,
    )
    parser.add_argument("--embed-max-steps", type=int, default=Config.embed_max_steps)
    parser.add_argument("--embed-min-epochs", type=int, default=Config.embed_min_epochs)
    parser.add_argument("--embed-stage2-epochs", type=int, default=Config.embed_stage2_epochs)
    parser.add_argument("--embed-stage2-steps-per-epoch", type=int, default=Config.embed_stage2_steps_per_epoch)
    parser.add_argument("--embed-stage2-max-steps", type=int, default=Config.embed_stage2_max_steps)
    parser.add_argument("--embed-stage2-min-epochs", type=int, default=Config.embed_stage2_min_epochs)
    parser.add_argument(
        "--embed-early-stopping-patience", type=int,
        default=Config.embed_early_stopping_patience,
    )
    parser.add_argument("--embed-grad-accum", type=int, default=Config.embed_grad_accum); parser.add_argument("--embed-learning-rate", type=float, default=Config.embed_learning_rate)
    parser.add_argument("--embed-stage2-learning-rate", type=float, default=Config.embed_stage2_learning_rate)
    parser.add_argument("--embed-stage3-learning-rate", type=float, default=Config.embed_stage3_learning_rate)
    parser.add_argument("--embed-stage3-epochs", type=int, default=Config.embed_stage3_epochs)
    parser.add_argument("--embed-stage3-steps-per-epoch", type=int, default=Config.embed_stage3_steps_per_epoch)
    parser.add_argument("--embed-stage3-min-epochs", type=int, default=Config.embed_stage3_min_epochs)
    parser.add_argument("--hard-negatives", type=int, default=Config.hard_negatives)
    parser.add_argument("--embed-stage2-hard-negatives", type=int, default=Config.embed_stage2_hard_negatives)
    parser.add_argument("--embed-stage2-lexical-negative-ratio", type=float, default=Config.embed_stage2_lexical_negative_ratio)
    parser.add_argument("--embed-stage3-hard-negatives", type=int, default=Config.embed_stage3_hard_negatives)
    parser.add_argument("--embed-stage3-lora-rank", type=int, default=Config.embed_stage3_lora_rank)
    parser.add_argument("--embed-stage3-kd-weight", type=float, default=Config.embed_stage3_kd_weight)
    parser.add_argument("--embed-stage3-rehearsal-ratio", type=float, default=Config.embed_stage3_rehearsal_ratio)
    parser.add_argument("--embedding-temperature", type=float, default=Config.embedding_temperature)
    parser.add_argument("--embedding-kd-weight", type=float, default=Config.embedding_kd_weight)
    parser.add_argument("--train-groups", type=int, default=Config.train_groups)
    parser.add_argument("--dense-chunk-k", type=int, default=Config.dense_chunk_k); parser.add_argument("--candidate-docs", type=int, default=Config.candidate_docs)
    parser.add_argument("--dense-member-quota", type=int, default=Config.dense_member_quota)
    parser.add_argument("--chunks-per-doc", type=int, default=Config.chunks_per_doc); parser.add_argument("--rerank-top-k", type=int, default=Config.rerank_top_k)
    parser.add_argument("--ensemble-stage1-weight", type=float, default=Config.ensemble_stage1_weight)
    parser.add_argument("--reranker-max-length", type=int, default=Config.reranker_max_length); parser.add_argument("--rerank-batch-size", type=int, default=Config.rerank_batch_size)
    parser.add_argument("--rerank-train-groups", type=int, default=Config.rerank_train_groups); parser.add_argument("--rerank-epochs", type=int, default=Config.rerank_epochs)
    parser.add_argument("--rerank-mining-docs", type=int, default=Config.rerank_mining_docs)
    parser.add_argument("--rerank-hard-negatives", type=int, default=Config.rerank_hard_negatives)
    parser.add_argument("--rerank-steps-per-epoch", type=int, default=Config.rerank_steps_per_epoch)
    parser.add_argument("--rerank-max-steps", type=int, default=Config.rerank_max_steps)
    parser.add_argument("--rerank-min-epochs", type=int, default=Config.rerank_min_epochs)
    parser.add_argument(
        "--rerank-early-stopping-patience", type=int,
        default=Config.rerank_early_stopping_patience,
    )
    parser.add_argument("--rerank-grad-accum", type=int, default=Config.rerank_grad_accum); parser.add_argument("--rerank-learning-rate", type=float, default=Config.rerank_learning_rate)
    parser.add_argument("--rerank-pairwise-weight", type=float, default=Config.rerank_pairwise_weight)
    parser.add_argument("--rerank-margin", type=float, default=Config.rerank_margin)
    parser.add_argument("--rerank-stage2-epochs", type=int, default=Config.rerank_stage2_epochs)
    parser.add_argument("--rerank-stage2-min-epochs", type=int, default=Config.rerank_stage2_min_epochs)
    parser.add_argument("--rerank-stage2-steps-per-epoch", type=int, default=Config.rerank_stage2_steps_per_epoch)
    parser.add_argument("--rerank-stage2-learning-rate", type=float, default=Config.rerank_stage2_learning_rate)
    parser.add_argument("--rerank-stage2-hard-negatives", type=int, default=Config.rerank_stage2_hard_negatives)
    parser.add_argument("--rerank-stage2-margin", type=float, default=Config.rerank_stage2_margin)
    parser.add_argument("--rerank-train-chunks-per-doc", type=int, default=Config.rerank_train_chunks_per_doc)
    parser.add_argument(
        "--rerank-checkpoint-eval-queries", type=int,
        default=Config.rerank_checkpoint_eval_queries,
    )
    parser.add_argument(
        "--rerank-checkpoint-batch-size", type=int,
        default=Config.rerank_checkpoint_batch_size,
    )
    parser.add_argument("--rerank-stage2-rehearsal-ratio", type=float, default=Config.rerank_stage2_rehearsal_ratio)
    parser.add_argument("--calibration-size", type=int, default=Config.calibration_size)
    parser.add_argument("--target-recall-at-5", type=float, default=Config.target_recall_at_5)
    parser.add_argument("--top-configs-logged", type=int, default=Config.top_configs_logged)
    parser.add_argument("--no-bm25", dest="bm25_enabled", action="store_false")
    parser.set_defaults(bm25_enabled=Config.bm25_enabled)
    parser.add_argument("--lora-rank", type=int, default=Config.lora_rank)
    parser.add_argument(
        "--lora-target-profile", choices=("attention", "attention_ffn"),
        default=Config.lora_target_profile,
    )
    parser.add_argument("--use-rslora", dest="lora_use_rslora", action="store_true")
    parser.add_argument("--no-rslora", dest="lora_use_rslora", action="store_false")
    parser.set_defaults(lora_use_rslora=Config.lora_use_rslora)
    parser.add_argument("--warmup-ratio", type=float, default=Config.warmup_ratio)
    parser.add_argument("--no-train-embedding", action="store_true"); parser.add_argument("--no-train-reranker", action="store_true")
    parser.add_argument("--rebuild", action="store_true"); parser.add_argument("--self-test", action="store_true")
    parsed, unknown = parser.parse_known_args(argv)
    launcher = Path(sys.argv[0]).name.casefold()
    notebook_kernel = (
        "ipykernel" in sys.modules
        or launcher in {"colab_kernel_launcher.py", "ipykernel_launcher.py"}
    )
    if unknown and notebook_kernel:
        print(f"[args] Ignoring notebook kernel arguments: {' '.join(unknown)}", flush=True)
    elif unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    values = vars(parsed); values.pop("kernel", None)
    if values["stage"] == "c-r2":
        values["stage"] = "retrain-c-r2"
    return Config(**values)


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = parse_args(argv); paths = Paths(Path(cfg.work_dir))
    if cfg.self_test: run_self_tests(paths.root / "self_test"); return
    if cfg.stage == "loadmodel":
        run_loadmodel(cfg)
        return
    if cfg.stage == "retrain-c-r2":
        run(importlib.import_module("legalir.api"), cfg)
        return
    validate_config(cfg); validate_cuda_runtime(cfg)
    random.seed(cfg.seed); paths.root.mkdir(parents=True, exist_ok=True)
    restore_previous_artifacts(cfg, paths)
    atomic_json(paths.root / "run_config.json", asdict(cfg)); log(f"stage={cfg.stage} work_dir={paths.root.resolve()}")
    if cfg.stage in {
        "prepare", "eval-base", "train-embedding", "train-embedding-stage3", "index",
        "train-reranker", "train-reranker-stage2", "eval", "calibrate", "predict", "all",
    }:
        prepare_data(cfg, paths)
    if cfg.stage == "eval-base": evaluate_base_dense(cfg, paths)
    if cfg.stage in {"train-embedding", "all"}:
        log("BGE fine-tuning stage 1/3: independent embedding LoRA A")
        prepare_data(cfg, paths); build_dense_index(cfg, paths, base=True); mine_groups(cfg, paths); train_embedder(cfg, paths, stage=1)
        build_dense_index(cfg, paths, base=False, stage=1)
    if cfg.stage == "index":
        prepare_data(cfg, paths); build_dense_index(cfg, paths, base=False, stage=1)
    if cfg.stage in {"train-embedding", "all"}:
        log("BGE fine-tuning stage 2/3: specialize LoRA B from A on teacher/lexical hard negatives")
        mine_groups(cfg, paths, force_final=True); train_embedder(cfg, paths, stage=2)
        build_dense_index(cfg, paths, base=False, stage=2)
        evaluate_dense_ensemble(cfg, paths)
    if cfg.stage in {"train-embedding-stage3", "all"}:
        prepare_data(cfg, paths)
        log("BGE fine-tuning stage 3/3: independent LoRA C on A/B boundary failures + rehearsal")
        train_embedder(cfg, paths, stage=3)
        build_dense_index(cfg, paths, base=False, stage=3)
        evaluate_dense_ensemble(cfg, paths)
    if cfg.stage in {"train-reranker", "all"}:
        if not dense_calibration_is_valid(cfg, paths):
            log("Dense ensemble calibration missing or stale; calibrating for candidate recall")
            evaluate_dense_ensemble(cfg, paths)
        log("Reranker R1: BGE reranker v2 M3 LoRA with A/B/C ensemble negatives")
        train_reranker(cfg, paths)
    if cfg.stage in {"train-reranker-stage2", "all"}:
        if not valid_lora_adapter(paths.rerank_adapter):
            raise FileNotFoundError("R1 is required before R2")
        if not dense_calibration_is_valid(cfg, paths): evaluate_dense_ensemble(cfg, paths)
        log("Reranker R2: document-level repair continued from R1 top-5 errors")
        train_reranker_stage2(cfg, paths)
    if cfg.stage in {"eval", "calibrate", "all"}:
        if not dense_calibration_is_valid(cfg, paths):
            log("Dense ensemble calibration missing or stale; calibrating for candidate recall")
            evaluate_dense_ensemble(cfg, paths)
        evaluate(cfg, paths)
    if cfg.stage in {"predict", "all"}:
        if not paths.calibration.exists(): raise FileNotFoundError("Run --stage eval before prediction")
        predict_submission(cfg, paths)
