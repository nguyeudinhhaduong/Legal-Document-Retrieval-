"""Kiểm tra cấu hình và môi trường CUDA.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import os

from legalir.core.config import Config
from legalir.core.utils import log


def validate_config(cfg: Config) -> None:
    if cfg.eval_size != 1000: raise ValueError("This run intentionally requires --eval-size 1000")
    if cfg.chunk_chars < 300: raise ValueError("chunk_chars must be at least 300")
    if cfg.embedding_dim < 64 or cfg.embedding_max_length < 128: raise ValueError("Embedding dimension/max length too small")
    if cfg.chunks_per_doc < 1: raise ValueError("chunks_per_doc must be positive")
    if not 0 <= cfg.dense_member_quota * 3 <= cfg.candidate_docs:
        raise ValueError("Require 0 <= 3*dense_member_quota <= candidate_docs")
    if cfg.candidate_docs < cfg.rerank_top_k or cfg.rerank_top_k < 5: raise ValueError("Require candidate_docs >= rerank_top_k >= 5")
    if not 0.0 <= cfg.ensemble_stage1_weight <= 1.0: raise ValueError("ensemble_stage1_weight must be in [0, 1]")
    if cfg.hard_negatives < 1 or cfg.embed_stage2_hard_negatives < 1 or cfg.lora_rank < 1:
        raise ValueError("embedding hard-negative counts and lora_rank must be positive")
    if cfg.candidate_docs <= cfg.embed_stage2_hard_negatives:
        raise ValueError("candidate_docs must exceed embed_stage2_hard_negatives")
    if cfg.rerank_mining_docs <= max(cfg.embed_stage3_hard_negatives, cfg.rerank_stage2_hard_negatives):
        raise ValueError("rerank_mining_docs must exceed stage-3/R2 hard-negative counts")
    if cfg.embedding_temperature <= 0.0 or not 0.0 <= cfg.embedding_kd_weight <= 1.0:
        raise ValueError("embedding_temperature must be positive and embedding_kd_weight in [0, 1]")
    if cfg.embed_stage3_kd_weight != 0.0:
        raise ValueError("Independent LoRA C requires embed_stage3_kd_weight=0")
    if not 0.0 <= cfg.embed_stage3_rehearsal_ratio <= 1.0:
        raise ValueError("embed_stage3_rehearsal_ratio must be in [0, 1]")
    if not 0.0 <= cfg.rerank_stage2_rehearsal_ratio <= 1.0:
        raise ValueError("rerank_stage2_rehearsal_ratio must be in [0, 1]")
    if not 1 <= cfg.rerank_train_chunks_per_doc <= cfg.chunks_per_doc:
        raise ValueError(
            "Require 1 <= rerank_train_chunks_per_doc <= chunks_per_doc"
        )
    if not 1 <= cfg.rerank_checkpoint_eval_queries <= cfg.eval_size:
        raise ValueError(
            "Require 1 <= rerank_checkpoint_eval_queries <= eval_size"
        )
    if cfg.rerank_checkpoint_batch_size < 1:
        raise ValueError("rerank_checkpoint_batch_size must be positive")
    if not 0.0 <= cfg.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if cfg.rerank_hard_negatives < 1 or cfg.rerank_mining_docs <= cfg.rerank_hard_negatives:
        raise ValueError("Require rerank_mining_docs > rerank_hard_negatives >= 1")
    if cfg.rerank_pairwise_weight < 0.0 or cfg.rerank_margin < 0.0:
        raise ValueError("rerank_pairwise_weight and rerank_margin must be non-negative")
    if cfg.rerank_epochs < 1 or cfg.embed_steps_per_epoch < 1 or cfg.rerank_steps_per_epoch < 1 or cfg.rerank_max_steps < 1:
        raise ValueError("Training step limits must be positive")
    if not (
        cfg.embed_min_epochs * cfg.embed_steps_per_epoch
        <= cfg.embed_max_steps
        <= cfg.embed_epochs * cfg.embed_steps_per_epoch
    ):
        raise ValueError(
            "Require embed_min_epochs*steps_per_epoch <= embed_max_steps "
            "<= embed_epochs*steps_per_epoch"
        )
    if not 1 <= cfg.embed_min_epochs <= cfg.embed_epochs:
        raise ValueError("Require 1 <= embed_min_epochs <= embed_epochs")
    if cfg.embed_stage2_steps_per_epoch < 1:
        raise ValueError("embed_stage2_steps_per_epoch must be positive")
    if not (
        cfg.embed_stage2_min_epochs * cfg.embed_stage2_steps_per_epoch
        <= cfg.embed_stage2_max_steps
        <= cfg.embed_stage2_epochs * cfg.embed_stage2_steps_per_epoch
    ):
        raise ValueError(
            "Require embed_stage2_min_epochs*steps_per_epoch <= "
            "embed_stage2_max_steps <= embed_stage2_epochs*steps_per_epoch"
        )
    if not 1 <= cfg.embed_stage2_min_epochs <= cfg.embed_stage2_epochs:
        raise ValueError("Require 1 <= embed_stage2_min_epochs <= embed_stage2_epochs")
    if cfg.embed_early_stopping_patience < 1:
        raise ValueError("embed_early_stopping_patience must be positive")
    if cfg.embed_stage3_lora_rank < 1:
        raise ValueError("embed_stage3_lora_rank must be positive")
    if cfg.lora_target_profile not in {"attention", "attention_ffn"}:
        raise ValueError("lora_target_profile must be attention or attention_ffn")
    if not 0.0 <= cfg.embed_stage2_lexical_negative_ratio <= 1.0:
        raise ValueError("embed_stage2_lexical_negative_ratio must be in [0, 1]")
    if not 1 <= cfg.rerank_min_epochs <= cfg.rerank_epochs:
        raise ValueError("Require 1 <= rerank_min_epochs <= rerank_epochs")
    if cfg.rerank_early_stopping_patience < 1:
        raise ValueError("rerank_early_stopping_patience must be positive")
    requested_rerank_steps = cfg.rerank_epochs * cfg.rerank_steps_per_epoch
    if cfg.rerank_max_steps < requested_rerank_steps:
        raise ValueError(
            "rerank_max_steps must be at least rerank_epochs * rerank_steps_per_epoch "
            f"({requested_rerank_steps:,})"
        )
    if cfg.embed_learning_rate <= 0 or cfg.embed_stage2_learning_rate <= 0: raise ValueError("Embedding learning rates must be positive")
    if cfg.embed_stage3_learning_rate <= 0 or cfg.rerank_stage2_learning_rate <= 0:
        raise ValueError("Stage-3/R2 learning rates must be positive")
    if not 1 <= cfg.embed_stage3_min_epochs <= cfg.embed_stage3_epochs:
        raise ValueError("Require 1 <= embed_stage3_min_epochs <= embed_stage3_epochs")
    if cfg.rerank_stage2_epochs < 1 or cfg.rerank_stage2_steps_per_epoch < 1:
        raise ValueError("R2 epochs/steps must be positive")
    if not 1 <= cfg.rerank_stage2_min_epochs <= cfg.rerank_stage2_epochs:
        raise ValueError("Require 1 <= rerank_stage2_min_epochs <= rerank_stage2_epochs")
    if not 1 <= cfg.calibration_size < cfg.eval_size:
        raise ValueError("calibration_size must split validation into two non-empty sets")
    if not 0.0 < cfg.target_recall_at_5 <= 1.0:
        raise ValueError("target_recall_at_5 must be in (0, 1]")
    if cfg.top_configs_logged < 1:
        raise ValueError("top_configs_logged must be positive")
    if cfg.candidate_docs <= cfg.embed_stage3_hard_negatives:
        raise ValueError("candidate_docs must exceed embed_stage3_hard_negatives")


def validate_cuda_runtime(cfg: Config) -> None:
    """Fail before expensive preprocessing when a GPU stage has no CUDA runtime."""
    gpu_stages = {
        "eval-base", "train-embedding", "train-embedding-stage3", "index",
        "train-reranker", "train-reranker-stage2", "eval", "calibrate", "predict", "all",
    }
    if cfg.stage not in gpu_stages:
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing; use a Kaggle GPU image with PyTorch installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable, so BGE-M3 indexing/fine-tuning and BGE reranking cannot run.\n"
            "Kaggle fix: open Notebook -> Settings -> Accelerator -> select GPU, then restart "
            "the session and Run All.\n"
            f"Diagnostic: torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}.\n"
            "Do not reinstall torch in the notebook; Kaggle supplies the CUDA-enabled build "
            "after the GPU accelerator is enabled. Use --stage prepare only if you intentionally "
            "want a CPU-only preprocessing run."
        )
    properties = torch.cuda.get_device_properties(0)
    memory_gib = properties.total_memory / (1024 ** 3)
    log(
        f"CUDA ready: {torch.cuda.get_device_name(0)}; VRAM={memory_gib:.1f} GiB; "
        f"torch CUDA={torch.version.cuda}"
    )
