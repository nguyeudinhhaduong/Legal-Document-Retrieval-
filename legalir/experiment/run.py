"""Điểm vào train C/R2 từ run nguồn A/B/R1 đã đóng băng.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import math
import numpy as np
import random

from legalir.experiment.experiment import Experiment


def run(api, cfg):
    if cfg.cr2_prepare_only and cfg.cr2_eval_only:
        raise ValueError("Choose prepare-only or eval-only")
    for name in ("cr2_c_epochs", "cr2_c_steps", "cr2_r2_epochs", "cr2_r2_steps", "cr2_negatives", "cr2_c_negatives", "cr2_c_positive_shortlist", "cr2_c_evidence", "cr2_r2_refresh_steps",
                 "embed_train_batch_size", "embed_batch_size", "query_batch_size", "rerank_batch_size",
                 "embed_grad_accum", "rerank_grad_accum", "embed_early_stopping_patience",
                 "rerank_early_stopping_patience"):
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= cfg.cr2_c_replay_fraction <= .5:
        raise ValueError("C replay fraction must be between 0 and 0.5")
    if not math.isfinite(cfg.cr2_r2_boundary_weight) or cfg.cr2_r2_boundary_weight < 0:
        raise ValueError("R2 boundary weight must be finite and non-negative")
    if cfg.cr2_learning_rate <= 0 or cfg.cr2_c_learning_rate <= 0 or cfg.embedding_temperature <= 0:
        raise ValueError("Learning rate and embedding temperature must be positive")
    experiment = Experiment(api, cfg)
    experiment.prepare()
    if cfg.cr2_prepare_only:
        experiment.verify_frozen()
        api.log("CPU preparation complete; launch without --cr2-prepare-only to train C and R2")
        return
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("C/R2 training requires a CUDA GPU with BF16 (recommended RTX PRO 6000 96 GB)")
    random.seed(experiment.cfg.seed); np.random.seed(experiment.cfg.seed); torch.manual_seed(experiment.cfg.seed)
    torch.cuda.manual_seed_all(experiment.cfg.seed)
    try:
        if cfg.cr2_eval_only:
            c = experiment.completed("c")
            if not c or not experiment.completed("r2"):
                raise FileNotFoundError("Eval-only requires completed C and R2 receipts in this work-dir")
            pool = c["pool"]
        else:
            pool = experiment.train_c()
            experiment.train_r2(pool)
        experiment.export(pool)
    finally:
        experiment.verify_frozen()
