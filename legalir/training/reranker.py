"""Huấn luyện reranker R1 và R2 (sửa lỗi vùng biên top-5).

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import json
import math
import random
from typing import Any, Sequence

from legalir.core.config import Config, Paths
from legalir.core.utils import atomic_json, log, progress, stable_key
from legalir.core.artifacts import artifact_identity, chunking_fingerprint, embedding_ensemble_identity, valid_lora_adapter
from legalir.models.lora import require_peft, resolved_lora_alpha
from legalir.retrieval.dense_fusion import DENSE_FUSION_VERSION, dense_ensemble_fingerprint, resolved_dense_ensemble_weight, resolved_dense_ensemble_weights, resolved_dense_member_quota
from legalir.models.reranker import BGEReranker, evaluate_reranker_groups, training_reranker_document_scores
from legalir.training.mining import mine_groups
from legalir.training.mining_repair import mine_reranker_stage2_groups
from legalir.training.reranker_checkpoints import load_or_build_reranker_checkpoint_pool, load_reranker_checkpoint_r1_scores, paired_reranker_pool_metrics, reranker_checkpoint_metric_key, reranker_pool_metrics, save_reranker_checkpoint_r1_scores, score_reranker_checkpoint_pool


def top5_boundary_reranker_loss(
    torch: Any, scores: Any, pairwise_weight: float, margin: float,
    negative_sources: Sequence[str] | None = None,
) -> Any:
    """Listwise CE plus a smooth, top-5-heavy positive/negative objective."""
    listwise = torch.nn.functional.cross_entropy(
        scores.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=scores.device)
    )
    count = max(0, int(scores.numel()) - 1)
    if count == 0:
        return listwise
    source_weights = {
        "fused_top5_error": 3.0, "r1_hard": 2.0,
        "dense_boundary": 1.5, "dense_tail": 0.75,
    }
    if negative_sources:
        weights = [source_weights.get(str(source), 1.0) for source in negative_sources]
    else:
        # Mining is hardest-first; emphasize the first five distractors that
        # most directly determine whether the positive enters the top five.
        weights = [2.0 if index < 5 else 1.0 for index in range(count)]
    weight_tensor = torch.tensor(weights, dtype=scores.dtype, device=scores.device)
    pairwise = (
        torch.nn.functional.softplus(margin - (scores[0] - scores[1:]))
        * weight_tensor
    ).sum() / weight_tensor.sum().clamp_min(1.0)
    return listwise + pairwise_weight * pairwise


def reranker_training_recipe(cfg: Config, paths: Paths) -> dict[str, Any]:
    return {
        "objective": "document_max_groupwise_ce_plus_top5_weighted_softplus_v5",
        "mining_method": "base_reranker_teacher_document_evidence_v4",
        "dense_fusion_version": DENSE_FUSION_VERSION,
        "mining_docs": cfg.rerank_mining_docs,
        "hard_negatives": cfg.rerank_hard_negatives,
        "dense_chunk_k": cfg.dense_chunk_k,
        "train_group_limit": cfg.rerank_train_groups,
        "learning_rate": cfg.rerank_learning_rate,
        "pairwise_weight": cfg.rerank_pairwise_weight,
        "pairwise_margin": cfg.rerank_margin,
        "train_chunks_per_doc": cfg.rerank_train_chunks_per_doc,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": resolved_lora_alpha(cfg.lora_rank, cfg.lora_use_rslora),
        "lora_use_rslora": cfg.lora_use_rslora,
        "lora_target_profile": cfg.lora_target_profile,
        "grad_accum": cfg.rerank_grad_accum,
        "warmup_ratio": cfg.warmup_ratio,
        "ensemble_stage1_weight": resolved_dense_ensemble_weight(cfg, paths),
        "ensemble_weights": list(resolved_dense_ensemble_weights(cfg, paths)),
        "dense_member_quota": resolved_dense_member_quota(cfg, paths),
        "epochs": cfg.rerank_epochs,
        "steps_per_epoch": cfg.rerank_steps_per_epoch,
        "min_epochs": cfg.rerank_min_epochs,
        "early_stopping_patience": cfg.rerank_early_stopping_patience,
        "split_seed": cfg.seed,
        "eval_size": cfg.eval_size,
        "checkpoint_selector": "true_validation_candidate_recall_at_5_v1",
        "checkpoint_eval_queries": cfg.rerank_checkpoint_eval_queries,
        "checkpoint_batch_size": cfg.rerank_checkpoint_batch_size,
        # Under the A->B->C->R1->R2 order, R1 mining transparently picks up C
        # via dense_ensemble_members auto-detection once C exists. Without
        # this field, a stale R1 trained before C existed would not be
        # invalidated when C later appears, since nothing else in this recipe
        # dict changes.
        "dense_ensemble_fingerprint": dense_ensemble_fingerprint(paths),
    }


def train_reranker(cfg: Config, paths: Paths) -> None:
    if cfg.no_train_reranker: log("Skipping reranker fine-tuning by request"); return
    if (paths.rerank_adapter / "adapter_config.json").exists() and not cfg.rebuild:
        metadata_path = paths.rerank_adapter / "pipeline_config.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        saved_model = metadata.get("base_model")
        saved_architecture = metadata.get("reranker_architecture")
        saved_epochs = metadata.get("requested_epochs")
        saved_steps_per_epoch = metadata.get("steps_per_epoch")
        saved_recipe = metadata.get("training_recipe")
        requested_steps = cfg.rerank_epochs * cfg.rerank_steps_per_epoch
        saved_steps = metadata.get("completed_micro_steps")
        chunking_matches = metadata.get("training_chunking") == chunking_fingerprint(paths.chunk_meta)
        if (
            saved_model == cfg.reranker_model
            and saved_architecture == "AutoModelForSequenceClassification"
            and saved_epochs == cfg.rerank_epochs
            and saved_steps_per_epoch == cfg.rerank_steps_per_epoch
            and (
                saved_steps == requested_steps
                or (
                    bool(metadata.get("stopped_early"))
                    and isinstance(saved_steps, int)
                    and 0 < saved_steps < requested_steps
                )
            )
            and saved_recipe == reranker_training_recipe(cfg, paths)
            and chunking_matches
        ):
            log("Reusing trained reranker adapter"); return
        reason = "chunking changed" if not chunking_matches else "recipe mismatch"
        log(f"Reranker checkpoint stale ({reason}); retraining")
    groups = mine_groups(cfg, paths, force_ensemble=True)
    sources = sorted({row.get("source_qid", row["qid"]) for row in groups}, key=lambda qid: stable_key(cfg.seed + 29, qid))
    holdout_sources = set(sources[:min(200, max(1, len(sources) // 25))])
    holdout = [row for row in groups if row.get("source_qid", row["qid"]) in holdout_sources]
    groups = [row for row in groups if row.get("source_qid", row["qid"]) not in holdout_sources]
    groups = sorted(groups, key=lambda row: stable_key(cfg.seed + 31, row["qid"]))
    if cfg.rerank_train_groups:
        groups = groups[:cfg.rerank_train_groups]
    checkpoint_records, checkpoint_candidates, checkpoint_chunk_texts = (
        load_or_build_reranker_checkpoint_pool(cfg, paths)
    )
    scorer = BGEReranker(cfg, training=True); torch = scorer.torch
    parameters = [value for value in scorer.model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.rerank_learning_rate, weight_decay=0.01)
    steps_per_epoch = min(len(groups), cfg.rerank_steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError("No reranker groups are available after the holdout split")
    planned_micro_steps = cfg.rerank_epochs * steps_per_epoch
    log(
        f"Reranker training recipe: {cfg.rerank_epochs} epochs x "
        f"{steps_per_epoch:,} steps = {planned_micro_steps:,} total steps"
    )
    updates = cfg.rerank_epochs * math.ceil(steps_per_epoch / cfg.rerank_grad_accum)
    from transformers import get_cosine_schedule_with_warmup
    peft = require_peft()
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, round(updates * cfg.warmup_ratio)), max(1, updates)
    )
    baseline = evaluate_reranker_groups(scorer, holdout)
    baseline_pool_scores = score_reranker_checkpoint_pool(
        scorer, checkpoint_records, checkpoint_candidates, checkpoint_chunk_texts,
        "R1 epoch 0 true Recall@5",
    )
    baseline_pool = reranker_pool_metrics(
        checkpoint_records, checkpoint_candidates, baseline_pool_scores
    )
    best_metrics = baseline; best_pool_metrics = baseline_pool; best_epoch = 0
    best_pool_scores = baseline_pool_scores
    best_state = {key: value.detach().cpu().clone() for key, value in peft.get_peft_model_state_dict(scorer.model).items()}
    history: list[dict[str, Any]] = [{
        "epoch": 0, "group_validation": baseline,
        "candidate_pool_validation": baseline_pool,
    }]
    rng = random.Random(cfg.seed); optimizer.zero_grad(set_to_none=True)
    micro = 0; pending_micro_steps = 0
    epochs_completed = 0; epochs_without_improvement = 0; stopped_early = False
    for epoch in range(cfg.rerank_epochs):
        rng.shuffle(groups); epoch_loss = 0.0; epoch_batches = 0
        epoch_groups = groups[:steps_per_epoch]
        train_progress = progress(epoch_groups, desc=f"Reranker LoRA {epoch + 1}")
        for row in train_progress:
            raw_scores = training_reranker_document_scores(
                scorer, row, cfg.rerank_train_chunks_per_doc
            )
            loss = top5_boundary_reranker_loss(
                torch, raw_scores, cfg.rerank_pairwise_weight, cfg.rerank_margin,
                row.get("negative_sources"),
            ) / cfg.rerank_grad_accum
            batch_loss = float(loss.detach().item()) * cfg.rerank_grad_accum
            epoch_loss += batch_loss; epoch_batches += 1
            if hasattr(train_progress, "set_postfix"):
                train_progress.set_postfix(
                    loss=f"{batch_loss:.4f}", avg_loss=f"{epoch_loss / epoch_batches:.4f}"
                )
            loss.backward(); micro += 1; pending_micro_steps += 1
            if pending_micro_steps >= cfg.rerank_grad_accum:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                pending_micro_steps = 0
        if pending_micro_steps:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            pending_micro_steps = 0
        average_loss = epoch_loss / max(1, epoch_batches)
        validation = evaluate_reranker_groups(scorer, holdout)
        pool_scores = score_reranker_checkpoint_pool(
            scorer, checkpoint_records, checkpoint_candidates, checkpoint_chunk_texts,
            f"R1 epoch {epoch + 1} true Recall@5",
        )
        pool_validation = reranker_pool_metrics(
            checkpoint_records, checkpoint_candidates, pool_scores
        )
        history.append({
            "epoch": epoch + 1, "train_loss": average_loss,
            "group_validation": validation,
            "candidate_pool_validation": pool_validation,
        })
        improved = reranker_checkpoint_metric_key(pool_validation) > reranker_checkpoint_metric_key(
            best_pool_metrics
        )
        if improved:
            best_metrics = validation; best_pool_metrics = pool_validation; best_epoch = epoch + 1
            best_pool_scores = pool_scores
            best_state = {key: value.detach().cpu().clone() for key, value in peft.get_peft_model_state_dict(scorer.model).items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epochs_completed = epoch + 1
        log(
            f"Reranker epoch {epoch + 1}: avg_loss={average_loss:.4f}, "
            f"group_validation={validation}, true_pool_R5={pool_validation}"
        )
        if (
            epochs_completed >= cfg.rerank_min_epochs
            and epochs_without_improvement >= cfg.rerank_early_stopping_patience
        ):
            stopped_early = True
            log(
                f"Reranker early stopping after epoch {epochs_completed}; "
                f"restoring best epoch {best_epoch}"
            )
            break
    peft.set_peft_model_state_dict(scorer.model, best_state)
    paths.rerank_adapter.mkdir(parents=True, exist_ok=True)
    scorer.model.save_pretrained(paths.rerank_adapter, safe_serialization=True); scorer.tokenizer.save_pretrained(paths.rerank_adapter)
    atomic_json(paths.rerank_adapter / "pipeline_config.json", {
        "pipeline_stage": "downstream - BGE reranker v2 M3 LoRA after BGE ensemble",
        "base_model": cfg.reranker_model,
        "reranker_architecture": "AutoModelForSequenceClassification",
        "embedding_ensemble": embedding_ensemble_identity(paths),
        "training_recipe": reranker_training_recipe(cfg, paths),
        "training_chunking": chunking_fingerprint(paths.chunk_meta),
        "loss": (
            "document-max groupwise cross-entropy + top-5-weighted smooth "
            "pairwise loss"
        ),
        "max_length": cfg.reranker_max_length,
        "train_groups": len(groups), "holdout_groups": len(holdout),
        "requested_epochs": cfg.rerank_epochs,
        "completed_epochs": epochs_completed,
        "steps_per_epoch": cfg.rerank_steps_per_epoch,
        "actual_steps_per_epoch": steps_per_epoch,
        "completed_micro_steps": micro, "max_micro_steps": cfg.rerank_max_steps,
        "best_epoch": best_epoch, "best_group_validation": best_metrics,
        "best_candidate_pool_validation": best_pool_metrics, "history": history,
        "stopped_early": stopped_early,
        "guardrail": (
            "Best epoch retained by true Recall@5 on the immutable validation "
            "candidate pool; AllGold/MRR/Precision break ties"
        ),
    })
    # R2 starts from this exact selected R1 adapter and uses the same immutable
    # pool. Persist its logits so stage 2 saves one full forward pass.
    save_reranker_checkpoint_r1_scores(
        cfg, paths, checkpoint_records, best_pool_scores
    )
    scorer.close()


def reranker_stage2_is_accepted(paths: Paths) -> bool:
    metadata_path = paths.second_rerank_adapter / "pipeline_config.json"
    if not valid_lora_adapter(paths.second_rerank_adapter) or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return bool(metadata.get("accepted")) and metadata.get("parent_r1") == list(
            artifact_identity(paths.rerank_adapter / "adapter_model.safetensors")[1:]
        )
    except (json.JSONDecodeError, OSError, TypeError):
        return False


def train_reranker_stage2(cfg: Config, paths: Paths) -> None:
    """Continue R1 on actual top-5 failures with a rehearsal guardrail."""
    if cfg.no_train_reranker:
        log("Skipping reranker R2 fine-tuning by request"); return
    parent_identity = list(artifact_identity(paths.rerank_adapter / "adapter_model.safetensors")[1:])
    recipe = {
        "version": "r2_continued_document_top5_repair_v5", "mining_teacher_r1": parent_identity,
        "initialization": "continued_from_r1",
        "dense": dense_ensemble_fingerprint(paths), "hard_negatives": cfg.rerank_stage2_hard_negatives,
        "epochs": cfg.rerank_stage2_epochs, "min_epochs": cfg.rerank_stage2_min_epochs,
        "early_stopping_patience": cfg.rerank_early_stopping_patience,
        "split_seed": cfg.seed,
        "eval_size": cfg.eval_size,
        "checkpoint_selector": "best_r1_r2_pair_true_validation_recall_at_5_v1",
        "checkpoint_eval_queries": cfg.rerank_checkpoint_eval_queries,
        "checkpoint_batch_size": cfg.rerank_checkpoint_batch_size,
        "dense_chunk_k": cfg.dense_chunk_k,
        "steps_per_epoch": cfg.rerank_stage2_steps_per_epoch,
        "learning_rate": cfg.rerank_stage2_learning_rate, "grad_accum": cfg.rerank_grad_accum,
        "train_chunks_per_doc": cfg.rerank_train_chunks_per_doc,
        "rehearsal_ratio": cfg.rerank_stage2_rehearsal_ratio,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": resolved_lora_alpha(cfg.lora_rank, cfg.lora_use_rslora),
        "lora_use_rslora": cfg.lora_use_rslora,
        "lora_target_profile": cfg.lora_target_profile,
        "warmup_ratio": cfg.warmup_ratio,
        "pairwise_weight": cfg.rerank_pairwise_weight, "margin": cfg.rerank_stage2_margin,
        "chunking": chunking_fingerprint(paths.chunk_meta),
    }
    metadata_path = paths.second_rerank_adapter / "pipeline_config.json"
    if valid_lora_adapter(paths.second_rerank_adapter) and metadata_path.is_file() and not cfg.rebuild:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("training_recipe") == recipe:
            log("Reusing trained reranker R2 adapter"); return
    if not valid_lora_adapter(paths.rerank_adapter):
        raise FileNotFoundError("Reranker R1 is required before training R2")
    groups = mine_reranker_stage2_groups(cfg, paths)
    sources = sorted(
        {row.get("source_qid", row["qid"]) for row in groups},
        key=lambda qid: stable_key(cfg.seed + 71, str(qid)),
    )
    holdout_sources = set(sources[:min(200, max(1, len(sources) // 25))])
    holdout = [row for row in groups if row.get("source_qid", row["qid"]) in holdout_sources]
    train = [row for row in groups if row.get("source_qid", row["qid"]) not in holdout_sources]
    steps_per_epoch = min(len(train), cfg.rerank_stage2_steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError("No R2 groups are available after the holdout split")
    checkpoint_records, checkpoint_candidates, checkpoint_chunk_texts = (
        load_or_build_reranker_checkpoint_pool(cfg, paths)
    )
    # Epoch 0 is the proven R1 checkpoint. Fine-tuning can therefore repair
    # boundary errors without giving up a safe non-regression fallback.
    scorer = BGEReranker(cfg, training=True, adapter=paths.rerank_adapter)
    torch = scorer.torch
    parameters = [value for value in scorer.model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.rerank_stage2_learning_rate, weight_decay=0.01)
    from transformers import get_cosine_schedule_with_warmup
    peft = require_peft()
    updates = cfg.rerank_stage2_epochs * math.ceil(
        steps_per_epoch / cfg.rerank_grad_accum
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, round(updates * cfg.warmup_ratio)), max(1, updates)
    )
    baseline = evaluate_reranker_groups(scorer, holdout)
    r1_pool_scores = load_reranker_checkpoint_r1_scores(
        cfg, paths, checkpoint_records, checkpoint_candidates
    )
    if r1_pool_scores is None:
        r1_pool_scores = score_reranker_checkpoint_pool(
            scorer, checkpoint_records, checkpoint_candidates, checkpoint_chunk_texts,
            "R2 epoch 0 / fixed R1 true Recall@5",
        )
        save_reranker_checkpoint_r1_scores(
            cfg, paths, checkpoint_records, r1_pool_scores
        )
    baseline_pool = reranker_pool_metrics(
        checkpoint_records, checkpoint_candidates, r1_pool_scores
    )
    baseline_pair, baseline_pair_weight = paired_reranker_pool_metrics(
        checkpoint_records, checkpoint_candidates, r1_pool_scores, r1_pool_scores
    )
    best_metrics = baseline
    best_pool_metrics = baseline_pool
    best_pair_metrics = baseline_pair
    best_pair_weight = baseline_pair_weight
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in peft.get_peft_model_state_dict(scorer.model).items()}
    history: list[dict[str, Any]] = [{
        "epoch": 0, "group_validation": baseline,
        "candidate_pool_validation": baseline_pool,
        "r1_r2_pair_validation": baseline_pair,
        "r2_pair_weight": baseline_pair_weight,
    }]
    rng = random.Random(cfg.seed + 73); optimizer.zero_grad(set_to_none=True)
    micro = 0; pending_micro_steps = 0
    epochs_completed = 0; epochs_without_improvement = 0
    for epoch in range(cfg.rerank_stage2_epochs):
        rng.shuffle(train); total_loss = 0.0
        epoch_groups = train[:steps_per_epoch]
        for row in progress(epoch_groups, desc=f"Reranker R2 epoch {epoch + 1}"):
            scores = training_reranker_document_scores(
                scorer, row, cfg.rerank_train_chunks_per_doc
            )
            loss = top5_boundary_reranker_loss(
                torch, scores, cfg.rerank_pairwise_weight,
                cfg.rerank_stage2_margin, row.get("negative_sources"),
            ) / cfg.rerank_grad_accum
            total_loss += float(loss.detach().item()) * cfg.rerank_grad_accum
            loss.backward(); micro += 1; pending_micro_steps += 1
            if pending_micro_steps >= cfg.rerank_grad_accum:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending_micro_steps = 0
        if pending_micro_steps:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending_micro_steps = 0
        validation = evaluate_reranker_groups(scorer, holdout)
        pool_scores = score_reranker_checkpoint_pool(
            scorer, checkpoint_records, checkpoint_candidates, checkpoint_chunk_texts,
            f"R2 epoch {epoch + 1} true Recall@5",
        )
        pool_validation = reranker_pool_metrics(
            checkpoint_records, checkpoint_candidates, pool_scores
        )
        pair_validation, pair_weight = paired_reranker_pool_metrics(
            checkpoint_records, checkpoint_candidates, r1_pool_scores, pool_scores
        )
        history.append({
            "epoch": epoch + 1, "train_loss": total_loss / len(epoch_groups),
            "group_validation": validation,
            "candidate_pool_validation": pool_validation,
            "r1_r2_pair_validation": pair_validation,
            "r2_pair_weight": pair_weight,
        })
        if reranker_checkpoint_metric_key(pair_validation) > reranker_checkpoint_metric_key(
            best_pair_metrics
        ):
            best_metrics = validation
            best_pool_metrics = pool_validation
            best_pair_metrics = pair_validation
            best_pair_weight = pair_weight
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in peft.get_peft_model_state_dict(scorer.model).items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epochs_completed = epoch + 1
        log(
            f"Reranker R2 epoch {epoch + 1}: group_validation={validation}, "
            f"standalone_true_pool_R5={pool_validation}, "
            f"best_R1_R2_pair_R5={pair_validation}, pair_r2_weight={pair_weight:.1f}"
        )
        if (
            epochs_completed >= cfg.rerank_stage2_min_epochs
            and epochs_without_improvement >= cfg.rerank_early_stopping_patience
        ):
            log(f"Reranker R2 early stopping after epoch {epochs_completed}; restoring epoch {best_epoch}")
            break
    peft.set_peft_model_state_dict(scorer.model, best_state)
    paths.second_rerank_adapter.mkdir(parents=True, exist_ok=True)
    scorer.model.save_pretrained(paths.second_rerank_adapter, safe_serialization=True)
    scorer.tokenizer.save_pretrained(paths.second_rerank_adapter)
    accepted = best_epoch > 0 and reranker_checkpoint_metric_key(
        best_pair_metrics
    ) > reranker_checkpoint_metric_key(baseline_pair)
    atomic_json(metadata_path, {
        "pipeline_stage": "R2 continued document-level top-5 repair", "base_model": cfg.reranker_model,
        "parent_r1": parent_identity, "training_recipe": recipe, "accepted": accepted,
        "baseline_group_validation": baseline,
        "best_group_validation": best_metrics,
        "baseline_candidate_pool_validation": baseline_pool,
        "best_candidate_pool_validation": best_pool_metrics,
        "baseline_r1_r2_pair_validation": baseline_pair,
        "best_r1_r2_pair_validation": best_pair_metrics,
        "best_pair_r2_weight": best_pair_weight,
        "best_epoch": best_epoch, "history": history, "completed_micro_steps": micro,
        "guardrail": (
            "R2 starts from R1 and is retained only when an R1/R2 RRF pair "
            "improves true Recall@5 on the immutable validation candidate pool"
        ),
    })
    scorer.close()
    log(
        f"Reranker R2 {'accepted' if accepted else 'rejected; R1 fallback retained'}: "
        f"best_epoch={best_epoch}, pair_R5={best_pair_metrics}, "
        f"pair_r2_weight={best_pair_weight:.1f}"
    )
