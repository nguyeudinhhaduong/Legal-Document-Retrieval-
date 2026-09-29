"""Huấn luyện LoRA embedding A/B/C (InfoNCE + distillation).

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Sequence

from legalir._deps import faiss
from legalir.core.config import Config, Paths
from legalir.core.utils import atomic_json, log, progress, read_jsonl
from legalir.core.artifacts import artifact_identity, chunking_fingerprint, valid_lora_adapter
from legalir.models.lora import require_peft, resolved_lora_alpha
from legalir.models.dense_encoder import DenseEncoder
from legalir.data.chunk_store import load_chunk_arrays
from legalir.retrieval.dense_index import base_index_path, dense_candidates, groups_in_current_train_split, train_groups_path
from legalir.retrieval.dense_fusion import DENSE_FUSION_VERSION
from legalir.training.mining import mine_groups
from legalir.training.mining_repair import mine_embedding_stage3_groups
from legalir.evaluation.metrics import task_metrics


def dense_contrastive_distillation_loss(
    torch: Any, query: Any, positive: Any, negative: Any,
    temperature: float, kd_weight: float,
    teacher_relevance: Any | None = None,
) -> Any:
    """Full-1024 InfoNCE plus optional cross-encoder listwise distillation."""
    query = torch.nn.functional.normalize(query, dim=1)
    positive = torch.nn.functional.normalize(positive, dim=1)
    negative = torch.nn.functional.normalize(negative, dim=2)
    positive_matrix = query @ positive.T
    own_negative_scores = torch.einsum("bd,bkd->bk", query, negative)
    logits = torch.cat([positive_matrix, own_negative_scores], dim=1) / temperature
    labels = torch.arange(query.shape[0], device=query.device)
    contrastive = torch.nn.functional.cross_entropy(logits, labels)
    if teacher_relevance is None or kd_weight <= 0.0:
        return contrastive
    local_student = torch.cat([
        positive_matrix.diagonal().unsqueeze(1), own_negative_scores
    ], dim=1) / temperature
    teacher_probability = torch.softmax(teacher_relevance.float(), dim=1)
    distillation = -(teacher_probability * torch.log_softmax(local_student, dim=1)).sum(dim=1).mean()
    return (1.0 - kd_weight) * contrastive + kd_weight * distillation


def evaluate_embedder_groups(
    encoder: DenseEncoder, groups: Sequence[dict[str, Any]], batch_size: int
) -> dict[str, float]:
    torch = encoder.torch; was_training = encoder.model.training; encoder.model.eval()
    hits = 0; hits_at_5 = 0; ndcg_at_5 = 0.0; reciprocal = 0.0; margin = 0.0
    for start in progress(range(0, len(groups), batch_size), desc="Embed holdout"):
        batch = groups[start:start + batch_size]
        q = encoder.tensor([row["query"] for row in batch], True)
        p = encoder.tensor([row["positive"] for row in batch], False)
        flat = [text for row in batch for text in row["negatives"]]
        n = encoder.tensor(flat, False).reshape(len(batch), -1, q.shape[-1])
        positive_scores = torch.einsum("bd,bd->b", q, p)
        negative_scores = torch.einsum("bd,bkd->bk", q, n)
        ranks = 1 + (negative_scores > positive_scores[:, None]).sum(dim=1)
        hits += int((ranks == 1).sum().item())
        hits_at_5 += int((ranks <= 5).sum().item())
        ndcg_at_5 += float(
            torch.where(
                ranks <= 5,
                1.0 / torch.log2(ranks.float() + 1.0),
                torch.zeros_like(ranks, dtype=torch.float32),
            ).sum().item()
        )
        reciprocal += float((1.0 / ranks.float()).sum().item())
        margin += float((positive_scores - negative_scores.max(dim=1).values).sum().item())
    if was_training: encoder.model.train()
    count = max(1, len(groups))
    return {
        "Accuracy@1": hits / count, "Hit@5": hits_at_5 / count,
        "NDCG@5": ndcg_at_5 / count, "MRR": reciprocal / count,
        "margin": margin / count,
    }


def evaluate_pretrain_dense_validation(
    cfg: Config, paths: Paths, encoder: DenseEncoder
) -> dict[str, float]:
    """Measure the untouched base retriever on the official validation split."""
    records = read_jsonl(paths.validation)
    _, chunk_cids, _ = load_chunk_arrays(paths.chunks)
    index_path = base_index_path(cfg, paths)
    dense_index = faiss.read_index(str(index_path))
    if dense_index.ntotal != len(chunk_cids):
        raise ValueError(
            f"FAISS/chunk mismatch before validation: index has {dense_index.ntotal:,} "
            f"vectors but {paths.chunks} has {len(chunk_cids):,} chunks"
        )
    if dense_index.d != cfg.embedding_dim:
        raise ValueError(
            f"FAISS dimension mismatch before validation: index dimension is "
            f"{dense_index.d}, but --embedding-dim is {cfg.embedding_dim}"
        )
    was_training = encoder.model.training
    encoder.model.eval()
    try:
        candidates = dense_candidates(
            cfg, encoder, dense_index, [row["question"] for row in records], chunk_cids
        )
    finally:
        if was_training:
            encoder.model.train()
    rankings = [[row["cid"] for row in rows] for rows in candidates]
    metrics = task_metrics(rankings, records, 5)
    log(f"Pre-train dense validation ({len(records)} queries): {json.dumps(metrics, ensure_ascii=False)}")
    return metrics


def embedding_training_recipe(cfg: Config, stage: int) -> dict[str, Any]:
    rank = cfg.embed_stage3_lora_rank if stage == 3 else cfg.lora_rank
    lora_recipe = {
        "lora_rank": rank,
        "lora_alpha": resolved_lora_alpha(rank, cfg.lora_use_rslora),
        "lora_use_rslora": cfg.lora_use_rslora,
        "lora_target_profile": cfg.lora_target_profile,
        "checkpoint_selector": "Hit@5_then_NDCG@5_then_MRR_v1",
        "batch_false_negative_guard": "unique_source_and_positive_cid_v1",
        "split_seed": cfg.seed,
        "eval_size": cfg.eval_size,
    }
    if stage == 3:
        return {
            "version": "dense_stage3_independent_ab_boundary_rehearsal_v5", "stage": 3,
            "dense_fusion_version": DENSE_FUSION_VERSION,
            "objective": "full_dimension_inbatch_infonce_no_teacher_distillation",
            "temperature": cfg.embedding_temperature,
            "teacher_kd_weight": cfg.embed_stage3_kd_weight,
            "hard_negatives": cfg.embed_stage3_hard_negatives,
            "negative_mining": "a_or_b_top5_boundary_plus_broad_rehearsal",
            "rehearsal_ratio": cfg.embed_stage3_rehearsal_ratio,
            "initialization": "base_bge_m3_independent",
            "dense_chunk_k": cfg.dense_chunk_k,
            "epochs": cfg.embed_stage3_epochs,
            "steps_per_epoch_limit": cfg.embed_stage3_steps_per_epoch,
            "min_epochs": cfg.embed_stage3_min_epochs,
            "early_stopping_patience": cfg.embed_early_stopping_patience,
            "learning_rate": cfg.embed_stage3_learning_rate,
            **lora_recipe, "grad_accum": cfg.embed_grad_accum,
            "warmup_ratio": cfg.warmup_ratio,
        }
    if stage == 2:
        return {
            "version": "dense_full1024_teacher_distill_lexical_stratified_v4",
            "stage": 2,
            "objective": "full_dimension_inbatch_infonce",
            "temperature": cfg.embedding_temperature,
            "teacher_kd_weight": cfg.embedding_kd_weight,
            "hard_negatives": cfg.embed_stage2_hard_negatives,
            "lexical_negative_ratio": cfg.embed_stage2_lexical_negative_ratio,
            "negative_mining": "base_bge_reranker_teacher_lexical_stratified",
            "dense_chunk_k": cfg.dense_chunk_k,
            "epochs": cfg.embed_stage2_epochs,
            "steps_per_epoch_limit": cfg.embed_stage2_steps_per_epoch,
            "max_micro_steps": cfg.embed_stage2_max_steps,
            "min_epochs": cfg.embed_stage2_min_epochs,
            "early_stopping_patience": cfg.embed_early_stopping_patience,
            "learning_rate": cfg.embed_stage2_learning_rate,
            **lora_recipe,
            "grad_accum": cfg.embed_grad_accum,
            "warmup_ratio": cfg.warmup_ratio,
        }
    return {
        "version": "dense_full1024_teacher_distill_v3",
        "stage": stage,
        "objective": "full_dimension_inbatch_infonce",
        "temperature": cfg.embedding_temperature,
        "teacher_kd_weight": 0.0,
        "hard_negatives": cfg.hard_negatives,
        "negative_mining": "supplied_groups",
        "epochs": cfg.embed_epochs,
        "steps_per_epoch_limit": cfg.embed_steps_per_epoch,
        "max_micro_steps": cfg.embed_max_steps,
        "min_epochs": cfg.embed_min_epochs,
        "early_stopping_patience": cfg.embed_early_stopping_patience,
        "learning_rate": cfg.embed_learning_rate,
        **lora_recipe,
        "grad_accum": cfg.embed_grad_accum,
        "warmup_ratio": cfg.warmup_ratio,
    }


def source_unique_batches(
    rows: Sequence[dict[str, Any]], batch_size: int, rng: random.Random,
    allow_shared_gold: bool = False,
) -> list[list[dict[str, Any]]]:
    """Avoid in-batch false negatives from duplicate query or positive CID."""
    def positive_key(row: dict[str, Any]) -> str:
        if allow_shared_gold:
            # C's all-gold loss masks shared CIDs explicitly. Only duplicate
            # queries need exclusion; shared laws should not shrink batches.
            return str(row.get("source_qid", row["qid"]))
        return str(row.get("positive_cid") or " ".join(str(row["positive"]).split()))

    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        queues[str(row.get("source_qid", row["qid"]))].append(row)
    for values in queues.values(): rng.shuffle(values)
    batches: list[list[dict[str, Any]]] = []
    active = [source for source, values in queues.items() if values]
    while active:
        # Randomize ties, then consume the largest queues first. This packs
        # secondary positives into normal-size later batches instead of
        # leaving a long tail of tiny batches.
        rng.shuffle(active)
        active.sort(key=lambda source: len(queues[source]), reverse=True)
        batch: list[dict[str, Any]] = []
        used_positive_keys: set[str] = set()
        for source in active:
            # InfoNCE treats other rows' positives as negatives. The same
            # positive law/document must therefore appear at most once.
            candidate_index = next((
                index for index in range(len(queues[source]) - 1, -1, -1)
                if positive_key(queues[source][index]) not in used_positive_keys
            ), None)
            if candidate_index is None:
                continue
            row = queues[source].pop(candidate_index)
            used_positive_keys.add(positive_key(row))
            batch.append(row)
            if len(batch) >= batch_size:
                break
        if not batch:
            # Starting a new batch makes its first positive eligible, so this
            # fallback is only a defensive guarantee against an infinite loop.
            source = active[0]
            batch.append(queues[source].pop())
        batches.append(batch)
        active = [source for source in active if queues[source]]
    return batches


def train_embedder(cfg: Config, paths: Paths, stage: int = 1) -> None:
    if stage not in {1, 2, 3}:
        raise ValueError("Embedding LoRA stage must be 1, 2 or 3")
    if cfg.no_train_embedding: log("Skipping embedding fine-tuning by request"); return
    adapter_path = {1: paths.embed_adapter, 2: paths.second_embed_adapter, 3: paths.third_embed_adapter}[stage]
    adapter_name = {1: "A", 2: "B", 3: "C"}[stage]
    requested_recipe = embedding_training_recipe(cfg, stage)
    group_path = {
        1: train_groups_path(cfg, paths),
        2: paths.second_stage_groups,
        3: paths.third_stage_groups,
    }[stage]
    expected_group_identity = list(artifact_identity(group_path)[1:])
    if valid_lora_adapter(adapter_path) and not cfg.rebuild:
        metadata_path = adapter_path / "training_meta.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        parent_lora_a = list(artifact_identity(paths.embed_adapter / "adapter_model.safetensors")[1:])
        expected_initialization = (
            "continued_from_lora_a" if stage == 2
            else "base_bge_m3"
        )
        initialization_matches = metadata.get("initialization") == expected_initialization
        parent_matches = stage in {1, 3} or metadata.get("parent_lora_a") == parent_lora_a
        if stage == 3:
            current_teacher_ab = {
                name: {
                    "adapter": list(artifact_identity(adapter / "adapter_model.safetensors")[1:]),
                    "index": list(artifact_identity(index)[1:]),
                }
                for name, (adapter, index) in zip(("a", "b"), (
                    (paths.embed_adapter, paths.final_index),
                    (paths.second_embed_adapter, paths.second_index),
                ))
            }
            parent_matches = metadata.get("mining_teacher_ab") == current_teacher_ab
        chunking_matches = metadata.get("training_chunking") == chunking_fingerprint(paths.chunk_meta)
        if (
            cfg.reuse_trained_embeddings
            and metadata.get("training_recipe") == requested_recipe
            and metadata.get("training_groups_identity") == expected_group_identity
            and initialization_matches and parent_matches and chunking_matches
        ):
            log(f"Reusing trained BGE-M3 LoRA {adapter_name}: {adapter_path}"); return
        reason = "chunking changed" if not chunking_matches else "recipe mismatch"
        log(f"LoRA {adapter_name} checkpoint stale ({reason}); retraining")
    groups = (
        groups_in_current_train_split(read_jsonl(train_groups_path(cfg, paths)), paths)
        if stage == 1 else mine_groups(cfg, paths, force_final=True)
        if stage == 2 else mine_embedding_stage3_groups(cfg, paths)
    )
    rng = random.Random(cfg.seed + {1: 0, 2: 1009, 3: 2027}[stage])
    sources = sorted({row.get("source_qid", row["qid"]) for row in groups})
    rng.shuffle(sources); holdout_sources = set(sources[:min(250, max(1, len(sources) // 20))])
    holdout = [row for row in groups if row.get("source_qid", row["qid"]) in holdout_sources]
    train = [row for row in groups if row.get("source_qid", row["qid"]) not in holdout_sources]
    initial_adapter = paths.embed_adapter if stage == 2 else None
    stage_lora_rank = cfg.embed_stage3_lora_rank if stage == 3 else cfg.lora_rank
    encoder = DenseEncoder(cfg, training=True, adapter=initial_adapter, lora_rank=stage_lora_rank)
    torch = encoder.torch
    parameters = [value for value in encoder.model.parameters() if value.requires_grad]
    learning_rate = {
        1: cfg.embed_learning_rate, 2: cfg.embed_stage2_learning_rate,
        3: cfg.embed_stage3_learning_rate,
    }[stage]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    available_steps_per_epoch = math.ceil(len(train) / cfg.embed_train_batch_size)
    if available_steps_per_epoch <= 0:
        raise ValueError(f"No embedding LoRA {adapter_name} groups remain after holdout")
    requested_steps_per_epoch = {
        1: cfg.embed_steps_per_epoch, 2: cfg.embed_stage2_steps_per_epoch, 3: cfg.embed_stage3_steps_per_epoch,
    }[stage]
    requested_epochs = {
        1: cfg.embed_epochs, 2: cfg.embed_stage2_epochs, 3: cfg.embed_stage3_epochs,
    }[stage]
    minimum_epochs = {
        1: cfg.embed_min_epochs, 2: cfg.embed_stage2_min_epochs, 3: cfg.embed_stage3_min_epochs,
    }[stage]
    # A/B must run the configured number of optimizer micro-steps even when
    # one pass over the groups is shorter. Cycle through freshly reshuffled,
    # source-unique batches; do not silently cap the configured limit to one pass.
    steps_per_epoch = (
        requested_steps_per_epoch if stage in {1, 2}
        else min(available_steps_per_epoch, requested_steps_per_epoch)
    )
    maximum_micro_steps = {
        1: cfg.embed_max_steps,
        2: cfg.embed_stage2_max_steps,
        3: steps_per_epoch * requested_epochs,
    }[stage]
    planned_micro_steps = min(
        maximum_micro_steps, steps_per_epoch * requested_epochs
    )
    full_windows, remainder = divmod(planned_micro_steps, steps_per_epoch)
    updates = (
        full_windows * math.ceil(steps_per_epoch / cfg.embed_grad_accum)
        + (math.ceil(remainder / cfg.embed_grad_accum) if remainder else 0)
    )
    from transformers import get_cosine_schedule_with_warmup
    peft = require_peft()
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, round(updates * cfg.warmup_ratio)), max(1, updates)
    )
    log(
        f"BGE LoRA {adapter_name} training budget: {requested_epochs} epochs x "
        f"up to {steps_per_epoch:,} micro-steps/epoch, {planned_micro_steps:,} "
        f"micro-steps total; one unique pass has about "
        f"{available_steps_per_epoch:,} batches"
    )
    pretrain_dense_validation = (
        evaluate_pretrain_dense_validation(cfg, paths, encoder) if stage == 1 else None
    )
    baseline = evaluate_embedder_groups(encoder, holdout, cfg.embed_train_batch_size)
    log(f"BGE LoRA {adapter_name} epoch 0 baseline: {baseline}")
    best_metrics = baseline; best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in peft.get_peft_model_state_dict(encoder.model).items()}
    history: list[dict[str, Any]] = [{"epoch": 0, "validation": baseline}]
    optimizer.zero_grad(set_to_none=True); step = 0; pending_micro_steps = 0
    epochs_completed = 0; epochs_without_improvement = 0; stopped_early = False
    for epoch in range(requested_epochs):
        current_epoch_steps = min(steps_per_epoch, maximum_micro_steps - step)
        if current_epoch_steps <= 0:
            break
        epoch_loss = 0.0; epoch_batches = 0
        batches: list[list[dict[str, Any]]] = []
        while len(batches) < current_epoch_steps:
            cycle = source_unique_batches(train, cfg.embed_train_batch_size, rng)
            if not cycle:
                raise ValueError(f"Cannot construct LoRA {adapter_name} training batches")
            batches.extend(cycle[:current_epoch_steps - len(batches)])
        train_progress = progress(
            batches, desc=f"Embed LoRA {adapter_name} epoch {epoch + 1}"
        )
        for batch_number, batch in enumerate(train_progress, start=1):
            queries = encoder.tensor([row["query"] for row in batch], True, grad=True)
            positives = encoder.tensor([row["positive"] for row in batch], False, grad=True)
            negative_counts = {len(row["negatives"]) for row in batch}
            if len(negative_counts) != 1:
                raise ValueError(f"Embedding batch has inconsistent negative counts: {negative_counts}")
            negative_count = next(iter(negative_counts))
            flat_neg = [text for row in batch for text in row["negatives"]]
            negatives = encoder.tensor(flat_neg, False, grad=True).reshape(len(batch), negative_count, -1)
            teacher_relevance = None
            if stage == 2 and all(
                row.get("positive_teacher_score") is not None
                and len(row.get("negative_teacher_scores", [])) == negative_count
                for row in batch
            ):
                teacher_relevance = torch.tensor([
                    [float(row["positive_teacher_score"]), *map(float, row["negative_teacher_scores"])]
                    for row in batch
                ], dtype=torch.float32, device=encoder.device)
            loss = dense_contrastive_distillation_loss(
                torch, queries, positives, negatives,
                cfg.embedding_temperature,
                cfg.embedding_kd_weight if stage == 2 else cfg.embed_stage3_kd_weight if stage == 3 else 0.0,
                teacher_relevance,
            ) / cfg.embed_grad_accum
            batch_loss = float(loss.detach().item()) * cfg.embed_grad_accum
            epoch_loss += batch_loss; epoch_batches += 1
            if hasattr(train_progress, "set_postfix"):
                train_progress.set_postfix(
                    loss=f"{batch_loss:.4f}", avg_loss=f"{epoch_loss / epoch_batches:.4f}"
                )
            loss.backward(); step += 1; pending_micro_steps += 1
            if pending_micro_steps >= cfg.embed_grad_accum or batch_number == len(batches):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                pending_micro_steps = 0
        average_loss = epoch_loss / max(1, epoch_batches)
        validation = evaluate_embedder_groups(encoder, holdout, cfg.embed_train_batch_size)
        history.append({"epoch": epoch + 1, "train_loss": average_loss, "validation": validation})
        # The competition target is Recall@5. The supplied log restored A at
        # epoch 4 on Accuracy@1 even though later epochs had better MRR, so all
        # dense stages now select checkpoints at the top-five boundary.
        improved = (
            validation["Hit@5"], validation["NDCG@5"], validation["MRR"],
            validation["Accuracy@1"], validation["margin"],
        ) > (
            best_metrics["Hit@5"], best_metrics["NDCG@5"], best_metrics["MRR"],
            best_metrics["Accuracy@1"], best_metrics["margin"],
        )
        if improved:
            best_metrics = validation; best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in peft.get_peft_model_state_dict(encoder.model).items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epochs_completed = epoch + 1
        log(f"BGE LoRA {adapter_name} epoch {epoch + 1}: avg_loss={average_loss:.4f}, validation={validation}")
        if (
            epochs_completed >= minimum_epochs
            and epochs_without_improvement >= cfg.embed_early_stopping_patience
        ):
            stopped_early = True
            log(
                f"BGE LoRA {adapter_name} early stopping after epoch {epochs_completed}; "
                f"restoring best epoch {best_epoch}"
            )
            break
    peft.set_peft_model_state_dict(encoder.model, best_state)
    adapter_path.mkdir(parents=True, exist_ok=True)
    encoder.model.save_pretrained(adapter_path, safe_serialization=True); encoder.tokenizer.save_pretrained(adapter_path)
    atomic_json(adapter_path / "training_meta.json", {
        "pipeline_stage": f"{stage}/3 - BGE-M3 embedding LoRA {adapter_name}",
        "initialization": (
            "continued_from_lora_a" if stage == 2
            else "base_bge_m3"
        ),
        "parent_lora_a": (
            list(artifact_identity(paths.embed_adapter / "adapter_model.safetensors")[1:])
            if stage == 2 else None
        ),
        "learning_rate": learning_rate,
        "base_model": cfg.embedding_model, "train_groups": len(train), "holdout_groups": len(holdout),
        "requested_epochs": requested_epochs, "completed_epochs": epochs_completed,
        "steps_per_epoch": steps_per_epoch,
        "max_micro_steps": maximum_micro_steps,
        "planned_micro_steps": planned_micro_steps,
        "completed_micro_steps": step,
        "loss": (
            "Full-1024 in-batch InfoNCE (no A/B distillation)" if stage == 3
            else "Full-1024 in-batch InfoNCE + teacher listwise distillation"
        ),
        "training_recipe": requested_recipe,
        "training_groups_identity": list(artifact_identity(group_path)[1:]),
        "training_chunking": chunking_fingerprint(paths.chunk_meta),
        "best_epoch": best_epoch,
        "pretrain_dense_validation": (
            {"eval_queries": cfg.eval_size, "metrics": pretrain_dense_validation}
            if pretrain_dense_validation is not None else None
        ),
        "negative_source": (
            "supplied groups" if stage == 1
            else "base-reranker-teacher hardest + lexical-overlap-stratified negatives from B's own pool" if stage == 2
            else "A/B top-5 boundary failures plus deterministic broad-query rehearsal"
        ),
        "mining_teacher_ab": ({
            name: {
                "adapter": list(artifact_identity(adapter / "adapter_model.safetensors")[1:]),
                "index": list(artifact_identity(index)[1:]),
            }
            for name, (adapter, index) in zip(("a", "b"), (
                (paths.embed_adapter, paths.final_index),
                (paths.second_embed_adapter, paths.second_index),
            ))
        } if stage == 3 else None),
        "best_validation": best_metrics, "history": history,
        "stopped_early": stopped_early,
        "guardrail": (
            "Best held-out epoch retained (epoch 0 is base BGE-M3)" if stage in {1, 3}
            else "Best held-out epoch retained (epoch 0 is the unchanged LoRA A checkpoint)"
        ),
        "config": asdict(cfg),
    }); encoder.close()
    if stage == 1:
        stale_artifacts = (paths.final_index, paths.second_stage_groups)
    elif stage == 2:
        stale_artifacts = (paths.second_index, paths.reranker_groups, paths.dense_calibration)
    else:
        stale_artifacts = (paths.third_index, paths.second_reranker_groups, paths.dense_calibration)
    for stale in stale_artifacts:
        if stale.exists():
            stale.unlink()
