"""Huấn luyện reranker R2 tiếp tục từ R1.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import copy
import math
from collections import Counter

from legalir.experiment.constants import ADAPTERS, VERSION
from legalir.experiment.utils import adapter_identity, balanced_schedule, copy_adapter, read
from legalir.experiment.losses import multi_gold_loss
from legalir.experiment.c_helpers import c_step_learning_rate
from legalir.rerank.fusion import FIXED


class TrainR2Mixin:
    def train_r2(self, pool):
        if self.completed("r2"):
            self.api.log("Reusing completed R2 under identical source, code and recipe")
            return
        a, cfg = self.api, self.cfg
        dev_rows = self.scored_pool(self.dev, self.members(self.dev), pool)
        baseline_rows = [[{**r, "r2": r["r1"]} for r in group] for group in dev_rows]
        old_rows = self.score(self.dev, dev_rows, self.source / ADAPTERS[4], "r2")
        baseline = self.subgroup_metrics(baseline_rows, self.dev)
        old = self.subgroup_metrics(old_rows, self.dev)
        def key(m): return (m["all"]["Recall@5"], m["multi"]["AllGold@5"], m["all"]["AllGold@5"], m["all"]["MRR@5"])
        best_metric, best_source, best_label = baseline, self.root / ADAPTERS[3], "R1_epoch0"
        if key(old) > key(best_metric):
            best_metric, best_source, best_label = old, self.source / ADAPTERS[4], "source_R2"
        reference = copy.deepcopy(best_metric)
        history = [{"checkpoint": "R1_epoch0", "metrics": baseline}, {"checkpoint": "source_R2", "metrics": old}]
        # Reuse cached A/B query rankings; only selected C needs new retrieval.
        # Exclude impossible all-gold pools BEFORE expensive R1 forwards.
        train_rows = self.dense_pool(self.members(self.train), pool)
        eligible_records, eligible_rows, excluded = [], [], []
        for record, rows in zip(self.train, train_rows):
            missing = set(record["answers"]) - {str(r["cid"]) for r in rows}
            if missing:
                excluded.append({"qid": record["qid"], "missing_pool": sorted(missing),
                                 "accepted": False, "reason": "missing_pool_gold"})
            else:
                eligible_records.append(record); eligible_rows.append(rows)
        a.atomic_json(self.root / "r2_pool_audit.json", {
            "train_queries": len(self.train), "r1_scored_queries": len(eligible_records),
            "skipped_before_r1": excluded})
        if not eligible_records:
            raise ValueError("No feasible all-gold R2 pools; inspect r2_pool_audit.json")
        a.log(f"R2 uses cached A/B + selected C; R1 scores {len(eligible_records)} queries, skips {len(excluded)} missing-gold pools")
        train_rows = self.score(eligible_records, eligible_rows, self.root / ADAPTERS[3], "r1")
        groups = self.mine(eligible_records, train_rows, "r2")
        mining_audit = read(self.root / "r2_mining_audit.json")
        a.atomic_json(self.root / "r2_mining_audit.json", [*excluded, *mining_audit])
        if not groups:
            raise ValueError("No feasible clean R2 groups; inspect r2_mining_audit.json")
        records_by_id = {str(r["qid"]): r for r in eligible_records}
        rows_by_id = {str(r["qid"]): rows for r, rows in zip(eligible_records, train_rows)}
        a.atomic_json(self.root / "r2_training_recipe.json", {
            "steps_per_epoch": cfg.cr2_r2_steps, "lr_decay": cfg.cr2_r2_lr_decay,
            "lr_floor_ratio": .1, "warmup_ratio": .1,
            "boundary_weight": cfg.cr2_r2_boundary_weight, "remine_r2": cfg.cr2_remine_r2,
            "refresh_steps": cfg.cr2_r2_refresh_steps, "fixed_weights": FIXED,
            "eligible_groups": len(groups), "teacher": "frozen_R1",
            "pool_and_evidence": "fixed_selected_C", "sampling_strata": "initial_B_R1_baseline"})
        model = a.BGEReranker(cfg, training=True, adapter=self.root / ADAPTERS[3])
        torch = model.torch
        optimizer = torch.optim.AdamW([p for p in model.model.parameters() if p.requires_grad], lr=cfg.cr2_learning_rate)
        visits, update, stale_epochs = {}, 0, 0
        try:
            for epoch in range(1, cfg.cr2_r2_epochs + 1):
                schedule = balanced_schedule(groups, cfg.cr2_r2_steps, visits, cfg.seed + 200 + epoch)
                self.save_schedule("r2", epoch, groups, schedule, visits)
                model.model.train(); optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                loss_totals = Counter()
                active_groups = {}
                for step, index in enumerate(a.progress(schedule, desc=f"Train R2 epoch {epoch}")):
                    if cfg.cr2_remine_r2 and step % cfg.cr2_r2_refresh_steps == 0:
                        upcoming = [groups[i] for i in schedule[step:step + cfg.cr2_r2_refresh_steps]]
                        active_groups = self.refresh_r2_groups(model, upcoming, records_by_id, rows_by_id,
                            epoch, step // cfg.cr2_r2_refresh_steps + 1, update)
                    row = active_groups[str(groups[index]["qid"])] if cfg.cr2_remine_r2 else groups[index]
                    documents = row["positive_chunks"] + row["negative_chunks"]
                    texts = [self.texts[i] for ids in documents for i in ids]
                    flat = torch.cat([model.logits([row["question"]] * len(texts[j:j + cfg.rerank_batch_size]),
                                                   texts[j:j + cfg.rerank_batch_size])
                                      for j in range(0, len(texts), cfg.rerank_batch_size)])
                    scores = a.aggregate_document_max_scores(torch, flat, [len(ids) for ids in documents])
                    loss, components = multi_gold_loss(torch, scores, len(row["answers"]), row["teacher"],
                        boundary_weight=cfg.cr2_r2_boundary_weight, return_components=True)
                    for name, value in zip(components, torch.stack(list(components.values())).detach().cpu().tolist()):
                        loss_totals[name] += value
                    group_size = min(cfg.rerank_grad_accum, len(schedule) - (step // cfg.rerank_grad_accum) * cfg.rerank_grad_accum)
                    (loss / group_size).backward(); total_loss += float(loss.detach())
                    if (step + 1) % cfg.rerank_grad_accum == 0 or step + 1 == len(schedule):
                        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0, error_if_nonfinite=True)
                        update += 1
                        total_updates = cfg.cr2_r2_epochs * math.ceil(
                            min(cfg.cr2_r2_steps, 2 * len(groups)) / cfg.rerank_grad_accum)
                        warmup = max(1, math.ceil(total_updates * 0.1))
                        for group in optimizer.param_groups:
                            group["lr"] = (c_step_learning_rate(update, total_updates, cfg.cr2_learning_rate)
                                           if cfg.cr2_r2_lr_decay else
                                           cfg.cr2_learning_rate * min(1.0, update / warmup))
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
                target = self.root / "r2_checkpoints" / f"epoch{epoch}"
                model.model.save_pretrained(target); model.model.eval()
                scored = self.score(self.dev, dev_rows, target, "r2")
                metrics = self.subgroup_metrics(scored, self.dev)
                # Fixed weights are the user's reproducible anchor. No selection
                # on an R2 weight which is later silently replaced at prediction.
                safe = (metrics["single"]["Recall@5"] >= reference["single"]["Recall@5"] - 0.002
                        and metrics["multi"]["Recall@5"] >= reference["multi"]["Recall@5"]
                        and metrics["multi"]["AllGold@5"] >= reference["multi"]["AllGold@5"])
                history.append({"checkpoint": f"epoch{epoch}", "metrics": metrics, "guardrail": safe,
                                "loss": total_loss / len(schedule),
                                "loss_components": {name: value / len(schedule) for name, value in loss_totals.items()},
                                "optimizer_updates": update, "learning_rate": optimizer.param_groups[0]["lr"],
                                "remine_r2": cfg.cr2_remine_r2, "boundary_weight": cfg.cr2_r2_boundary_weight})
                if safe and key(metrics) > key(best_metric):
                    best_metric, best_source, best_label = metrics, target, f"epoch{epoch}"
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                a.atomic_json(self.root / "r2_checkpoint_report.json", history)
                a.log(f"R2 epoch {epoch}: fixed Recall@5={metrics['all']['Recall@5']:.6f}; multi AllGold@5={metrics['multi']['AllGold@5']:.6f}; selected={best_label}")
                if stale_epochs >= cfg.rerank_early_stopping_patience:
                    a.log(f"R2 stopped after {stale_epochs} epochs without an accepted improvement")
                    break
        finally:
            model.close()
        target = self.root / ADAPTERS[4]
        copy_adapter(best_source, target)
        a.atomic_json(target / "pipeline_config.json", {
            "training_recipe": VERSION, "initialization": "R1", "selected": best_label,
            "selected_from": str(best_source), "run_manifest": self.signature,
            "max_length": cfg.reranker_max_length, "fixed_weights": FIXED,
            "boundary_weight": cfg.cr2_r2_boundary_weight, "remine_r2": cfg.cr2_remine_r2,
            "refresh_steps": cfg.cr2_r2_refresh_steps,
            "source_chunk_meta": read(self.source / "prepared" / "chunk_meta.json")})
        a.atomic_json(self.root / "r2_complete.json", {"run": self.signature, "adapter": adapter_identity(target),
                      "selected": best_label, "metrics": best_metric, "history": history, "fixed_weights": FIXED})
