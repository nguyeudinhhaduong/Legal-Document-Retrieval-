"""Huấn luyện dense C tiếp tục từ B.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import copy
import math
import random

from legalir.experiment.constants import ADAPTERS, INDEXES, VERSION
from legalir.experiment.utils import adapter_identity, balanced_schedule, copy_adapter, read
from legalir.experiment.losses import dense_document_loss
from legalir.experiment.c_helpers import c_checkpoint_key, c_checkpoint_safe, c_mining_candidates, c_repair_targets, c_step_learning_rate
from legalir.rerank.fusion import FIXED


class TrainCMixin:
    def train_c(self):
        if self.completed("c"):
            self.api.log("Reusing completed C under identical source, code and recipe")
            self.build_index(self.root / ADAPTERS[2], INDEXES[2])
            return read(self.root / "c_complete.json")["pool"]
        cfg, a = self.cfg, self.api
        ab_dev = [self.retrieve(self.dev, self.root / ADAPTERS[i], self.source / INDEXES[i]) for i in (0, 1)]
        old = self.retrieve(self.dev, self.source / ADAPTERS[2], self.source / INDEXES[2])
        old_pool = self.choose_pool([*ab_dev, old], self.dev)
        b_pool = self.choose_pool([*ab_dev, ab_dev[1]], self.dev)
        ab_dev_rows = self.pool([*ab_dev, ab_dev[1]], b_pool["weights"], b_pool["quota"])
        old_pool.update(self.evaluate_c([*ab_dev, old], old_pool, ab_dev_rows))
        b_pool.update(self.evaluate_c([*ab_dev, ab_dev[1]], b_pool, ab_dev_rows))
        history = [{"checkpoint": "source_C", **old_pool}, {"checkpoint": "B_epoch0", **b_pool}]
        key = c_checkpoint_key
        def safe(p): return c_checkpoint_safe(p, old_pool, b_pool)
        best_pool = old_pool
        best_source = self.source / ADAPTERS[2]
        best_label = "source_C"
        if safe(b_pool) and key(b_pool) > key(best_pool):
            best_pool, best_source, best_label = b_pool, self.root / ADAPTERS[1], "B_epoch0"
        # Read source A/B indexes directly, encode only queries for initial mining.
        # Use the AB selector from development; combine the duplicated B vote.
        ab_train = [self.retrieve(self.train, self.root / ADAPTERS[i], self.source / INDEXES[i]) for i in (0, 1)]
        ab_weights = (b_pool["weights"][0], sum(b_pool["weights"][1:]))
        candidate_rows = a.fuse_dense_ensemble_members(ab_train, ab_weights,
            cfg.candidate_docs, cfg.chunks_per_doc, b_pool["quota"])
        # Freeze B's own failures separately from union failures.
        b_failures = c_repair_targets(self.train, ab_train[1], candidate_rows, cfg.candidate_docs)
        a.atomic_json(self.root / "c_b_repair_targets.json", {
            qid: {name: sorted(cids) for name, cids in failures.items()} for qid, failures in b_failures.items()})
        candidate_rows = c_mining_candidates(self.train, candidate_rows, ab_train[1], {},
                                            cfg.cr2_c_negatives, cfg.chunks_per_doc)
        groups = self.mine(self.train, candidate_rows, "c")
        replay_groups = {str(r["qid"]): copy.deepcopy(r) for r in groups}
        a.atomic_json(self.root / "c_mining_source.json", {
            "repair_target": "B_top5_and_B_pool", "query_weights": {"pool_miss": 2., "top5_miss": 1.5, "rehearsal": 1.},
            "requested_negatives": cfg.cr2_c_negatives, "evidence_per_document": cfg.cr2_c_evidence,
            "positive_lexical_shortlist": cfg.cr2_c_positive_shortlist,
            "indexes": [str(self.source / INDEXES[i]) for i in (0, 1)],
            "index_sha256": [self.file_digest(self.source / INDEXES[i]) for i in (0, 1)],
            "weights": {"a": ab_weights[0], "b": ab_weights[1]}, "quota": b_pool["quota"],
            "groups": len(groups), "remine_with_current_c": cfg.cr2_remine_c,
            "b_replay_fraction": cfg.cr2_c_replay_fraction,
            "corpus_reencoded_for_A_B": False})
        a.log(f"C mining: {len(groups)} groups from existing A/B indexes; remine_C={cfg.cr2_remine_c}")
        # C always starts from B even when the old C is the fallback selector.
        model = a.DenseEncoder(cfg, training=True, adapter=self.root / ADAPTERS[1])
        torch = model.torch
        optimizer = torch.optim.AdamW([p for p in model.model.parameters() if p.requires_grad], lr=cfg.cr2_c_learning_rate)
        visits, update, stale_epochs = {}, 0, 0
        total_updates = None
        current_adapter, current_index = self.root / ADAPTERS[1], self.source / INDEXES[1]
        try:
            for epoch in range(1, cfg.cr2_c_epochs + 1):
                if epoch > 1 and cfg.cr2_remine_c:
                    current = self.retrieve(self.train, current_adapter, current_index)
                    candidate_rows = a.fuse_dense_ensemble_members([*ab_train, current],
                        (ab_weights[0] * .8, ab_weights[1] * .8, .2),
                        cfg.candidate_docs, cfg.chunks_per_doc, 3)
                    candidate_rows = c_mining_candidates(self.train, candidate_rows, current,
                        replay_groups if cfg.cr2_c_replay_fraction else {}, cfg.cr2_c_negatives, cfg.chunks_per_doc)
                    groups = self.mine(self.train, candidate_rows, "c")
                for row in groups:
                    failures = b_failures[str(row["qid"])]
                    row["repair"] = bool(failures["top5"])
                schedule = balanced_schedule(groups, cfg.cr2_c_steps * cfg.embed_train_batch_size,
                                             dict(visits), cfg.seed + epoch)
                # Every gold is supervised in every query visit. A/B misses
                # receive bounded relative emphasis within multi-gold queries.
                rows = []
                for i in schedule:
                    r = groups[i]; qid = str(r["qid"])
                    failures = b_failures[qid]
                    documents = [ids[:cfg.cr2_c_evidence] for ids in r["positive_chunks"] + r["negative_chunks"]]
                    rows.append({"qid": qid, "source_qid": qid, "gold_cids": r["answers"],
                        "query_weight": 2. if failures["pool"] else 1.5 if failures["top5"] else 1.,
                        "query": r["question"], "document_chunks": documents,
                        "document_cids": r["answers"] + r["negative_cids"],
                        "gold_weights": [2. if cid in failures["pool"] else
                                         1.5 if cid in failures["top5"] else 1.
                                         for cid in r["answers"]]})
                batches = a.source_unique_batches(rows, cfg.embed_train_batch_size, random.Random(cfg.seed + epoch),
                                                  allow_shared_gold=True)
                batches = batches[:cfg.cr2_c_steps]
                if not batches:
                    raise ValueError("No C batches after mining and packing")
                if total_updates is None:
                    total_updates = cfg.cr2_c_epochs * math.ceil(len(batches) / cfg.embed_grad_accum)
                group_index = {str(r["qid"]): i for i, r in enumerate(groups)}
                actual_schedule = [group_index[r["qid"]] for batch in batches for r in batch]
                for i in actual_schedule:
                    qid = str(groups[i]["qid"])
                    visits[qid] = visits.get(qid, 0) + 1
                self.save_schedule("c", epoch, groups, actual_schedule, visits)
                model.model.train(); optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for step, batch in enumerate(a.progress(batches, desc=f"Train C epoch {epoch}")):
                    query = model.tensor([r["query"] for r in batch], True, grad=True)
                    texts = [self.texts[i] for r in batch for ids in r["document_chunks"] for i in ids]
                    evidence = torch.cat([model.tensor(texts[j:j + cfg.embed_batch_size], False, grad=True)
                                          for j in range(0, len(texts), cfg.embed_batch_size)])
                    loss = dense_document_loss(torch, query, evidence, batch, cfg.embedding_temperature)
                    group_size = min(cfg.embed_grad_accum, len(batches) - (step // cfg.embed_grad_accum) * cfg.embed_grad_accum)
                    (loss / group_size).backward(); total_loss += float(loss.detach())
                    if (step + 1) % cfg.embed_grad_accum == 0 or step + 1 == len(batches):
                        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0, error_if_nonfinite=True)
                        update += 1
                        for group in optimizer.param_groups:
                            group["lr"] = c_step_learning_rate(update, total_updates, cfg.cr2_c_learning_rate)
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
                current_adapter = self.root / "c_checkpoints" / f"epoch{epoch}"
                model.model.save_pretrained(current_adapter)
                model.model.eval()
                current_index = self.build_index(current_adapter, "c_current.faiss")
                dev_c = self.retrieve(self.dev, current_adapter, current_index)
                pool = self.choose_pool([*ab_dev, dev_c], self.dev)
                pool.update(self.evaluate_c([*ab_dev, dev_c], pool, ab_dev_rows))
                history.append({"checkpoint": f"epoch{epoch}", "loss": total_loss / len(batches),
                                "guardrail": safe(pool), "optimizer_updates": update,
                                "learning_rate": optimizer.param_groups[0]["lr"], **pool})
                if safe(pool) and key(pool) > key(best_pool):
                    best_pool, best_source, best_label = pool, current_adapter, f"epoch{epoch}"
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                a.atomic_json(self.root / "c_checkpoint_report.json", history)
                a.log(f"C epoch {epoch}: pool Recall@{cfg.candidate_docs}={pool['pool_recall']:.6f}; "
                      f"C-B standalone Recall@5={pool['standalone']['delta_c_minus_b']['Recall@5']:+.6f}; "
                      f"frozen R1 Recall@5={pool['anchor']['all']['Recall@5']:.6f}; "
                      f"recovered/lost gold vs A/B={pool['complement']['recovered_gold_vs_ab']}/"
                      f"{pool['complement']['lost_gold_vs_ab']}; selected={best_label}")
                if stale_epochs >= cfg.embed_early_stopping_patience:
                    a.log(f"C stopped after {stale_epochs} epochs without checkpoint improvement")
                    break
        finally:
            model.close()
        target = self.root / ADAPTERS[2]
        copy_adapter(best_source, target)
        selected_index = self.build_index(target, INDEXES[2])
        # Better pool recall alone need not improve a cross-encoder: changing C
        # also changes evidence. Check the selected C with frozen R1 before R2.
        old_anchor_rows = self.scored_pool(self.dev, [*ab_dev, old], old_pool)
        new_c = self.retrieve(self.dev, target, selected_index)
        new_anchor_rows = self.scored_pool(self.dev, [*ab_dev, new_c], best_pool)
        def anchor_metrics(rows):
            return self.subgroup_metrics([[{**r, "r2": r["r1"]} for r in group] for group in rows], self.dev)
        old_anchor, new_anchor = anchor_metrics(old_anchor_rows), anchor_metrics(new_anchor_rows)
        guard_passed = (new_anchor["all"]["Recall@5"] >= old_anchor["all"]["Recall@5"] - .002
                and new_anchor["multi"]["Recall@5"] >= old_anchor["multi"]["Recall@5"] - .002)
        a.atomic_json(self.root / "c_frozen_r1_guardrail.json", {
            "source_C": old_anchor, "candidate_C": new_anchor, "passed": guard_passed,
            "candidate": best_label, "tolerance": .002, "split": "development", "weights": FIXED})
        if not guard_passed:
            best_pool, best_source, best_label = old_pool, self.source / ADAPTERS[2], "source_C_guardrail_fallback"
            copy_adapter(best_source, target)
            self.build_index(target, INDEXES[2])
            a.log("Selected C changed evidence and reduced frozen R1 Recall@5; restored source C")
        a.atomic_json(target / "training_meta.json", {
            "training_recipe": VERSION, "initialization": "B", "selected": best_label,
            "selected_from": str(best_source), "run_manifest": self.signature,
            "selection": "development frozen R1 Recall@5 then pool recall; subgroup and pool guards; old C and B fallback",
            "source_chunk_meta": read(self.source / "prepared" / "chunk_meta.json")})
        a.atomic_json(self.root / "c_complete.json", {"run": self.signature, "adapter": adapter_identity(target),
                      "selected": best_label, "pool": best_pool, "history": history})
        return best_pool
