"""Mine hard negative cho C và R2.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import copy
import math

from legalir.experiment.constants import ADAPTERS
from legalir.experiment.utils import identity
from legalir.experiment.c_helpers import c_evidence_selection
from legalir.rerank.fusion import FIXED, rank_features, rankings


class MiningMixin:
    def mine(self, records, rows, stage, audit_label=None):
        negative_limit = self.cfg.cr2_c_negatives if stage == "c" else self.cfg.cr2_negatives
        recipe = {"records": records, "rows": rows, "stage": stage,
                  "negatives": negative_limit, "c_evidence": self.cfg.cr2_c_evidence,
                  "c_positive_shortlist": self.cfg.cr2_c_positive_shortlist,
                  "r1": self.frozen[ADAPTERS[3]]}
        def compute():
            groups, audit = [], []
            teacher = (self.api.BGEReranker(self.cfg, training=False, adapter=self.root / ADAPTERS[3])
                       if stage == "c" else None)
            # Ignore the document-name line for duplicate-evidence detection.
            def text_key(index):
                text = self.texts[index].split("\n", 1)[-1]
                return identity(" ".join(text.casefold().split()))
            fingerprints = [text_key(i) for i in range(len(self.texts))]
            try:
                for record, candidates in self.api.progress(list(zip(records, rows)), desc=f"Mine {stage}"):
                    gold = set(map(str, record["answers"]))
                    lookup = {str(row["cid"]): i for i, row in enumerate(candidates)}
                    missing = gold - lookup.keys()
                    item = {"qid": record["qid"], "missing_pool": sorted(missing), "accepted": False}
                    audit.append(item)
                    if stage == "r2" and missing:
                        # A cross-encoder cannot retrieve missing documents. C is
                        # responsible for these examples; R2 learns feasible pools.
                        item["reason"] = "missing_pool_gold"
                        continue
                    if stage == "r2":
                        scores = [row["r1"] for row in candidates]
                        best_chunks = [row["chunks"][0] for row in candidates]
                    else:
                        scores, best_chunks, _ = self.api.teacher_candidate_scores(
                            teacher, record["question"], candidates, self.texts)
                    teacher_order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]["cid"]))
                    terms = self.api.informative_tokens(record["question"])
                    lexical = sorted(range(len(candidates)), key=lambda i: (
                        -len(terms & self.api.informative_tokens(self.texts[best_chunks[i]])), candidates[i]["cid"]))
                    if stage == "r2":
                        provisional = [[{**row, "r1": score, "r2": row.get("r2", score)}
                                        for row, score in zip(candidates, scores)]]
                        f, ids = rank_features(provisional)
                        fused = rankings(f, ids, FIXED)[0]
                        order = [lookup[cid] for cid in fused]
                    else:
                        order = sorted(range(len(candidates)), key=lambda i: (
                            -float(candidates[i].get("_c_dense", candidates[i]["dense"])), str(candidates[i]["cid"])))
                    item["missing_top5"] = sorted(gold - {str(candidates[i]["cid"]) for i in order[:5]})
                    positive_chunks = []
                    for cid in sorted(gold):
                        if stage == "r2":
                            positive_chunks.append(list(candidates[lookup[cid]]["chunks"]))
                        else:
                            local = sorted(self.doc_chunks[cid], key=lambda i: (
                                -len(terms & self.api.informative_tokens(self.texts[i])), i))[:self.cfg.cr2_c_positive_shortlist]
                            retrieved = candidates[lookup[cid]]["chunks"] if cid in lookup else []
                            shortlist = list(dict.fromkeys([*retrieved, *local]))
                            values = teacher.scores([record["question"]] * len(shortlist), [self.texts[i] for i in shortlist])
                            positive_chunks.append(c_evidence_selection(
                                shortlist, values, retrieved, self.cfg.cr2_c_evidence))
                    positive_text = {fingerprints[i] for cid in gold for i in self.doc_chunks[cid]}
                    seen = set(positive_text)
                    selected, selected_ids, sources = [], set(), []
                    def add(i, source):
                        row = candidates[i]
                        hashes = {fingerprints[j] for j in row["chunks"]}
                        if str(row["cid"]) in gold or i in selected_ids or hashes & seen or not hashes:
                            return False
                        selected.append(i); selected_ids.add(i); seen.update(hashes); sources.append(source)
                        return True
                    if stage == "r2":
                        strata = (("fused_top5", order[:5], 5), ("boundary6_15", order[5:15], 5),
                                  ("r1_hard", teacher_order, 4), ("tail", order[15:], 2))
                    else:
                        replay_order = [i for i in order if candidates[i].get("_c_replay")]
                        replay_count = int(negative_limit * self.cfg.cr2_c_replay_fraction) if replay_order else 0
                        remaining = negative_limit - replay_count
                        dense_count = remaining // 2
                        teacher_count = (remaining - dense_count) // 2
                        strata = (("b_replay", replay_order, replay_count), ("dense_hard", order[:30], dense_count),
                                  ("r1_hard", teacher_order, teacher_count),
                                  ("lexical", lexical, remaining - dense_count - teacher_count))
                    for label, indices, quota in strata:
                        added = 0
                        for i in indices:
                            if len(selected) >= negative_limit or added >= quota:
                                break
                            if add(i, label):
                                added += 1
                    for i in teacher_order:
                        if len(selected) >= negative_limit:
                            break
                        add(i, "fill_teacher")
                    item["requested_negatives"] = negative_limit
                    item["selected_negatives"] = len(selected)
                    # Do not discard a useful repair merely because extra clean negatives ran out.
                    minimum_negatives = min(16, negative_limit) if stage == "c" else negative_limit
                    if len(selected) < minimum_negatives:
                        item["reason"] = "insufficient_clean_negatives"
                        continue
                    # Dense failure may be caused by a different chunk from
                    # the teacher's favorite. Keep both views for max pooling.
                    negatives = [list(candidates[i]["chunks"]) if stage == "r2" else
                        list(dict.fromkeys([candidates[i]["chunks"][0], best_chunks[i], *candidates[i]["chunks"]]))[:self.cfg.cr2_c_evidence]
                        for i in selected]
                    group = {**record, "answers": sorted(gold), "positive_chunks": positive_chunks, "negative_chunks": negatives,
                             "negative_cids": [str(candidates[i]["cid"]) for i in selected],
                             "negative_sources": sources, "repair": bool(item["missing_top5"])}
                    if stage == "r2":
                        group["teacher"] = [scores[lookup[cid]] for cid in sorted(gold)] + [scores[i] for i in selected]
                    groups.append(group); item["accepted"] = True
            finally:
                if teacher is not None:
                    teacher.close()
            return {"groups": groups, "audit": audit}
        result = self.cached("mining", recipe, compute)
        self.api.atomic_json(self.root / f"{audit_label or stage + '_mining_audit'}.json", result["audit"])
        return result["groups"]

    def refresh_r2_groups(self, model, scheduled_groups, records_by_id, rows_by_id, epoch, block, updates):
        """Score only the next block's queries with the live, pre-update R2.

        Candidate CIDs/evidence, dense votes and R1 teacher scores stay fixed.
        Sampling strata stay tied to the initial baseline across refreshes.
        """
        baseline = {str(row["qid"]): row for row in scheduled_groups}
        records = [records_by_id[qid] for qid in baseline]
        rows = copy.deepcopy([rows_by_id[qid] for qid in baseline])
        self.validate_pools(records, rows)
        was_training = model.model.training
        try:
            model.model.eval()
            with model.torch.inference_mode():
                for record, candidates in self.api.progress(list(zip(records, rows)), desc=f"Refresh R2 epoch {epoch} block {block}"):
                    values, _, _ = self.api.teacher_candidate_scores(model, record["question"], candidates, self.texts)
                    if len(values) != len(candidates) or not all(math.isfinite(float(v)) for v in values):
                        raise ValueError(f"Invalid current R2 scores for qid={record['qid']}")
                    for row, value in zip(candidates, values):
                        row["r2"] = float(value)
        finally:
            model.model.train(was_training)
        refreshed = self.mine(records, rows, "r2", audit_label=f"r2_refresh_epoch{epoch}_block{block}_mining_audit")
        by_id = {str(row["qid"]): row for row in refreshed}
        fallback, changed = [], []
        for qid, original in baseline.items():
            if qid not in by_id:
                # Greedy duplicate-evidence filtering can exhaust a different
                # ordering. Keep the already-validated original group explicitly.
                by_id[qid] = copy.deepcopy(original)
                fallback.append(qid)
            else:
                by_id[qid]["repair"] = original["repair"]
                if set(by_id[qid]["negative_cids"]) != set(original["negative_cids"]):
                    changed.append(qid)
        self.api.atomic_json(self.root / f"r2_refresh_epoch{epoch}_block{block}.json", {
            "epoch": epoch, "block": block, "optimizer_updates_before_scoring": updates,
            "unique_queries": len(records), "scheduled_visits": len(scheduled_groups),
            "qids": list(baseline), "changed_negative_set_qids": changed,
            "fallback_to_initial_group_qids": fallback, "weights": FIXED,
            "teacher": "frozen_R1", "mining_ranker": "current_R2_eval_mode"})
        self.api.log(f"R2 refresh epoch {epoch} block {block}: scored={len(records)}, "
                     f"changed negative sets={len(changed)}, initial-group fallback={len(fallback)}")
        return by_id
