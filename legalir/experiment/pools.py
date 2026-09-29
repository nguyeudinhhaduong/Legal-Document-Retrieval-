"""Truy hồi dense A/B/C, dựng pool và chấm R1/R2 cho Experiment.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from legalir.experiment.constants import ADAPTERS, INDEXES
from legalir.experiment.utils import adapter_identity, identity, read
from legalir.rerank.fusion import FIXED, SIGNALS, rank_features, rankings


class PoolMixin:
    def retrieve(self, records, adapter, index_path):
        # A query's vector/ranking does not depend on gold labels or which split
        # requests it. Reuse train across C/R2 and dev when exporting full val.
        key = identity({"run": self.signature, "adapter": adapter_identity(adapter),
                        "index": self.file_digest(index_path), "kind": "per_query_retrieval_v1"})
        cache_path = self.cache / f"retrieval_queries_{key[:24]}.json"
        saved = read(cache_path) if cache_path.exists() else {"key": key, "queries": {}}
        if saved.get("key") != key:
            raise ValueError(f"Retrieval cache identity mismatch: {cache_path}")
        entries = saved["queries"]
        query_keys = [identity({"qid": str(r["qid"]), "question": r["question"]}) for r in records]
        missing = {}
        for record, qkey in zip(records, query_keys):
            if qkey not in entries:
                missing[qkey] = record
        self.api.log(f"Dense {Path(adapter).name}: cached={len(records) - sum(k not in entries for k in query_keys)}, new_queries={len(missing)}; source_index={index_path}")
        if missing:
            index = self.api.faiss.read_index(str(index_path))
            if index.ntotal != len(self.cids) or index.d != self.cfg.embedding_dim:
                raise ValueError(f"FAISS vector/dimension mismatch: {index_path}")
            encoder = self.api.DenseEncoder(self.cfg, adapter=Path(adapter))
            try:
                retrieved = self.api.dense_candidates(self.cfg, encoder, index,
                    [r["question"] for r in missing.values()], self.cids, limit=self.cfg.candidate_docs * 3)
            finally:
                encoder.close()
            if len(retrieved) != len(missing):
                raise ValueError("Dense retrieval did not cover all requested queries")
            entries.update(zip(missing, retrieved))
            self.api.atomic_json(cache_path, saved)
        # Downstream mining/scoring may annotate rows; never mutate saved votes.
        return copy.deepcopy([entries[k] for k in query_keys])

    def build_index(self, adapter, name):
        path = self.root / name
        receipt = path.with_suffix(path.suffix + ".cr2.json")
        key = identity({"run": self.signature, "adapter": adapter_identity(adapter)})
        if path.exists() and receipt.exists() and read(receipt).get("key") == key:
            if read(receipt).get("sha256") == self.file_digest(path):
                return path
        unchanged = next((i for i in range(3) if adapter_identity(adapter) == self.frozen[ADAPTERS[i]]), None)
        if unchanged is not None:
            # Fallback B/old C already has a verified matching corpus index.
            shutil.copy2(self.source / INDEXES[unchanged], path)
        else:
            encoder = self.api.DenseEncoder(self.cfg, adapter=Path(adapter))
            index = self.api.faiss.IndexFlatIP(self.cfg.embedding_dim)
            try:
                for start in self.api.progress(range(0, len(self.texts), self.cfg.embed_batch_size), desc=name):
                    values = encoder.tensor(self.texts[start:start + self.cfg.embed_batch_size], False).cpu().numpy()
                    values = values[:, :self.cfg.embedding_dim].astype("float32")
                    self.api.faiss.normalize_L2(values); index.add(values)
                temp = path.with_suffix(".tmp")
                self.api.faiss.write_index(index, str(temp)); temp.replace(path)
            finally:
                encoder.close()
        self.api.atomic_json(receipt, {"key": key, "sha256": self.file_digest(path)})
        weights = next(p for p in Path(adapter).iterdir() if p.name in {"adapter_model.safetensors", "adapter_model.bin"})
        chunk_meta = read(self.source / "prepared" / "chunk_meta.json")
        self.api.atomic_json(path.with_suffix(path.suffix + ".meta.json"), {
            "schema": 1, "base_model": self.cfg.embedding_model, "stage": 3,
            "adapter_identity": [weights.stat().st_size, weights.stat().st_mtime_ns],
            "embedding_dim": self.cfg.embedding_dim, "embedding_max_length": self.cfg.embedding_max_length,
            "pooling": "cls", "query_instruction": False,
            "chunking": {k: chunk_meta[k] for k in ("schema", "chunks", "config")}})
        return path

    def pool(self, members, weights, quota):
        return self.api.fuse_dense_ensemble_members(members, weights, self.cfg.candidate_docs,
                                                   self.cfg.chunks_per_doc, quota)

    def pool_metrics(self, rows, records):
        ranks = [[r["cid"] for r in group] for group in rows]
        return self.api.task_metrics(ranks, records, self.cfg.candidate_docs), self.api.task_metrics(ranks, records, 5)

    def choose_pool(self, members, records):
        choices = []
        for b in (0.4, 0.6, 0.8, 1.0):
            for c in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
                for quota in (0, 3):
                    weights = ((1 - b) * (1 - c), b * (1 - c), c)
                    rows = self.pool(members, weights, quota)
                    mp, m5 = self.pool_metrics(rows, records)
                    choices.append((mp[f"Recall@{self.cfg.candidate_docs}"], m5["Recall@5"], weights, quota))
        winner = max(choices, key=lambda x: x[:2])
        return {"weights": winner[2], "quota": winner[3], "pool_recall": winner[0], "dense_recall5": winner[1]}

    def evaluate_c(self, members, pool, ab_rows):
        rows = self.scored_pool(self.dev, members, pool)
        anchor = self.subgroup_metrics([[{**r, "r2": r["r1"]} for r in group] for group in rows], self.dev)
        recovered, lost, details = 0, 0, []
        for record, before, after in zip(self.dev, ab_rows, rows):
            gold = set(map(str, record["answers"]))
            before_gold = gold & {str(r["cid"]) for r in before}
            after_gold = gold & {str(r["cid"]) for r in after}
            gains, losses = sorted(after_gold - before_gold), sorted(before_gold - after_gold)
            recovered += len(gains); lost += len(losses)
            if gains or losses:
                details.append({"qid": record["qid"], "recovered_gold": gains, "lost_gold": losses})
        standalone = {"pool_depth": self.cfg.candidate_docs, "b": {}, "c": {}, "delta_c_minus_b": {}}
        for k in sorted({5, 20, self.cfg.candidate_docs}):
            for label, rankings_ in (("b", members[1]), ("c", members[2])):
                standalone[label].update(self.api.task_metrics([[str(r["cid"]) for r in group] for group in rankings_], self.dev, k))
            for metric in (f"Recall@{k}", f"AllGold@{k}", f"MRR@{k}"):
                standalone["delta_c_minus_b"][metric] = standalone["c"][metric] - standalone["b"][metric]
        return {"anchor": anchor, "standalone": standalone, "complement": {"recovered_gold_vs_ab": recovered,
                "lost_gold_vs_ab": lost, "queries": details, "split": "development"}}

    def dense_pool(self, members, pool_config):
        rows = self.pool(members, pool_config["weights"], pool_config["quota"])
        for q, group in enumerate(rows):
            for s, member in zip(SIGNALS[:3], members):
                by_cid = {r["cid"]: r["dense"] for r in member[q]}
                floor = min(by_cid.values(), default=0.0) - 0.001
                for row in group:
                    row[f"{s}_present"] = row["cid"] in by_cid
                    row[s] = by_cid.get(row["cid"], floor)
        return rows

    def scored_pool(self, records, members, pool_config, r2=None):
        rows = self.dense_pool(members, pool_config)
        self.validate_pools(records, rows)
        rows = self.score(records, rows, self.root / ADAPTERS[3], "r1")
        return self.score(records, rows, r2, "r2") if r2 else rows

    def validate_pools(self, records, rows):
        if len(records) != len(rows):
            raise ValueError("Candidate pools do not cover all queries")
        for record, group in zip(records, rows):
            cids = [str(row["cid"]) for row in group]
            if len(set(cids)) < 5 or len(set(cids)) != len(cids):
                raise ValueError(f"Invalid candidate pool for qid={record['qid']}: "
                                 f"rows={len(cids)}, unique_documents={len(set(cids))}; "
                                 "need at least five unique documents before reranking")

    def score(self, records, rows, adapter, signal):
        if (signal == "r2" and adapter_identity(adapter) == adapter_identity(self.root / ADAPTERS[3])
                and all("r1" in row for group in rows for row in group)):
            self.api.log("R2 equals R1 checkpoint; reusing R1 scores")
            return [[{**row, "r2": row["r1"]} for row in group] for group in rows]
        recipe = {"queries": records, "rows": rows, "adapter": adapter_identity(adapter), "signal": signal}
        def compute():
            scored = copy.deepcopy(rows)
            model = self.api.BGEReranker(self.cfg, training=False, adapter=Path(adapter))
            try:
                for record, group in self.api.progress(list(zip(records, scored)), desc=f"Score {signal}"):
                    scores, _, _ = self.api.teacher_candidate_scores(model, record["question"], group, self.texts)
                    for row, value in zip(group, scores):
                        row[signal] = value
            finally:
                model.close()
            return scored
        return self.cached("scores", recipe, compute)

    def members(self, records):
        return [self.retrieve(records, self.root / ADAPTERS[i],
                              (self.source if i < 2 else self.root) / INDEXES[i]) for i in range(3)]

    def subgroup_metrics(self, rows, records, weights=FIXED):
        f, ids = rank_features(rows)
        predicted = rankings(f, ids, weights, 5)
        result = {"all": self.api.task_metrics(predicted, records, 5)}
        for label, predicate in (("single", lambda n: n == 1), ("multi", lambda n: n > 1)):
            ix = [i for i, r in enumerate(records) if predicate(len(set(r["answers"])))]
            result[label] = {"queries": len(ix), **self.api.task_metrics([predicted[i] for i in ix], [records[i] for i in ix], 5)}
        return result
