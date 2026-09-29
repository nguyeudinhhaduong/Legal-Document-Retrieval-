"""Tìm trọng số fusion trên development và xuất 3 submission.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import json
import zipfile

from legalir.experiment.constants import ADAPTERS
from legalir.experiment.utils import adapter_identity, identity, read
from legalir.rerank.fusion import SIGNALS, rank_features, rankings
from legalir.rerank.weight_search import select_three


class ExportMixin:
    def export(self, pool):
        a = self.api
        rows = self.scored_pool(self.validation, self.members(self.validation), pool, self.root / ADAPTERS[4])
        features, ids = rank_features(rows)
        dev_f = features[self.dev_indices]
        dev_ids = [ids[i] for i in self.dev_indices]
        search_path = self.root / "weight_search.json"
        score_identity = identity(rows)
        if search_path.exists() and read(search_path).get("score_identity") == score_identity:
            saved = read(search_path)
            weights, search_report = saved["weights"], saved["search"]
        else:
            weights, search_report = select_three(a, dev_f, dev_ids, self.dev)
            a.atomic_json(search_path, {"score_identity": score_identity, "weights": weights, "search": search_report})
        report = {"pool": pool, "weights": {name: dict(zip(SIGNALS, w)) for name, w in weights.items()},
                  "search": search_report, "validation": {}, "ablations": {}, "run": self.signature}
        report["r2_equals_r1"] = adapter_identity(self.root / ADAPTERS[4]) == adapter_identity(self.root / ADAPTERS[3])
        report["fixed_without_r2"] = self.subgroup_metrics(rows, self.validation, (0., .4, 0., .6, 0.))
        report["selected_checkpoints"] = {stage: read(self.root / f"{stage}_complete.json").get("selected")
            for stage in ("c", "r2") if (self.root / f"{stage}_complete.json").exists()}
        for signal in SIGNALS:
            w = tuple(float(s == signal) for s in SIGNALS)
            report["ablations"][signal] = self.subgroup_metrics(rows, self.validation, w)
        for name, w in weights.items():
            pred = rankings(features, ids, w)
            entry = {"full": self.subgroup_metrics(rows, self.validation, w)}
            for label, ix in (("development", self.dev_indices), ("locked_report_only", self.locked_indices)):
                entry[label] = self.subgroup_metrics([rows[i] for i in ix], [self.validation[i] for i in ix], w)
            report["validation"][name] = entry
            a.atomic_json(self.root / f"eval_submission_{name}.json", {
                str(r["qid"]): {"answer": ranking[:5]} for r, ranking in zip(self.validation, pred)})
            a.write_jsonl(self.root / f"eval_wrong_{name}.jsonl", [
                {"qid": r["qid"], "question": r["question"], "gold": r["answers"], "prediction": ranking[:5],
                 "missing_pool": sorted(set(r["answers"]) - set(ranking)),
                 "gold_ranks": {cid: (ranking.index(cid) + 1 if cid in ranking else None) for cid in r["answers"]}}
                for r, ranking in zip(self.validation, pred) if set(r["answers"]) - set(ranking[:5])])
            a.log(f"{name}: weights={report['weights'][name]}; full={entry['full']['all']}; locked={entry['locked_report_only']['all']}")
        a.atomic_json(self.root / "submission_experiments.json", report)
        public_rows = self.scored_pool(self.public, self.members(self.public), pool, self.root / ADAPTERS[4])
        pf, pids = rank_features(public_rows)
        for name, w in weights.items():
            pred = rankings(pf, pids, w)
            submission = {str(r["qid"]): {"answer": ranking[:5]} for r, ranking in zip(self.public, pred)}
            if len(submission) != len(self.public) or any(len(v["answer"]) != 5 or
                    len(set(v["answer"])) != 5 or set(v["answer"]) - set(self.doc_chunks) for v in submission.values()):
                raise ValueError("Invalid submission coverage, IDs or top-five uniqueness")
            a.atomic_json(self.root / f"submission_{name}.json", submission)
            with zipfile.ZipFile(self.root / f"submission_{name}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("submission.json", json.dumps(submission, ensure_ascii=False))
            a.write_jsonl(self.root / f"public_top10_{name}.jsonl", [
                {"qid": record["qid"], "question": record["question"], "submission_answer": ranking[:5],
                 "weights": dict(zip(SIGNALS, w)),
                 "top10": [dict(next(row for row in group if row["cid"] == cid),
                                  evidence=[self.texts[i] for i in next(row for row in group if row["cid"] == cid)["chunks"]])
                           for cid in ranking[:10]]}
                for record, group, ranking in zip(self.public, public_rows, pred)])
            a.log(f"Saved {self.root / ('submission_' + name + '.zip')}")
