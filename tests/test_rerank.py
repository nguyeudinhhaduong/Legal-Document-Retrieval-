"""Kiểm thuật toán rerank Weighted RRF (CPU, không cần model)."""
import json
import random
import unittest
from pathlib import Path

from legalir.rerank import fusion
from legalir.rerank.rrf import Candidate, fused_scores, ranks_from_rrf_votes, rerank, signal_ranks

ROOT = Path(__file__).resolve().parents[1]


def random_pool(rng, size=45):
    rows = []
    for i in range(size):
        row = {"cid": str(rng.randrange(10**6)) + f"_{i}"}
        for s in fusion.SIGNALS:
            row[s] = round(rng.gauss(0, 1), 2)  # làm tròn để có điểm hòa
        for s in fusion.SIGNALS[:3]:
            row[f"{s}_present"] = rng.random() > 0.3
        rows.append(row)
    return rows


def to_candidates(rows):
    return [Candidate(cid=r["cid"], scores={s: r[s] for s in fusion.SIGNALS},
                      present={s: r[f"{s}_present"] for s in fusion.SIGNALS[:3]}) for r in rows]


class RerankTests(unittest.TestCase):
    def test_matches_vectorized_fusion_used_in_training(self):
        rng = random.Random(7)
        weights_list = [fusion.FIXED, (0.1, 0.2, 0.3, 0.4, 0.5), (1, 0, 0, 0, 0), (0.3, 0.3, 0.1, 0.7, 0.9)]
        pools = [random_pool(rng) for _ in range(40)]
        features, ids = fusion.rank_features(pools)
        for weights in weights_list:
            expected = fusion.rankings(features, ids, weights, k=10)
            named = dict(zip(fusion.SIGNALS, weights))
            for pool, exp in zip(pools, expected):
                got = [cid for cid, _ in rerank(to_candidates(pool), named, top_k=10)]
                self.assertEqual(got, exp)

    def test_absent_dense_document_gets_no_vote(self):
        cands = [Candidate(str(i), {"a": 1.0 - i / 10, "b": 0, "c": 0, "r1": 0, "r2": 0}) for i in range(5)]
        cands[0].present = {"a": False}
        ranks = signal_ranks(cands, "a")
        self.assertNotIn("0", ranks)
        self.assertEqual(ranks["1"], 1)
        self.assertEqual(fused_scores(cands, {"a": 1})["0"], 0.0)

    def test_ties_break_by_cid(self):
        cands = [Candidate(c, {s: 1.0 for s in fusion.SIGNALS}) for c in ("9", "3", "5", "1", "7")]
        self.assertEqual([c for c, _ in rerank(cands)], ["1", "3", "5", "7", "9"])

    def test_ranks_recovered_from_saved_votes(self):
        self.assertEqual(ranks_from_rrf_votes({"b": 1 / 11, "r1": 1 / 27, "a": 0.0}), {"b": 1, "r1": 17, "a": None})

    def test_example_file_reproduces_submitted_top5(self):
        path = ROOT / "examples" / "sample_candidates.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cands = [Candidate(c["cid"], c["scores"], c["present"], c["ranks"]) for c in row["candidates"]]
            self.assertEqual([cid for cid, _ in rerank(cands)], row["submission_answer"])


if __name__ == "__main__":
    unittest.main()
