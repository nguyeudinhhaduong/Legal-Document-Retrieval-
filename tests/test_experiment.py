"""CPU regression tests. No pretrained models, downloads, or CUDA required."""
import copy
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ["LEGALIR_NO_AUTO_INSTALL"] = "1"
import numpy as np
import torch
import legalir.api as api
import legalir.cli as cli_mod
import legalir.experiment.base as base_mod
import legalir.experiment.export as export_mod
import legalir.experiment.pools as pools_mod
cr = api


class BRepairChangesTests(unittest.TestCase):
    def test_defaults_and_decay_control(self):
        cfg = api.parse_args([])
        self.assertEqual((cfg.cr2_c_negatives, cfg.cr2_c_evidence, cfg.cr2_c_positive_shortlist), (24, 4, 32))
        self.assertEqual((cfg.cr2_negatives, cfg.cr2_r2_steps), (16, 2000))
        self.assertTrue(cfg.cr2_r2_lr_decay)
        self.assertFalse(api.parse_args(["--no-cr2-r2-lr-decay"]).cr2_r2_lr_decay)
        self.assertAlmostEqual(api.c_step_learning_rate(125, 1250, 2e-6), 2e-6)
        self.assertAlmostEqual(api.c_step_learning_rate(1250, 1250, 2e-6), 2e-7)

    def test_single_gold_query_emphasis_changes_gradient(self):
        losses, gradients = [], []
        for weight in (1., 2.):
            query = torch.tensor([[.2, .5]], requires_grad=True)
            evidence = torch.tensor([[.3, .1], [.1, .4]])
            rows = [{"gold_cids": ["g"], "document_cids": ["g", "n"],
                     "document_chunks": [[0], [1]], "query_weight": weight}]
            loss = api.dense_document_loss(torch, query, evidence, rows, .02)
            loss.backward(); losses.append(loss.detach()); gradients.append(query.grad)
        torch.testing.assert_close(losses[1], 2 * losses[0])
        torch.testing.assert_close(gradients[1], 2 * gradients[0])

    def test_evidence_preserves_teacher_and_dense_with_four_unique_chunks(self):
        selected = api.c_evidence_selection([0, 1, 2, 3, 4], [0., 1., 2., 3., 4.], [0, 1], 4)
        self.assertEqual(selected, [4, 0, 3, 2])
        self.assertEqual(api.c_evidence_selection([0], [1.], [0], 4), [0])

    def test_mining_keeps_repair_with_17_clean_negatives_and_four_positive_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = object.__new__(api.Experiment)
            exp.cfg = api.Config()
            exp.root, exp.api = Path(temp), api
            exp.frozen = {api.ADAPTERS[3]: {}}
            exp.texts = [f"Document {d}\nDistinct document {d} evidence {i}" for d in range(18) for i in range(6)]
            exp.doc_chunks = {str(d): list(range(d * 6, d * 6 + 6)) for d in range(18)}
            exp.cached = lambda label, recipe, make: make()
            class Teacher:
                def __init__(self, *args, **kwargs): self.cfg = exp.cfg
                def scores(self, queries, texts): return [float(exp.texts.index(t)) for t in texts]
                def close(self): pass
            candidates = [{"cid": str(d), "dense": float(18-d), "chunks": exp.doc_chunks[str(d)][:4]} for d in range(18)]
            with patch.object(api, "BGEReranker", Teacher), patch.object(api, "progress", side_effect=lambda x, **kw: x):
                groups = exp.mine([{"qid": "q", "question": "evidence", "answers": ["0"]}], [candidates], "c")
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(groups[0]["negative_cids"]), 17)
            self.assertNotIn("0", groups[0]["negative_cids"])
            self.assertEqual(groups[0]["positive_chunks"], [[5, 0, 4, 3]])
            audit = api.read(exp.root / "c_mining_audit.json")[0]
            self.assertEqual((audit["requested_negatives"], audit["selected_negatives"], audit["accepted"]), (24, 17, True))


class LogicTests(unittest.TestCase):
    def test_c_all_gold_max_evidence_and_cross_query_masking(self):
        q = torch.ones((2, 1), requires_grad=True)
        evidence = torch.tensor([[0.], [2.], [-1.], [3.], [1.], [5.], [0.]], requires_grad=True)
        rows = [
            {"gold_cids": ["a", "b"], "document_cids": ["a", "b", "c"],
             "document_chunks": [[0, 1], [2], [3, 4]]},
            {"gold_cids": ["b"], "document_cids": ["b", "d"], "document_chunks": [[5], [6]]}]
        loss = cr.dense_document_loss(torch, q, evidence, rows, 1.)
        # q1 must not compete against q2's high-scoring b, another known gold.
        first = (torch.nn.functional.softplus(torch.tensor(1.)) + torch.nn.functional.softplus(torch.tensor(4.))) / 2
        second = torch.logsumexp(torch.tensor([0., -5., -3.]), 0)
        self.assertAlmostEqual(loss.item(), ((first + second) / 2).item(), places=6)
        loss.backward()
        self.assertLess(evidence.grad[1].item(), 0)
        self.assertLess(evidence.grad[2].item(), 0)
        self.assertGreater(evidence.grad[3].item(), 0)
        self.assertEqual(evidence.grad[0].item(), 0)  # max selects the other chunk
        self.assertTrue(torch.isfinite(q.grad).all())
        reverse = cr.dense_document_loss(torch, q.flip(0), torch.cat([evidence[5:], evidence[:5]]), rows[::-1], 1.)
        self.assertAlmostEqual(loss.item(), reverse.item(), places=6)

    def test_c_gold_weights_and_invalid_layout(self):
        row = {"gold_cids": ["a", "b"], "document_cids": ["a", "b", "c"],
               "document_chunks": [[0], [1], [2]], "gold_weights": [1., 2.]}
        q = torch.ones((1, 1))
        e = torch.tensor([[2.], [-1.], [3.]])
        expected = (torch.nn.functional.softplus(torch.tensor(1.)) + 2 * torch.nn.functional.softplus(torch.tensor(4.))) / 3
        self.assertAlmostEqual(cr.dense_document_loss(torch, q, e, [row], 1.).item(), expected.item(), places=6)
        with self.assertRaisesRegex(ValueError, "every gold"):
            cr.dense_document_loss(torch, q, e, [{**row, "gold_cids": ["a", "a"]}], 1.)
        with self.assertRaisesRegex(ValueError, "align"):
            cr.dense_document_loss(torch, q, e[:2], [row], 1.)

    def test_c_selector_prefers_downstream_gain_and_guards_pool(self):
        def metric(recall, pool):
            return {"pool_recall": pool, "dense_recall5": .85, "anchor": {
                "all": {"Recall@5": recall, "MRR@5": .8},
                "single": {"Recall@5": .96}, "multi": {"Recall@5": .90, "AllGold@5": .8}}}
        reference = metric(.94, .985)
        high_pool_bad_top5 = metric(.93, .995)
        better_top5 = metric(.945, .986)
        self.assertGreater(cr.c_checkpoint_key(better_top5), cr.c_checkpoint_key(high_pool_bad_top5))
        self.assertTrue(cr.c_checkpoint_safe(better_top5, reference))
        self.assertFalse(cr.c_checkpoint_safe(metric(.95, .97), reference))
        broken_multi = copy.deepcopy(better_top5)
        broken_multi["anchor"]["multi"]["Recall@5"] = .85
        self.assertFalse(cr.c_checkpoint_safe(broken_multi, reference))

    def test_c_packing_shared_gold_and_learning_rate(self):
        import random
        rows = [{"qid": str(i), "gold_cids": ["same"]} for i in range(4)] * 2
        batches = cr.source_unique_batches(rows, 4, random.Random(0), allow_shared_gold=True)
        self.assertEqual([len(b) for b in batches], [4, 4])
        self.assertTrue(all(len({r["qid"] for r in b}) == len(b) for b in batches))
        rates = [cr.c_step_learning_rate(i, 100, 1.) for i in range(1, 101)]
        self.assertAlmostEqual(rates[0], .1)
        self.assertEqual(rates[9], 1.)
        self.assertLess(rates[50], rates[10])
        self.assertAlmostEqual(rates[-1], .1)

    def test_dense_expands_chunk_dominated_query_without_reencoding(self):
        # The first six chunk hits all belong to a single long document.
        cfg = api.Config(embedding_dim=2, dense_chunk_k=6, candidate_docs=5, chunks_per_doc=2)
        cids = ["long"] * 12 + [str(i) for i in range(6)]
        index = api.faiss.IndexFlatIP(2)
        index.add(np.asarray([[1. - i * .01, 0.] for i in range(len(cids))], dtype="float32"))
        from unittest.mock import Mock
        encoder = Mock()
        encoder.encode.return_value = np.asarray([[1., 0.]], dtype="float32")
        with patch.object(api, "log"):
            result = api.dense_candidates(cfg, encoder, index, ["question"], cids)
        self.assertEqual(len(result[0]), 5)
        self.assertEqual([r["cid"] for r in result[0]], ["long", "0", "1", "2", "3"])
        self.assertEqual(result[0][0]["chunks"], [0, 1])
        encoder.encode.assert_called_once()

    def test_pool_validation_identifies_query_before_scoring(self):
        exp = object.__new__(cr.Experiment)
        with self.assertRaisesRegex(ValueError, "qid=bad.*unique_documents=1"):
            exp.validate_pools([{"qid": "bad"}], [[{"cid": "x"}] * 5])
        with self.assertRaisesRegex(ValueError, "cover all queries"):
            exp.validate_pools([{"qid": "missing"}], [])

    def test_identical_r2_reuses_r1_scores_without_model(self):
        exp = object.__new__(cr.Experiment)
        exp.root, exp.api = Path("unused"), api
        rows = [[{"cid": "a", "r1": 2.5}]]
        with patch.object(pools_mod, "adapter_identity", return_value="same"), patch.object(api, "log"), patch.object(api, "BGEReranker", side_effect=AssertionError("must not load R2")):
            result = exp.score([{"qid": "q"}], rows, exp.root / cr.ADAPTERS[4], "r2")
        self.assertEqual(result[0][0]["r2"], 2.5)
        self.assertNotIn("r2", rows[0][0])

    def test_multi_gold_gradients_and_permutation(self):
        x = torch.tensor([2., -1., 3., .5, 0., -2., 1.], requires_grad=True)
        loss = cr.multi_gold_loss(torch, x, 2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((x.grad[:2] < 0).all())  # both golds improve simultaneously
        self.assertTrue((x.grad[2:] > 0).all())
        self.assertGreater(abs(x.grad[1].item()), abs(x.grad[0].item()))
        perm = torch.tensor([-1., 2., 1., -2., 0., .5, 3.])
        self.assertAlmostEqual(loss.item(), cr.multi_gold_loss(torch, perm, 2).item(), places=6)

    def test_dense_masks_secondary_gold(self):
        q = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        p = torch.tensor([[1., 0.], [1., 1.]], requires_grad=True)
        n = torch.tensor([[[-1., 0.]], [[0., -1.]]], requires_grad=True)
        rows = [{"gold_cids": ["a", "b"], "positive_cid": "a", "negative_cids": ["c"]},
                {"gold_cids": ["b"], "positive_cid": "b", "negative_cids": ["c"]}]
        loss = cr.dense_gold_loss(torch, q, p, n, rows, 1.)
        expected = (torch.nn.functional.cross_entropy(torch.tensor([[1., -1.]]), torch.tensor([0]))
                    + torch.nn.functional.cross_entropy(torch.tensor([[0., 1., -1.]]), torch.tensor([1]))) / 2
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_sampling_covers_queries_without_tiny_stratum_domination(self):
        groups = [{"qid": str(i), "answers": ["a", "b"] if i < 2 else ["a"], "repair": i < 2} for i in range(30)]
        visits = {}
        for epoch in range(6):
            ix = cr.balanced_schedule(groups, 12, visits, epoch)
            self.assertLessEqual(max(ix.count(i) for i in ix), 2)
            self.assertLessEqual(sum(i < 2 for i in ix), 4)
        self.assertEqual(set(visits), {str(i) for i in range(30)})

    def test_no_validation_text_leak_merge_gold(self):
        train = [{"qid": "1", "question": " Câu trùng ", "answers": ["a"]},
                 {"qid": "2", "question": "CÂU TRÙNG", "answers": ["b"]},
                 {"qid": "3", "question": "câu val", "answers": ["a"]}]
        out, report = cr.clean_records(api, train, [{"qid": "val", "question": " CÂU VAL "}], {"a", "b"})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["answers"], ["a", "b"])
        self.assertEqual(report["validation_overlap_removed"], ["3"])

    def test_full_grid_100000_and_fixed_scale(self):
        rng = np.random.default_rng(4)
        rows = [[{"cid": str(i), **dict(zip(cr.SIGNALS, rng.normal(size=5)))} for i in range(7)] for _ in range(3)]
        records = [{"answers": ["1", "4"]}, {"answers": ["2"]}, {"answers": ["3", "6"]}]
        f, ids = cr.rank_features(rows)
        with patch.object(api, "log"):
            weights, report = cr.search_weights(api, f, ids, records,
                itertools.product(range(1, 11), repeat=5), "test", batch_size=256)
        self.assertEqual(report["tried"], 100000)
        self.assertTrue(all(1 <= w <= 10 for w in weights))
        pred = cr.rankings(f, ids, weights)
        self.assertEqual(pred, cr.rankings(f, ids, np.asarray(weights) / 10))
        self.assertEqual(cr.rankings(f, ids, cr.FIXED), cr.rankings(f, ids, np.asarray(cr.FIXED) * 7))
        self.assertEqual(tuple(np.round(cr.metric_key(api, pred, records), 12)), report["development_key"])
        small = list(itertools.product((.1, .5, 1.), repeat=5))
        with patch.object(api, "log"):
            best, _ = cr.search_weights(api, f, ids, records, small, "brute_check")
        brute = max(small, key=lambda w: cr.metric_key(api, cr.rankings(f, ids, w), records))
        self.assertEqual(best, brute)

    def test_ties_and_padding_do_not_insert_missing_documents(self):
        rows = [[{"cid": str(i), **dict.fromkeys(cr.SIGNALS, 1.)} for i in range(n)] for n in (5, 8)]
        f, ids = cr.rank_features(rows)
        self.assertEqual(cr.rankings(f, ids, cr.FIXED)[0], ["0", "1", "2", "3", "4"])

    def test_dense_rrf_does_not_vote_for_unretrieved_documents(self):
        rows = [[{"cid": str(i), **dict.fromkeys(cr.SIGNALS, 0.),
                  "a_present": i == 4} for i in range(5)]]
        f, ids = cr.rank_features(rows)
        self.assertEqual(cr.rankings(f, ids, (1., 0., 0., 0., 0.))[0][0], "4")
        self.assertTrue((f[0, :4, 0] == 0).all())
        with self.assertRaisesRegex(ValueError, "positive sum"):
            cr.rankings(f, ids, (0., 0., 0., 0., 0.))

    def test_default_stage_is_loadmodel_and_c_r2_alias_routes_to_experiment(self):
        cfg = api.parse_args([])
        self.assertEqual(cfg.stage, "loadmodel")
        self.assertEqual(api.parse_args(["--stage", "c-r2"]).stage, "retrain-c-r2")
        self.assertTrue(cfg.cr2_remine_c)
        self.assertFalse(api.parse_args(["--no-cr2-remine-c"]).cr2_remine_c)
        self.assertTrue(api.parse_args(["--cr2-remine-c"]).cr2_remine_c)
        self.assertEqual(cr.FIXED, (0.0, 0.4, 0.0, 0.4, 0.2))
        with patch.object(cli_mod, "run") as run, patch.object(cli_mod, "train_embedder", side_effect=AssertionError),                 patch.object(cli_mod, "restore_previous_artifacts", side_effect=AssertionError):
            api.main(["--stage", "c-r2"])
            self.assertEqual(run.call_count, 1)


def make_source(root):
    src = root / "source"
    (src / "prepared").mkdir(parents=True)
    for name in cr.ADAPTERS:
        p = src / name; p.mkdir()
        api.atomic_json(p / "adapter_config.json", {"base_model_name_or_path": "BAAI/bge-m3", "r": 4})
        (p / "adapter_model.safetensors").write_bytes(b"fake_adapter_for_cpu_test")
    api.atomic_json(src / cr.ADAPTERS[3] / "pipeline_config.json", {"max_length": 128})
    chunk_meta = {"schema": 10, "chunks": 20, "config": {"chunk_chars": 1400, "max_chunks_per_doc": 0},
                  "split": {"seed": 2026, "eval_size": 4}}
    api.atomic_json(src / "prepared" / "chunk_meta.json", chunk_meta)
    chunks = [{"chunk_id": i, "cid": str(i), "text": f"Văn bản: {i}\nEvidence distinct {i}", "part": 0} for i in range(20)]
    api.write_jsonl(src / "prepared" / "chunks.jsonl", chunks)
    api.write_jsonl(src / "prepared" / "documents.jsonl", [{"cid": str(i)} for i in range(20)])
    for name, count in (("train", 8), ("validation_1000", 4), ("public_test", 2)):
        api.write_jsonl(src / "prepared" / f"{name}.jsonl", [{"qid": f"{name}-{i}", "question": f"{name} question {i}", "answers": ["1", "2"] if i % 2 else ["1"]} for i in range(count)])
    for i, name in enumerate(cr.INDEXES):
        index = api.faiss.IndexFlatIP(2)
        index.add(np.random.default_rng(i).normal(size=(20, 2)).astype("float32"))
        api.faiss.write_index(index, str(src / name))
        side = {"chunking": {k: chunk_meta[k] for k in ("schema", "chunks", "config")},
                "pooling": "cls", "query_instruction": False, "embedding_dim": 2, "embedding_max_length": 16,
                "base_model": "BAAI/bge-m3", "adapter_identity": [24, 123]}
        side["adapter_identity"][0] = (src / cr.ADAPTERS[i] / "adapter_model.safetensors").stat().st_size
        api.atomic_json(src / (name + ".meta.json"), side)
    return src


class ArtifactTests(unittest.TestCase):
    def test_c_mining_keeps_dense_and_teacher_negative_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp); src = make_source(root)
            cfg = api.Config(cr2_source=str(src), work_dir=str(root / "out"), calibration_size=2,
                             cr2_negatives=3, cr2_c_negatives=3, cr2_c_evidence=2)
            exp = cr.Experiment(api, cfg); exp.prepare()
            exp.texts.append("Văn bản: 3\nAdditional distinct teacher evidence")
            exp.doc_chunks["3"].append(20)
            candidates = [{"cid": str(i), "dense": 1. - i / 100,
                           "chunks": [3, 20] if i == 3 else [i]} for i in [3, 0, 1, 2, 4, 5, 6, 7]]
            with patch.object(api, "BGEReranker") as teacher, patch.object(api, "teacher_candidate_scores",
                    return_value=([0.] * 8, [20, 0, 1, 2, 4, 5, 6, 7], None)):
                teacher.return_value.scores.side_effect = lambda questions, texts: [1.] * len(texts)
                groups = exp.mine([{"qid": "train", "question": "test", "answers": ["1"]}], [candidates], "c")
            self.assertIn("3", groups[0]["negative_cids"])
            teacher_negative = groups[0]["negative_chunks"][groups[0]["negative_cids"].index("3")]
            self.assertEqual(teacher_negative, [3, 20])
            self.assertNotIn("1", groups[0]["negative_cids"])

    def test_retrieval_reuses_subset_queries_without_loading_models(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp); src = make_source(root)
            exp = cr.Experiment(api, api.Config(cr2_source=str(src), work_dir=str(root / "out"), calibration_size=2))
            exp.prepare()
            seen = []
            def retrieve(cfg, encoder, index, questions, cids, limit):
                seen.extend(questions)
                return [[{"cid": str(i), "chunks": [i], "dense": 1. - i / 30} for i in range(20)] for q in questions]
            adapter, index = exp.root / cr.ADAPTERS[0], src / cr.INDEXES[0]
            with patch.object(api, "DenseEncoder") as model, patch.object(api, "dense_candidates", side_effect=retrieve):
                exp.retrieve(exp.dev, adapter, index)
                exp.retrieve(exp.validation, adapter, index)
                self.assertEqual(len(seen), len(exp.validation))
                self.assertEqual(model.call_count, 2)
                shuffled = copy.deepcopy(exp.validation[::-1])
                shuffled[0]["answers"] = ["19"]  # labels never affect query encoding
                data = exp.retrieve(shuffled, adapter, index)
                data[0][0]["dense"] = -999
                reloaded = exp.retrieve(shuffled, adapter, index)
                self.assertNotEqual(reloaded[0][0]["dense"], -999)
                self.assertEqual(model.call_count, 2)
                shuffled[0]["question"] += " changed"
                exp.retrieve(shuffled, adapter, index)
                self.assertEqual(model.call_count, 3)
                self.assertEqual(len(seen), len(exp.validation) + 1)
            with patch.object(base_mod, "digest", side_effect=AssertionError("Repeated source index hash")):
                exp.file_digest(index); exp.file_digest(index)

    def test_c_and_r2_training_loops_with_tiny_cpu_models(self):
        self._exercise_training(dynamic=True, boundary_weight=0.)

    def test_static_r2_old_boundary_control_training(self):
        self._exercise_training(dynamic=False, boundary_weight=.3)

    def test_static_r2_no_boundary_control_training(self):
        self._exercise_training(dynamic=False, boundary_weight=0.)

    def _exercise_training(self, dynamic, boundary_weight):
        """Exercise real optimizer/loss/mining/checkpoint code with small models."""
        training_sources = []
        class TinyModel(torch.nn.Module):
            def __init__(self, adapter):
                super().__init__()
                self.proj = torch.nn.Linear(2, 2)
                with torch.no_grad():
                    self.proj.weight.copy_(torch.eye(2)); self.proj.bias.zero_()
                path = Path(adapter) / "adapter_model.safetensors"
                if path.stat().st_size > 100:
                    self.load_state_dict(torch.load(path, weights_only=True))
            def save_pretrained(self, path):
                path = Path(path); path.mkdir(parents=True, exist_ok=True)
                api.atomic_json(path / "adapter_config.json", {"base_model_name_or_path": "BAAI/bge-m3", "r": 4})
                torch.save(self.state_dict(), path / "adapter_model.safetensors")
        def inputs(texts):
            return torch.tensor([[int(cr.identity(t)[:4], 16) / 32768. - 1.,
                                  int(cr.identity(t)[4:8], 16) / 32768. - 1.] for t in texts])
        class TinyDense:
            def __init__(self, cfg, training=False, adapter=None, **kwargs):
                self.cfg, self.torch = cfg, torch
                self.adapter = Path(adapter)
                self.model = TinyModel(adapter); self.model.train(training)
                if training:
                    training_sources.append(("c", Path(adapter).name))
            def tensor(self, texts, queries, grad=False):
                if not queries and not grad and self.adapter.name in cr.ADAPTERS[:2]:
                    raise AssertionError("Frozen A/B corpus must not be re-encoded")
                with torch.set_grad_enabled(grad):
                    return torch.nn.functional.normalize(self.model.proj(inputs(texts)), dim=1)
            def encode(self, texts, queries, batch_size, desc):
                return self.tensor(texts, queries).detach().numpy()
            def close(self): pass
        class TinyReranker(TinyDense):
            def __init__(self, cfg, training=False, adapter=None):
                super().__init__(cfg, False, adapter)
                self.model.train(training)
                self.training_adapter = training
                if training:
                    training_sources.append(("r2", Path(adapter).name))
            def logits(self, questions, docs):
                if self.training_adapter:
                    assert self.model.training == torch.is_grad_enabled(), "R2 train/eval mode or gradients not restored"
                return (self.model.proj(inputs(questions)) * inputs(docs)).sum(dim=1)
            def scores(self, questions, docs):
                with torch.no_grad():
                    return self.logits(questions, docs).tolist()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp); src = make_source(root)
            cfg = api.Config(cr2_source=str(src), work_dir=str(root / "out"), calibration_size=2,
                cr2_c_epochs=2, cr2_c_steps=2, cr2_r2_epochs=5, cr2_r2_steps=3,
                cr2_remine_r2=dynamic, cr2_r2_boundary_weight=boundary_weight, cr2_r2_refresh_steps=1,
                cr2_negatives=3, cr2_c_negatives=3, embed_train_batch_size=2, embed_grad_accum=3, rerank_grad_accum=2,
                candidate_docs=20, dense_chunk_k=20, chunks_per_doc=1)
            exp = cr.Experiment(api, cfg); exp.prepare()
            with patch.object(api, "DenseEncoder", TinyDense), patch.object(api, "BGEReranker", TinyReranker), patch.object(api, "progress", side_effect=lambda x, **kw: x), patch.object(api, "log"), patch.object(exp, "mine", wraps=exp.mine) as mining:
                pool = exp.train_c()
                exp.train_r2(pool)
                self.assertIsNotNone(exp.completed("c"))
                self.assertIsNotNone(exp.completed("r2"))
                exp.train_c(); exp.train_r2(pool)  # completed stages do not train again
                self.assertEqual(sum(c.args[2] == "c" for c in mining.call_args_list), 2)
                self.assertEqual(sum(c.args[2] == "r2" for c in mining.call_args_list), 7 if dynamic else 1)
            self.assertEqual(training_sources, [("c", cr.ADAPTERS[1]), ("r2", cr.ADAPTERS[3])])
            exp.verify_frozen()
            history = cr.read(exp.root / "r2_checkpoint_report.json")
            self.assertEqual(len(history), 4)
            self.assertTrue(all(np.isfinite(h["loss"]) for h in history if "loss" in h))
            for row in history[2:]:
                self.assertAlmostEqual(sum(row["loss_components"].values()), row["loss"], places=5)
                self.assertEqual(row["boundary_weight"], boundary_weight)
                if boundary_weight == 0:
                    self.assertEqual(row["loss_components"]["boundary"], 0.)
            self.assertEqual(history[-1]["optimizer_updates"], 4)  # two updates per three-step epoch
            if dynamic:
                first = cr.read(exp.root / "r2_refresh_epoch1_block1.json")
                third = cr.read(exp.root / "r2_refresh_epoch1_block3.json")
                self.assertEqual(first["optimizer_updates_before_scoring"], 0)
                self.assertEqual(third["optimizer_updates_before_scoring"], 1)
            mining_source = cr.read(exp.root / "c_mining_source.json")
            self.assertEqual(mining_source["indexes"], [str(src / name) for name in cr.INDEXES[:2]])
            self.assertTrue(mining_source["remine_with_current_c"])
            self.assertFalse(mining_source["corpus_reencoded_for_A_B"])

    def test_prepare_preserves_source_schema_and_frozen_weights(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp); src = make_source(root)
            cfg = api.Config(cr2_source=str(src), work_dir=str(root / "out"), calibration_size=2)
            exp = cr.Experiment(api, cfg); exp.prepare(); exp.verify_frozen()
            self.assertEqual(exp.cfg.embedding_max_length, 16)
            self.assertEqual(exp.cfg.reranker_max_length, 128)
            self.assertEqual(len(exp.train), 8)
            self.assertEqual(len(exp.dev), 2)
            self.assertEqual(cr.read(src / "prepared" / "chunk_meta.json")["schema"], 10)
            exp2 = cr.Experiment(api, cfg); exp2.prepare()
            notebook_cfg = api.Config(
                cr2_source=str(src), work_dir=str(root / "notebook_out"), calibration_size=2
            )
            with patch.object(base_mod, "__file__", None):
                notebook_exp = cr.Experiment(api, notebook_cfg)
                notebook_exp.prepare()
                notebook_exp.verify_frozen()
            self.assertTrue((notebook_exp.root / "cr2_manifest.json").is_file())
            (exp.root / cr.ADAPTERS[0] / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                exp.verify_frozen()

    def test_reject_changed_source_and_stale_index_recipe(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp); src = make_source(root)
            cfg = api.Config(cr2_source=str(src), work_dir=str(root / "out"), calibration_size=2)
            exp = cr.Experiment(api, cfg); exp.prepare()
            (src / "prepared" / "public_test.jsonl").write_text('{}\n', encoding="utf8")
            with self.assertRaisesRegex(ValueError, "another source"):
                cr.Experiment(api, cfg).prepare()
            side_path = src / (cr.INDEXES[0] + ".meta.json")
            meta = cr.read(side_path); meta["chunking"]["schema"] = 11; api.atomic_json(side_path, meta)
            with self.assertRaisesRegex(ValueError, "does not match"):
                cr.Experiment(api, cfg).prepare()

    def test_export_three_zips_same_pool_exact_weights_and_locked_unused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp); src = make_source(root)
            cfg = api.Config(cr2_source=str(src), work_dir=str(root / "out"), calibration_size=2)
            exp = cr.Experiment(api, cfg); exp.prepare()
            for i in (2, 4):
                cr.copy_adapter(src / cr.ADAPTERS[i], exp.root / cr.ADAPTERS[i])
            rng = np.random.default_rng(43)
            def score(records, members, pool, r2):
                return [[{"cid": str(i), "chunks": [i], **dict(zip(cr.SIGNALS, rng.normal(size=5)))} for i in range(8)] for r in records]
            chosen = {"best_val": (0., .5, 0., .5, 0.), "fixed": cr.FIXED, "positive_grid": (.1, .2, .3, .4, .5)}
            seen = []
            def select(api_, f, ids, records):
                seen.extend(r["qid"] for r in records)
                return chosen, {"test": True}
            with patch.object(exp, "scored_pool", side_effect=score), patch.object(exp, "members", return_value=[]), patch.object(export_mod, "select_three", side_effect=select):
                exp.export({"weights": (.2, .8, 0), "quota": 0})
            self.assertEqual(set(seen), {r["qid"] for r in exp.dev})
            import zipfile
            for name in chosen:
                with zipfile.ZipFile(exp.root / f"submission_{name}.zip") as z:
                    self.assertEqual(z.namelist(), ["submission.json"])
                    result = json.loads(z.read("submission.json"))
                    self.assertEqual(len(result), 2)
                    self.assertTrue(all(len(r["answer"]) == 5 for r in result.values()))
            saved = cr.read(exp.root / "submission_experiments.json")
            self.assertEqual(saved["weights"]["fixed"], dict(zip(cr.SIGNALS, cr.FIXED)))


if __name__ == "__main__":
    unittest.main()
