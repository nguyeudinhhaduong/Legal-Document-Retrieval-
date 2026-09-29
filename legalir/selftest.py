"""Self-test CPU cho tiền xử lý, metric, cache và submission.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from legalir.core.config import Config, Paths
from legalir.core.utils import atomic_json, log, read_jsonl, write_jsonl
from legalir.models.lora import resolved_lora_alpha, resolved_lora_targets
from legalir.data.text import clean_legal_text
from legalir.data.chunking import chunk_document, document_name_from_context, split_long_block
from legalir.data.loading import exact_validation_split
from legalir.data.chunk_store import load_chunk_cids_only
from legalir.retrieval.dense_fusion import fuse_dense_ensemble_members
from legalir.training.negatives import bm25_candidate_scores, select_stratified_negatives, teacher_hard_negative_rows
from legalir.retrieval.fused_order import fused_ranking, missing_gold_from_top_k
from legalir.models.reranker import reranker_group_documents
from legalir.rerank.cache import load_legacy_b_reranker_cache, write_complete_rerank_cache
from legalir.evaluation.metrics import save_eval_prediction_artifacts, task_metrics, validate_submission
from legalir.training.embedder import dense_contrastive_distillation_loss, embedding_training_recipe, source_unique_batches
from legalir.evaluation.dense_eval import calibration_locked_indices, dense_weight_grid
from legalir.training.reranker_checkpoints import load_reranker_checkpoint_r1_scores, paired_reranker_pool_metrics, reranker_pool_metrics, save_reranker_checkpoint_r1_scores
from legalir.rerank.repair_fusion import REPAIR_SIGNALS, fused_repair_ranking, fused_repair_rankings, repair_signal_scores, repair_weight_grid
from legalir.rerank.meta_ranker import crossfit_listwise_meta_ranker, meta_ranker_rankings
from legalir.rerank.continuous_opt import continuous_optimize_repair_config
from legalir.pipeline.submission import save_public_top10_debug


def run_self_tests(tmp: Path) -> None:
    test_paths = Paths(tmp)
    assert isinstance(test_paths.calibration, Path)
    atomic_json(test_paths.calibration, {"ok": True})
    assert test_paths.calibration.is_file()
    noisy = "Điều 1.\r\n  Quyền  sử dụng đất\r\n\r\nĐiều 1.\nQuyền sử dụng đất"
    cleaned = clean_legal_text(noisy); assert "Điều 1" in cleaned and "Quyền sử dụng đất" in cleaned
    inferred_title = document_name_from_context({
        "name": "",
        "link": (
            "https://thuvienphapluat.vn/tcvn/Nong-nghiep/"
            "TCVN-13268-1-2021-Bao-ve-thuc-vat-919415.aspx"
        ),
    })
    assert inferred_title == "TCVN 13268 1 2021 Bao ve thuc vat"
    assert document_name_from_context({"name": "Luật Đất đai", "link": "bad"}) == "Luật Đất đai"
    document_groups = reranker_group_documents({
        "positive": "p0", "positive_chunks": ["p0", "p1", "p2"],
        "negatives": ["n0", "m0"],
        "negative_chunk_groups": [["n0", "n1", "n2"], ["m0"]],
    }, 2)
    assert document_groups == [["p0", "p1"], ["n0", "n1"], ["m0"]]
    footer = (
        "Điều 6a. Quy định mới\nNội dung pháp lý.\n\nNơi nhận:\n\n- Bộ Tư pháp;\n\n"
        "-\n\nCác đơn vị trực\n\nthuộc, Công báo;\n\nKT. BỘ TRƯỞNG\n\nLưu: VT, PC"
    )
    footer_cleaned = clean_legal_text(footer)
    assert "Điều 6a" in footer_cleaned and "Nội dung pháp lý" in footer_cleaned
    assert "Nơi nhận" not in footer_cleaned and "Công báo" not in footer_cleaned
    assert "BỘ TRƯỞNG" not in footer_cleaned and "Lưu: VT" not in footer_cleaned
    # Chunk boundaries must align with Điều boundaries: even two short,
    # adjacent Điều remain separate; an oversized Điều is split at Khoản
    # boundaries with the Điều header carried into every piece.
    short_articles = "Điều 1. Quyền sử dụng đất theo quy định pháp luật.\n\nĐiều 2. Nghĩa vụ của người sử dụng đất."
    packed = chunk_document("Luật đất đai", short_articles, 500)
    assert len(packed) == 2
    assert "Điều 1" in packed[0] and "Điều 2" not in packed[0]
    assert "Điều 2" in packed[1] and "Điều 1" not in packed[1]
    long_khoan = "\n\n".join(f"{n}. Khoản {n} quy định về nghĩa vụ tài chính của người sử dụng đất và các bên liên quan trong giao dịch." for n in range(1, 10))
    oversized_article = f"Điều 5. Nghĩa vụ tài chính\n\n{long_khoan}"
    split_pieces = chunk_document("Luật đất đai", oversized_article, 300)
    assert len(split_pieces) > 1
    assert all(len(piece) <= 300 + len("Văn bản: Luật đất đai\n") for piece in split_pieces)
    assert all("Điều 5" in piece for piece in split_pieces)
    suffixed = chunk_document("Luật sửa đổi", "Điều 12b. Nội dung được bổ sung.", 300)
    assert len(suffixed) == 1 and "Điều 12b" in suffixed[0]
    hierarchical = chunk_document(
        "Luật phân cấp",
        "Chương I\nQUY ĐỊNH CHUNG\n\nMục 1\nPHẠM VI\n\n"
        "Điều 1. Phạm vi điều chỉnh.\n\nĐiều 2. Đối tượng áp dụng.",
        300,
    )
    assert all(
        "Ngữ cảnh pháp lý:\nChương I: QUY ĐỊNH CHUNG\nMục 1: PHẠM VI" in piece
        for piece in hierarchical if "Điều" in piece
    )
    separated_title = chunk_document(
        "Luật tiêu đề",
        "Chương II\n\nQUYỀN VÀ NGHĨA VỤ\n\nĐiều 3. Nội dung kiểm tra.", 500,
    )
    assert len(separated_title) == 1
    assert "Chương II: QUYỀN VÀ NGHĨA VỤ" in separated_title[0]
    assert separated_title[0].count("QUYỀN VÀ NGHĨA VỤ") == 1
    fragmented_hierarchy = chunk_document(
        "Luật thi hành",
        "Chương 6\nQUY ĐỊNH CHUNG\n\nĐiều 36. Nội dung chương cũ.\n\n"
        "Chương\n\n10:\n\nĐIỀU KHOẢN THI HÀNH\n\n"
        "Điều 37. Nội dung chương mới.",
        500,
    )
    assert len(fragmented_hierarchy) == 2
    assert "Chương 6: QUY ĐỊNH CHUNG" in fragmented_hierarchy[0]
    assert "Chương 10: ĐIỀU KHOẢN THI HÀNH" in fragmented_hierarchy[1]
    assert all(piece.count("Ngữ cảnh pháp lý:") == 1 for piece in fragmented_hierarchy)
    long_table = "Mục 1. " + ("ô dữ liệu " * 800) + ". Kết thúc."
    assert max(map(len, split_long_block(long_table, 300))) <= 300
    tiny_heading = chunk_document(
        "Luật thử nghiệm",
        "Mục 1. TIÊU ĐỀ NGẮN\n\nĐiều 1. " + ("Nội dung quy định đầy đủ. " * 30),
        300,
    )
    assert all(len(piece.split("\n", 1)[-1].strip()) >= 50 for piece in tiny_heading)
    rows = [{"qid": str(i), "answers": ["1"] if i % 5 else ["1", "2"]} for i in range(7000)]
    train, validation = exact_validation_split(rows, 1000, 2026); assert len(train) == 6000 and len(validation) == 1000
    metrics = task_metrics([["1", "3"]], [{"answers": ["1", "2"]}], 5); assert metrics["Recall@5"] == 0.5 and metrics["Precision@5"] == 0.5
    checkpoint_records = [{"qid": "checkpoint", "answers": ["gold"]}]
    checkpoint_candidates = [[
        {"cid": cid, "chunks": [index]}
        for index, cid in enumerate(["n1", "n2", "n3", "n4", "n5", "gold"])
    ]]
    r1_checkpoint_scores = [[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
    r2_checkpoint_scores = [[1.0, 2.0, 3.0, 4.0, 5.0, 10.0]]
    assert reranker_pool_metrics(
        checkpoint_records, checkpoint_candidates, r1_checkpoint_scores
    )["Recall@5"] == 0.0
    pair_metrics, pair_weight = paired_reranker_pool_metrics(
        checkpoint_records, checkpoint_candidates,
        r1_checkpoint_scores, r2_checkpoint_scores,
    )
    assert pair_metrics["Recall@5"] == 1.0 and pair_weight > 0.0
    save_reranker_checkpoint_r1_scores(
        Config(rerank_checkpoint_eval_queries=1), test_paths,
        checkpoint_records, r1_checkpoint_scores,
    )
    assert load_reranker_checkpoint_r1_scores(
        Config(rerank_checkpoint_eval_queries=1), test_paths,
        checkpoint_records, checkpoint_candidates,
    ) == r1_checkpoint_scores
    fusion_rows = [
        {"cid": "dense-first", "dense": 1.0, "rerank_max": 0.0, "rerank_mean": 0.0},
        {"cid": "cross-first", "dense": 0.9, "rerank_max": 4.0, "rerank_mean": 3.0},
        {"cid": "outside", "dense": 0.8, "rerank_max": 9.0, "rerank_mean": 9.0},
    ]
    assert fused_ranking(fusion_rows, 0.0, 2, 1.0, "score") == ["dense-first", "cross-first"]
    assert fused_ranking(fusion_rows, 1.0, 2, 1.0, "score")[0] == "cross-first"
    assert fused_ranking(fusion_rows, 1.0, 2, 0.5, "rrf")[0] == "cross-first"
    advanced_fusion = [
        {"cid": "dense", "dense": 1.0, "bm25": 0.0, "rerank_max": 1.0,
         "rerank_r1_chunks": [1.0, 1.0, 1.0], "rerank_r2_chunks": [1.0, 1.0, 1.0]},
        {"cid": "lexical", "dense": 0.5, "bm25": 8.0, "rerank_max": 0.0,
         "rerank_r1_chunks": [0.0, 0.0, 0.0], "rerank_r2_chunks": [0.0, 0.0, 9.0]},
    ]
    assert fused_ranking(advanced_fusion, 0.0, 2, 1.0, "rrf", 0.0, 0.0, 2)[0] == "dense"
    assert fused_ranking(advanced_fusion, 0.0, 2, 1.0, "rrf", 0.0, 1.0, 2)[0] == "lexical"
    assert fused_ranking(advanced_fusion, 1.0, 2, 1.0, "rrf", 1.0, 0.0, 4)[0] == "lexical"
    member_a = [[{"cid": "a", "dense": 1.0, "chunks": [0]}, {"cid": "b", "dense": 0.9, "chunks": [1]}]]
    member_b = [[{"cid": "b", "dense": 1.0, "chunks": [1]}, {"cid": "a", "dense": 0.9, "chunks": [0]}]]
    member_c = [[{"cid": "c", "dense": 1.0, "chunks": [2]}, {"cid": "a", "dense": 0.9, "chunks": [0]}]]
    two_member = fuse_dense_ensemble_members([member_a, member_b], (0.6, 0.4), 2, 4)
    three_member_zero_c = fuse_dense_ensemble_members([member_a, member_b, member_c], (0.6, 0.4, 0.0), 2, 4)
    assert [row["cid"] for row in two_member[0]] == [row["cid"] for row in three_member_zero_c[0]]
    quota_a = [[*member_a[0], {"cid": "d", "dense": 0.8, "chunks": [3]}]]
    quota_b = [[*member_b[0], {"cid": "d", "dense": 0.8, "chunks": [3]}]]
    quota_union = fuse_dense_ensemble_members(
        [quota_a, quota_b, member_c], (0.6, 0.4, 0.0), 3, 4, member_quota=1,
    )
    assert "c" in {row["cid"] for row in quota_union[0]}
    shared_members = [
        [[{"cid": "shared", "dense": 1.0, "chunks": [base, base + 1, base + 2, base + 3]}]]
        for base in (10, 20, 30)
    ]
    shared_fused = fuse_dense_ensemble_members(
        shared_members, (1 / 3, 1 / 3, 1 / 3), 1, 4,
    )
    assert set(shared_fused[0][0]["chunks"][:3]) == {10, 20, 30}
    b_only_shared = fuse_dense_ensemble_members(
        shared_members, (0.0, 1.0, 0.0), 1, 4,
    )
    assert b_only_shared[0][0]["chunks"] == [20, 21, 22, 23]
    assert len(dense_weight_grid(3)) == 66 and all(math.isclose(sum(weights), 1.0) for weights in dense_weight_grid(3))
    split_rows = [{"qid": f"s{index}"} for index in range(1000)]
    calibration_a, locked_a = calibration_locked_indices(split_rows, 500, 2026)
    calibration_b, locked_b = calibration_locked_indices(split_rows, 500, 2026)
    assert calibration_a == calibration_b and locked_a == locked_b
    assert len(calibration_a) == len(locked_a) == 500 and not set(calibration_a) & set(locked_a)
    bm25_scores, bm25_chunks = bm25_candidate_scores(
        "", [{"cid": "a", "chunks": [0]}, {"cid": "b", "chunks": [1]}],
        ["điều luật đất đai", "thuế doanh nghiệp"], 4,
    )
    assert bm25_scores == [0.0, 0.0] and bm25_chunks == [0, 1]
    batching_rows = [
        {"qid": f"q{index}:{positive}", "source_qid": f"q{index}", "positive": positive}
        for index in range(7) for positive in (("a", "b") if index < 2 else ("a",))
    ]
    batches = source_unique_batches(batching_rows, 4, random.Random(2026))
    assert sum(map(len, batches)) == len(batching_rows)
    assert all(len({row["source_qid"] for row in batch}) == len(batch) for batch in batches)
    assert all(len({row["positive"] for row in batch}) == len(batch) for batch in batches)
    # Flat dense/reranker RRF fusion: pure B vs. pure reranker-weighted
    # pure reranker-weighted must pick opposite winners on a row pair
    # where dense and reranker signals disagree.
    five_signal_rows = [
        {"cid": "1", "dense": 0.9, "dense_b": 0.9, "dense_a": 0.9, "dense_c": 0.9, "rerank_r1_chunks": [0.1], "rerank_r2_chunks": [0.1]},
        {"cid": "2", "dense": 0.1, "dense_b": 0.1, "dense_a": 0.1, "dense_c": 0.1, "rerank_r1_chunks": [0.9], "rerank_r2_chunks": [0.9]},
    ]
    b_leaning = fused_repair_ranking(five_signal_rows, {"weights": {"b": 1.0}, "evidence_k": 1, "rrf_constant": 10.0})
    assert b_leaning[0] == "1"
    r_leaning = fused_repair_ranking(five_signal_rows, {"weights": {"r1": 1.0, "r2": 1.0}, "evidence_k": 1, "rrf_constant": 10.0})
    assert r_leaning[0] == "2"
    # Dense C carries information B/R1/R2 lack: with C weighted in, a
    # document C strongly favors can enter the top-5 even though B/R1/R2
    # rank it low.
    dense_ac_rows = [
        {
            "cid": str(rank), "dense": 1.0 - rank * 0.01, "dense_b": 1.0 - rank * 0.01,
            "dense_a": 1.0 - rank * 0.01,
            "dense_c": 20.0 if rank == 9 else 1.0 - rank * 0.01,
            "rerank_r1_chunks": [1.0], "rerank_r2_chunks": [1.0],
        }
        for rank in range(1, 11)
    ]
    b_only_config = {"weights": {"b": 1.0}, "evidence_k": 1, "rrf_constant": 10.0}
    without_c = fused_repair_ranking(dense_ac_rows, b_only_config)
    assert "9" not in without_c[:5]
    with_c = fused_repair_ranking(dense_ac_rows, {**b_only_config, "weights": {"b": 0.2, "c": 0.8}})
    assert "9" in with_c[:5]
    # repair_weight_grid must enumerate a valid signal simplex (each point
    # sums to one, uses exactly REPAIR_SIGNALS as keys).
    signal_grid = repair_weight_grid(10)
    parts = len(REPAIR_SIGNALS)
    assert len(signal_grid) == math.comb(10 + parts - 1, parts - 1)
    assert all(math.isclose(sum(point.values()), 1.0, abs_tol=1e-9) for point in signal_grid)
    assert set(signal_grid[0]) == set(REPAIR_SIGNALS)
    assert len(repair_weight_grid(20)) == math.comb(20 + parts - 1, parts - 1)
    evidence_rows = [
        {"cid": "spike", "rerank_r1_chunks": [10.0, -10.0]},
        {"cid": "supported", "rerank_r1_chunks": [7.0, 7.0]},
    ]
    assert repair_signal_scores(evidence_rows, "r1", 2) == [10.0, 7.0]
    assert repair_signal_scores(evidence_rows, "r1_top2", 2) == [0.0, 7.0]
    # Nelder-Mead must escape a plateau seed (all weight on b/r1/r2) and find
    # a continuous weight vector no grid step in this file tests (dense_c
    # fractional weight) that flips gold into the top-5.
    nelder_mead_rows = [
        {
            "cid": str(rank), "dense": 1.0 - rank * 0.01, "dense_b": 1.0 - rank * 0.01,
            "dense_a": 1.0 - rank * 0.01,
            "dense_c": 12.3 if rank == 9 else 1.0 - rank * 0.01,
            "rerank_r1_chunks": [1.0 - rank * 0.01], "rerank_r2_chunks": [1.0 - rank * 0.01],
        }
        for rank in range(1, 41)
    ]
    nelder_mead_records = [{"qid": "q0", "answers": ["9"]}]
    nelder_mead_rows_by_query = [nelder_mead_rows]
    nelder_mead_seed = {
        "weights": {"b": 0.4, "a": 0.0, "c": 0.0, "r1": 0.2, "r2": 0.4},
        "evidence_k": 1, "rrf_constant": 10.0,
    }
    seed_recall = task_metrics(
        fused_repair_rankings(nelder_mead_rows_by_query, nelder_mead_seed), nelder_mead_records, 5
    )["Recall@5"]
    assert seed_recall == 0.0

    best_scores: list[float] = []

    def recording_evaluate_config(config: dict[str, Any], stage: str) -> float:
        rankings = fused_repair_rankings(nelder_mead_rows_by_query, config)
        score = task_metrics(rankings, nelder_mead_records, 5)["Recall@5"]
        best_scores.append(score)
        return score

    continuous_optimize_repair_config([nelder_mead_seed], recording_evaluate_config)
    assert max(best_scores) == 1.0
    # The listwise stacker must learn a repeatable rank-6 rescue pattern using
    # out-of-fold predictions, even when BM25 favors a lexical decoy.
    meta_rows: list[list[dict[str, Any]]] = []
    meta_records: list[dict[str, Any]] = []
    for query_index in range(20):
        query_rows = []
        for candidate_index in range(8):
            gold_candidate = candidate_index == 5
            query_rows.append({
                "cid": f"m{query_index}-{candidate_index}",
                "dense": 1.0 - 0.05 * candidate_index,
                "dense_b": 1.0 - 0.05 * candidate_index,
                "dense_a": 1.0 - 0.05 * candidate_index,
                "dense_c": 1.0 - 0.05 * candidate_index,
                "rerank_r1_chunks": [5.0 if gold_candidate else 1.0 - 0.05 * candidate_index],
                "rerank_r2_chunks": [6.0 if gold_candidate else 1.0 - 0.05 * candidate_index],
                "bm25": 10.0 if candidate_index == 1 else 0.0,
            })
        meta_rows.append(query_rows)
        meta_records.append({"qid": f"meta-{query_index}", "answers": [f"m{query_index}-5"]})
    meta_calibration = list(range(10)); meta_locked = list(range(10, 20))
    synthetic_model, synthetic_oof, synthetic_report = crossfit_listwise_meta_ranker(
        Config(chunks_per_doc=1), meta_rows, meta_records, meta_calibration, meta_locked
    )
    assert synthetic_report["oof_metrics"]["Recall@5"] == 1.0
    assert task_metrics(synthetic_oof, meta_records, 5)["Recall@5"] == 1.0
    assert all(ranking[0].endswith("-5") for ranking in meta_ranker_rankings(meta_rows, synthetic_model))
    mini_chunks = tmp / "mini_chunks.jsonl"
    mini_chunks.write_text(
        '{"chunk_id": 0, "cid": "10", "text": "x"}\n'
        '{"chunk_id": 1, "cid": "20", "text": "y"}\n',
        encoding="utf-8",
    )
    assert load_chunk_cids_only(mini_chunks) == ["10", "20"]
    migration_paths = Paths(tmp / "migration")
    migration_paths.root.mkdir(parents=True, exist_ok=True)
    atomic_json(migration_paths.dense_calibration, {"best_weights": [0.0, 1.0, 0.0]})
    migration_records = [{"qid": "cached-q"}]
    migration_rows = [[
        {"cid": "1", "dense": 1.0, "chunks": [0],
         "rerank_r1_chunks": [0.1], "rerank_r2_chunks": [0.2]},
        {"cid": "2", "dense": 0.9, "chunks": [1],
         "rerank_r1_chunks": [0.3], "rerank_r2_chunks": [0.4]},
    ]]
    old_cache = migration_paths.root / "reranked_1q_old.jsonl"
    write_complete_rerank_cache(old_cache, migration_records, migration_rows)
    migrated = load_legacy_b_reranker_cache(
        Config(candidate_docs=2, rerank_top_k=2), migration_paths,
        migration_records, migration_paths.root / "reranked_1q_new.jsonl",
    )
    assert migrated is not None and not migrated[1]
    assert [row["cid"] for row in migrated[0][0]] == ["1", "2"]
    import torch
    q = torch.nn.functional.normalize(torch.randn(2, 8), dim=1).requires_grad_()
    p = torch.nn.functional.normalize(torch.randn(2, 8), dim=1).requires_grad_()
    n = torch.nn.functional.normalize(torch.randn(2, 3, 8), dim=2).requires_grad_()
    teacher = torch.tensor([[4.0, 2.0, 0.0, -1.0], [3.0, 1.0, 0.0, -2.0]])
    dense_loss = dense_contrastive_distillation_loss(torch, q, p, n, 0.02, 0.3, teacher)
    assert torch.isfinite(dense_loss); dense_loss.backward()
    hard_candidates = [
        {"cid": "gold", "dense": 1.0, "chunks": [3]},
        {"cid": "exact-number", "dense": 0.9, "chunks": [0]},
        {"cid": "semantic", "dense": 0.95, "chunks": [2]},
        {"cid": "noise", "dense": 0.8, "chunks": [1]},
    ]
    selected_hard = teacher_hard_negative_rows(
        hard_candidates, {"gold"}, [10.0, 8.0, 9.0, 1.0], [3, 0, 2, 1], 2, 2026
    )
    assert len(selected_hard) == 2 and all(row["cid"] != "gold" for row in selected_hard)
    assert selected_hard[0]["teacher_score"] >= selected_hard[1]["teacher_score"]
    # "lexical-decoy" has a low teacher score (so the teacher-hardest stratum
    # never picks it) but the highest lexical overlap; a positive
    # lexical_ratio must still surface it as a dedicated negative, and with
    # lexical_ratio=0 it must be excluded (the pre-existing 70/30 behavior).
    lexical_candidates = [
        {"cid": "gold", "dense": 1.0, "chunks": [0]},
        {"cid": "teacher-hardest", "dense": 0.9, "chunks": [1]},
        {"cid": "teacher-second", "dense": 0.7, "chunks": [2]},
        {"cid": "lexical-decoy", "dense": 0.5, "chunks": [3]},
        {"cid": "irrelevant", "dense": 0.4, "chunks": [4]},
    ]
    lexical_selected = teacher_hard_negative_rows(
        lexical_candidates, {"gold"}, [9.0, 8.0, 7.0, 0.1, 0.0], [0, 1, 2, 3, 4], 2, 2026,
        lexical_scores=[0.0, 0.1, 0.05, 0.9, 0.0], lexical_ratio=0.5,
    )
    assert {row["cid"] for row in lexical_selected} == {"teacher-hardest", "lexical-decoy"}
    lexical_disabled = teacher_hard_negative_rows(
        lexical_candidates, {"gold"}, [9.0, 8.0, 7.0, 0.1, 0.0], [0, 1, 2, 3, 4], 2, 2026,
        lexical_scores=[0.0, 0.1, 0.05, 0.9, 0.0], lexical_ratio=0.0,
    )
    assert {row["cid"] for row in lexical_disabled} == {"teacher-hardest", "teacher-second"}
    # The C curriculum helper still needs to identify gold documents outside
    # a teacher top five; production mining applies this to both A and B.
    dense_a_candidates = [
        {"cid": "a-hard-1", "dense_a": 0.95, "chunks": [2]},
        {"cid": "a-hard-2", "dense_a": 0.90, "chunks": [0]},
        {"cid": "a-easy", "dense_a": 0.10, "chunks": [1]},
        {"cid": "noise-1", "dense_a": 0.09, "chunks": [4]},
        {"cid": "noise-2", "dense_a": 0.08, "chunks": [5]},
        {"cid": "gold", "dense_a": 0.07, "chunks": [3]},
    ]
    assert missing_gold_from_top_k(dense_a_candidates, {"gold"}, 5) == {"gold"}
    a_teacher_scores = [float(row["dense_a"]) for row in dense_a_candidates]
    a_teacher_chunks = [int(row["chunks"][0]) for row in dense_a_candidates]
    c_negatives = teacher_hard_negative_rows(
        dense_a_candidates, {"gold"}, a_teacher_scores, a_teacher_chunks, 2, 2026,
    )
    assert [row["cid"] for row in c_negatives] == ["a-hard-1", "a-hard-2"]
    c_recipe = embedding_training_recipe(Config(), 3)
    assert c_recipe["initialization"] == "base_bge_m3_independent"
    assert c_recipe["teacher_kd_weight"] == 0.0
    assert resolved_lora_alpha(64, True) == 16
    assert resolved_lora_alpha(64, False) == 128
    assert "intermediate.dense" in resolved_lora_targets(Config(), True)
    stratified = select_stratified_negatives(
        hard_candidates, {"gold"}, [10.0, 8.0, 9.0, 1.0], [3, 0, 2, 1],
        [0.0, 3.0, 2.0, 1.0], [3, 0, 2, 1],
        (("top5_error", 1, [1, 2]), ("bm25", 1, [1, 2, 3])), 2,
        ["exact", "noise", "semantic", "gold"], {"gold"},
    )
    assert len(stratified) == 2 and len({row["cid"] for row in stratified}) == 2
    assert all(row["cid"] != "gold" for row in stratified)
    validate_submission({"q1": {"answer": ["10"]}}, [{"qid": "q1"}], {"10"})
    eval_records = [
        {"qid": "eval-ok", "question": "Câu đúng", "answers": ["10"]},
        {"qid": "eval-wrong", "question": "Câu còn thiếu", "answers": ["10", "20"]},
    ]
    eval_rankings = [["10", "30"], ["10", "30", "40", "50", "60", "20"]]
    wrong_count = save_eval_prediction_artifacts(
        test_paths, eval_records, eval_rankings, {"10", "20", "30", "40", "50", "60"}
    )
    assert wrong_count == 1
    saved_eval = json.loads(test_paths.eval_submission_json.read_text(encoding="utf-8"))
    saved_wrong = read_jsonl(test_paths.eval_wrong_predictions)
    assert saved_eval["eval-wrong"]["answer"] == ["10", "30", "40", "50", "60"]
    assert saved_wrong[0]["missing_gold"] == ["20"]
    assert saved_wrong[0]["gold_candidate_ranks"]["20"] == 6
    debug_paths = Paths(tmp / "public_debug")
    debug_paths.prepared.mkdir(parents=True, exist_ok=True)
    write_jsonl(debug_paths.chunks, [
        {"chunk_id": 0, "cid": "10", "part": 0, "text": "Văn bản: Luật A\nĐiều 1. Nội dung A"},
        {"chunk_id": 1, "cid": "20", "part": 0, "text": "Văn bản: Luật B\nĐiều 2. Nội dung B"},
    ])
    write_jsonl(debug_paths.documents, [
        {"cid": "10", "name": "Luật A"}, {"cid": "20", "name": "Luật B"},
    ])
    debug_rows = [[
        {"cid": "10", "dense": 0.7, "dense_a": 0.8, "dense_b": 0.9, "dense_c": 0.6,
         "chunks": [0], "rerank_r1_chunks": [1.2], "rerank_r2_chunks": [1.1], "bm25": 2.0},
        {"cid": "20", "dense": 0.6, "dense_a": 0.7, "dense_b": 0.5, "dense_c": 0.9,
         "chunks": [1], "rerank_r1_chunks": [0.8], "rerank_r2_chunks": [1.4], "bm25": 3.0},
    ]]
    debug_config = {"weights": {"b": 1.0}, "evidence_k": 1, "rrf_constant": 10.0}
    debug_rankings = fused_repair_rankings(debug_rows, debug_config)
    save_public_top10_debug(
        Config(chunks_per_doc=1), debug_paths,
        [{"qid": "public-q", "question": "Câu hỏi debug"}],
        debug_rows, debug_rankings, "rrf_baseline", debug_config, None,
    )
    assert debug_paths.public_debug_jsonl.is_file() and debug_paths.public_debug_zip.is_file()
    saved_public_debug = read_jsonl(debug_paths.public_debug_jsonl)
    assert len(saved_public_debug[0]["top10"]) == 2
    assert saved_public_debug[0]["submission_answer"] == debug_rankings[0][:5]
    assert saved_public_debug[0]["top10"][0]["chunks"][0]["text"].startswith("Văn bản:")
    tmp.mkdir(parents=True, exist_ok=True); log("Self-tests passed")
