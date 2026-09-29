"""Tiện ích huấn luyện C: lịch LR, chọn evidence, mine ứng viên.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import copy
import math


def c_step_learning_rate(update, total_updates, base_lr):
    """Warm up then linearly decay; budget comes from actual packed batches."""
    warmup = max(1, math.ceil(total_updates * .1))
    if update <= warmup:
        return base_lr * update / warmup
    return base_lr * max(.1, (total_updates - update) / max(1, total_updates - warmup))


def c_evidence_selection(shortlist, scores, retrieved, limit):
    """Preserve the teacher's best passage and one current dense passage."""
    if len(shortlist) != len(scores) or limit < 1 or not shortlist:
        raise ValueError("Invalid C evidence shortlist")
    if not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("Nonfinite C evidence score")
    ordered = [cid for _, cid in sorted(zip(scores, shortlist), key=lambda p: (-p[0], p[1]))]
    dense = [cid for cid in retrieved[:1] if cid in set(shortlist)]
    return list(dict.fromkeys([ordered[0], *dense, *ordered]))[:limit]


def c_checkpoint_key(pool):
    m = pool["anchor"]
    return (m["all"]["Recall@5"], pool["pool_recall"], m["multi"]["AllGold@5"],
            m["all"]["MRR@5"], pool["dense_recall5"])


def c_checkpoint_safe(pool, reference, b_reference=None):
    if b_reference is not None:
        for k, tolerance in ((5, .005), (b_reference["standalone"]["pool_depth"], .002)):
            metric = f"Recall@{k}"
            if pool["standalone"]["c"][metric] < b_reference["standalone"]["b"][metric] - tolerance:
                return False
    return (pool["pool_recall"] >= reference["pool_recall"] - .002
            and all(pool["anchor"][label]["Recall@5"] >= reference["anchor"][label]["Recall@5"] - .002
                    for label in ("single", "multi")))


def c_repair_targets(records, b_rows, ab_rows, pool_depth):
    """Do not let A rescuing a gold hide B's own retrieval failure."""
    if len(records) != len(b_rows) or len(records) != len(ab_rows):
        raise ValueError("C repair rankings must cover all training queries")
    result = {}
    for record, b, ab in zip(records, b_rows, ab_rows):
        gold = set(map(str, record["answers"]))
        result[str(record["qid"])] = {
            "pool": gold - {str(r["cid"]) for r in b[:pool_depth]},
            "top5": gold - {str(r["cid"]) for r in b[:5]},
            "ab_pool": gold - {str(r["cid"]) for r in ab[:pool_depth]},
            "ab_top5": gold - {str(r["cid"]) for r in ab[:5]}}
    return result


def c_mining_candidates(records, fused_rows, active_rows, replay_groups, negatives, chunks_per_doc):
    """Expose active retriever mistakes and retain clean old negatives.

    This is a training-only pool. Appended rows never enter inference pools.
    Replayed documents are re-scored and re-filtered by mine(), not trusted.
    """
    if len(records) != len(fused_rows) or len(records) != len(active_rows):
        raise ValueError("C mining pools must cover all training queries")
    output = []
    for record, fused, active in zip(records, fused_rows, active_rows):
        by_cid = {str(r["cid"]): copy.deepcopy(r) for r in fused}
        active_map = {str(r["cid"]): r for r in active}
        for row in active[:negatives]:
            by_cid.setdefault(str(row["cid"]), copy.deepcopy(row))
        replay = replay_groups.get(str(record["qid"]), {})
        replay_map = dict(zip(replay.get("negative_cids", []), replay.get("negative_chunks", [])))
        for cid, chunks in replay_map.items():
            by_cid.setdefault(str(cid), {"cid": str(cid), "dense": 0., "chunks": list(chunks)})
        for cid, row in by_cid.items():
            current = active_map.get(cid, {})
            current_chunks = current.get("chunks", [])
            previous = replay_map.get(cid, [])
            preferred = [*current_chunks[:1], *previous[:1], *current_chunks, *row["chunks"], *previous]
            row["chunks"] = list(dict.fromkeys(preferred))[:chunks_per_doc]
            row["_c_dense"] = float(current.get("dense", -1e30))
            row["_c_replay"] = bool(previous)
        output.append(list(by_cid.values()))
    return output
