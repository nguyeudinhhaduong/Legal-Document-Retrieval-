"""Tìm trọng số fusion theo Recall@5 trên development.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import itertools
import numpy as np
import time

from legalir.rerank.fusion import FIXED, SIGNALS


def metric_key(api, predictions, records):
    m = api.task_metrics(predictions, records, 5)
    return (m["Recall@5"], m["AllGold@5"], m["MRR@5"])


def search_weights(api, features, ids, records, grid, label, batch_size=128):
    """Exhaustive, memory-bounded search. Never consult locked labels.

    We keep raw integer grid weights during ranking; scaling by .1 has no
    mathematical effect on RRF. Equal metrics retain the first configuration.
    """
    gold = np.zeros(features.shape[:2], dtype=np.float64)
    counts = np.asarray([len(set(r["answers"])) for r in records], dtype=np.float64)
    for q, (cids, row) in enumerate(zip(ids, records)):
        answers = set(map(str, row["answers"]))
        gold[q, :len(cids)] = [cid in answers for cid in cids]
    best_key, best, tried, last_log = None, None, 0, 0.0
    iterator = iter(grid)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        weights = np.asarray(batch, dtype=np.float64)
        weights /= weights.sum(axis=1, keepdims=True)
        metrics = np.zeros((len(batch), 3), dtype=np.float64)
        # Query blocks avoid allocating grid x all queries x all documents.
        for start in range(0, len(records), 32):
            f = features[start:start + 32]
            scores = np.round(np.einsum("qds,bs->bqd", f, weights, optimize=False), 12)
            top = np.argsort(-scores, axis=2, kind="stable")[:, :, :5]
            hit = np.take_along_axis(np.broadcast_to(gold[start:start + 32], scores.shape), top, axis=2)
            total = hit.sum(axis=2)
            metrics[:, 0] += (total / counts[None, start:start + 32]).sum(axis=1)
            metrics[:, 1] += (total == counts[None, start:start + 32]).sum(axis=1)
            metrics[:, 2] += (hit / np.arange(1, 6)[None, None, :]).max(axis=2).sum(axis=1)
        metrics /= len(records)
        for w, m in zip(batch, metrics):
            key = tuple(np.round(m, 12))
            if best_key is None or key > best_key:
                best_key, best = key, tuple(map(float, w))
        tried += len(batch)
        if time.monotonic() - last_log > 30:
            api.log(f"{label}: searched {tried:,} weight vectors; best={best_key}")
            last_log = time.monotonic()
    if best is None:
        raise ValueError("Empty weight search")
    return best, {"tried": tried, "development_key": best_key, "raw_weights": dict(zip(SIGNALS, best))}


def select_three(api, features, ids, records):
    integer_grid = itertools.product(range(1, 11), repeat=5)
    grid_weights, grid_report = search_weights(api, features, ids, records, integer_grid, "positive_grid")
    grid_weights = tuple(round(w / 10, 1) for w in grid_weights)
    # Unrestricted nonnegative simplex: deterministic coarse search, then local
    # coordinate refinement. This is an empirical search, not a global proof.
    coarse = [tuple(x / 10 for x in w) for w in itertools.product(range(11), repeat=5) if sum(w) == 10]
    coarse += [FIXED, grid_weights]
    best, report = search_weights(api, features, ids, records, coarse, "best_val")
    best = np.asarray(best) / sum(best)
    for step in (0.05, 0.02, 0.01, 0.005):
        for _ in range(4):
            candidates = [tuple(best)]
            for i in range(5):
                for j in range(5):
                    if i != j and best[j] >= step:
                        w = best.copy(); w[i] += step; w[j] -= step
                        candidates.append(tuple(w))
            new, _ = search_weights(api, features, ids, records, candidates, f"refine {step}")
            if np.allclose(best, new, rtol=0, atol=1e-12):
                break
            best = np.asarray(new)
    grid_report["raw_weights"] = dict(zip(SIGNALS, grid_weights))
    return {"best_val": tuple(best), "fixed": FIXED, "positive_grid": grid_weights}, {
        "positive_grid": grid_report, "best_val": report,
        "selection": "development only; Recall@5, AllGold@5, MRR@5; fixed is user supplied",
        "best_val_method": "simplex coarse + coordinate refinement, includes positive_grid winner",
        "fixed_sum": sum(FIXED), "rrf_constant": 10.0,
    }
