"""Thuật toán rerank cuối: weighted RRF trên 5 tín hiệu A/B/C/R1/R2.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import math
import numpy as np


SIGNALS = ("a", "b", "c", "r1", "r2")


FIXED = (0.0, 0.4, 0.0, 0.4, 0.2)


def rank_features(rows_by_query):
    """Five document RRF votes, max over the SAME evidence for R1 and R2."""
    if not rows_by_query:
        raise ValueError("Cannot rank an empty query set")
    size = max(map(len, rows_by_query))
    features = np.zeros((len(rows_by_query), size, 5), dtype=np.float64)
    ids = []
    for q, rows in enumerate(rows_by_query):
        ordered = sorted(rows, key=lambda r: str(r["cid"]))
        cids = [str(r["cid"]) for r in ordered]
        if len(set(cids)) != len(cids) or len(cids) < 5:
            raise ValueError("Each candidate pool must contain at least five unique documents")
        ids.append(cids)
        for s, signal in enumerate(SIGNALS):
            values = [float(r[signal]) for r in ordered]
            if not all(map(math.isfinite, values)):
                raise ValueError(f"Nonfinite {signal} score")
            # Dense retrieval members only vote for documents present in
            # their own retrieved list. Assigning the shared floor values a
            # normal rank would award arbitrary RRF points in CID order.
            eligible = [
                i for i, row in enumerate(ordered)
                if signal not in SIGNALS[:3] or row.get(f"{signal}_present", True)
            ]
            order = sorted(eligible, key=lambda i: (-values[i], cids[i]))
            for rank, i in enumerate(order, 1):
                features[q, i, s] = 1.0 / (10.0 + rank)
    return features, ids


def rankings(features, ids, weights, k=None):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(SIGNALS),) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError(f"Weights must contain {len(SIGNALS)} finite values with a positive sum")
    scores = np.round(np.einsum("qds,s->qd", features, weights / weights.sum()), 12)
    # IDs already sorted; stable sorting is an exact deterministic CID tie break.
    return [[cids[i] for i in np.argsort(-scores[q, :len(cids)], kind="stable")[:k]]
            for q, cids in enumerate(ids)]
