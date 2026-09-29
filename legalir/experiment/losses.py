"""Hàm loss multi-gold cho reranker và dense.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import math
from collections import defaultdict


def multi_gold_loss(torch, scores, gold_count, teacher=None, *, boundary_weight=.30, return_components=False):
    """Each gold competes only with negatives; average once per query.

    Top-5 allows at most 5-G negatives above the weakest of G golds (G<=5).
    A configurable boundary term and weak-gold term complement the negative
    list loss. The helper keeps the legacy coefficient; train_r2 passes the
    experiment's coefficient explicitly (zero by default).
    R1 distillation preserves relative scores, with a small weight.
    """
    if not math.isfinite(boundary_weight) or boundary_weight < 0:
        raise ValueError("R2 boundary weight must be finite and non-negative")
    if scores.ndim != 1 or not 1 <= gold_count < scores.numel():
        raise ValueError("R2 needs positive documents and negative documents")
    positives, negatives = scores[:gold_count].float(), scores[gold_count:].float()
    differences = negatives[None, :] - positives[:, None]
    losses = torch.logsumexp(torch.cat([torch.zeros_like(positives[:, None]), differences], dim=1), dim=1)
    boundary = scores.new_zeros((), dtype=torch.float32)
    if boundary_weight:
        threshold_index = min(max(0, 5 - gold_count), negatives.numel() - 1)
        threshold = negatives.sort(descending=True).values[threshold_index]
        boundary = boundary_weight * torch.nn.functional.softplus(threshold + 0.15 - positives).mean()
    components = {"supervised": losses.mean(), "weak_gold": .20 * losses.max(),
                  "boundary": boundary, "teacher": scores.new_zeros((), dtype=torch.float32)}
    if teacher is not None:
        teacher = torch.as_tensor(teacher, device=scores.device, dtype=torch.float32)
        if teacher.shape != scores.shape:
            raise ValueError("R2 teacher scores must align with document scores")
        # Centered regression allows the relevance offset to move while keeping
        # evidence-aligned teacher ordering. Gold supervision remains dominant.
        components["teacher"] = 0.02 * torch.nn.functional.smooth_l1_loss(
            scores - scores.mean(), teacher - teacher.mean())
    loss = sum(components.values())
    return (loss, components) if return_components else loss


def dense_gold_loss(torch, query, positive, negative, rows, temperature):
    """InfoNCE with all query golds excluded from in-batch false negatives."""
    in_batch = query @ positive.T
    own_negative = torch.einsum("bd,bnd->bn", query, negative)
    logits = torch.cat([in_batch, own_negative], dim=1) / temperature
    mask = []
    for i, row in enumerate(rows):
        gold = set(row["gold_cids"])
        mask.append([j != i and other["positive_cid"] in gold for j, other in enumerate(rows)]
                    + [cid in gold for cid in row["negative_cids"]])
    logits = logits.masked_fill(torch.tensor(mask, dtype=torch.bool, device=query.device), -torch.inf)
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(rows), device=query.device))


def dense_document_loss(torch, query, evidence, rows, temperature):
    """All golds learn together; max evidence matches document retrieval.

    Each row provides gold documents first, then negatives. Evidence is flat
    in that order. Other queries' positives supply in-batch negatives, with
    every gold CID masked. Average gold losses per query, not per document.
    """
    if temperature <= 0 or len(rows) != len(query) or not rows:
        raise ValueError("Invalid C loss batch or temperature")
    if any(len(row["document_cids"]) != len(row["document_chunks"]) for row in rows):
        raise ValueError("C document CIDs must align with chunks within each query")
    counts = [len(chunks) for row in rows for chunks in row["document_chunks"]]
    if not counts or min(counts) < 1 or sum(counts) != len(evidence):
        raise ValueError("C evidence must align with nonempty document chunks")
    chunk_scores = query.float() @ evidence.float().T / temperature
    # Keep dimensions [batch, document]; splitting and slicing the chunk axis
    # are equivalent here when the resulting pieces are stacked on dim=1.
    doc_ranges, cursor = [], 0
    for chunk_count in counts:
        doc_ranges.append((cursor, cursor + chunk_count))
        cursor += chunk_count
    doc_scores = torch.stack([
        chunk_scores[:, start:end].max(dim=1).values for start, end in doc_ranges
    ], dim=1)
    cids = [str(cid) for row in rows for cid in row["document_cids"]]
    if len(cids) != len(counts):
        raise ValueError("C document CIDs do not align with evidence")
    offsets, positive_columns, offset = [], [], 0
    for row in rows:
        n = len(row["gold_cids"])
        if (n < 1 or len(set(map(str, row["gold_cids"]))) != n or len(row["document_cids"]) <= n
                or list(row["document_cids"][:n]) != list(row["gold_cids"])):
            raise ValueError("C requires every gold first and at least one negative")
        offsets.append(offset)
        positive_columns.extend(range(offset, offset + n))
        offset += len(row["document_cids"])
    losses = []
    for i, row in enumerate(rows):
        gold = set(map(str, row["gold_cids"]))
        start, n = offsets[i], len(gold)
        negative_columns = list(range(start + n, start + len(row["document_cids"]))) + positive_columns
        by_cid = defaultdict(list)
        for column in negative_columns:
            if cids[column] not in gold:
                by_cid[cids[column]].append(column)
        if not by_cid:
            raise ValueError("C batch has no non-gold negatives")
        negatives = torch.stack([doc_scores[i, cols].max() for cols in by_cid.values()])
        positives = doc_scores[i, start:start + n]
        differences = negatives[None, :] - positives[:, None]
        per_gold = torch.logsumexp(torch.cat([torch.zeros_like(positives[:, None]), differences], dim=1), dim=1)
        weights = torch.as_tensor(row.get("gold_weights", [1.] * n), device=query.device, dtype=torch.float32)
        if weights.numel() != n or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Invalid C gold weights")
        losses.append((per_gold * weights).sum() / weights.sum())
    query_weights = torch.as_tensor([row.get("query_weight", 1.) for row in rows],
                                    device=query.device, dtype=torch.float32)
    if not torch.isfinite(query_weights).all() or (query_weights <= 0).any():
        raise ValueError("C query weights must be finite and positive")
    # Fixed scale, not normalized per batch: a singleton repair keeps its emphasis.
    return (torch.stack(losses) * query_weights).mean()
