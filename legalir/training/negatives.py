"""Chọn hard negative (teacher, lexical, phân tầng).

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import math
from typing import Any, Sequence

from legalir._deps import BM25Okapi
from legalir.core.utils import stable_key
from legalir.data.text import clean_legal_text, informative_tokens, legal_tokens


def best_positive_chunk(question: str, answer_cids: set[str], candidates: Sequence[dict[str, Any]],
                        chunk_texts: Sequence[str], doc_to_chunks: dict[str, list[int]]) -> int:
    q = informative_tokens(question)
    candidate_chunks = [chunk for row in candidates if row["cid"] in answer_cids for chunk in row["chunks"]]
    if not candidate_chunks: candidate_chunks = [chunk for cid in answer_cids for chunk in doc_to_chunks.get(cid, [])]
    if not candidate_chunks: raise ValueError(f"No chunks for positive documents {answer_cids}")
    return max(candidate_chunks, key=lambda idx: (len(q & informative_tokens(chunk_texts[idx])) / max(1, len(q)), -idx))


def teacher_hard_negative_rows(
    candidates: Sequence[dict[str, Any]],
    positive_cids: set[str],
    teacher_scores: Sequence[float],
    teacher_chunks: Sequence[int],
    count: int,
    seed: int,
    lexical_scores: Sequence[float] | None = None,
    lexical_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    """Mix teacher-ranked false positives with deterministic mid-rank negatives.

    Vietnamese legal text repeats near-identical boilerplate (same điều/khoản
    numbers, same statute name) across unrelated articles, so a document can
    look like a false positive purely from token overlap even when the base
    teacher reranker already scores it low. When lexical_ratio > 0, a third
    stratum reserves that share of the budget for the highest lexical-overlap
    non-gold candidates not already claimed by the teacher-hardest stratum,
    so training explicitly sees these lexical near-duplicates as negatives
    instead of relying on the teacher to have already ranked them low.
    """
    if not (len(candidates) == len(teacher_scores) == len(teacher_chunks)):
        raise ValueError("Teacher scores/chunks must align with candidates")
    if lexical_scores is not None and len(lexical_scores) != len(candidates):
        raise ValueError("Lexical scores must align with candidates")
    if not 0.0 <= lexical_ratio <= 1.0:
        raise ValueError("lexical_ratio must be in [0, 1]")
    ranked: list[dict[str, Any]] = []
    for dense_rank, (row, score, chunk) in enumerate(
        zip(candidates, teacher_scores, teacher_chunks), start=1
    ):
        if row["cid"] in positive_cids:
            continue
        item = dict(row)
        item["teacher_score"] = float(score)
        item["teacher_chunk"] = int(chunk)
        item["dense_rank"] = dense_rank
        if lexical_scores is not None:
            item["lexical_score"] = float(lexical_scores[dense_rank - 1])
        ranked.append(item)
    ranked.sort(key=lambda row: (-row["teacher_score"], row["dense_rank"], row["cid"]))
    lexical_count = (
        min(len(ranked), math.floor(count * lexical_ratio))
        if lexical_scores is not None and lexical_ratio > 0.0 else 0
    )
    hard_count = min(len(ranked), max(1, math.ceil((count - lexical_count) * 0.7)))
    selected = ranked[:hard_count]
    selected_cids = {row["cid"] for row in selected}
    if lexical_count:
        lexical_pool = [row for row in ranked if row["cid"] not in selected_cids]
        lexical_pool.sort(key=lambda row: (-row["lexical_score"], row["dense_rank"], row["cid"]))
        for row in lexical_pool[:lexical_count]:
            selected.append(row); selected_cids.add(row["cid"])
    remaining = [row for row in ranked if row["cid"] not in selected_cids]
    remaining.sort(key=lambda row: stable_key(seed, row["cid"]))
    selected.extend(remaining[:max(0, count - len(selected))])
    return selected[:count]


def teacher_candidate_scores(
    scorer: Any,
    question: str,
    candidates: Sequence[dict[str, Any]],
    chunk_texts: Sequence[str],
) -> tuple[list[float], list[int], list[list[int]]]:
    queries: list[str] = []
    documents: list[str] = []
    owners: list[int] = []
    chunks: list[int] = []
    for owner, row in enumerate(candidates):
        for chunk in row["chunks"]:
            queries.append(question)
            documents.append(chunk_texts[chunk])
            owners.append(owner)
            chunks.append(chunk)
    best_scores = [-math.inf] * len(candidates)
    best_chunks = [int(row["chunks"][0]) for row in candidates]
    scored_chunks: list[list[tuple[float, int]]] = [[] for _ in candidates]
    for start in range(0, len(queries), scorer.cfg.rerank_batch_size):
        batch_scores = scorer.scores(
            queries[start:start + scorer.cfg.rerank_batch_size],
            documents[start:start + scorer.cfg.rerank_batch_size],
        )
        for owner, chunk, score in zip(
            owners[start:start + scorer.cfg.rerank_batch_size],
            chunks[start:start + scorer.cfg.rerank_batch_size],
            batch_scores,
        ):
            scored_chunks[owner].append((float(score), int(chunk)))
            if float(score) > best_scores[owner]:
                best_scores[owner] = float(score)
                best_chunks[owner] = chunk
    ranked_chunks = [
        [chunk for _, chunk in sorted(values, key=lambda item: (-item[0], item[1]))]
        for values in scored_chunks
    ]
    return best_scores, best_chunks, ranked_chunks


def hardest_row_chunk(question: str, row: dict[str, Any], chunk_texts: Sequence[str]) -> int:
    query_terms = informative_tokens(question)
    return max(
        row["chunks"],
        key=lambda chunk: (
            len(query_terms & informative_tokens(chunk_texts[chunk])) / max(1, len(query_terms)),
            -chunk,
        ),
    )


def bm25_candidate_scores(
    question: str, candidates: Sequence[dict[str, Any]], chunk_texts: Sequence[str],
    evidence_k: int = 4,
) -> tuple[list[float], list[int]]:
    """Score only the dense candidate evidence; BM25 never expands the global pool."""
    query_tokens = legal_tokens(question)
    owners: list[int] = []
    chunk_ids: list[int] = []
    corpus: list[list[str]] = []
    for owner, row in enumerate(candidates):
        for chunk in row.get("chunks", [])[:evidence_k]:
            owners.append(owner); chunk_ids.append(int(chunk)); corpus.append(legal_tokens(chunk_texts[int(chunk)]))
    best_scores = [0.0] * len(candidates)
    best_chunks = [int(row["chunks"][0]) for row in candidates]
    if not corpus or not query_tokens:
        return best_scores, best_chunks
    scores = BM25Okapi(corpus).get_scores(query_tokens)
    for owner, chunk, score in zip(owners, chunk_ids, scores):
        if float(score) > best_scores[owner]:
            best_scores[owner] = float(score); best_chunks[owner] = chunk
    return best_scores, best_chunks


def select_stratified_negatives(
    candidates: Sequence[dict[str, Any]], positive_cids: set[str],
    teacher_scores: Sequence[float], teacher_chunks: Sequence[int],
    lexical_scores: Sequence[float], lexical_chunks: Sequence[int],
    quotas: Sequence[tuple[str, int, Sequence[int]]], count: int,
    chunk_texts: Sequence[str], positive_texts: set[str],
) -> list[dict[str, Any]]:
    """Fill deterministic hard-negative strata while enforcing CID/text uniqueness."""
    selected: list[dict[str, Any]] = []; seen_cids: set[str] = set(); seen_texts = set(positive_texts)

    def add(index: int, source: str) -> bool:
        row = candidates[index]; cid = str(row["cid"])
        chunk = int(teacher_chunks[index] if source != "bm25" else lexical_chunks[index])
        normalized = clean_legal_text(chunk_texts[chunk]).casefold()
        if cid in positive_cids or cid in seen_cids or normalized in seen_texts:
            return False
        item = dict(row)
        item.update({
            "teacher_score": float(teacher_scores[index]), "teacher_chunk": chunk,
            "lexical_score": float(lexical_scores[index]), "dense_rank": index + 1,
            "negative_source": source,
        })
        selected.append(item); seen_cids.add(cid); seen_texts.add(normalized)
        return True

    for source, quota, indices in quotas:
        added = 0
        for index in indices:
            if add(int(index), source):
                added += 1
                if added == quota: break
    fallback = sorted(
        range(len(candidates)),
        key=lambda index: (-float(teacher_scores[index]), index, str(candidates[index]["cid"])),
    )
    for index in fallback:
        if len(selected) >= count: break
        add(index, "teacher_fallback")
    return selected[:count]
