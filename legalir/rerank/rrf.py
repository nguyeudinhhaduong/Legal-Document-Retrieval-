"""Thuật toán rerank cuối cùng: Weighted Reciprocal Rank Fusion (RRF).

Mỗi câu hỏi có một candidate pool (mặc định 60 văn bản) do 3 dense retriever
BGE-M3 LoRA A/B/C tìm ra. Mỗi văn bản mang 5 tín hiệu:

    a, b, c : cosine của dense retriever A/B/C (chỉ có khi văn bản nằm trong
              danh sách truy hồi của chính retriever đó)
    r1, r2  : điểm cross-encoder BGE-reranker-v2-m3 LoRA R1/R2, lấy MAX trên
              tối đa 4 chunk evidence của văn bản

Điểm hợp nhất của văn bản d:

    fused(d) = sum_s  w_s / sum(w)  *  1 / (K + rank_s(d)),     K = 10

trong đó rank_s(d) là hạng 1-based của d khi sắp pool theo tín hiệu s giảm dần
(hòa thì CID nhỏ hơn đứng trước). Văn bản vắng mặt trong danh sách của một dense
retriever không nhận phiếu của retriever đó. Top-5 theo fused (hòa → CID tăng
dần) là đáp án nộp. Bộ trọng số nộp (public 0.9488):

    A = 0.0, B = 0.4, C = 0.0, R1 = 0.4, R2 = 0.2

Module này chỉ dùng thư viện chuẩn, chạy được trên CPU không cần GPU/model.
Kết quả trùng khớp `legalir.rerank.fusion.rank_features` + `rankings` (bản
vector hóa bằng NumPy dùng khi train/tìm trọng số).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

SIGNALS: tuple[str, ...] = ("a", "b", "c", "r1", "r2")
DENSE_SIGNALS: tuple[str, ...] = ("a", "b", "c")
FIXED_WEIGHTS: dict[str, float] = {"a": 0.0, "b": 0.4, "c": 0.0, "r1": 0.4, "r2": 0.2}
RRF_K = 10.0


@dataclass
class Candidate:
    """Một văn bản trong candidate pool của một câu hỏi.

    `scores`  : điểm thô theo tín hiệu (a/b/c/r1/r2).
    `present` : dense retriever nào thực sự truy hồi được văn bản (mặc định có).
    `ranks`   : hạng đã tính sẵn trên pool đầy đủ (tùy chọn). Dùng khi chỉ lưu
                lại top-N của pool nhưng vẫn muốn tái lập đúng hạng gốc.
    """

    cid: str
    scores: dict[str, float] = field(default_factory=dict)
    present: dict[str, bool] = field(default_factory=dict)
    ranks: dict[str, int | None] | None = None
    title: str = ""


def document_score(chunk_scores: Iterable[float]) -> float:
    """Điểm cross-encoder của văn bản = max trên các chunk evidence."""
    values = [float(v) for v in chunk_scores]
    if not values:
        raise ValueError("A document needs at least one evidence chunk score")
    return max(values)


def signal_ranks(candidates: Sequence[Candidate], signal: str) -> dict[str, int]:
    """Hạng 1-based của từng văn bản theo một tín hiệu.

    Dense signal chỉ xếp hạng các văn bản có mặt trong danh sách retriever đó;
    văn bản vắng mặt không có hạng (không được phiếu).
    """
    eligible = []
    for cand in candidates:
        if signal in DENSE_SIGNALS and not cand.present.get(signal, True):
            continue
        value = float(cand.scores[signal])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {signal} score for cid={cand.cid}")
        eligible.append((-value, cand.cid))
    eligible.sort()
    return {cid: rank for rank, (_, cid) in enumerate(eligible, start=1)}


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    values = {s: float(weights.get(s, 0.0)) for s in SIGNALS}
    total = sum(values.values())
    if any(not math.isfinite(v) or v < 0 for v in values.values()) or total <= 0:
        raise ValueError(f"Weights must be finite, non-negative and have a positive sum: {weights}")
    return {s: v / total for s, v in values.items()}


def fused_scores(
    candidates: Sequence[Candidate],
    weights: Mapping[str, float] = FIXED_WEIGHTS,
    k: float = RRF_K,
) -> dict[str, float]:
    """Điểm Weighted RRF của mọi văn bản trong pool."""
    cids = [c.cid for c in candidates]
    if len(set(cids)) != len(cids):
        raise ValueError("Candidate pool contains duplicate CIDs")
    norm = normalize_weights(weights)
    total = {cid: 0.0 for cid in cids}
    for signal in SIGNALS:
        if norm[signal] == 0.0:
            continue
        if all(c.ranks is not None for c in candidates):
            ranks = {c.cid: c.ranks.get(signal) for c in candidates if c.ranks.get(signal)}
        else:
            ranks = signal_ranks(candidates, signal)
        for cid, rank in ranks.items():
            total[cid] += norm[signal] / (k + rank)
    # Làm tròn 12 chữ số như bản NumPy để thứ tự hòa điểm là tất định.
    return {cid: round(value, 12) for cid, value in total.items()}


def rerank(
    candidates: Sequence[Candidate],
    weights: Mapping[str, float] = FIXED_WEIGHTS,
    top_k: int | None = 5,
    k: float = RRF_K,
) -> list[tuple[str, float]]:
    """Sắp xếp lại pool, trả về [(cid, fused_score)] theo thứ tự giảm dần."""
    if len(candidates) < 5:
        raise ValueError("Each candidate pool must contain at least five documents")
    scores = fused_scores(candidates, weights, k)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered if top_k is None else ordered[:top_k]


def ranks_from_rrf_votes(votes: Mapping[str, float], k: float = RRF_K) -> dict[str, int | None]:
    """Khôi phục hạng gốc từ phiếu RRF đã lưu (vote = 1 / (k + rank))."""
    return {s: (round(1.0 / v - k) if v > 0 else None) for s, v in votes.items()}


def candidate_from_debug_row(row: Mapping) -> Candidate:
    """Đọc một dòng candidate trong file debug `private_top20_fixed.jsonl`
    (output của `python -m legalir --stage loadmodel`)."""
    chunks = row.get("chunks") or []
    title = ""
    if chunks and isinstance(chunks[0], Mapping):
        first = str(chunks[0].get("text", "")).split("\n", 1)[0]
        title = first.removeprefix("Văn bản:").strip()
    return Candidate(
        cid=str(row["cid"]),
        scores={s: float(v) for s, v in row["raw_scores"].items()},
        present={s: bool(v) for s, v in row.get("dense_present", {}).items()},
        ranks=ranks_from_rrf_votes(row["rrf_votes"]) if "rrf_votes" in row else None,
        title=row.get("title", title),
    )
