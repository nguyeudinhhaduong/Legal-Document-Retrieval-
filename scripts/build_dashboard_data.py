"""Dựng dữ liệu cho dashboard kết quả cuối và file ví dụ nhỏ.

Input: file debug do `python -m legalir --stage loadmodel` sinh ra
(`private_top20_fixed.jsonl`). Output:
    dashboard/results.js              top-10 sau rerank của mọi câu hỏi (gọn, không có text chunk)
    examples/sample_candidates.jsonl  vài câu hỏi ở định dạng gọn cho scripts/rerank.py

    python scripts/build_dashboard_data.py --input path/to/private_top20_fixed.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legalir.rerank.rrf import FIXED_WEIGHTS, RRF_K, SIGNALS, candidate_from_debug_row, rerank  # noqa: E402

SAMPLE_QIDS = ("44474", "13582", "87044")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dashboard", type=Path, default=ROOT / "dashboard" / "results.js")
    parser.add_argument("--sample", type=Path, default=ROOT / "examples" / "sample_candidates.jsonl")
    parser.add_argument("--depth", type=int, default=10)
    args = parser.parse_args()

    titles: dict[str, str] = {}
    queries, samples = [], []
    agree = {s: 0 for s in SIGNALS}
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pool = next(v for k, v in row.items() if k.startswith("top") and isinstance(v, list))
            cands = [candidate_from_debug_row(c) for c in pool]
            ranked = rerank(cands, FIXED_WEIGHTS, top_k=args.depth)
            by_cid = {c.cid: c for c in cands}
            items = []
            for cid, score in ranked:
                cand = by_cid[cid]
                titles.setdefault(cid, cand.title)
                items.append([cid, score, *[cand.ranks.get(s) for s in SIGNALS]])
            top1 = ranked[0][0]
            for s in SIGNALS:
                agree[s] += by_cid[top1].ranks.get(s) == 1
            queries.append([str(row["qid"]), row["question"], items])
            if str(row["qid"]) in SAMPLE_QIDS:
                samples.append({
                    "qid": str(row["qid"]),
                    "question": row["question"],
                    "submission_answer": row["submission_answer"],
                    "candidates": [{
                        "cid": c.cid, "title": c.title,
                        "scores": {s: round(v, 6) for s, v in c.scores.items()},
                        "present": c.present, "ranks": c.ranks,
                    } for c in cands],
                })

    payload = {
        "meta": {
            "signals": list(SIGNALS), "weights": FIXED_WEIGHTS, "rrf_k": RRF_K,
            "queries": len(queries), "depth": args.depth,
            "top1_agreement": {s: round(agree[s] / len(queries), 4) for s in SIGNALS},
        },
        "titles": titles,
        "queries": queries,
    }
    args.dashboard.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    args.dashboard.write_text(f"window.LEGALIR_RESULTS = {data};\n", encoding="utf-8")
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    with args.sample.open("w", encoding="utf-8") as out:
        for sample in samples:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
    size = args.dashboard.stat().st_size / 1e6
    print(f"{len(queries)} queries -> {args.dashboard} ({size:.1f} MB); samples={len(samples)} -> {args.sample}")


if __name__ == "__main__":
    main()
