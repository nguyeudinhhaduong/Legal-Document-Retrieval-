"""Rerank candidate pool bằng Weighted RRF và ghi submission (chạy trên CPU).

Input là JSONL, mỗi dòng một câu hỏi. Chấp nhận hai định dạng:

1. File debug do `python -m legalir --stage loadmodel` sinh ra
   (`private_top20_fixed.jsonl`: khóa `top20` với `raw_scores`, `dense_present`, `rrf_votes`).
2. Định dạng gọn: {"qid", "question", "candidates": [{"cid", "scores", "present", "ranks", "title"}]}

Ví dụ:
    python scripts/rerank.py --input examples/sample_candidates.jsonl --show
    python scripts/rerank.py --input private_top20_fixed.jsonl --output outputs/rerank
    python scripts/rerank.py --input ... --weights b=0.5,r1=0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legalir.rerank.rrf import FIXED_WEIGHTS, SIGNALS, Candidate, candidate_from_debug_row, rerank  # noqa: E402


def parse_weights(text: str | None) -> dict[str, float]:
    if not text:
        return dict(FIXED_WEIGHTS)
    weights = {s: 0.0 for s in SIGNALS}
    for part in text.split(","):
        name, value = part.split("=")
        if name.strip() not in weights:
            raise SystemExit(f"Unknown signal {name!r}; expected one of {SIGNALS}")
        weights[name.strip()] = float(value)
    return weights


def load_queries(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "candidates" in row:
                cands = [Candidate(cid=str(c["cid"]), scores=c["scores"], present=c.get("present", {}),
                                   ranks=c.get("ranks"), title=c.get("title", "")) for c in row["candidates"]]
            else:
                pool = next(v for k, v in row.items() if k.startswith("top") and isinstance(v, list))
                cands = [candidate_from_debug_row(c) for c in pool]
            yield row, cands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/rerank"))
    parser.add_argument("--weights", help="vd. b=0.4,r1=0.4,r2=0.2 (mặc định bộ fixed)")
    parser.add_argument("--show", action="store_true", help="In top-5 của từng câu hỏi")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    weights = parse_weights(args.weights)
    submission, agree, total = {}, 0, 0
    for row, cands in load_queries(args.input):
        top = rerank(cands, weights)
        answer = [cid for cid, _ in top]
        submission[str(row["qid"])] = {"answer": answer}
        if "submission_answer" in row:
            total += 1
            agree += answer == list(map(str, row["submission_answer"]))
        if args.show:
            titles = {c.cid: c.title for c in cands}
            print(f"\nqid={row['qid']} | {row.get('question', '')}")
            for rank, (cid, score) in enumerate(top, 1):
                print(f"  #{rank} cid={cid:<7} rrf={score:.6f}  {titles.get(cid, '')[:90]}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "submission.json").write_text(json.dumps(submission, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(args.output / "submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", json.dumps(submission, ensure_ascii=False))
    print(f"\nWeights: {weights}")
    print(f"Queries: {len(submission)} -> {args.output / 'submission.zip'}")
    if total:
        print(f"Top-5 khớp submission_answer trong file input: {agree}/{total}")


if __name__ == "__main__":
    main()
