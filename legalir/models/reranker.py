"""Cross-encoder BGE-reranker-v2-m3 và gộp điểm evidence theo văn bản.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Sequence

from legalir.core.config import Config
from legalir.core.utils import progress
from legalir.models.lora import log_lora_parameter_summary, make_lora_config, require_peft


class BGEReranker:
    """BGE v2 M3 cross-encoder with an optional sequence-classification LoRA."""

    def __init__(self, cfg: Config, training: bool, adapter: Path | None = None):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise SystemExit("Install transformers, accelerate and peft") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("BGE reranker requires CUDA")
        adapter_exists = bool(adapter and (adapter / "adapter_config.json").exists())
        peft = require_peft() if training or adapter_exists else None
        self.torch, self.cfg = torch, cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.reranker_model, cache_dir=cfg.cache_dir or None
        )
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg.reranker_model, cache_dir=cfg.cache_dir or None, dtype=dtype,
            low_cpu_mem_usage=True,
        ).to("cuda")
        if base.config.num_labels != 1:
            raise ValueError(
                f"Expected a one-logit BGE reranker, got num_labels={base.config.num_labels}"
            )
        if training:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
        if adapter_exists:
            assert peft is not None and adapter is not None
            base = peft.PeftModel.from_pretrained(base, str(adapter), is_trainable=training)
        elif training:
            assert peft is not None
            base = peft.get_peft_model(base, make_lora_config(
                peft, cfg, peft.TaskType.SEQ_CLS, cfg.lora_rank, True,
            ))
        if training:
            log_lora_parameter_summary(base, "Reranker", cfg, cfg.lora_rank, True)
        self.model = base
        self.model.train(training)
        self.device = next(base.parameters()).device

    def encode(self, queries: Sequence[str], documents: Sequence[str]) -> dict[str, Any]:
        if len(queries) != len(documents):
            raise ValueError("queries and documents must have equal lengths")
        pairs = list(zip(queries, documents))
        batch = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.cfg.reranker_max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in batch.items()}

    def logits(self, queries: Sequence[str], documents: Sequence[str]) -> Any:
        return self.model(**self.encode(queries, documents), return_dict=True).logits.view(-1).float()

    def scores(self, queries: Sequence[str], documents: Sequence[str]) -> list[float]:
        with self.torch.inference_mode():
            logits = self.logits(queries, documents)
        return logits.cpu().tolist()

    def close(self) -> None:
        del self.model; gc.collect(); self.torch.cuda.empty_cache()


def reranker_group_documents(
    row: dict[str, Any], chunks_per_doc: int,
) -> list[list[str]]:
    """Return positive then negative evidence groups with legacy fallback."""
    positive = list(row.get("positive_chunks") or [row["positive"]])[:chunks_per_doc]
    stored_negatives = list(row.get("negative_chunk_groups") or [])
    negatives: list[list[str]] = []
    for index, fallback in enumerate(row["negatives"]):
        values = (
            list(stored_negatives[index])
            if index < len(stored_negatives) and stored_negatives[index]
            else [fallback]
        )
        negatives.append(values[:chunks_per_doc])
    return [positive or [row["positive"]], *negatives]


def aggregate_document_max_scores(torch: Any, flat_scores: Any, sizes: Sequence[int]) -> Any:
    """Aggregate chunk logits exactly as inference does: max per document."""
    outputs = []
    offset = 0
    for size in sizes:
        if size < 1:
            raise ValueError("Every reranker document must have at least one evidence chunk")
        outputs.append(flat_scores[offset:offset + size].max())
        offset += size
    if offset != int(flat_scores.numel()):
        raise ValueError("Reranker chunk-group sizes do not match the flat score tensor")
    return torch.stack(outputs)


def training_reranker_document_scores(
    scorer: BGEReranker, row: dict[str, Any], chunks_per_doc: int,
) -> Any:
    groups = reranker_group_documents(row, chunks_per_doc)
    documents = [text for values in groups for text in values]
    raw_parts = []
    for start in range(0, len(documents), scorer.cfg.rerank_batch_size):
        batch = documents[start:start + scorer.cfg.rerank_batch_size]
        raw_parts.append(scorer.logits([row["query"]] * len(batch), batch))
    flat_scores = scorer.torch.cat(raw_parts)
    return aggregate_document_max_scores(
        scorer.torch, flat_scores, [len(values) for values in groups]
    )


def evaluate_reranker_groups(scorer: BGEReranker, groups: Sequence[dict[str, Any]]) -> dict[str, float]:
    was_training = scorer.model.training; scorer.model.eval()
    hits = 0; hits_at_5 = 0; ndcg_at_5 = 0.0; reciprocal = 0.0; margin = 0.0
    for row in progress(groups, desc="Reranker holdout"):
        document_groups = reranker_group_documents(
            row, scorer.cfg.rerank_train_chunks_per_doc
        )
        documents = [text for values in document_groups for text in values]
        flat_scores: list[float] = []
        batch_size = scorer.cfg.rerank_checkpoint_batch_size
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            flat_scores.extend(scorer.scores([row["query"]] * len(batch), batch))
        scores = []
        offset = 0
        for values in document_groups:
            scores.append(max(flat_scores[offset:offset + len(values)]))
            offset += len(values)
        rank = 1 + sum(score > scores[0] for score in scores[1:])
        hits += int(rank == 1)
        hits_at_5 += int(rank <= 5)
        ndcg_at_5 += (1.0 / math.log2(rank + 1)) if rank <= 5 else 0.0
        reciprocal += 1.0 / rank; margin += scores[0] - max(scores[1:])
    if was_training: scorer.model.train()
    count = max(1, len(groups))
    return {
        "Accuracy@1": hits / count, "Hit@5": hits_at_5 / count,
        "NDCG@5": ndcg_at_5 / count, "MRR": reciprocal / count,
        "margin": margin / count,
    }
