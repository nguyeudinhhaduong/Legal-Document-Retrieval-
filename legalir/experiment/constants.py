"""Hằng số của thí nghiệm B-repair (C/R2).

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations


VERSION = "20260913_b_repair_evidence4_neg24_r2_2000_decay_v9"


ADAPTERS = ("embedding_adapter", "embedding_adapter_stage2", "embedding_adapter_stage3",
            "bge_reranker_v2_m3_adapter", "bge_reranker_v2_m3_adapter_stage2")


INDEXES = ("dense_finetuned.faiss", "dense_finetuned_stage2.faiss", "dense_finetuned_stage3.faiss")
