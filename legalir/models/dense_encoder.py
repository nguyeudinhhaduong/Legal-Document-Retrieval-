"""Bộ mã hóa dense BGE-M3 (+LoRA) cho query và chunk.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Sequence

from legalir.core.constants import EMBED_INSTRUCTION
from legalir.core.config import Config
from legalir.core.utils import log, progress
from legalir.models.lora import log_lora_parameter_summary, make_lora_config, require_peft


def require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        return torch, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install torch and transformers>=4.51 first") from exc


def last_token_pool(hidden: Any, attention_mask: Any) -> Any:
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding: return hidden[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return hidden[range(hidden.shape[0]), sequence_lengths]


def query_prompt(question: str) -> str:
    return f"Instruct: {EMBED_INSTRUCTION}\nQuery: {question}"


def is_bge_m3_model(model_name_or_path: str) -> bool:
    return "bge-m3" in model_name_or_path.casefold() or "bge_m3" in model_name_or_path.casefold()


def load_bge_m3_tokenizer_without_sentencepiece(model_path: str, maximum_length: int) -> Any:
    """Build the XLM-R fast tokenizer from its .model file using protobuf only."""
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import XLMRobertaConverter

    root = Path(model_path)
    candidates = [root / "sentencepiece.bpe.model", *sorted(root.glob("*.model"))]
    vocab_file = next((path for path in candidates if path.is_file()), None)
    if vocab_file is None:
        raise FileNotFoundError(
            f"BGE-M3 tokenizer model not found under {root}; expected sentencepiece.bpe.model"
        )

    class XLMRobertaTokenizerStub:
        add_prefix_space = True
        legacy = True

        def __init__(self, path: Path):
            self.vocab_file = str(path)

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            return {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}.get(token, 3)

    backend = XLMRobertaConverter(XLMRobertaTokenizerStub(vocab_file)).converted()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>", cls_token="<s>",
        eos_token="</s>", sep_token="</s>",
        unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
        model_max_length=maximum_length,
        padding_side="right",
    )
    log(f"Loaded BGE-M3 tokenizer from {vocab_file} without the sentencepiece package")
    return tokenizer


def load_embedding_tokenizer(cfg: Config, AutoTokenizer: Any) -> Any:
    padding_side = "right" if is_bge_m3_model(cfg.embedding_model) else "left"
    try:
        return AutoTokenizer.from_pretrained(
            cfg.embedding_model, cache_dir=cfg.cache_dir or None, padding_side=padding_side
        )
    except ValueError as exc:
        missing_backend = "Couldn't instantiate the backend tokenizer" in str(exc)
        if not is_bge_m3_model(cfg.embedding_model) or not missing_backend:
            raise
        return load_bge_m3_tokenizer_without_sentencepiece(
            cfg.embedding_model, cfg.embedding_max_length
        )


class DenseEncoder:
    def __init__(
        self, cfg: Config, training: bool = False, adapter: Path | None = None,
        lora_rank: int | None = None,
    ):
        torch, AutoModel, AutoTokenizer = require_torch()
        if not torch.cuda.is_available(): raise RuntimeError("Dense embedding training/indexing requires CUDA")
        adapter_exists = bool(adapter and (adapter / "adapter_config.json").exists())
        peft = require_peft() if training or adapter_exists else None
        self.torch, self.cfg = torch, cfg
        self.is_bge_m3 = is_bge_m3_model(cfg.embedding_model)
        rank = lora_rank if lora_rank is not None else cfg.lora_rank
        self.tokenizer = load_embedding_tokenizer(cfg, AutoTokenizer)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        base = AutoModel.from_pretrained(
            cfg.embedding_model, cache_dir=cfg.cache_dir or None, dtype=dtype,
            low_cpu_mem_usage=True, attn_implementation="sdpa",
        ).to("cuda")
        if training:
            # This must also run when B continues adapter A. Previously the
            # adapter_exists branch skipped checkpointing and used much more
            # VRAM than the independently initialized A/C branches.
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
            # New adapters use an explicit rank. Continued adapter B retains
            # the rank encoded in its parent A checkpoint.
            base = peft.get_peft_model(base, make_lora_config(
                peft, cfg, peft.TaskType.FEATURE_EXTRACTION, rank, self.is_bge_m3,
            ))
        if training:
            log_lora_parameter_summary(base, "Dense", cfg, rank, self.is_bge_m3)
        self.model = base; self.model.train(training); self.device = next(base.parameters()).device

    def tensor(self, texts: Sequence[str], queries: bool, grad: bool = False) -> Any:
        torch = self.torch
        # BGE-M3 was trained to encode queries directly without an instruction.
        values = list(texts) if self.is_bge_m3 or not queries else [query_prompt(text) for text in texts]
        batch = self.tokenizer(values, padding=True, truncation=True, max_length=self.cfg.embedding_max_length, pad_to_multiple_of=8, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        context = torch.enable_grad() if grad else torch.inference_mode()
        with context, torch.autocast("cuda", dtype=next(self.model.parameters()).dtype):
            output = self.model(**batch, use_cache=False, return_dict=True)
            pooled = (
                output.last_hidden_state[:, 0]
                if self.is_bge_m3
                else last_token_pool(output.last_hidden_state, batch["attention_mask"])
            )
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        return pooled

    def encode(self, texts: Sequence[str], queries: bool, batch_size: int, desc: str) -> Any:
        import numpy as np
        batches = [self.tensor(texts[start:start + batch_size], queries).cpu().numpy() for start in progress(range(0, len(texts), batch_size), desc=desc)]
        return np.concatenate(batches).astype("float32")

    def close(self) -> None:
        del self.model; gc.collect(); self.torch.cuda.empty_cache()
