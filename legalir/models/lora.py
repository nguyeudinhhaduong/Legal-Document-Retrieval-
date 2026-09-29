"""Cấu hình LoRA/rsLoRA cho BGE-M3 và BGE-reranker.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import inspect
import math
from typing import Any

from legalir.core.config import Config
from legalir.core.utils import log


def require_peft() -> Any:
    """Import PEFT while ignoring an incompatible optional TorchAO backend.

    Some Kaggle GPU images currently bundle torchao 0.10.0.  Recent PEFT
    releases inspect that optional backend while creating an ordinary LoRA
    adapter and raise because they require torchao >= 0.16.0.  This pipeline
    does not quantize with TorchAO, so disabling only PEFT's TorchAO dispatcher
    is both sufficient and preferable to reinstalling the Kaggle PyTorch stack.
    """
    try:
        import peft
    except ImportError as exc:
        raise SystemExit("Install peft (and transformers/accelerate) before LoRA training") from exc

    try:
        from peft import import_utils as peft_import_utils
        from peft.tuners.lora import torchao as peft_lora_torchao

        checker = getattr(peft_lora_torchao, "is_torchao_available", None)
        if checker is not None:
            try:
                checker()
            except ImportError as exc:
                if "torchao" not in str(exc).casefold():
                    raise

                def torchao_disabled() -> bool:
                    return False

                # dispatch_torchao() holds the function imported into this
                # module, while future PEFT modules may read import_utils.
                peft_lora_torchao.is_torchao_available = torchao_disabled
                peft_import_utils.is_torchao_available = torchao_disabled
                log("Ignoring incompatible optional torchao; PEFT LoRA will use standard torch layers")
    except ModuleNotFoundError:
        # Older PEFT versions have no TorchAO dispatcher and need no workaround.
        pass
    return peft


def resolved_lora_alpha(rank: int, use_rslora: bool) -> int:
    """Keep the effective adapter scale near 2 in either LoRA mode."""
    return max(8, round(2.0 * math.sqrt(rank))) if use_rslora else rank * 2


def resolved_lora_targets(cfg: Config, is_bge_m3: bool) -> list[str]:
    if cfg.lora_target_profile == "attention":
        return (
            ["query", "key", "value"] if is_bge_m3
            else ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
    if cfg.lora_target_profile == "attention_ffn":
        return (
            [
                "query", "key", "value", "attention.output.dense",
                "intermediate.dense", "output.dense",
            ] if is_bge_m3
            else [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj",
            ]
        )
    raise ValueError(f"Unknown LoRA target profile: {cfg.lora_target_profile}")


def make_lora_config(
    peft: Any, cfg: Config, task_type: Any, rank: int, is_bge_m3: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "task_type": task_type,
        "r": rank,
        "lora_alpha": resolved_lora_alpha(rank, cfg.lora_use_rslora),
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": resolved_lora_targets(cfg, is_bge_m3),
    }
    if cfg.lora_use_rslora:
        if "use_rslora" not in inspect.signature(peft.LoraConfig).parameters:
            raise RuntimeError(
                "This run enables rsLoRA, but the installed PEFT is too old. "
                "Upgrade PEFT or pass --no-rslora."
            )
        kwargs["use_rslora"] = True
    return peft.LoraConfig(**kwargs)


def log_lora_parameter_summary(
    model: Any, label: str, cfg: Config, rank: int, is_bge_m3: bool,
) -> None:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    ratio = 100.0 * trainable / max(1, total)
    log(
        f"{label} LoRA parameters: trainable={trainable:,}/{total:,} ({ratio:.3f}%); "
        f"rank={rank}; alpha={resolved_lora_alpha(rank, cfg.lora_use_rslora)}; "
        f"rsLoRA={cfg.lora_use_rslora}; targets={resolved_lora_targets(cfg, is_bge_m3)}"
    )
