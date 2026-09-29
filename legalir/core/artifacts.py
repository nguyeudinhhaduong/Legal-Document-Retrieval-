"""Nhận diện và khôi phục artifact từ run trước (adapter, index, chunk).

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from legalir.core.constants import CHUNK_SCHEMA
from legalir.core.config import Config, Paths
from legalir.core.utils import atomic_json, log


def valid_lora_adapter(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(
        (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def chunking_fingerprint(chunk_meta_path: Path) -> dict[str, Any] | None:
    """Identity of a run's chunk boundaries, read from its chunk_meta.json.

    Every cache keyed on chunk_id or containing chunk text verbatim (mining
    group caches, dense_calibration, calibration.json, the rerank cache) must
    embed this so a cache built under older chunk boundaries -- e.g. before
    chunk_document's Điều/Khoản-atomic rewrite (2026-08-19) -- is judged
    stale by content, not just by whether adapter/index files happen to
    still exist. Returns None when chunk_meta.json is missing or unreadable,
    which never equals a real fingerprint dict, so any cache compared
    against None is correctly treated as unmatched.
    """
    if not chunk_meta_path.is_file():
        return None
    try:
        meta = json.loads(chunk_meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return {
        "schema": meta.get("schema"), "chunks": meta.get("chunks"),
        "config": meta.get("config"),
    }


def discover_previous_run(cfg: Config, destination: Path) -> Path | None:
    """Find an attached Kaggle output containing the trained A/B LoRAs."""
    if cfg.resume_artifacts_dir:
        configured = Path(cfg.resume_artifacts_dir)
        # Accept the run root, an embedding_adapter directory, or a file inside
        # that directory (for example the tokenizer_config.json path copied
        # from Kaggle's Output browser).
        if configured.is_file(): configured = configured.parent
        if configured.name in {
            "embedding_adapter", "embedding_adapter_stage2", "embedding_adapter_stage3",
            "bge_reranker_v2_m3_adapter", "bge_reranker_v2_m3_adapter_stage2",
        }:
            configured = configured.parent
        if not configured.is_dir():
            raise FileNotFoundError(f"--resume-artifacts-dir does not exist: {configured}")
        candidates = [configured]
    else:
        kaggle_input = Path("/kaggle/input")
        if not kaggle_input.is_dir(): return None
        candidates = sorted({
            config.parent.parent
            for config in kaggle_input.rglob("adapter_config.json")
            if config.parent.name == "embedding_adapter"
        })
    candidates = [root for root in candidates if root.resolve() != destination.resolve()]
    if not candidates: return None

    def score(root: Path) -> tuple[int, int]:
        artifacts = (
            valid_lora_adapter(root / "embedding_adapter"),
            valid_lora_adapter(root / "embedding_adapter_stage2"),
            valid_lora_adapter(root / "bge_reranker_v2_m3_adapter"),
            (root / "dense_finetuned.faiss").is_file(),
            (root / "dense_finetuned_stage2.faiss").is_file(),
            (root / "dense_base.faiss").is_file(),
        )
        return sum(artifacts), int((root / "embedding_adapter" / "training_meta.json").is_file())

    selected = max(candidates, key=lambda root: (score(root), str(root)))
    if not valid_lora_adapter(selected / "embedding_adapter"):
        if cfg.resume_artifacts_dir:
            raise FileNotFoundError(
                f"LoRA A is incomplete under {selected}; adapter_config.json and adapter weights are required"
            )
        return None
    return selected


def link_or_copy_file(source: Path, target: Path) -> str:
    if target.exists() or target.is_symlink(): return "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source.resolve())
        return "symlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def source_chunking_matches_current(cfg: Config, source: Path) -> bool:
    """Check whether a previous run's chunk_meta.json matches the chunking
    this run would produce, before trusting any of its chunk-text-derived
    artifacts (FAISS indexes, mining group caches, calibration, submission).

    chunk_document's canonical hierarchy/title rewrite (current chunk schema)
    changes chunk_id boundaries at the same chunk_chars value the old
    raw-paragraph greedy pack used, so a source run built under the old
    chunker must never have its chunk-derived artifacts silently reused even
    though prepare_data itself will correctly rebuild prepared/chunks.jsonl.
    """
    fingerprint = chunking_fingerprint(source / "prepared" / "chunk_meta.json")
    if fingerprint is None:
        return False
    return (
        fingerprint["schema"] == CHUNK_SCHEMA
        and (fingerprint["config"] or {}).get("chunk_chars") == cfg.chunk_chars
        and (fingerprint["config"] or {}).get("max_chunks_per_doc", 0)
        == cfg.max_chunks_per_doc
    )


def restore_previous_artifacts(cfg: Config, paths: Paths) -> None:
    """Restore prior checkpoints without copying multi-GB FAISS files when possible."""
    if cfg.rebuild: return
    source = discover_previous_run(cfg, paths.root)
    if source is None:
        log("No attached previous LoRA A/B output found; training will run if local checkpoints are absent")
        return
    chunking_matches = source_chunking_matches_current(cfg, source)
    restored: list[str] = []
    for name, target in (
        ("embedding_adapter", paths.embed_adapter),
        ("embedding_adapter_stage2", paths.second_embed_adapter),
        ("embedding_adapter_stage3", paths.third_embed_adapter),
        ("bge_reranker_v2_m3_adapter", paths.rerank_adapter),
        ("bge_reranker_v2_m3_adapter_stage2", paths.second_rerank_adapter),
    ):
        # LoRA weights are not tied to chunk boundaries (they only depend on
        # the base model + training recipe), so these remain safe to restore
        # regardless of chunking. Only chunk-text-derived artifacts below are
        # gated on chunking_matches.
        origin = source / name
        if valid_lora_adapter(origin) and not valid_lora_adapter(target):
            if target.exists(): shutil.rmtree(target)
            shutil.copytree(origin, target)
            atomic_json(target / "restored_from.json", {"source": str(origin), "trusted": True})
            restored.append(name)

    if not chunking_matches:
        log(
            f"Previous run at {source} was built under different chunking "
            "(or has no chunk_meta.json); skipping FAISS indexes, mining "
            "group caches, calibration, and submission restore -- these "
            "will be rebuilt from scratch"
        )
    else:
        for name, target in (
            ("dense_base.faiss", paths.base_index),
            ("dense_finetuned.faiss", paths.final_index),
            ("dense_finetuned_stage2.faiss", paths.second_index),
            ("dense_finetuned_stage3.faiss", paths.third_index),
            ("dense_base.faiss.meta.json", dense_index_meta_path(paths.base_index)),
            ("dense_finetuned.faiss.meta.json", dense_index_meta_path(paths.final_index)),
            ("dense_finetuned_stage2.faiss.meta.json", dense_index_meta_path(paths.second_index)),
            ("dense_finetuned_stage3.faiss.meta.json", dense_index_meta_path(paths.third_index)),
            ("train_groups.jsonl", paths.groups),
            ("reranker_groups_bgem3.jsonl", paths.second_stage_groups),
            ("embedding_groups_stage3.jsonl", paths.third_stage_groups),
            ("reranker_groups_ensemble.jsonl", paths.reranker_groups),
            ("reranker_groups_stage2.jsonl", paths.second_reranker_groups),
            ("dense_ensemble_calibration.json", paths.dense_calibration),
            ("calibration.json", paths.calibration),
            ("submission.zip", paths.baseline_submission_zip),
        ):
            origin = source / name
            if origin.is_file():
                method = link_or_copy_file(origin, target)
                if method != "existing": restored.append(f"{name}({method})")
        # Rerank caches are the most expensive inference artifact after the
        # FAISS indexes.  They are safe only under identical chunking, but when
        # safe they let a new calibration/meta-ranker run without rescoring
        # A/C/R1/R2 or retraining any LoRA.  Copy every fingerprinted cache:
        # load_complete_rerank_cache/load_legacy_b_reranker_cache will still
        # validate QID coverage and required score fields before reuse.
        for origin in sorted(source.glob("reranked_*q_*.jsonl")):
            target = paths.root / origin.name
            method = link_or_copy_file(origin, target)
            if method != "existing": restored.append(f"{origin.name}({method})")

    source_prepared = source / "prepared"
    destination_prepared = paths.prepared
    prepared_names = (
        "train.jsonl", "validation_1000.jsonl", "public_test.jsonl", "documents.jsonl",
        *(("chunks.jsonl", "chunk_meta.json") if chunking_matches else ()),
    )
    if source_prepared.is_dir():
        for name in prepared_names:
            origin, target = source_prepared / name, destination_prepared / name
            if origin.is_file():
                method = link_or_copy_file(origin, target)
                if method != "existing": restored.append(f"prepared/{name}({method})")

    if restored:
        log(f"Restored previous artifacts from {source}: {', '.join(restored)}")
    else:
        log(f"Previous artifact root found at {source}; all matching local artifacts already exist")


def dense_index_meta_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".meta.json")


def artifact_identity(path: Path) -> tuple[str, int | None, int | None]:
    if not path.exists(): return str(path), None, None
    stat = path.stat(); return str(path), stat.st_size, stat.st_mtime_ns


def embedding_ensemble_identity(paths: Paths) -> dict[str, list[Any]]:
    return {
        "stage1_adapter": list(artifact_identity(paths.embed_adapter / "adapter_model.safetensors")),
        "stage2_adapter": list(artifact_identity(paths.second_embed_adapter / "adapter_model.safetensors")),
        "stage1_index": list(artifact_identity(paths.final_index)),
        "stage2_index": list(artifact_identity(paths.second_index)),
    }
