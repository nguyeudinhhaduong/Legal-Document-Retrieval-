"""Khởi tạo, kiểm artifact nguồn và cache của Experiment.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import copy
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from legalir.experiment.constants import ADAPTERS, INDEXES, VERSION
from legalir.experiment.utils import adapter_identity, clean_records, copy_adapter, digest, identity, implementation_digest, read


class ExperimentBase:
    def __init__(self, api, cfg):
        self.api, self.cfg = api, copy.deepcopy(cfg)
        self.source = Path(cfg.cr2_source.replace("\\_", "_"))
        if cfg.work_dir == api.Config.work_dir:
            self.cfg.work_dir = "/kaggle/working/legalir_brepair" if Path("/kaggle/working").is_dir() else "runs/legalir_brepair"
        self.root = Path(self.cfg.work_dir)
        source, target = self.source.resolve(), self.root.resolve()
        if source == target or source in target.parents or target in source.parents:
            raise ValueError("Source and output must be separate, non-nested directories")
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = api.Paths(self.root)
        self.cache = self.root / "cr2_cache"
        self.cache.mkdir(exist_ok=True)
        self._file_hashes = {}

    def file_digest(self, path):
        """Hash unchanged files once per process; rehash any observed change."""
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self._file_hashes:
            self._file_hashes[key] = digest(path)
        return self._file_hashes[key]

    def prepare(self):
        a, cfg, src = self.api, self.cfg, self.source
        for name in ADAPTERS:
            if not a.valid_lora_adapter(src / name):
                raise FileNotFoundError(f"Attach the complete source run; missing {src / name}")
        for name in ("train.jsonl", "validation_1000.jsonl", "public_test.jsonl", "documents.jsonl", "chunks.jsonl", "chunk_meta.json"):
            if not (src / "prepared" / name).is_file():
                raise FileNotFoundError(src / "prepared" / name)
        saved = read(src / "run_config.json") if (src / "run_config.json").exists() else {}
        meta = read(src / "prepared" / "chunk_meta.json")
        sidecars = [read(src / (name + ".meta.json")) for name in INDEXES]
        for stage, side in enumerate(sidecars):
            expected = {"schema": meta["schema"], "chunks": meta["chunks"], "config": meta["config"]}
            if side.get("chunking") != expected or side.get("pooling") != "cls" or side.get("query_instruction"):
                raise ValueError(f"Source index {INDEXES[stage]} does not match source chunking / BGE CLS recipe")
            weights = next((p for p in (src / ADAPTERS[stage]).iterdir() if p.name in {"adapter_model.safetensors", "adapter_model.bin"}), None)
            old = side.get("adapter_identity")
            if not old or weights is None or old[0] != weights.stat().st_size:
                raise ValueError(f"Source adapter/index identity mismatch: {ADAPTERS[stage]}")
        for name in ("embedding_dim", "embedding_max_length"):
            values = {s[name] for s in sidecars}
            if len(values) != 1:
                raise ValueError(f"Source indexes disagree on {name}")
            setattr(cfg, name, values.pop())
        if len({s["base_model"] for s in sidecars}) != 1:
            raise ValueError("Source A/B/C indexes disagree on the base embedding model")
        for name in ("chunk_chars", "max_chunks_per_doc"):
            setattr(cfg, name, meta["config"].get(name, 0))
        split = meta.get("split", {})
        if "seed" not in split:
            raise ValueError("Missing source split seed; cannot safely reconstruct development/locked")
        cfg.seed = split["seed"]
        cfg.eval_size = split["eval_size"]
        cfg.calibration_size = saved.get("calibration_size", cfg.calibration_size)
        # The old pipeline config records the actual R1 token budget.
        r1_meta = read(src / ADAPTERS[3] / "pipeline_config.json") if (src / ADAPTERS[3] / "pipeline_config.json").exists() else {}
        r1_cfg = r1_meta.get("config", r1_meta)
        source_max_length = saved.get("reranker_max_length", r1_cfg.get("reranker_max_length", r1_meta.get("max_length")))
        if source_max_length is None:
            raise ValueError("Missing source R1 max_length in run_config.json / pipeline_config.json")
        cfg.reranker_max_length = source_max_length
        for field in ("candidate_docs", "dense_chunk_k", "chunks_per_doc"):
            setattr(cfg, field, saved.get(field, getattr(cfg, field)))
        cfg.rerank_top_k = cfg.candidate_docs
        cfg.rerank_train_chunks_per_doc = cfg.chunks_per_doc
        if cfg.candidate_docs < 5 or cfg.chunks_per_doc < 1 or cfg.dense_chunk_k < cfg.candidate_docs:
            raise ValueError("Invalid source candidate/evidence retrieval budget")
        for field, source_model in (("embedding_model", sidecars[0]["base_model"]),
                                    ("reranker_model", saved.get("reranker_model", read(src / ADAPTERS[3] / "adapter_config.json")["base_model_name_or_path"]))):
            requested = getattr(cfg, field)
            # Explicit model override wins. Otherwise keep source model when it
            # exists, or the existing default Kaggle mount for the same BGE model.
            if requested == getattr(a.Config, field) and Path(source_model).is_dir():
                setattr(cfg, field, source_model)
        source_hashes = {name: adapter_identity(src / name) for name in ADAPTERS}
        a.log("Hashing source prepared data and indexes for cache identity (once per launch)")
        source_files = [*[(src / "prepared" / name) for name in (
            "train.jsonl", "validation_1000.jsonl", "public_test.jsonl", "chunks.jsonl", "chunk_meta.json", "documents.jsonl")],
                        *[src / name for name in INDEXES]]
        hashes = {str(p.relative_to(src)): self.file_digest(p) for p in source_files}
        recipe_cfg = asdict(cfg)
        for name in ("cr2_prepare_only", "cr2_eval_only", "rebuild"):
            recipe_cfg.pop(name, None)
        implementation = implementation_digest(sys.modules[__name__])
        base_implementation = implementation if a is sys.modules[__name__] else implementation_digest(a)
        manifest = {"version": VERSION, "implementation": implementation,
                    "base_implementation": base_implementation, "source_adapters": source_hashes,
                    "source_files": hashes, "config": recipe_cfg,
                    "legacy_identity_note": "Source sidecars contain size/mtime, not content hashes. Export mtime is tolerated; hashes lock this imported run from now on."}
        self.signature = identity(manifest)
        old_path = self.root / "cr2_manifest.json"
        if old_path.exists() and read(old_path) != manifest:
            raise ValueError("Output belongs to another source/recipe/code version. Use a NEW --work-dir.")
        a.atomic_json(old_path, manifest)
        self.frozen = source_hashes
        for i in (0, 1, 3):
            target = self.root / ADAPTERS[i]
            if target.exists():
                if adapter_identity(target) != source_hashes[ADAPTERS[i]]:
                    raise ValueError(f"Frozen output adapter was modified: {target}")
            else:
                copy_adapter(src / ADAPTERS[i], target)
        self.texts, self.cids, _ = a.load_chunk_arrays(src / "prepared" / "chunks.jsonl")
        if len(self.cids) != meta["chunks"]:
            raise ValueError("Source chunks.jsonl count does not match chunk_meta.json")
        for name in INDEXES:
            index = a.faiss.read_index(str(src / name))
            if index.ntotal != len(self.cids) or index.d != cfg.embedding_dim:
                raise ValueError(f"Invalid source FAISS dimensions/count: {name}")
            del index
        self.doc_chunks = defaultdict(list)
        for i, cid in enumerate(self.cids):
            self.doc_chunks[cid].append(i)
        self.validation = a.read_jsonl(src / "prepared" / "validation_1000.jsonl")
        self.public = a.read_jsonl(src / "prepared" / "public_test.jsonl")
        self.train, audit = clean_records(a, a.read_jsonl(src / "prepared" / "train.jsonl"), self.validation, set(self.cids))
        for records in (self.validation, self.public):
            if len({str(r["qid"]) for r in records}) != len(records):
                raise ValueError("Duplicate QID in validation/public")
        if len(self.validation) != cfg.eval_size or not 0 < cfg.calibration_size < len(self.validation):
            raise ValueError("Validation/development size mismatch")
        for r in self.validation:
            if not r["answers"] or set(map(str, r["answers"])) - set(self.doc_chunks):
                raise ValueError(f"Invalid validation gold: {r['qid']}")
        self.dev_indices, self.locked_indices = a.calibration_locked_indices(self.validation, cfg.calibration_size, cfg.seed)
        self.dev = [self.validation[i] for i in self.dev_indices]
        self.train_by_id = {str(r["qid"]): r for r in self.train}
        a.atomic_json(self.root / "cr2_data_audit.json", audit)
        a.atomic_json(self.root / "run_config.json", asdict(cfg))
        a.log(f"Frozen A/B/R1 verified. Keep source chunk schema {meta['schema']}; clean train={len(self.train)}; dev={len(self.dev)}; locked={len(self.locked_indices)}")

    def cached(self, label, recipe, make):
        key = identity({"run": self.signature, "recipe": recipe})
        path = self.cache / f"{label}_{key[:24]}.json"
        if path.exists():
            value = read(path)
            if value.get("key") != key:
                raise ValueError(f"Cache identity mismatch {path}")
            return value["data"]
        data = make()
        self.api.atomic_json(path, {"key": key, "data": data})
        return data

    def verify_frozen(self):
        for i in (0, 1, 3):
            name = ADAPTERS[i]
            if adapter_identity(self.source / name) != self.frozen[name] or adapter_identity(self.root / name) != self.frozen[name]:
                raise RuntimeError(f"Frozen adapter changed: {name}")
        self.api.atomic_json(self.root / "frozen_verified.json", {"A_B_R1_unchanged": True, "sha256": self.frozen})

    def completed(self, stage):
        path = self.root / f"{stage}_complete.json"
        if not path.exists():
            return None
        receipt = read(path)
        target = self.root / ADAPTERS[2 if stage == "c" else 4]
        if receipt["run"] != self.signature or adapter_identity(target) != receipt["adapter"]:
            raise ValueError(f"Invalid {stage} completion receipt")
        return receipt

    def save_schedule(self, stage, epoch, groups, schedule, visits):
        selected = [groups[i] for i in schedule]
        self.api.atomic_json(self.root / f"{stage}_schedule_epoch{epoch}.json", {
            "qids": [r["qid"] for r in selected], "steps_or_queries": len(selected),
            "multi_gold": sum(len(r["answers"]) > 1 for r in selected),
            "repair": sum(bool(r["repair"]) for r in selected),
            "unique_queries": len({r["qid"] for r in selected}),
            "maximum_repeats": max(Counter(r["qid"] for r in selected).values(), default=0),
        })
        self.api.atomic_json(self.root / f"{stage}_sampling_coverage.json", {
            "visits": visits, "eligible_queries": len(groups),
            "unseen_eligible_qids": [r["qid"] for r in groups if visits.get(str(r["qid"]), 0) == 0],
            "unseen_clean_train_qids": [r["qid"] for r in self.train if visits.get(str(r["qid"]), 0) == 0],
        })
