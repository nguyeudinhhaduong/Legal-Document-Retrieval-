"""Băm artifact, đọc JSON, làm sạch dữ liệu train và lịch lấy mẫu.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def implementation_digest(module):
    """Hash script bytes, or executable notebook definitions without source cache."""
    import types
    import sys

    filename = getattr(module, "__file__", None)
    if filename and Path(filename).is_file():
        return digest(filename)
    # Dataclasses generate methods with filename <string>; their source is
    # never in linecache. Kaggle may also evict cell source. Fingerprint code
    # structurally, excluding filenames/line numbers that change per kernel.
    def encode(value):
        if isinstance(value, types.CodeType):
            fields = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount",
                      "co_nlocals", "co_stacksize", "co_flags", "co_code",
                      "co_consts", "co_names", "co_varnames", "co_freevars",
                      "co_cellvars", "co_exceptiontable")
            return {key: encode(getattr(value, key, None)) for key in fields}
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return {"set": sorted((encode(item) for item in value), key=repr)}
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, type(re.compile(""))):
            return {"pattern": value.pattern, "flags": value.flags}
        return {"type": type(value).__qualname__}

    def function(value):
        return {"code": encode(value.__code__), "defaults": encode(value.__defaults__),
                "kwdefaults": encode(value.__kwdefaults__)}

    definitions = {}
    for name, value in list(vars(module).items()):
        if name.isupper():
            definitions[name] = encode(value)
        if getattr(value, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(value):
            definitions[name] = function(value)
        elif inspect.isclass(value):
            members = {}
            for key, member in vars(value).items():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if inspect.isfunction(member):
                    members[key] = function(member)
                elif isinstance(member, property):
                    members[key] = [function(f) if f else None
                                    for f in (member.fget, member.fset, member.fdel)]
                elif not key.startswith("__"):
                    members[key] = encode(member)
            definitions[name] = members
    if not definitions:
        raise RuntimeError("No implementation definitions found for cache fingerprint")
    return identity({"notebook_code_v2": definitions, "python": list(sys.version_info[:2])})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def adapter_identity(path):
    files = [Path(path) / "adapter_config.json"]
    files += [p for p in Path(path).iterdir() if p.name in {
        "adapter_model.safetensors", "adapter_model.bin"}]
    if len(files) < 2:
        raise FileNotFoundError(f"Incomplete adapter: {path}")
    return {p.name: digest(p) for p in files}


def copy_adapter(source, target):
    """Only model/config files; never copy an old score cache or training receipt."""
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        return
    target.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = target / name
        if p.exists() and not (source / name).exists():
            p.unlink()
    for p in source.iterdir():
        if p.is_file() and (p.suffix in {".json", ".safetensors", ".bin", ".model"}
                            or p.name == "README.md"):
            shutil.copy2(p, target / p.name)


def clean_records(api, train, validation, corpus_cids):
    """Remove validation by QID AND normalized text; merge all duplicate golds."""
    key = lambda s: " ".join(api.clean_legal_text(s).casefold().split())
    held_text = {key(r["question"]) for r in validation}
    held_ids = {str(r["qid"]) for r in validation}
    merged = {}
    excluded = []
    for row in train:
        text_key = key(row["question"])
        if str(row["qid"]) in held_ids or text_key in held_text:
            excluded.append(str(row["qid"]))
            continue
        gold = set(map(str, row["answers"]))
        if not gold or gold - corpus_cids:
            raise ValueError(f"Invalid train gold: {row['qid']}, missing={gold - corpus_cids}")
        if text_key not in merged:
            merged[text_key] = {**row, "answers": set(), "source_qids": []}
        merged[text_key]["answers"].update(gold)
        merged[text_key]["source_qids"].append(str(row["qid"]))
    output = [{**r, "answers": sorted(r["answers"])} for r in merged.values()]
    return output, {"raw_queries": len(train), "clean_queries": len(output),
                    "validation_overlap_removed": excluded,
                    "duplicates_merged": len(train) - len(excluded) - len(output),
                    "multi_gold_queries": sum(len(r["answers"]) > 1 for r in output)}


def balanced_schedule(groups, count, visits, seed, repeat_cap=2):
    """Equal target shares over (multi-gold, repair), bounded repeated queries.

    Least-exposed queries are first across epochs. Small strata cannot consume
    60% of every epoch indefinitely. Exhausted slots go to other strata.
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for i, row in enumerate(groups):
        buckets[(len(row["answers"]) > 1, bool(row["repair"]))].append(i)
    if not buckets:
        raise ValueError("No clean groups for training")
    selected, local = [], Counter()
    count = min(count, repeat_cap * len(groups))
    while len(selected) < count:
        keys = list(buckets)
        rng.shuffle(keys)
        advanced = False
        for key in keys:
            choices = [i for i in buckets[key] if local[i] < repeat_cap]
            if not choices:
                continue
            rng.shuffle(choices)
            chosen = min(choices, key=lambda i: (visits.get(str(groups[i]["qid"]), 0), local[i]))
            selected.append(chosen)
            local[chosen] += 1
            qid = str(groups[chosen]["qid"])
            visits[qid] = visits.get(qid, 0) + 1
            advanced = True
            if len(selected) == count:
                break
        if not advanced:
            break
    rng.shuffle(selected)
    return selected
