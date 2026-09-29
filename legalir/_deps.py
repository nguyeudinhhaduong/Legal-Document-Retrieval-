"""Cài/nạp FAISS và rank-bm25 (ưu tiên wheel offline trên Kaggle).

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def bootstrap_retrieval_dependencies() -> None:
    """Install FAISS/BM25 once, preferring attached offline Kaggle wheels."""
    required = {"faiss": "faiss-cpu==1.15.0", "rank_bm25": "rank-bm25==0.2.2"}
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if not missing or os.environ.get("LEGALIR_NO_AUTO_INSTALL") == "1":
        return

    script_dir = Path(__file__).resolve().parents[1]
    search_roots = [script_dir]
    configured_kaggle_wheels = Path("/kaggle/input/datasets/hnhhctiliu/faissdsc")
    kaggle_input = Path("/kaggle/input")
    if configured_kaggle_wheels.is_dir():
        search_roots.insert(0, configured_kaggle_wheels)
    if kaggle_input.is_dir() and configured_kaggle_wheels not in search_roots:
        search_roots.insert(0, kaggle_input)

    wheels: list[Path] = []
    # The bundled FAISS wheel is Linux/manylinux for Kaggle; never try to install
    # it into the user's Windows interpreter.
    if sys.platform.startswith("linux"):
        for root in search_roots:
            for pattern in ("faiss_cpu-*.whl", "rank_bm25-*.whl"):
                wheels.extend(root.rglob(pattern))
    wheels = sorted(set(wheels))

    if not wheels and sys.platform.startswith("linux"):
        bundles = [
            bundle for root in search_roots
            for bundle in root.rglob("kaggle_offline_faiss.zip")
        ]
        if bundles:
            extract_dir = (
                Path("/kaggle/working/legalir_offline_wheels")
                if Path("/kaggle/working").is_dir()
                else script_dir / ".legalir_offline_wheels"
            )
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(bundles[0]) as archive:
                safe_members = [
                    name for name in archive.namelist()
                    if Path(name).name == name and name.endswith(".whl")
                ]
                if not safe_members:
                    raise RuntimeError(f"Offline bundle is incomplete: {bundles[0]}")
                for member in safe_members:
                    archive.extract(member, extract_dir)
            wheels = sorted(extract_dir.glob("*.whl"))

    command = [sys.executable, "-m", "pip", "install"]
    if wheels:
        command += ["--no-index", "--no-deps", *map(str, wheels)]
        print(f"[bootstrap] Installing offline retrieval wheels from {wheels[0].parent}", flush=True)
    else:
        # Kaggle/current environments already provide NumPy and packaging. Do
        # not let pip silently replace NumPy while installing retrieval helpers.
        command += ["--no-deps", *missing]
        print(f"[bootstrap] Offline wheels not found; installing {missing} from pip", flush=True)
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Cannot install retrieval dependencies. Attach kaggle_offline_faiss.zip "
            "or enable Internet, then rerun Baseline.py."
        ) from exc
    importlib.invalidate_caches()


bootstrap_retrieval_dependencies()


import faiss


from rank_bm25 import BM25Okapi
