"""Lớp Experiment ghép từ các mixin (base, pools, mining, train_c, train_r2, export).

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

from legalir.experiment.base import ExperimentBase
from legalir.experiment.pools import PoolMixin
from legalir.experiment.mining import MiningMixin
from legalir.experiment.train_c import TrainCMixin
from legalir.experiment.train_r2 import TrainR2Mixin
from legalir.experiment.export import ExportMixin


class Experiment(PoolMixin, MiningMixin, TrainCMixin, TrainR2Mixin, ExportMixin, ExperimentBase):
    """Frozen A/B/R1, B-to-C and multi-gold R2 experiment."""
