"""Tối ưu liên tục trọng số fusion (Nelder-Mead).

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import math
from typing import Any, Sequence

from legalir.core.utils import log
from legalir.rerank.repair_fusion import REPAIR_SIGNALS


def _softmax(values: Sequence[float]) -> list[float]:
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _softplus(value: float) -> float:
    # Numerically stable softplus: avoids overflow for large positive value
    # and underflow noise for large negative value.
    return value + math.log1p(math.exp(-value)) if value > 0 else math.log1p(math.exp(value))


def _repair_config_from_vector(vector: Sequence[float], seed: dict[str, Any]) -> dict[str, Any]:
    """Decode an unconstrained real vector into a valid repair config.

    Weights go through softmax (always non-negative, always sums to one);
    rrf_constant goes through softplus (always positive). This lets
    Nelder-Mead search freely without ever needing to reject
    a point for violating the weight-simplex or positivity constraints.
    """
    signal_count = len(REPAIR_SIGNALS)
    if len(vector) != signal_count + 1:
        raise ValueError("Repair vector must contain one logit per signal plus RRF constant")
    weight_values = _softmax(vector[:signal_count])
    weights = dict(zip(REPAIR_SIGNALS, weight_values))
    return {
        **seed, "weights": weights,
        "rrf_constant": 1.0 + _softplus(vector[signal_count]),
    }


def _vector_from_repair_config(config: dict[str, Any]) -> list[float]:
    """Best-effort inverse of _repair_config_from_vector for a warm start."""
    weight_values = [max(1e-6, float(config["weights"].get(signal, 0.0))) for signal in REPAIR_SIGNALS]
    log_weights = [math.log(value) for value in weight_values]
    mean_log_weight = sum(log_weights) / len(log_weights)
    centered = [value - mean_log_weight for value in log_weights]

    def inverse_softplus(value: float) -> float:
        value = max(1e-9, float(value))
        return math.log(math.expm1(value)) if value < 30 else value

    return [*centered, inverse_softplus(max(1e-6, float(config.get("rrf_constant", 10.0)) - 1.0))]


def continuous_optimize_repair_config(
    seeds: Sequence[dict[str, Any]],
    evaluate_config: Any,
) -> None:
    """Refine each seed's continuous knobs jointly with Nelder-Mead.

    Recall@5 is piecewise-constant in these weights (only the induced
    ranking matters), so there is no usable gradient; Nelder-Mead only needs
    function evaluations and still converges on step functions because it
    tracks a simplex of trial points rather than a local slope.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        log("scipy unavailable; skipping continuous Nelder-Mead refinement")
        return

    def make_objective(seed: dict[str, Any]) -> Any:
        def objective_function(vector: Sequence[float]) -> float:
            config = _repair_config_from_vector(vector, seed)
            recall_at_5 = evaluate_config(config, "nelder_mead_refine")
            return -recall_at_5
        return objective_function

    # A grid-search seed typically has one or two signals near zero weight
    # (logit around -6..-20) and one dominant signal (logit near +7..+11).
    # scipy's default initial simplex perturbs each coordinate by only ~5%,
    # which cannot climb a ~15-unit logit gap and leaves every simplex vertex
    # on the same ranking (a flat plateau of the underlying step function),
    # so the default run reports false convergence at the seed itself. A
    # manually sized simplex with large per-axis steps is required to let a
    # currently-near-zero signal actually overtake the dominant one.
    step = 15.0
    for seed in seeds:
        objective_function = make_objective(seed)
        x0 = _vector_from_repair_config(seed)
        dimensions = len(x0)
        initial_simplex = [x0] + [
            [value + (step if axis == coordinate else 0.0) for coordinate, value in enumerate(x0)]
            for axis in range(dimensions)
        ]
        try:
            minimize(
                objective_function, x0, method="Nelder-Mead",
                options={
                    "maxiter": 400, "xatol": 1e-3, "fatol": 1e-5,
                    "initial_simplex": initial_simplex,
                },
            )
        except Exception as exc:
            log(f"Nelder-Mead refinement failed for a seed, skipping: {exc}")
