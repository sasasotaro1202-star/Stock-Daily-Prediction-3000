from __future__ import annotations

from collections.abc import Sequence

import numpy as np

REGIMES = ("trend_up", "range", "trend_down", "shock", "unknown")


def _as_finite_matrix(values, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def normalized_entropy(probabilities, *, axis: int = -1) -> np.ndarray:
    """Return entropy normalized to [0, 1] for diagnostics."""
    p = _as_finite_matrix(probabilities, name="probabilities")
    if np.any(p < 0.0):
        raise ValueError("probabilities must be non-negative")
    row_sum = p.sum(axis=axis, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("each probability row must have positive mass")
    p = p / row_sum
    k = p.shape[axis]
    if k <= 1:
        return np.zeros(p.shape[0], dtype=float)
    entropy = -np.sum(
        np.where(p > 0.0, p * np.log(np.clip(p, 1e-12, 1.0)), 0.0),
        axis=axis,
    )
    return np.clip(entropy / np.log(float(k)), 0.0, 1.0)


def ensemble_disagreement(predictions) -> np.ndarray:
    """Mean per-case standard deviation across model probabilities."""
    p = _as_finite_matrix(predictions, name="predictions")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("predictions must lie in [0, 1]")
    if p.shape[1] < 2:
        return np.zeros(p.shape[0], dtype=float)
    return np.clip(np.std(p, axis=1), 0.0, 1.0)


def estimate_transition_matrix(
    states: Sequence[str],
    *,
    categories: Sequence[str] = REGIMES,
    alpha: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Estimate a smoothed Markov transition matrix from an ordered history.

    The function consumes an already-cut historical prefix. It never inspects
    timestamps or future rows itself, so chronological callers can enforce PIT
    by passing only observations available before the prediction cutoff.
    """
    cats = tuple(str(x) for x in categories)
    if not cats:
        raise ValueError("categories must be non-empty")
    if len(set(cats)) != len(cats):
        raise ValueError("categories must be unique")
    if not np.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("alpha must be positive and finite")

    normalized = [str(state) for state in states]
    unknown = sorted(set(normalized).difference(cats))
    if unknown:
        raise ValueError(f"states outside categories: {unknown}")

    counts = {
        source: {target: float(alpha) for target in cats}
        for source in cats
    }
    for source, target in zip(normalized[:-1], normalized[1:]):
        counts[source][target] += 1.0

    matrix: dict[str, dict[str, float]] = {}
    for source in cats:
        total = float(sum(counts[source].values()))
        matrix[source] = {
            target: float(counts[source][target] / total)
            for target in cats
        }
    return matrix


def chronological_transition_forecast(
    states: Sequence[str],
    *,
    categories: Sequence[str] = REGIMES,
    alpha: float = 1.0,
) -> list[dict[str, object]]:
    """Produce PIT-safe one-step state-transition distributions.

    Row t is generated from transitions observed strictly before t. Therefore
    the transition from states[t] to states[t+1] can only affect row t+1 or
    later, never row t itself.
    """
    normalized = [str(state) for state in states]
    cats = tuple(str(x) for x in categories)
    if not normalized:
        return []

    outputs: list[dict[str, object]] = []
    for t, state in enumerate(normalized):
        history = normalized[:t]
        matrix = estimate_transition_matrix(
            history,
            categories=cats,
            alpha=alpha,
        )
        posterior = matrix.get(state)
        if posterior is None:
            raise ValueError(f"state outside categories: {state}")
        outputs.append(
            {
                "index": int(t),
                "state": state,
                "history_rows": int(t),
                "transition": posterior,
            }
        )
    return outputs


def heuristic_regime_posterior(
    *,
    trend_score,
    volatility_score,
    liquidity_score=None,
    event_score=None,
    temperature: float = 1.0,
) -> np.ndarray:
    """Research-only soft regime posterior from normalized diagnostics.

    This is a transparent baseline for experimentation, not a production
    regime model. It intentionally exposes the component scores so a future
    trained regime estimator can replace it without changing downstream
    contracts.
    """
    trend = np.asarray(trend_score, dtype=float)
    vol = np.asarray(volatility_score, dtype=float)
    if trend.ndim != 1 or vol.ndim != 1 or len(trend) != len(vol):
        raise ValueError("trend_score and volatility_score must be aligned 1-D arrays")

    liq = np.zeros_like(trend) if liquidity_score is None else np.asarray(liquidity_score, dtype=float)
    event = np.zeros_like(trend) if event_score is None else np.asarray(event_score, dtype=float)
    if liq.shape != trend.shape or event.shape != trend.shape:
        raise ValueError("optional regime scores must align with trend_score")
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be positive and finite")
    values = np.column_stack(
        [
            trend - 0.50 * vol - 0.20 * event,
            -np.abs(trend) - 0.20 * vol + 0.10 * liq,
            -trend - 0.50 * vol - 0.20 * event,
            0.80 * vol + 0.80 * event - 0.20 * liq,
            -0.50 * np.abs(trend) - 0.25 * vol,
        ]
    )
    values = values / temp
    values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(values, -50.0, 50.0))
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def uncertainty_components(
    *,
    model_disagreement,
    probability_entropy,
    data_uncertainty,
    regime_uncertainty,
    ood_score,
    volatility_uncertainty,
) -> dict[str, np.ndarray]:
    """Normalize independent uncertainty channels without collapsing them."""
    names = {
        "model_disagreement": model_disagreement,
        "probability_entropy": probability_entropy,
        "data_uncertainty": data_uncertainty,
        "regime_uncertainty": regime_uncertainty,
        "ood_score": ood_score,
        "volatility_uncertainty": volatility_uncertainty,
    }
    arrays = {k: np.asarray(v, dtype=float) for k, v in names.items()}
    length = {len(v) for v in arrays.values()}
    if len(length) != 1:
        raise ValueError("all uncertainty components must have equal length")
    for key, value in arrays.items():
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite and 1-D")
    return {
        key: np.clip(value, 0.0, 1.0).astype(float)
        for key, value in arrays.items()
    }


def aggregate_uncertainty(components: dict[str, np.ndarray]) -> np.ndarray:
    """Equal-weight diagnostic aggregate; components remain separately auditable."""
    if not components:
        raise ValueError("components must be non-empty")
    arrays = list(components.values())
    length = {len(x) for x in arrays}
    if len(length) != 1:
        raise ValueError("uncertainty components must be aligned")
    matrix = np.column_stack(arrays)
    if not np.isfinite(matrix).all():
        raise ValueError("uncertainty components must be finite")
    return np.clip(np.mean(matrix, axis=1), 0.0, 1.0)


def choose_information_action(
    *,
    aggregate_uncertainty,
    data_quality,
    ood_score,
    regime_stability,
) -> np.ndarray:
    """Research-only selective policy: PREDICT / ACQUIRE_MORE / WAIT / ABSTAIN."""
    u = np.asarray(aggregate_uncertainty, dtype=float)
    dq = np.asarray(data_quality, dtype=float)
    ood = np.asarray(ood_score, dtype=float)
    stability = np.asarray(regime_stability, dtype=float)
    if not (u.ndim == dq.ndim == ood.ndim == stability.ndim == 1):
        raise ValueError("policy inputs must be 1-D")
    if len({len(u), len(dq), len(ood), len(stability)}) != 1:
        raise ValueError("policy inputs must be aligned")
    if not np.isfinite(np.column_stack([u, dq, ood, stability])).all():
        raise ValueError("policy inputs must be finite")

    actions = np.full(len(u), "PREDICT", dtype=object)
    actions[dq < 0.60] = "ACQUIRE_MORE"
    actions[(stability < 0.40) & (u >= 0.55)] = "WAIT"
    actions[ood >= 0.85] = "ABSTAIN"
    actions[u >= 0.80] = "ABSTAIN"
    return actions


def scenario_probabilities(
    up_probability,
    *,
    shock_probability,
    disagreement,
) -> np.ndarray:
    """Construct a transparent four-way scenario mixture for research.

    Scenarios are continuation-up, continuation-down, range/neutral, and
    shock. The function is a policy baseline only; it is not a fitted model.
    """
    p = np.asarray(up_probability, dtype=float)
    shock = np.asarray(shock_probability, dtype=float)
    disagree = np.asarray(disagreement, dtype=float)
    if p.ndim != 1 or shock.shape != p.shape or disagree.shape != p.shape:
        raise ValueError("scenario inputs must be aligned 1-D arrays")
    if not np.isfinite(np.column_stack([p, shock, disagree])).all():
        raise ValueError("scenario inputs must be finite")
    if np.any((p < 0.0) | (p > 1.0) | (shock < 0.0) | (shock > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    shock_mass = np.clip(0.55 * shock + 0.35 * disagree, 0.0, 1.0)
    directional_mass = 1.0 - shock_mass
    confidence = np.clip(2.0 * np.abs(p - 0.5), 0.0, 1.0)
    neutral = directional_mass * (1.0 - confidence)
    up = directional_mass * confidence * p
    down = directional_mass * confidence * (1.0 - p)
    neutral += directional_mass - (up + down + neutral)
    out = np.column_stack([up, down, neutral, shock_mass])
    return out / out.sum(axis=1, keepdims=True)


__all__ = [
    "REGIMES",
    "aggregate_uncertainty",
    "choose_information_action",
    "chronological_transition_forecast",
    "ensemble_disagreement",
    "estimate_transition_matrix",
    "heuristic_regime_posterior",
    "normalized_entropy",
    "scenario_probabilities",
    "uncertainty_components",
]
