from __future__ import annotations

"""Research-only frontier prediction-pattern suite.

This module evaluates many deterministic / prior-only prediction patterns on the
same chronological OOS bank. It is deliberately isolated from production.

Core rule:
    current-fold outcomes may score a pattern, but may never tune a pattern
    or its parameters. Any learned component is fitted on strictly prior folds.
"""

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6
DEFAULT_LOCKED_FOLDS = 2
MIN_STACK_ROWS = 80
STACK_C_VALUES = (0.25, 1.0)
TEMPERATURE_GRID = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)


def _safe_probability(p: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.size and not np.isfinite(arr).all():
        raise ValueError("non-finite probability")
    return np.clip(arr, EPS, 1.0 - EPS)


def _logit(p: np.ndarray | Sequence[float]) -> np.ndarray:
    x = _safe_probability(p)
    return np.log(x) - np.log1p(-x)


def _sigmoid(z: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(z, dtype=float)
    out = np.empty_like(arr, dtype=float)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_z = np.exp(arr[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return np.clip(out, EPS, 1.0 - EPS)


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=int)
    p = _safe_probability(p)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p <= right if right >= 1.0 else p < right)
        if not mask.any():
            continue
        total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    p = _safe_probability(p)
    if len(y) == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }
    pred = p >= 0.5
    accuracy = float(np.mean(pred == y))
    logloss = float(
        -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    )
    brier = float(np.mean((p - y) ** 2))
    return {
        "n": int(len(y)),
        "accuracy": accuracy,
        "logloss": logloss,
        "brier": brier,
        "ece": _ece(y, p),
    }


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), EPS)
    x = np.asarray(values, dtype=float) / temp
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.clip(exp_x.sum(), EPS, None)


def _fold_probabilities(fold: Mapping[str, Any], models: Sequence[str]) -> np.ndarray:
    y = np.asarray(fold.get("y", []), dtype=int)
    predictions = fold.get("predictions")
    if not isinstance(predictions, Mapping):
        raise ValueError("missing predictions")
    cols = []
    for model in models:
        if model not in predictions:
            raise ValueError(f"missing model prediction: {model}")
        p = np.asarray(predictions[model], dtype=float)
        if p.ndim != 1 or len(p) != len(y):
            raise ValueError(f"misaligned predictions: {model}")
        cols.append(_safe_probability(p))
    return np.column_stack(cols)


def _risk_matrix(fold: Mapping[str, Any], n: int) -> np.ndarray:
    raw = fold.get("risk_matrix")
    if raw is None:
        return np.zeros((n, 0), dtype=float)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError("invalid risk_matrix")
    return arr


def _incomplete_score(risk: np.ndarray) -> np.ndarray:
    if risk.shape[1] == 0:
        return np.zeros(len(risk), dtype=float)
    return 1.0 - np.clip(np.isfinite(risk).mean(axis=1), 0.0, 1.0)


def _risk_ood(current_risk: np.ndarray, prior_risks: Sequence[np.ndarray]) -> np.ndarray:
    if current_risk.shape[1] == 0 or not prior_risks:
        return np.zeros(len(current_risk), dtype=float)
    dim = current_risk.shape[1]
    prior = []
    for arr in prior_risks:
        x = np.asarray(arr, dtype=float)
        if x.ndim != 2 or x.shape[1] != dim:
            continue
        prior.append(x)
    if not prior:
        return np.zeros(len(current_risk), dtype=float)
    hist = np.vstack(prior)
    med = np.nanmedian(hist, axis=0)
    sd = np.nanstd(hist, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    cur = np.nan_to_num(current_risk, nan=med[None, :])
    z = np.abs((cur - med[None, :]) / sd[None, :])
    return np.clip(np.sqrt(np.mean(z * z, axis=1)) / 4.0, 0.0, 1.0)


def _difficulty(p_matrix: np.ndarray, risk: np.ndarray) -> np.ndarray:
    mean_p = p_matrix.mean(axis=1)
    uncertainty = p_matrix.std(axis=1)
    confidence = 1.0 - 2.0 * np.abs(mean_p - 0.5)
    incompleteness = _incomplete_score(risk)
    agreement = 1.0 - np.mean(
        (p_matrix >= 0.5) != (mean_p[:, None] >= 0.5),
        axis=1,
    )
    return np.clip(
        0.50 * uncertainty
        + 0.25 * (1.0 - np.clip(agreement, 0.0, 1.0))
        + 0.15 * (1.0 - confidence)
        + 0.10 * incompleteness,
        0.0,
        1.0,
    )


def _history_rows(folds: Sequence[Mapping[str, Any]], models: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = [_fold_probabilities(f, models) for f in folds]
    ys = [np.asarray(f.get("y", []), dtype=int) for f in folds]
    risks = [_risk_matrix(f, len(y)) for f, y in zip(folds, ys)]
    if not matrices:
        return np.zeros((0, len(models))), np.zeros(0, dtype=int), np.zeros((0, 0))
    common_risk_dim = max((r.shape[1] for r in risks), default=0)
    padded = []
    for r in risks:
        if r.shape[1] == common_risk_dim:
            padded.append(r)
        else:
            x = np.full((len(r), common_risk_dim), np.nan)
            if r.shape[1]:
                x[:, :r.shape[1]] = r
            padded.append(x)
    return np.vstack(matrices), np.concatenate(ys), np.vstack(padded)


def _prior_model_losses(folds: Sequence[Mapping[str, Any]], models: Sequence[str]) -> np.ndarray:
    losses = []
    for model in models:
        vals = []
        for fold in folds:
            y = np.asarray(fold.get("y", []), dtype=int)
            p = _fold_probabilities(fold, models)[:, models.index(model)]
            if len(y):
                vals.append(_metrics(y, p)["logloss"])
        losses.append(float(np.mean(vals)) if vals else math.log(2.0))
    return np.asarray(losses, dtype=float)


def _prior_recency_losses(folds: Sequence[Mapping[str, Any]], models: Sequence[str], half_life: float = 2.0) -> np.ndarray:
    if not folds:
        return np.full(len(models), math.log(2.0))
    ages = np.arange(len(folds) - 1, -1, -1, dtype=float)
    fold_weights = np.exp(-np.log(2.0) * ages / max(half_life, EPS))
    out = []
    for model in models:
        numerator = 0.0
        denominator = 0.0
        idx = models.index(model)
        for fold, weight in zip(folds, fold_weights):
            y = np.asarray(fold.get("y", []), dtype=int)
            if not len(y):
                continue
            p = _fold_probabilities(fold, models)[:, idx]
            numerator += float(_metrics(y, p)["logloss"]) * float(weight)
            denominator += float(weight)
        out.append(float(numerator / max(denominator, EPS)))
    return np.asarray(out, dtype=float)


def _regime_weights(
    folds: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    current_frame: Any,
) -> np.ndarray:
    global_losses = _prior_model_losses(folds, models)
    result = []
    current_regimes = np.asarray(current_frame.get("regime", ["unknown"] * len(current_frame)), dtype=str)
    for regime in current_regimes:
        per_model = []
        counts = []
        for model in models:
            vals = []
            for fold in folds:
                frame = fold.get("frame")
                if frame is None or "regime" not in frame:
                    continue
                mask = frame["regime"].astype(str).to_numpy() == regime
                if not mask.any():
                    continue
                y = np.asarray(fold.get("y", []), dtype=int)[mask]
                p = _fold_probabilities(fold, models)[mask, models.index(model)]
                if len(y):
                    vals.append(float(_metrics(y, p)["logloss"]))
            per_model.append(float(np.mean(vals)) if vals else float("nan"))
            counts.append(len(vals))
        usable = np.isfinite(per_model) & (np.asarray(counts) >= 1)
        losses = global_losses.copy()
        losses[usable] = np.asarray(per_model)[usable]
        result.append(_softmax(-losses, 0.05))
    return np.asarray(result, dtype=float)


def _stacker_predict(
    prior_X: np.ndarray,
    prior_y: np.ndarray,
    current_X: np.ndarray,
    C: float,
) -> np.ndarray:
    if len(prior_y) < MIN_STACK_ROWS or len(np.unique(prior_y)) < 2:
        return np.full(len(current_X), float(np.mean(prior_y)) if len(prior_y) else 0.5)
    if not np.isfinite(prior_X).all() or not np.isfinite(current_X).all():
        return np.full(len(current_X), float(np.mean(prior_y)))
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=float(C), max_iter=2000, random_state=13013)),
    ])
    model.fit(prior_X, prior_y)
    return _safe_probability(model.predict_proba(current_X)[:, 1])


def _fit_platt(prior_p: np.ndarray, prior_y: np.ndarray, current_p: np.ndarray) -> np.ndarray:
    if len(prior_y) < 30 or len(np.unique(prior_y)) < 2:
        return current_p
    x = _logit(prior_p).reshape(-1, 1)
    cur = _logit(current_p).reshape(-1, 1)
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=13013)
    model.fit(x, prior_y)
    return _safe_probability(model.predict_proba(cur)[:, 1])


def _fit_beta(prior_p: np.ndarray, prior_y: np.ndarray, current_p: np.ndarray) -> np.ndarray:
    if len(prior_y) < 30 or len(np.unique(prior_y)) < 2:
        return current_p
    p = _safe_probability(prior_p)
    cur = _safe_probability(current_p)
    X = np.column_stack([np.log(p), np.log1p(-p)])
    XC = np.column_stack([np.log(cur), np.log1p(-cur)])
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=13013)
    model.fit(X, prior_y)
    return _safe_probability(model.predict_proba(XC)[:, 1])


def _fit_isotonic(prior_p: np.ndarray, prior_y: np.ndarray, current_p: np.ndarray) -> np.ndarray:
    if len(prior_y) < 40 or len(np.unique(prior_y)) < 2:
        return current_p
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(prior_p, prior_y)
    return _safe_probability(model.predict(current_p))


def _fit_temperature(prior_p: np.ndarray, prior_y: np.ndarray, current_p: np.ndarray) -> tuple[np.ndarray, float]:
    if len(prior_y) < 30 or len(np.unique(prior_y)) < 2:
        return current_p, 1.0
    logits = _logit(prior_p)
    best_t = 1.0
    best_loss = float("inf")
    for t in TEMPERATURE_GRID:
        pred = _sigmoid(logits / float(t))
        loss = float(_metrics(prior_y, pred)["logloss"])
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return _sigmoid(_logit(current_p) / best_t), best_t


def _prior_weights(folds: Sequence[Mapping[str, Any]], models: Sequence[str], temperature: float) -> np.ndarray:
    if not folds:
        return np.full(len(models), 1.0 / len(models))
    losses = _prior_model_losses(folds, models)
    return _softmax(-losses, temperature)


def _rank_percentile(p_matrix: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(p_matrix, axis=0), axis=0)
    n = p_matrix.shape[0]
    pct = (ranks + 1.0) / (n + 2.0)
    return np.clip(pct.mean(axis=1), EPS, 1.0 - EPS)


def _session_cluster_bootstrap(
    y: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    session_ids: Sequence[Any],
    *,
    n_resamples: int = 1000,
    seed: int = 13013,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    candidate = _safe_probability(candidate)
    baseline = _safe_probability(baseline)
    clusters = np.asarray([str(x) for x in session_ids], dtype=str)
    if len(y) != len(candidate) or len(y) != len(baseline) or len(y) != len(clusters):
        raise ValueError("bootstrap inputs must be aligned")
    valid = clusters != ""
    if not valid.all():
        return {
            "status": "BLOCKED_MISSING_SESSION_CLUSTER",
            "n": int(len(y)),
            "clusters": int(len(np.unique(clusters[valid]))),
            "same_oos_cases": True,
            "selection_allowed": False,
        }
    unique, inverse = np.unique(clusters, return_inverse=True)
    if len(unique) < 5:
        return {
            "status": "INSUFFICIENT_SESSION_CLUSTERS",
            "n": int(len(y)),
            "clusters": int(len(unique)),
            "same_oos_cases": True,
            "selection_allowed": False,
        }
    cand_loss = -(y * np.log(candidate) + (1 - y) * np.log(1 - candidate))
    base_loss = -(y * np.log(baseline) + (1 - y) * np.log(1 - baseline))
    delta = cand_loss - base_loss
    rows = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    sums = np.asarray([delta[idx].sum() for idx in rows], dtype=float)
    sizes = np.asarray([len(idx) for idx in rows], dtype=int)
    observed = float(sums.sum() / max(sizes.sum(), 1))
    rng = np.random.default_rng(int(seed))
    draw = rng.integers(0, len(unique), size=(int(n_resamples), len(unique)))
    boot = sums[draw].sum(axis=1) / np.maximum(sizes[draw].sum(axis=1), 1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "status": "EXECUTED_CLUSTER_BOOTSTRAP",
        "n": int(len(y)),
        "clusters": int(len(unique)),
        "same_oos_cases": True,
        "selection_allowed": False,
        "observed_delta_candidate_minus_equal": observed,
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "p_two_sided": float(
            np.clip(
                2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0))),
                0.0,
                1.0,
            )
        ),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "cluster_unit": "session_date",
        "research_only": True,
    }


def _evaluate_pattern(
    name: str,
    category: str,
    y: np.ndarray,
    p: np.ndarray,
    baseline: np.ndarray,
    session_ids: Sequence[Any],
    *,
    meta: Mapping[str, Any] | None = None,
    active: np.ndarray | None = None,
) -> dict[str, Any]:
    p = _safe_probability(p)
    metrics = _metrics(y, p)
    row = {
        "name": name,
        "category": category,
        "metrics": metrics,
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "parameters": dict(meta or {}),
    }
    if active is not None:
        active = np.asarray(active, dtype=bool)
        active_metrics = _metrics(y[active], p[active]) if active.any() else _metrics(np.array([], dtype=int), np.array([]))
        row["selective"] = {
            "coverage": float(active.mean()) if len(active) else 0.0,
            "active_metrics": active_metrics,
            "abstain_count": int((~active).sum()),
        }
    seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
    row["cluster_bootstrap"] = _session_cluster_bootstrap(
        y, p, baseline, session_ids, seed=seed
    )
    return row


def run_frontier_pattern_suite(
    bank: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    locked_folds: int = DEFAULT_LOCKED_FOLDS,
    min_folds: int = 5,
) -> dict[str, Any]:
    """Evaluate a broad, fixed research matrix on a chronological OOS bank."""

    ordered = [dict(bank[k]) for k in sorted(bank, key=lambda x: int(x))] if isinstance(bank, Mapping) else [dict(x) for x in bank]
    base = {
        "schema_version": 1,
        "status": "BLOCKED",
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
    }
    if len(ordered) < int(min_folds):
        return {**base, "reason": f"need_at_least_{int(min_folds)}_chronological_folds"}

    common = set(ordered[0].get("predictions", {}) or {})
    for fold in ordered[1:]:
        common &= set(fold.get("predictions", {}) or {})
    models = sorted(common)
    if len(models) < 2:
        return {**base, "reason": "need_at_least_2_common_models"}

    locked = min(max(1, int(locked_folds)), len(ordered) - 1)
    locked_start = len(ordered) - locked

    per_pattern: dict[str, list[dict[str, Any]]] = {}
    training_trace: dict[str, list[dict[str, Any]]] = {}
    locked_rows: dict[str, list[float]] = {}
    locked_y: list[int] = []
    locked_sessions: list[str] = []

    def add_locked(name: str, p: np.ndarray) -> None:
        locked_rows.setdefault(name, []).extend(np.asarray(p, dtype=float).tolist())

    for t, fold in enumerate(ordered):
        y = np.asarray(fold.get("y", []), dtype=int)
        p_matrix = _fold_probabilities(fold, models)
        risk = _risk_matrix(fold, len(y))
        frame = fold.get("frame")
        session_ids = (
            frame.get("session_date", [""] * len(y)).astype(str).tolist()
            if frame is not None and "session_date" in frame
            else [str(t)] * len(y)
        )

        equal = np.mean(p_matrix, axis=1)
        patterns: dict[str, tuple[np.ndarray, str, dict[str, Any], np.ndarray | None]] = {}

        patterns["equal_mean"] = (equal, "baseline", {}, None)
        patterns["median_probability"] = (np.median(p_matrix, axis=1), "robust_aggregation", {}, None)

        if p_matrix.shape[1] >= 3:
            sorted_p = np.sort(p_matrix, axis=1)
            patterns["trimmed_mean_1"] = (
                sorted_p[:, 1:-1].mean(axis=1),
                "robust_aggregation",
                {"trim_each_side": 1},
                None,
            )
        else:
            patterns["trimmed_mean_1"] = (equal.copy(), "robust_aggregation", {"trim_each_side": 1, "fallback": "equal"}, None)

        patterns["logit_mean"] = (
            _sigmoid(np.mean(_logit(p_matrix), axis=1)),
            "probability_geometry",
            {"space": "logit"},
            None,
        )

        logit_matrix = _logit(p_matrix)
        row_q10 = np.quantile(logit_matrix, 0.10, axis=1)
        row_q90 = np.quantile(logit_matrix, 0.90, axis=1)
        winsorized = np.clip(logit_matrix, row_q10[:, None], row_q90[:, None])
        patterns["winsorized_logit_mean"] = (
            _sigmoid(np.mean(winsorized, axis=1)),
            "probability_geometry",
            {"space": "logit", "winsorize": [0.10, 0.90]},
            None,
        )

        for power in (0.50, 2.00):
            transformed = np.power(np.clip(p_matrix, EPS, 1.0), power).mean(axis=1)
            power_mean = np.power(np.clip(transformed, EPS, 1.0), 1.0 / power)
            patterns[f"power_mean_p{str(power).replace('.', '')}"] = (
                power_mean,
                "nonlinear_aggregation",
                {"power": power},
                None,
            )

        for temp in (0.02, 0.05, 0.10):
            weights = _prior_weights(ordered[:t], models, temp)
            patterns[f"prior_quality_t{int(temp*100):03d}"] = (
                p_matrix @ weights,
                "prior_quality_weighting",
                {"temperature": temp, "fit_policy": "strictly_prior_folds"},
                None,
            )

        prior_losses = _prior_model_losses(ordered[:t], models)
        top_k = min(2, len(models))
        top_idx = np.argsort(prior_losses)[:top_k] if len(prior_losses) else np.arange(top_k)
        top2 = p_matrix[:, top_idx].mean(axis=1)
        patterns["top2_prior_models"] = (
            top2,
            "sparse_routing",
            {"k": int(top_k), "selection": "prior_oos_logloss"},
            None,
        )

        recency_losses = _prior_recency_losses(ordered[:t], models, half_life=2.0)
        patterns["prior_recency_weighted"] = (
            p_matrix @ _softmax(-recency_losses, 0.05),
            "recency_routing",
            {"fold_half_life": 2.0},
            None,
        )

        if t:
            current_agreement = (p_matrix >= 0.5) == (equal[:, None] >= 0.5)
            row_conf = 1.0 - np.abs(p_matrix - 0.5) * 2.0
            row_weights = np.maximum(row_conf, 0.05)
            row_weights /= np.clip(row_weights.sum(axis=1, keepdims=True), EPS, None)
            patterns["row_confidence_weighted"] = (
                np.sum(p_matrix * row_weights, axis=1),
                "case_level_routing",
                {"row_weight": "distance_from_half", "floor": 0.05},
                None,
            )
            sparse_scores = np.exp(
                -np.abs(p_matrix - equal[:, None])
                / np.maximum(p_matrix.std(axis=1, keepdims=True), 0.02)
            )
            quality = _softmax(-recency_losses, 0.05)[None, :]
            joint_scores = sparse_scores * quality
            k = min(2, joint_scores.shape[1])
            idx = np.argpartition(joint_scores, -k, axis=1)[:, -k:]
            top_scores = np.take_along_axis(joint_scores, idx, axis=1)
            top_probs = np.take_along_axis(p_matrix, idx, axis=1)
            top_scores /= np.clip(top_scores.sum(axis=1, keepdims=True), EPS, None)
            patterns["sparse_consensus_top2"] = (
                np.sum(top_probs * top_scores, axis=1),
                "sparse_routing",
                {"k": int(k), "prior_quality": True, "selection": "rowwise_prior_quality_times_consensus"},
                None,
            )

        std = p_matrix.std(axis=1)
        for alpha in (0.25, 0.50, 0.75):
            shrunk = 0.5 + (equal - 0.5) * np.exp(-alpha * std / 0.10)
            patterns[f"disagreement_shrink_{alpha:.2f}"] = (
                shrunk,
                "uncertainty_shrinkage",
                {"alpha": alpha, "signal": "model_disagreement"},
                None,
            )

        completeness = 1.0 - _incomplete_score(risk)
        comp_shrink = 0.5 + (equal - 0.5) * (0.75 + 0.25 * completeness)
        patterns["information_completeness_shrink"] = (
            comp_shrink,
            "information_risk",
            {"missingness_fraction_weight": 0.25},
            None,
        )

        if t:
            prior_risks = [_risk_matrix(prev, len(np.asarray(prev.get("y", []), dtype=int))) for prev in ordered[:t]]

            prior_outcomes = np.concatenate([
                np.asarray(prev.get("y", []), dtype=int)
                for prev in ordered[:t]
                if len(np.asarray(prev.get("y", []), dtype=int))
            ]) if ordered[:t] else np.array([], dtype=int)
            prior_base_rate = float(np.mean(prior_outcomes)) if len(prior_outcomes) else 0.5
            for strength in (0.10, 0.25, 0.40):
                patterns[f"prior_outcome_base_rate_shrink_{int(strength*100)}"] = (
                    (1.0 - strength) * equal + strength * prior_base_rate,
                    "outcome_base_rate_calibration",
                    {"strength": strength, "fit_source": "strictly_prior_oos_outcomes"},
                    None,
                )
            risk_ood = _risk_ood(risk, prior_risks)
            patterns["risk_ood_shrink"] = (
                0.5 + (equal - 0.5) * np.exp(-0.75 * risk_ood),
                "ood_uncertainty",
                {"scale": 0.75, "fit_policy": "prior_risk_state_only"},
                None,
            )

            # Consensus gating: use a prior-only threshold on row disagreement.
            prior_disagreement = np.concatenate([
                _fold_probabilities(prev, models).std(axis=1)
                for prev in ordered[:t]
            ]) if ordered[:t] else np.array([])
            current_disagreement = p_matrix.std(axis=1)
            for q in (0.50, 0.75, 0.90):
                threshold = float(np.quantile(prior_disagreement, q)) if len(prior_disagreement) else 0.10
                shrink = np.clip(current_disagreement / max(threshold, 0.01), 0.0, 1.0)
                patterns[f"consensus_shrink_q{int(q*100)}"] = (
                    0.5 + (equal - 0.5) * (1.0 - 0.60 * shrink),
                    "case_level_uncertainty",
                    {"disagreement_quantile": q, "threshold_fit_source": "strictly_prior_oos"},
                    None,
                )

            # Regime × recentness: mix regime expert weights with global recent expert weights.
            reg_weights_arr = _regime_weights(
                ordered[:t],
                models,
                frame if frame is not None else {},
            )
            recent_weights = np.tile(
                _softmax(-recency_losses, 0.05)[None, :],
                (len(y), 1),
            )
            for mix in (0.10, 0.25, 0.50, 0.75, 0.90):
                mix_weights = mix * reg_weights_arr + (1.0 - mix) * recent_weights
                mix_weights /= np.clip(mix_weights.sum(axis=1, keepdims=True), EPS, None)
                patterns[f"regime_recent_mix_{int(mix*100)}"] = (
                    np.sum(p_matrix * mix_weights, axis=1),
                    "regime_recency_interaction",
                    {"regime_weight": mix, "recent_weight": 1.0 - mix, "fit_policy": "prior_oos_only"},
                    None,
                )

            # Prior failure/difficulty gate: calibrate toward 0.5 when the row is difficult.
            difficulty_scale = np.clip(difficulty, 0.0, 1.0)
            for strength in (0.20, 0.40, 0.60, 0.80):
                patterns[f"difficulty_shrink_{int(strength*100)}"] = (
                    0.5 + (equal - 0.5) * (1.0 - strength * difficulty_scale),
                    "case_level_risk_control",
                    {"strength": strength, "difficulty": "model_disagreement_plus_information"},
                    None,
                )
            reg_weights = reg_weights_arr
            patterns["regime_prior_expert"] = (
                np.sum(p_matrix * reg_weights, axis=1),
                "regime_routing",
                {"temperature": 0.05, "fit_policy": "strictly_prior_folds"},
                None,
            )

        prior_matrix, prior_y, _ = _history_rows(ordered[:t], models)
        if t:
            prior_equal_by_fold = [
                float(np.mean(_fold_probabilities(prev, models)))
                for prev in ordered[:t]
                if len(np.asarray(prev.get("y", []), dtype=int))
            ]
            if prior_equal_by_fold:
                long_run_direction = float(np.mean(prior_equal_by_fold))
                patterns["prior_base_rate_shrink"] = (
                    0.75 * equal + 0.25 * long_run_direction,
                    "base_rate_stabilization",
                    {"blend": [0.75, 0.25], "prior_only": True},
                    None,
                )
        if len(prior_y):
            prior_equal = prior_matrix.mean(axis=1)
            prior_logits = _logit(prior_matrix)
            current_X = p_matrix
            current_logit_X = _logit(p_matrix)
            for C in STACK_C_VALUES:
                patterns[f"stack_logistic_prob_C{str(C).replace('.', '')}"] = (
                    _stacker_predict(prior_matrix, prior_y, current_X, C),
                    "learned_prior_stacker",
                    {"C": C, "features": "raw_model_probabilities", "min_rows": MIN_STACK_ROWS},
                    None,
                )
                patterns[f"stack_logistic_logit_C{str(C).replace('.', '')}"] = (
                    _stacker_predict(prior_logits, prior_y, current_logit_X, C),
                    "learned_prior_stacker",
                    {"C": C, "features": "model_logits", "min_rows": MIN_STACK_ROWS},
                    None,
                )

            prior_cal_blends = {
                "equal_mean": prior_equal,
                "prior_quality": prior_matrix @ _prior_weights(ordered[:t], models, 0.05),
            }
            for source_name, prior_raw in prior_cal_blends.items():
                current_raw = (
                    equal
                    if source_name == "equal_mean"
                    else p_matrix @ _prior_weights(ordered[:t], models, 0.05)
                )
                patterns[f"platt_after_{source_name}"] = (
                    _fit_platt(prior_raw, prior_y, current_raw),
                    "post_blend_calibration",
                    {"calibrator": "platt", "base": source_name, "fit_policy": "strictly_prior_rows"},
                    None,
                )
                patterns[f"isotonic_after_{source_name}"] = (
                    _fit_isotonic(prior_raw, prior_y, current_raw),
                    "post_blend_calibration",
                    {"calibrator": "isotonic", "base": source_name, "fit_policy": "strictly_prior_rows"},
                    None,
                )
                temp_pred, best_t = _fit_temperature(prior_raw, prior_y, current_raw)
                patterns[f"temperature_after_{source_name}"] = (
                    temp_pred,
                    "post_blend_calibration",
                    {"calibrator": "temperature", "base": source_name, "selected_on_prior_rows": True, "selected_temperature": best_t},
                    None,
                )
                patterns[f"beta_after_{source_name}"] = (
                    _fit_beta(prior_raw, prior_y, current_raw),
                    "post_blend_calibration",
                    {"calibrator": "beta", "base": source_name, "fit_policy": "strictly_prior_rows"},
                    None,
                )

        if p_matrix.shape[1] >= 3:
            order = np.argsort(p_matrix, axis=1)
            low2 = np.take_along_axis(p_matrix, order[:, :2], axis=1).mean(axis=1)
            high2 = np.take_along_axis(p_matrix, order[:, -2:], axis=1).mean(axis=1)
            patterns["lower_tail_mean2"] = (
                low2,
                "tail_robust_aggregation",
                {"tail": "lower_2_models"},
                None,
            )
            patterns["upper_tail_mean2"] = (
                high2,
                "tail_robust_aggregation",
                {"tail": "upper_2_models"},
                None,
            )

            # Direction-vote probability with confidence-weighted vote strength.
            votes = (p_matrix >= 0.5).astype(float)
            vote_rate = votes.mean(axis=1)
            strength = np.mean(np.abs(p_matrix - 0.5) * 2.0, axis=1)
            patterns["vote_strength_probability"] = (
                np.clip(0.5 + (vote_rate - 0.5) * (0.60 + 0.40 * strength), EPS, 1.0 - EPS),
                "direction_vote",
                {"vote_weight": "confidence_strength"},
                None,
            )

        rank_p = _rank_percentile(p_matrix)
        patterns["rank_average_percentile"] = (
            rank_p,
            "cross_sectional_rank",
            {"mapping": "(rank+1)/(n+2)", "aggregation": "mean_model_rank"},
            None,
        )
        patterns["rank_average_logit"] = (
            _sigmoid(_logit(rank_p)),
            "cross_sectional_rank",
            {"mapping": "rank_percentile_then_logit"},
            None,
        )

        difficulty = _difficulty(p_matrix, risk)
        if t:
            prior_difficulty = np.concatenate([
                _difficulty(_fold_probabilities(prev, models), _risk_matrix(prev, len(np.asarray(prev.get("y", []), dtype=int))))
                for prev in ordered[:t]
            ])
            for q in (0.50, 0.60, 0.70, 0.75, 0.85, 0.90, 0.95, 0.975):
                threshold = float(np.quantile(prior_difficulty, q)) if len(prior_difficulty) else 1.0
                active = difficulty <= threshold
                patterns[f"selective_difficulty_q{int(q*100)}"] = (
                    equal,
                    "selective_prediction",
                    {"threshold_quantile": q, "threshold_fit_source": "strictly_prior_state_only"},
                    active,
                )

        if t >= locked_start:
            locked_y.extend(np.asarray(y, dtype=int).tolist())
            locked_sessions.extend([str(x) for x in session_ids])

        for name, (pred, category, meta, active) in patterns.items():
            row = _evaluate_pattern(
                name,
                category,
                y,
                pred,
                equal,
                session_ids,
                meta=meta,
                active=active,
            )
            row["fold"] = int(t)
            row["is_locked"] = bool(t >= locked_start)
            per_pattern.setdefault(name, []).append(row)

            if t >= locked_start:
                add_locked(name, pred, y, session_ids)

    locked_y_arr = np.asarray(locked_y, dtype=int)
    locked_session_arr = np.asarray(locked_sessions, dtype=str)
    equal_locked = np.asarray(locked_rows.get("equal_mean", []), dtype=float)

    summary = []
    for name, rows in sorted(per_pattern.items()):
        locked_metrics = [r["metrics"] for r in rows if r["is_locked"]]
        aggregate = {}
        for key in ("accuracy", "logloss", "brier", "ece"):
            vals = [float(r[key]) for r in locked_metrics if np.isfinite(r.get(key, np.nan))]
            aggregate[key] = float(np.mean(vals)) if vals else float("nan")
        candidate_locked = np.asarray(locked_rows.get(name, []), dtype=float)
        if len(candidate_locked) != len(equal_locked):
            continue
        relative = float(
            (aggregate["logloss"] - float(_metrics(locked_y_arr, equal_locked)["logloss"]))
            / max(abs(float(_metrics(locked_y_arr, equal_locked)["logloss"])), EPS)
        )
        selective_rows = [r.get("selective", {}) for r in rows if r["is_locked"] and "selective" in r]
        selective_summary = {}
        if selective_rows:
            cov = [float(r.get("coverage", float("nan"))) for r in selective_rows]
            al = [r.get("active_metrics", {}).get("logloss", float("nan")) for r in selective_rows]
            aa = [r.get("active_metrics", {}).get("accuracy", float("nan")) for r in selective_rows]
            selective_summary = {
                "coverage": float(np.nanmean(cov)) if cov else float("nan"),
                "active_logloss": float(np.nanmean(al)) if al else float("nan"),
                "active_accuracy": float(np.nanmean(aa)) if aa else float("nan"),
            }
        summary.append({
            "name": name,
            "category": rows[0]["category"],
            "locked_metrics": aggregate,
            "selective_locked": selective_summary,
            "relative_logloss_delta_vs_equal": relative,
            "locked_cases": int(len(candidate_locked)),
            "locked_fold_count": int(sum(1 for r in rows if r["is_locked"])),
            "parameters": rows[0].get("parameters", {}),
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
        })

    summary_sorted = sorted(
        summary,
        key=lambda r: (
            float(r["locked_metrics"].get("logloss", float("inf"))),
            float(r["locked_metrics"].get("brier", float("inf"))),
            float(r["locked_metrics"].get("ece", float("inf"))),
        ),
    )

    # Prequential development selection. Each development fold is scored only
    # after selecting a candidate from strictly earlier development folds.
    preq_scores = {}
    preq_decisions = []
    for eval_t in range(1, locked_start):
        eligible = {}
        for name, rows in per_pattern.items():
            prior_rows = [
                r for r in rows
                if (not r["is_locked"]) and int(r["fold"]) < eval_t
            ]
            vals = [
                float(r["logloss"])
                for r in prior_rows
                if np.isfinite(r.get("logloss", np.nan))
            ]
            if vals:
                eligible[name] = float(np.mean(vals))
        if not eligible:
            continue
        chosen = min(eligible, key=lambda name: (eligible[name], name))
        target = next(
            (r for r in per_pattern[chosen] if int(r["fold"]) == eval_t),
            None,
        )
        if target is None:
            continue
        ll = float(target["logloss"])
        if np.isfinite(ll):
            preq_scores.setdefault(chosen, []).append(ll)
            preq_decisions.append({
                "fold": int(eval_t),
                "selected_name": chosen,
                "selection_source": "strictly_prior_development_folds",
                "logloss": ll,
            })
    preq_rank = sorted(
        (
            {
                "name": name,
                "prequential_logloss": float(np.mean(vals)),
                "evaluated_folds": int(len(vals)),
            }
            for name, vals in preq_scores.items()
            if vals
        ),
        key=lambda row: (row["prequential_logloss"], row["name"]),
    )
    prequential_selected_name = preq_rank[0]["name"] if preq_rank else None

    development_candidates = []
    for name, rows in per_pattern.items():
        dev = [r["metrics"] for r in rows if not r["is_locked"]]
        if not dev:
            continue
        ll = [float(r["logloss"]) for r in dev if np.isfinite(r.get("logloss", np.nan))]
        br = [float(r["brier"]) for r in dev if np.isfinite(r.get("brier", np.nan))]
        ec = [float(r["ece"]) for r in dev if np.isfinite(r.get("ece", np.nan))]
        if not ll:
            continue
        development_candidates.append({
            "name": name,
            "development_metrics": {
                "logloss": float(np.mean(ll)),
                "brier": float(np.mean(br)) if br else float("nan"),
                "ece": float(np.mean(ec)) if ec else float("nan"),
                "folds": int(sum(1 for r in rows if not r["is_locked"])),
            },
        })
    development_sorted = sorted(
        development_candidates,
        key=lambda r: (
            r["development_metrics"]["logloss"],
            r["development_metrics"]["brier"] if np.isfinite(r["development_metrics"]["brier"]) else float("inf"),
            r["development_metrics"]["ece"] if np.isfinite(r["development_metrics"]["ece"]) else float("inf"),
            r["name"],
        ),
    )
    selected_name = prequential_selected_name or (development_sorted[0]["name"] if development_sorted else None)
    selected_locked = next((r for r in summary_sorted if r["name"] == selected_name), None)
    best_locked_diagnostic = summary_sorted[0] if summary_sorted else None

    diagnostics = {
        "oracle_best_per_case_accuracy": None,
        "oracle_best_per_case_logloss": None,
        "leave_one_model_out": {},
    }
    if len(locked_y_arr):
        locked_matrices = [_fold_probabilities(f, models) for f in ordered[locked_start:]]
        locked_matrix = np.vstack(locked_matrices)
        oracle_pred = np.clip(
            np.where(
                locked_y_arr[:, None] == 1,
                np.max(locked_matrix, axis=1),
                np.min(locked_matrix, axis=1),
            ),
            EPS,
            1.0 - EPS,
        )
        diagnostics["oracle_best_per_case_logloss"] = float(
            np.mean(-(locked_y_arr * np.log(oracle_pred) + (1 - locked_y_arr) * np.log(1 - oracle_pred)))
        )
        diagnostics["oracle_best_per_case_accuracy"] = float(
            np.mean(
                np.where(
                    locked_y_arr == 1,
                    np.max(locked_matrix, axis=1) >= 0.5,
                    np.min(locked_matrix, axis=1) < 0.5,
                )
            )
        )
        for i, model in enumerate(models):
            mask = [j for j in range(len(models)) if j != i]
            p_loo = locked_matrix[:, mask].mean(axis=1) if mask else locked_matrix[:, i]
            diagnostics["leave_one_model_out"][model] = _metrics(locked_y_arr, p_loo)

    return {
        **base,
        "status": "EXECUTED_RESEARCH_PATTERN_MATRIX",
        "evaluation_mode": "chronological_oos_same_observations",
        "models": models,
        "fold_count": len(ordered),
        "development_folds": locked_start,
        "locked_folds": locked,
        "pattern_count": len(summary_sorted),
        "patterns": summary_sorted,
        "best_research_pattern": selected_locked,
        "best_locked_diagnostic": best_locked_diagnostic,
        "selection": {
            "source": "prequential_development_only",
            "selected_name": selected_name,
            "development_ranking": development_sorted,
            "prequential_ranking": preq_rank,
            "prequential_decisions": preq_decisions,
        },
        "baseline": {
            "name": "equal_mean",
            "locked_metrics": _metrics(locked_y_arr, equal_locked) if len(equal_locked) else {},
        },
        "diagnostics": diagnostics,
        "contracts": {
            "current_fold_outcomes_used_for_pattern_tuning": False,
            "learned_patterns_fit_only_on_strictly_prior_folds": True,
            "calibration_fit_only_on_strictly_prior_rows": True,
            "selective_thresholds_fit_only_on_prior_state": True,
            "locked_outcomes_used_for_tuning": False,
            "same_locked_oos_observations_for_all_patterns": True,
            "frozen_holdout_used": False,
            "production_changed": False,
            "promotion_allowed": False,
            "locked_outcomes_used_for_selection": False,
            "best_research_pattern_selected_from_development_only": True,
            "winner_selection_is_prequential_development_only": True,
        },
    }


__all__ = ["run_frontier_pattern_suite"]
