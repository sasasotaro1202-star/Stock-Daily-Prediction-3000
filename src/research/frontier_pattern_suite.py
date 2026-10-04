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
    weights = np.exp(-np.log(2.0) * ages / max(half_life, EPS))
    weights /= np.clip(weights.sum(), EPS, None)
    out = []
    for model in models:
        vals = []
        for fold, w in zip(folds, weights):
            y = np.asarray(fold.get("y", []), dtype=int)
            p = _fold_probabilities(fold, models)[:, models.index(model)]
            if len(y):
                vals.append(float(_metrics(y, p)["logloss"]) * float(w))
        out.append(float(sum(vals) / max(sum(weights[-len(vals):]) if vals else 1.0, EPS)))
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

    def add_locked(name: str, p: np.ndarray, y: np.ndarray, session_ids: Sequence[Any]) -> None:
        locked_rows.setdefault(name, []).extend(np.asarray(p, dtype=float).tolist())
        if not locked_y:
            locked_y.extend(np.asarray(y, dtype=int).tolist())
            locked_sessions.extend([str(x) for x in session_ids])
        else:
            locked_y.extend(np.asarray(y, dtype=int).tolist())
            locked_sessions.extend([str(x) for x in session_ids])

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

        patterns["geometric_odds_mean"] = (
            _sigmoid(np.mean(_logit(p_matrix), axis=1)),
            "probability_geometry",
            {"space": "log_odds_geometric"},
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
            sparse_scores = np.exp(-np.abs(p_matrix - equal[:, None]) / np.maximum(p_matrix.std(axis=1, keepdims=True), 0.02))
            sparse_weights = sparse_scores * _softmax(-recency_losses, 0.05)[None, :]
            sparse_weights /= np.clip(sparse_weights.sum(axis=1, keepdims=True), EPS, None)
            patterns["sparse_consensus_top2"] = (
                np.sum(p_matrix * sparse_weights, axis=1),
                "sparse_routing",
                {"k": 2, "prior_quality": True, "consensus_scale": 0.02},
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
            reg_weights = _regime_weights(ordered[:t], models, frame if frame is not None else {})
            patterns["regime_prior_expert"] = (
                np.sum(p_matrix * reg_weights, axis=1),
                "regime_routing",
                {"temperature": 0.05, "fit_policy": "strictly_prior_folds"},
                None,
            )

        prior_matrix, prior_y, _ = _history_rows(ordered[:t], models)
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
            for q in (0.75, 0.85, 0.90):
                threshold = float(np.quantile(prior_difficulty, q)) if len(prior_difficulty) else 1.0
                active = difficulty <= threshold
                patterns[f"selective_difficulty_q{int(q*100)}"] = (
                    equal,
                    "selective_prediction",
                    {"threshold_quantile": q, "threshold_fit_source": "strictly_prior_state_only"},
                    active,
                )

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
        summary.append({
            "name": name,
            "category": rows[0]["category"],
            "locked_metrics": aggregate,
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

    best = summary_sorted[0] if summary_sorted else None
    best_name = best["name"] if best else None

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
        "best_research_pattern": best,
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
        },
    }


__all__ = ["run_frontier_pattern_suite"]
