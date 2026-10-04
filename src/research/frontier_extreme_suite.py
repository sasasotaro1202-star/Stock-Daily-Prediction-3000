from __future__ import annotations

"""Extreme breadth research suite for stock directional probability forecasts.

Every candidate is evaluated on the same chronological OOS bank.
Any learned/tuned component is fitted only on strictly prior folds/rows.

This module is research-only and never mutates production state.
"""

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.frontier_pattern_suite import (
    _ece,
    _fit_beta,
    _fit_isotonic,
    _fit_platt,
    _fold_probabilities,
    _metrics,
    _prior_model_losses,
    _prior_recency_losses,
    _safe_probability,
    _sigmoid,
    _softmax,
)

try:
    from scipy.special import ndtr, ndtri
except Exception:  # pragma: no cover - scipy is normally present through sklearn
    ndtr = None
    ndtri = None

EPS = 1e-6
MIN_META_ROWS = 100
TEMPS = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
HEDGE_RATES = (0.05, 0.10, 0.25, 0.50, 1.00)
FLOORS = (0.02, 0.05, 0.10, 0.20)


def _entropy(p: np.ndarray) -> np.ndarray:
    x = _safe_probability(p)
    return -(x * np.log(x) + (1.0 - x) * np.log(1.0 - x))


def _inverse_sigmoid_odds_mean(p_matrix: np.ndarray, power: float = 1.0) -> np.ndarray:
    p = _safe_probability(p_matrix)
    odds = p / (1.0 - p)
    avg = np.mean(np.power(odds, float(power)), axis=1)
    return _safe_probability(avg / (1.0 + avg))


def _probit_mean(p_matrix: np.ndarray, robust: bool = False) -> np.ndarray:
    if ndtri is None or ndtr is None:
        return np.mean(p_matrix, axis=1)
    p = _safe_probability(p_matrix)
    z = ndtri(p)
    if robust and z.shape[1] >= 3:
        z = np.sort(z, axis=1)[:, 1:-1]
    return _safe_probability(ndtr(np.mean(z, axis=1)))


def _arcsine_mean(p_matrix: np.ndarray) -> np.ndarray:
    p = _safe_probability(p_matrix)
    theta = np.arcsin(np.sqrt(p))
    theta_bar = np.mean(theta, axis=1)
    return _safe_probability(np.sin(theta_bar) ** 2)


def _rank_prob_hybrid(p_matrix: np.ndarray, alpha: float) -> np.ndarray:
    n = p_matrix.shape[1]
    ranks = np.argsort(np.argsort(p_matrix, axis=1), axis=1)
    rank_p = (ranks + 1.0) / (n + 2.0)
    rank_mean = rank_p.mean(axis=1)
    return _safe_probability(float(alpha) * p_matrix.mean(axis=1) + (1.0 - float(alpha)) * rank_mean)


def _prior_fold_loss_matrix(
    folds: Sequence[Mapping[str, Any]],
    models: Sequence[str],
) -> np.ndarray:
    rows = []
    for fold in folds:
        y = np.asarray(fold.get("y", []), dtype=int)
        if not len(y):
            continue
        p = _fold_probabilities(fold, models)
        rows.append(np.asarray([
            float(_metrics(y, p[:, j])["logloss"]) for j in range(len(models))
        ]))
    return np.vstack(rows) if rows else np.zeros((0, len(models)), dtype=float)


def _hedge_weights(
    folds: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    rate: float,
) -> np.ndarray:
    losses = _prior_fold_loss_matrix(folds, models)
    if losses.size == 0:
        return np.full(len(models), 1.0 / len(models))
    cumulative = np.sum(losses, axis=0)
    return _softmax(-float(rate) * cumulative, 1.0)


def _diversity_weights(
    folds: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    *,
    temperature: float,
    redundancy_penalty: float,
    floor: float,
) -> np.ndarray:
    losses = _prior_model_losses(folds, models)
    if not folds:
        return np.full(len(models), 1.0 / len(models))
    residuals = []
    for model in models:
        parts = []
        idx = models.index(model)
        for fold in folds:
            y = np.asarray(fold.get("y", []), dtype=int)
            p = _fold_probabilities(fold, models)[:, idx]
            if len(y):
                parts.extend((p - y).tolist())
        residuals.append(np.asarray(parts, dtype=float))
    corr = np.eye(len(models), dtype=float)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            if len(residuals[i]) >= 3 and len(residuals[j]) == len(residuals[i]):
                a, b = residuals[i], residuals[j]
                if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                    v = float(np.corrcoef(a, b)[0, 1])
                    v = 0.0 if not np.isfinite(v) else v
                else:
                    v = 0.0
                corr[i, j] = corr[j, i] = v
    redundancy = np.mean(np.abs(corr - np.eye(len(models))), axis=1)
    raw = np.exp(-(losses - np.min(losses)) / max(float(temperature), EPS))
    raw *= 1.0 / (1.0 + float(redundancy_penalty) * redundancy)
    raw = np.maximum(raw, float(floor))
    return raw / np.clip(raw.sum(), EPS, None)


def _model_quality_floor_weights(
    folds: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    *,
    temperature: float,
    floor: float,
) -> np.ndarray:
    losses = _prior_model_losses(folds, models)
    if not folds:
        return np.full(len(models), 1.0 / len(models))
    weights = np.exp(-(losses - np.min(losses)) / max(float(temperature), EPS))
    weights /= np.clip(weights.sum(), EPS, None)
    weights = np.maximum(weights, float(floor))
    return weights / np.clip(weights.sum(), EPS, None)


def _minimax_weights(
    folds: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    *,
    temperature: float,
) -> np.ndarray:
    losses = _prior_fold_loss_matrix(folds, models)
    if losses.size == 0:
        return np.full(len(models), 1.0 / len(models))
    worst = np.max(losses, axis=0)
    return _softmax(-worst, max(float(temperature), EPS))


def _sticky_champion(
    folds: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    current: np.ndarray,
    *,
    champion_mix: float,
) -> np.ndarray:
    if not folds:
        return current.mean(axis=1)
    losses = _prior_model_losses(folds, models)
    champion = int(np.argmin(losses))
    return _safe_probability(
        float(champion_mix) * current[:, champion]
        + (1.0 - float(champion_mix)) * current.mean(axis=1)
    )


def _individual_calibrated_matrix(
    prior_folds: Sequence[Mapping[str, Any]],
    current: np.ndarray,
    models: Sequence[str],
    method: str,
) -> np.ndarray:
    if not prior_folds:
        return current.copy()
    prior_parts = []
    prior_y_parts = []
    for fold in prior_folds:
        y = np.asarray(fold.get("y", []), dtype=int)
        if not len(y):
            continue
        prior_parts.append(_fold_probabilities(fold, models))
        prior_y_parts.append(y)
    if not prior_parts:
        return current.copy()
    prior_matrix = np.vstack(prior_parts)
    prior_y = np.concatenate(prior_y_parts)
    out = np.empty_like(current, dtype=float)
    for j in range(current.shape[1]):
        pp = prior_matrix[:, j]
        cp = current[:, j]
        if method == "platt":
            out[:, j] = _fit_platt(pp, prior_y, cp)
        elif method == "beta":
            out[:, j] = _fit_beta(pp, prior_y, cp)
        elif method == "isotonic":
            out[:, j] = _fit_isotonic(pp, prior_y, cp)
        elif method == "temperature":
            logits = np.log(_safe_probability(pp)) - np.log1p(-_safe_probability(pp))
            best_t = 1.0
            best_loss = float("inf")
            for t in TEMPS:
                pred = _sigmoid(logits / float(t))
                loss = float(_metrics(prior_y, pred)["logloss"])
                if loss < best_loss:
                    best_loss = loss
                    best_t = float(t)
            out[:, j] = _safe_probability(
                _sigmoid(
                    (np.log(_safe_probability(cp)) - np.log1p(-_safe_probability(cp)))
                    / best_t
                )
            )
        else:
            raise ValueError(f"unknown calibration method: {method}")
    return out


def _quantile_calibrate(
    prior_p: np.ndarray,
    prior_y: np.ndarray,
    current_p: np.ndarray,
    bins: int,
) -> np.ndarray:
    p = _safe_probability(prior_p)
    cp = _safe_probability(current_p)
    if len(prior_y) < max(30, bins * 4) or len(np.unique(prior_y)) < 2:
        return cp
    order = np.argsort(p)
    edges_idx = np.linspace(0, len(p), bins + 1).astype(int)
    left_values = []
    right_values = []
    targets = []
    for i in range(bins):
        lo, hi = int(edges_idx[i]), int(edges_idx[i + 1])
        if hi <= lo:
            continue
        block = order[lo:hi]
        if len(block) < 4:
            continue
        left_values.append(float(np.min(p[block])))
        right_values.append(float(np.max(p[block])))
        targets.append(float(np.mean(prior_y[block])))
    if not targets:
        return cp
    out = cp.copy()
    for lo, hi, target in zip(left_values, right_values, targets):
        mask = (out >= lo) & (out <= hi)
        out[mask] = target
    below = out < left_values[0]
    above = out > right_values[-1]
    out[below] = targets[0]
    out[above] = targets[-1]
    return _safe_probability(out)


def _base_features(p_matrix: np.ndarray, risk: np.ndarray) -> np.ndarray:
    mean_p = p_matrix.mean(axis=1)
    std_p = p_matrix.std(axis=1)
    range_p = np.ptp(p_matrix, axis=1)
    entropy = _entropy(mean_p)
    if risk.shape[1]:
        completeness = np.isfinite(risk).mean(axis=1)
        risk_mean = np.nanmean(np.where(np.isfinite(risk), risk, np.nan), axis=1)
        risk_mean = np.nan_to_num(risk_mean, nan=0.0)
    else:
        completeness = np.ones(len(p_matrix), dtype=float)
        risk_mean = np.zeros(len(p_matrix), dtype=float)
    return np.column_stack([
        mean_p,
        np.abs(mean_p - 0.5),
        std_p,
        range_p,
        entropy,
        completeness,
        risk_mean,
    ])


def _contextual_correctness_router(
    prior_folds: Sequence[Mapping[str, Any]],
    current_fold: Mapping[str, Any],
    models: Sequence[str],
    *,
    tree: bool = False,
) -> np.ndarray:
    current_y = np.asarray(current_fold.get("y", []), dtype=int)
    current = _fold_probabilities(current_fold, models)
    current_risk = np.asarray(current_fold.get("risk_matrix", np.zeros((len(current_y), 0))), dtype=float)
    current_state = _base_features(current, current_risk)
    predictions = []
    for j in range(len(models)):
        X_parts = []
        y_parts = []
        for fold in prior_folds:
            y = np.asarray(fold.get("y", []), dtype=int)
            if not len(y):
                continue
            pm = _fold_probabilities(fold, models)
            risk = np.asarray(fold.get("risk_matrix", np.zeros((len(y), 0))), dtype=float)
            state = _base_features(pm, risk)
            expert_p = pm[:, j]
            X_parts.append(np.column_stack([
                state,
                expert_p,
                np.abs(expert_p - 0.5),
                np.abs(expert_p - pm.mean(axis=1)),
            ]))
            y_parts.append(((expert_p >= 0.5) == y).astype(int))
        if not X_parts:
            predictions.append(np.full(len(current), 0.5))
            continue
        X = np.vstack(X_parts)
        yy = np.concatenate(y_parts)
        Xc = np.column_stack([
            current_state,
            current[:, j],
            np.abs(current[:, j] - 0.5),
            np.abs(current[:, j] - current.mean(axis=1)),
        ])
        if len(yy) < MIN_META_ROWS or len(np.unique(yy)) < 2 or not np.isfinite(X).all():
            prior_rate = float(np.mean(yy)) if len(yy) else 0.5
            predictions.append(np.full(len(current), np.clip(prior_rate, 0.01, 0.99)))
            continue
        if tree:
            model = ExtraTreesClassifier(
                n_estimators=120,
                max_depth=4,
                min_samples_leaf=20,
                random_state=13013,
                n_jobs=-1,
            )
        else:
            model = Pipeline([
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(
                    C=0.50,
                    max_iter=2000,
                    random_state=13013,
                )),
            ])
        model.fit(X, yy)
        predictions.append(np.clip(model.predict_proba(Xc)[:, 1], 0.01, 0.99))
    pred_correct = np.column_stack(predictions)
    weights = np.clip(pred_correct, 0.02, 0.98)
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), EPS, None)
    return np.sum(current * weights, axis=1)


def _pairwise_best(
    folds: Sequence[Mapping[str, Any]],
    current: np.ndarray,
    models: Sequence[str],
    k: int,
) -> np.ndarray:
    losses = _prior_model_losses(folds, models)
    if len(losses) == 0:
        return current.mean(axis=1)
    idx = np.argsort(losses)[:min(int(k), len(models))]
    return _safe_probability(current[:, idx].mean(axis=1))


def _best_pair_diversity(
    folds: Sequence[Mapping[str, Any]],
    current: np.ndarray,
    models: Sequence[str],
) -> np.ndarray:
    if len(models) < 2 or not folds:
        return current.mean(axis=1)
    losses = _prior_model_losses(folds, models)
    best_pair = None
    best_score = float("inf")
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            parts_i = []
            parts_j = []
            for fold in folds:
                y = np.asarray(fold.get("y", []), dtype=int)
                if not len(y):
                    continue
                pm = _fold_probabilities(fold, models)
                parts_i.extend((pm[:, i] - y).tolist())
                parts_j.extend((pm[:, j] - y).tolist())
            corr = 0.0
            if len(parts_i) >= 3 and np.std(parts_i) > 1e-12 and np.std(parts_j) > 1e-12:
                corr = abs(float(np.corrcoef(parts_i, parts_j)[0, 1]))
            score = 0.5 * (float(losses[i]) + float(losses[j])) + 0.05 * corr
            if score < best_score:
                best_score = score
                best_pair = (i, j)
    if best_pair is None:
        return current.mean(axis=1)
    return _safe_probability(current[:, list(best_pair)].mean(axis=1))


def _oracle_best_case(
    locked_folds: Sequence[Mapping[str, Any]],
    models: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    ys = []
    ps = []
    oracle = []
    for fold in locked_folds:
        y = np.asarray(fold.get("y", []), dtype=int)
        p = _fold_probabilities(fold, models)
        ys.extend(y.tolist())
        ps.extend(np.mean(p, axis=1).tolist())
        oracle.extend(np.where(y == 1, np.max(p, axis=1), np.min(p, axis=1)).tolist())
    return np.asarray(ys, dtype=int), np.asarray(oracle, dtype=float)


def run_extreme_pattern_suite(
    bank: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    locked_folds: int = 2,
    min_folds: int = 5,
    minimum_patterns: int = 80,
) -> dict[str, Any]:
    ordered = (
        [dict(bank[k]) for k in sorted(bank, key=lambda x: int(x))]
        if isinstance(bank, Mapping)
        else [dict(x) for x in bank]
    )
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

    pattern_rows: dict[str, list[dict[str, Any]]] = {}
    locked_predictions: dict[str, list[float]] = {}
    locked_y: list[int] = []
    locked_sessions: list[str] = []

    for t, fold in enumerate(ordered):
        y = np.asarray(fold.get("y", []), dtype=int)
        p_matrix = _fold_probabilities(fold, models)
        risk = np.asarray(
            fold.get("risk_matrix", np.zeros((len(y), 0))),
            dtype=float,
        )
        frame = fold.get("frame")
        if frame is not None and "session_date" in frame:
            sessions = frame["session_date"].astype(str).tolist()
        else:
            sessions = [str(t)] * len(y)
        equal = p_matrix.mean(axis=1)
        patterns: dict[str, tuple[np.ndarray, str, dict[str, Any], np.ndarray | None]] = {}

        # 1. Probability geometry.
        patterns["extreme_median"] = (
            np.median(p_matrix, axis=1),
            "geometry",
            {"aggregation": "median"},
            None,
        )
        patterns["extreme_logit_mean"] = (
            _sigmoid(np.mean(np.log(_safe_probability(p_matrix)) - np.log1p(-_safe_probability(p_matrix)), axis=1)),
            "geometry",
            {"space": "logit"},
            None,
        )
        patterns["extreme_logit_median"] = (
            _sigmoid(np.median(np.log(_safe_probability(p_matrix)) - np.log1p(-_safe_probability(p_matrix)), axis=1)),
            "geometry",
            {"space": "logit_median"},
            None,
        )
        patterns["extreme_probit_mean"] = (
            _probit_mean(p_matrix, robust=False),
            "geometry",
            {"space": "probit"},
            None,
        )
        patterns["extreme_probit_trimmed"] = (
            _probit_mean(p_matrix, robust=True),
            "geometry",
            {"space": "probit_trimmed"},
            None,
        )
        patterns["extreme_arcsine_mean"] = (
            _arcsine_mean(p_matrix),
            "geometry",
            {"space": "arcsine_sqrt"},
            None,
        )
        for power in (-1.0, -0.5, 0.5, 1.0, 2.0):
            if abs(power) < EPS:
                pred = np.exp(np.mean(np.log(_safe_probability(p_matrix)), axis=1))
            else:
                pred = np.power(np.mean(np.power(_safe_probability(p_matrix), power), axis=1), 1.0 / power)
            patterns[f"power_mean_{str(power).replace('-', 'm').replace('.', 'p')}"] = (
                pred,
                "geometry",
                {"power": power},
                None,
            )
        for power in (0.5, 1.0, 2.0):
            patterns[f"odds_power_mean_{str(power).replace('.', 'p')}"] = (
                _inverse_sigmoid_odds_mean(p_matrix, power),
                "odds_geometry",
                {"power": power},
                None,
            )

        # 2. Robust/cross-model aggregation.
        sorted_p = np.sort(p_matrix, axis=1)
        for trim in (1, 2):
            if p_matrix.shape[1] > 2 * trim:
                patterns[f"trimmed_{trim}"] = (
                    sorted_p[:, trim:-trim].mean(axis=1),
                    "robust_aggregation",
                    {"trim_each_side": trim},
                    None,
                )
        if p_matrix.shape[1] >= 4:
            patterns["middle_half"] = (
                sorted_p[:, p_matrix.shape[1] // 4 : p_matrix.shape[1] - p_matrix.shape[1] // 4].mean(axis=1),
                "robust_aggregation",
                {"middle": "central_half"},
                None,
            )
        for alpha in (0.10, 0.25, 0.50, 0.75, 0.90):
            weights = np.power(np.maximum(np.abs(p_matrix - 0.5), EPS), alpha)
            weights /= np.clip(weights.sum(axis=1, keepdims=True), EPS, None)
            patterns[f"confidence_power_{int(alpha*100)}"] = (
                np.sum(weights * p_matrix, axis=1),
                "case_level_aggregation",
                {"confidence_power": alpha},
                None,
            )
        for alpha in (0.25, 0.50, 0.75):
            patterns[f"rank_prob_hybrid_{int(alpha*100)}"] = (
                _rank_prob_hybrid(p_matrix, alpha),
                "cross_sectional_rank_hybrid",
                {"probability_weight": alpha},
                None,
            )

        # 3. Prior expert weighting: quality, recency, hedge, minimax, floors.
        for temp in (0.01, 0.02, 0.05, 0.10, 0.20):
            w = _model_quality_floor_weights(ordered[:t], models, temperature=temp, floor=0.02)
            patterns[f"quality_floor_{str(temp).replace('.', 'p')}"] = (
                p_matrix @ w,
                "prior_quality",
                {"temperature": temp, "floor": 0.02, "fit_policy": "prior_only"},
                None,
            )
        for half_life in (1.0, 2.0, 4.0, 8.0):
            losses = _prior_recency_losses(ordered[:t], models, half_life=half_life)
            w = _softmax(-losses, 0.05)
            patterns[f"recency_h{str(half_life).replace('.', 'p')}"] = (
                p_matrix @ w,
                "recency",
                {"half_life_folds": half_life},
                None,
            )
        for rate in HEDGE_RATES:
            w = _hedge_weights(ordered[:t], models, rate)
            patterns[f"hedge_{str(rate).replace('.', 'p')}"] = (
                p_matrix @ w,
                "online_expert",
                {"learning_rate": rate, "history_only": True},
                None,
            )
        for temp in (0.02, 0.05, 0.10):
            for penalty in (0.50, 1.00, 2.00):
                for floor in (0.02, 0.05):
                    w = _diversity_weights(
                        ordered[:t],
                        models,
                        temperature=temp,
                        redundancy_penalty=penalty,
                        floor=floor,
                    )
                    patterns[
                        f"diversity_t{str(temp).replace('.', 'p')}_p{int(penalty*100)}_f{int(floor*100)}"
                    ] = (
                        p_matrix @ w,
                        "diversity_routing",
                        {
                            "temperature": temp,
                            "redundancy_penalty": penalty,
                            "floor": floor,
                            "history_only": True,
                        },
                        None,
                    )
        for temp in (0.02, 0.05, 0.10, 0.20):
            w = _minimax_weights(ordered[:t], models, temperature=temp)
            patterns[f"minimax_t{str(temp).replace('.', 'p')}"] = (
                p_matrix @ w,
                "robust_expert_selection",
                {"objective": "worst_prior_fold_logloss", "temperature": temp},
                None,
            )

        for mix in (0.25, 0.50, 0.75, 0.90):
            patterns[f"sticky_champion_{int(mix*100)}"] = (
                _sticky_champion(ordered[:t], models, p_matrix, champion_mix=mix),
                "sticky_expert",
                {"champion_mix": mix},
                None,
            )
        for k in (1, 2, 3, 4):
            patterns[f"prior_top_{k}"] = (
                _pairwise_best(ordered[:t], p_matrix, models, k),
                "sparse_expert",
                {"top_k": k},
                None,
            )
        patterns["best_diverse_pair"] = (
            _best_pair_diversity(ordered[:t], p_matrix, models),
            "diverse_pair_selection",
            {"selection": "prior_loss_plus_residual_diversity"},
            None,
        )

        # 4. Individual calibration before blending.
        if t:
            for method in ("platt", "beta", "isotonic", "temperature"):
                calibrated = _individual_calibrated_matrix(
                    ordered[:t],
                    p_matrix,
                    models,
                    method,
                )
                for blend in ("equal", "quality"):
                    if blend == "equal":
                        pred = calibrated.mean(axis=1)
                    else:
                        cal_losses = []
                        for j in range(len(models)):
                            vals = []
                            for prev in ordered[:t]:
                                yy = np.asarray(prev.get("y", []), dtype=int)
                                if not len(yy):
                                    continue
                                raw = _fold_probabilities(prev, models)[:, j]
                                if method == "platt":
                                    cp = _fit_platt(
                                        np.vstack([_fold_probabilities(z, models) for z in ordered[:t]])[:, j],
                                        np.concatenate([np.asarray(z.get("y", []), dtype=int) for z in ordered[:t]]),
                                        np.vstack([_fold_probabilities(z, models) for z in ordered[:t]])[:, j],
                                    )
                                    cp = np.asarray(cp, dtype=float)
                                    vals.append(float(np.mean(-(yy * np.log(_safe_probability(raw)) + (1-yy) * np.log(1-_safe_probability(raw))))))
                                else:
                                    vals.append(float(_metrics(yy, raw)["logloss"]))
                            cal_losses.append(float(np.mean(vals)) if vals else math.log(2.0))
                        w = _softmax(-np.asarray(cal_losses), 0.05)
                        pred = calibrated @ w
                    patterns[f"individual_{method}_{blend}"] = (
                        _safe_probability(pred),
                        "individual_calibration_then_blend",
                        {"calibrator": method, "blend": blend, "fit_policy": "strictly_prior"},
                        None,
                    )

            prior_parts = []
            prior_y = []
            for prev in ordered[:t]:
                yy = np.asarray(prev.get("y", []), dtype=int)
                if len(yy):
                    prior_parts.append(_fold_probabilities(prev, models).mean(axis=1))
                    prior_y.append(yy)
            if prior_parts:
                pp = np.concatenate(prior_parts)
                yy = np.concatenate(prior_y)
                for bins in (5, 10, 20, 30):
                    pred = _quantile_calibrate(pp, yy, equal, bins)
                    patterns[f"quantile_calibration_{bins}"] = (
                        pred,
                        "quantile_calibration",
                        {"bins": bins, "fit_policy": "strictly_prior_rows"},
                        None,
                    )

                # Calibration ensemble: multiple monotone calibrators agree on current distribution.
                calibrators = [
                    _fit_platt(pp, yy, equal),
                    _fit_beta(pp, yy, equal),
                    _fit_isotonic(pp, yy, equal),
                    _sigmoid(
                        (
                            np.log(_safe_probability(equal))
                            - np.log1p(-_safe_probability(equal))
                        )
                        / 1.25
                    ),
                ]
                cal_stack = np.column_stack(calibrators)
                patterns["calibration_mean"] = (
                    _safe_probability(cal_stack.mean(axis=1)),
                    "calibration_ensemble",
                    {"members": ["platt", "beta", "isotonic", "temperature_1p25"]},
                    None,
                )
                patterns["calibration_median"] = (
                    _safe_probability(np.median(cal_stack, axis=1)),
                    "calibration_ensemble",
                    {"members": ["platt", "beta", "isotonic", "temperature_1p25"], "aggregation": "median"},
                    None,
                )
                if cal_stack.shape[1] >= 3:
                    patterns["calibration_trimmed"] = (
                        np.sort(cal_stack, axis=1)[:, 1:-1].mean(axis=1),
                        "calibration_ensemble",
                        {"trim_each_side": 1},
                        None,
                    )

        # 5. Contextual routers fit only to prior cases.
        if t:
            patterns["contextual_correctness_logistic"] = (
                _contextual_correctness_router(ordered[:t], fold, models, tree=False),
                "contextual_expert_router",
                {"target": "expert_direction_correctness", "learner": "logistic", "fit_policy": "strictly_prior"},
                None,
            )
            patterns["contextual_correctness_tree"] = (
                _contextual_correctness_router(ordered[:t], fold, models, tree=True),
                "contextual_expert_router",
                {"target": "expert_direction_correctness", "learner": "extra_trees", "fit_policy": "strictly_prior"},
                None,
            )

        # 6. Selective prediction / abstention variants.
        disagreement = p_matrix.std(axis=1)
        margin = np.abs(equal - 0.5)
        completeness = (
            np.isfinite(risk).mean(axis=1)
            if risk.shape[1]
            else np.ones(len(y), dtype=float)
        )
        if t:
            prior_disagreement = np.concatenate([
                _fold_probabilities(prev, models).std(axis=1)
                for prev in ordered[:t]
                if len(np.asarray(prev.get("y", []), dtype=int))
            ])
            prior_margin = np.concatenate([
                np.abs(_fold_probabilities(prev, models).mean(axis=1) - 0.5)
                for prev in ordered[:t]
                if len(np.asarray(prev.get("y", []), dtype=int))
            ])
            prior_missing = np.concatenate([
                1.0 - (
                    np.isfinite(
                        np.asarray(
                            prev.get("risk_matrix", np.zeros((len(np.asarray(prev.get("y", []), dtype=int)), 0))),
                            dtype=float,
                        )
                    ).mean(axis=1)
                    if np.asarray(
                        prev.get("risk_matrix", np.zeros((len(np.asarray(prev.get("y", []), dtype=int)), 0))),
                        dtype=float,
                    ).shape[1]
                    else np.ones(len(np.asarray(prev.get("y", []), dtype=int)))
                )
                for prev in ordered[:t]
                if len(np.asarray(prev.get("y", []), dtype=int))
            ])
            for q in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
                d_thr = float(np.quantile(prior_disagreement, q)) if len(prior_disagreement) else 1.0
                active = disagreement <= d_thr
                patterns[f"selective_disagreement_q{int(q*100)}"] = (
                    equal,
                    "selective_prediction",
                    {"signal": "model_disagreement", "threshold_quantile": q, "prior_only": True},
                    active,
                )
            for q in (0.50, 0.60, 0.70, 0.80, 0.90):
                m_thr = float(np.quantile(prior_margin, q)) if len(prior_margin) else 0.5
                active = margin >= m_thr
                patterns[f"selective_margin_q{int(q*100)}"] = (
                    equal,
                    "selective_prediction",
                    {"signal": "prediction_margin", "threshold_quantile": q, "prior_only": True},
                    active,
                )
            for q in (0.80, 0.90, 0.95):
                if len(prior_missing):
                    miss_thr = float(np.quantile(prior_missing, q))
                    active = (1.0 - completeness) <= miss_thr
                    patterns[f"selective_missing_q{int(q*100)}"] = (
                        equal,
                        "selective_prediction",
                        {"signal": "missingness", "threshold_quantile": q, "prior_only": True},
                        active,
                    )

        # 7. Cross-sectional direction/rank patterns.
        vote_rate = (p_matrix >= 0.5).mean(axis=1)
        for strength in (0.25, 0.50, 0.75, 1.00):
            pred = _safe_probability(
                0.5 + (vote_rate - 0.5) * (
                    strength + (1.0 - strength) * np.clip(2.0 * margin, 0.0, 1.0)
                )
            )
            patterns[f"vote_strength_{int(strength*100)}"] = (
                pred,
                "direction_vote",
                {"strength": strength},
                None,
            )

        for rank_weight in (0.25, 0.50, 0.75):
            rank = np.argsort(np.argsort(p_matrix, axis=1), axis=1)
            rank_p = (rank + 1.0) / (p_matrix.shape[1] + 2.0)
            patterns[f"rank_blend_{int(rank_weight*100)}"] = (
                _safe_probability(
                    (1.0 - rank_weight) * equal
                    + rank_weight * rank_p.mean(axis=1)
                ),
                "cross_sectional_rank",
                {"rank_weight": rank_weight},
                None,
            )

        # 8. Prior champion/base-rate and robust state shrinkage.
        prior_y = np.concatenate([
            np.asarray(prev.get("y", []), dtype=int)
            for prev in ordered[:t]
            if len(np.asarray(prev.get("y", []), dtype=int))
        ]) if t else np.array([], dtype=int)
        if len(prior_y):
            base_rate = float(np.mean(prior_y))
            for strength in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
                patterns[f"base_rate_shrink_{int(strength*100)}"] = (
                    _safe_probability((1.0 - strength) * equal + strength * base_rate),
                    "base_rate_stabilization",
                    {"strength": strength, "prior_only": True},
                    None,
                )

        # Register every pattern on the current fold.
        if t >= locked_start:
            locked_y.extend(y.tolist())
            locked_sessions.extend(sessions)

        for name, (pred, category, meta, active) in patterns.items():
            p = _safe_probability(pred)
            row = {
                "name": name,
                "fold": int(t),
                "category": category,
                "metrics": _metrics(y, p),
                "parameters": dict(meta),
                "research_only": True,
                "production_changed": False,
                "promotion_allowed": False,
                "is_locked": bool(t >= locked_start),
            }
            if active is not None:
                mask = np.asarray(active, dtype=bool)
                active_metrics = _metrics(y[mask], p[mask]) if mask.any() else _metrics(np.array([], dtype=int), np.array([]))
                row["selective"] = {
                    "coverage": float(mask.mean()) if len(mask) else 0.0,
                    "active_metrics": active_metrics,
                    "abstain_count": int((~mask).sum()),
                }
            pattern_rows.setdefault(name, []).append(row)
            if t >= locked_start:
                locked_predictions.setdefault(name, []).extend(p.tolist())

    y_locked = np.asarray(locked_y, dtype=int)
    session_locked = np.asarray(locked_sessions, dtype=str)
    summaries = []
    baseline_ll = float(_metrics(y_locked, np.asarray(locked_predictions.get("extreme_median", [0.5] * len(y_locked)), dtype=float))["logloss"]) if len(y_locked) else float("nan")

    for name, rows in sorted(pattern_rows.items()):
        locked_rows = [r for r in rows if r["is_locked"]]
        pred = np.asarray(locked_predictions.get(name, []), dtype=float)
        if len(pred) != len(y_locked):
            continue
        metrics = _metrics(y_locked, pred)
        rel = (
            float((metrics["logloss"] - baseline_ll) / max(abs(baseline_ll), EPS))
            if np.isfinite(baseline_ll) and np.isfinite(metrics["logloss"])
            else float("nan")
        )
        selective = [r["selective"] for r in locked_rows if "selective" in r]
        selective_summary = {}
        if selective:
            selective_summary = {
                "coverage": float(np.mean([s["coverage"] for s in selective])),
                "active_accuracy": float(np.nanmean([
                    s["active_metrics"]["accuracy"] for s in selective
                ])),
                "active_logloss": float(np.nanmean([
                    s["active_metrics"]["logloss"] for s in selective
                ])),
                "active_brier": float(np.nanmean([
                    s["active_metrics"]["brier"] for s in selective
                ])),
                "active_ece": float(np.nanmean([
                    s["active_metrics"]["ece"] for s in selective
                ])),
            }
        summaries.append({
            "name": name,
            "category": rows[0]["category"],
            "locked_metrics": metrics,
            "relative_logloss_delta_vs_extreme_median": rel,
            "selective_locked": selective_summary,
            "locked_cases": int(len(pred)),
            "locked_fold_count": int(len(locked_rows)),
            "parameters": rows[0]["parameters"],
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
        })

    summaries.sort(
        key=lambda r: (
            float(r["locked_metrics"].get("logloss", float("inf"))),
            float(r["locked_metrics"].get("brier", float("inf"))),
            float(r["locked_metrics"].get("ece", float("inf"))),
        )
    )

    locked_folds_data = ordered[locked_start:]
    oracle_y, oracle_p = _oracle_best_case(locked_folds_data, models)
    diagnostics: dict[str, Any] = {
        "oracle_best_case_logloss": float(_metrics(oracle_y, _safe_probability(oracle_p))["logloss"]) if len(oracle_y) else None,
        "oracle_best_case_accuracy": float(np.mean(((oracle_p >= 0.5) == oracle_y))) if len(oracle_y) else None,
        "leave_one_model_out": {},
        "model_count": len(models),
    }
    if len(oracle_y):
        matrix = np.vstack([_fold_probabilities(f, models) for f in locked_folds_data])
        for j, model in enumerate(models):
            keep = [k for k in range(len(models)) if k != j]
            p = matrix[:, keep].mean(axis=1) if keep else matrix[:, j]
            diagnostics["leave_one_model_out"][model] = _metrics(oracle_y, p)

    pattern_count = len(summaries)
    status = "EXECUTED_EXTREME_PATTERN_MATRIX" if pattern_count >= int(minimum_patterns) else "BLOCKED_INSUFFICIENT_PATTERN_BREADTH"
    best = summaries[0] if summaries else None
    return {
        **base,
        "status": status,
        "evaluation_mode": "chronological_oos_same_locked_cases",
        "models": models,
        "fold_count": len(ordered),
        "development_folds": int(locked_start),
        "locked_folds": int(locked),
        "pattern_count": int(pattern_count),
        "minimum_pattern_count": int(minimum_patterns),
        "patterns": summaries,
        "best_research_pattern": best,
        "baseline": {
            "name": "extreme_median",
            "locked_metrics": _metrics(
                y_locked,
                np.asarray(locked_predictions.get("extreme_median", [0.5] * len(y_locked)), dtype=float),
            ) if len(y_locked) else {},
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
            "external_method_transfer": {
                "FAME_like_sparse_routing": "mechanism_inspired_only_not_directly_adopted",
                "expert_loss_integration": "mechanism_inspired_only_not_directly_adopted",
                "time_series_conformal_dependence": "evaluation_inspired_only_not_claimed_as_guarantee",
            },
        },
        "session_cluster": {
            "cluster_unit": "session_date",
            "n_clusters": int(len(np.unique(session_locked))) if len(session_locked) else 0,
        },
    }


__all__ = ["run_extreme_pattern_suite"]
