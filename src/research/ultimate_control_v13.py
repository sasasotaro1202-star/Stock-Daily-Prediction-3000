from __future__ import annotations

"""Research-only v13 predictive control plane.

This module consumes a chronological OOS prediction bank produced by the v6
research runner. It adds PIT-safe, history-only control signals for:
- model disagreement
- predictability / future predictability proxy
- future model failure / time-to-failure
- regime transition
- historical prototype/failure retrieval
- uncertainty decomposition
- OOD / extrapolation risk
- strategy and prediction-action selection
- prediction trajectory / revision policy
- adaptive compute and selective prediction

No production artifact is modified. Missing provenance required for a stronger
claim is represented as BLOCKED/UNAVAILABLE rather than inferred.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6
STRATEGIES = ("standard", "ensemble", "retrieval", "risk_control", "abstain")
ACTIONS = ("maintain", "micro_revision", "recompute", "deep_recompute", "abstain")

STATE_COLS = (
    "mean_probability",
    "std_probability",
    "probability_range",
    "prediction_entropy",
    "agreement",
    "data_completeness",
    "feature_reliability",
    "feature_drift",
    "information_shock",
)


def safe_probability(p: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.size == 0:
        return arr.astype(float)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite probability")
    return np.clip(arr, EPS, 1.0 - EPS)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=int)
    p = safe_probability(p)
    if len(y) == 0:
        return float("nan")
    out = 0.0
    for lo, hi in zip(
        np.linspace(0.0, 1.0, bins + 1)[:-1],
        np.linspace(0.0, 1.0, bins + 1)[1:],
    ):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(out)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

    y = np.asarray(y, dtype=int)
    p = safe_probability(p)
    if len(y) == 0:
        return {"accuracy": float("nan"), "logloss": float("nan"), "brier": float("nan"), "ece": float("nan")}
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece(y, p)),
    }


def block_bootstrap_ci(
    values: Iterable[float],
    *,
    block_length: int = 3,
    draws: int = 1000,
    seed: int = 13013,
) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return ((float(x[0]), float(x[0])) if len(x) else (float("nan"), float("nan")))
    rng = np.random.default_rng(seed)
    block_length = min(max(1, int(block_length)), len(x))
    starts = np.arange(len(x) - block_length + 1)
    means = []
    for _ in range(draws):
        sample: list[float] = []
        while len(sample) < len(x):
            s = int(rng.choice(starts))
            sample.extend(x[s : s + block_length].tolist())
        means.append(float(np.mean(sample[: len(x)])))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _state_from_probabilities(
    probabilities: np.ndarray,
    risk: np.ndarray,
    history_risk: np.ndarray | None,
) -> pd.DataFrame:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("need at least two base models")
    safe_probability(p)
    mean_p = p.mean(axis=1)
    std_p = p.std(axis=1)
    cls = p >= 0.5
    agreement = np.maximum(cls.mean(axis=1), 1.0 - cls.mean(axis=1))
    entropy = -(safe_probability(mean_p) * np.log(safe_probability(mean_p))
                + (1.0 - safe_probability(mean_p)) * np.log(1.0 - safe_probability(mean_p)))

    if risk.ndim != 2 or risk.shape[0] != len(p):
        raise ValueError("invalid risk context")
    completeness = np.isfinite(risk).mean(axis=1) if risk.shape[1] else np.ones(len(p))
    completeness = np.clip(completeness, 0.0, 1.0)

    if history_risk is None or history_risk.size == 0:
        drift = np.zeros(len(p), dtype=float)
    else:
        ref = np.asarray(history_risk, dtype=float)
        if ref.ndim != 2 or ref.shape[1] != risk.shape[1]:
            drift = np.zeros(len(p), dtype=float)
        else:
            ref_med = np.nanmedian(ref, axis=0)
            ref_sd = np.nanstd(ref, axis=0)
            ref_sd[~np.isfinite(ref_sd) | (ref_sd < 1e-6)] = 1.0
            cur_med = np.nanmedian(risk, axis=0)
            drift = np.full(len(p), float(
                np.mean(np.clip(np.abs(cur_med - ref_med) / ref_sd, 0.0, 8.0) / 8.0)
            ))

    feature_reliability = np.clip(completeness * (1.0 - 0.5 * drift), 0.0, 1.0)
    if risk.shape[1] >= 3:
        gap = np.abs(np.nan_to_num(risk[:, 2], nan=0.0))
        volume = np.nan_to_num(risk[:, 1], nan=0.0)
        shock = np.maximum((gap - 0.03) / 0.05, np.maximum((volume - 3.0) / 6.0, 0.0))
        shock = np.clip(shock, 0.0, 1.0)
    else:
        shock = np.zeros(len(p), dtype=float)

    return pd.DataFrame({
        "mean_probability": mean_p,
        "std_probability": std_p,
        "probability_range": p.max(axis=1) - p.min(axis=1),
        "prediction_entropy": entropy,
        "agreement": agreement,
        "data_completeness": completeness,
        "feature_reliability": feature_reliability,
        "feature_drift": drift,
        "information_shock": shock,
    })


def _row_ood(current: pd.DataFrame, history: list[pd.DataFrame]) -> np.ndarray:
    cols = [c for c in STATE_COLS if c in current.columns]
    if not history:
        return np.zeros(len(current), dtype=float)
    hist = pd.concat(history, ignore_index=True)[cols].to_numpy(dtype=float)
    cur = current[cols].to_numpy(dtype=float)
    med = np.nanmedian(hist, axis=0)
    sd = np.nanstd(hist, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    z = (np.nan_to_num(cur, nan=0.0) - med) / sd
    return np.clip(np.sqrt(np.mean(z * z, axis=1)) / 4.0, 0.0, 1.0)


def _history_quality_weights(folds: list[dict], models: list[str]) -> np.ndarray:
    if not folds:
        return np.full(len(models), 1.0 / len(models))
    losses = []
    for name in models:
        values = []
        for fold in folds:
            p = np.asarray(fold["predictions"][name], dtype=float)
            y = np.asarray(fold["y"], dtype=int)
            values.append(metrics(y, p)["logloss"])
        losses.append(float(np.mean(values)) if values else math.log(2.0))
    x = np.exp(-(np.asarray(losses) - np.min(losses)) / 0.10)
    x = np.maximum(x, 0.03)
    return x / x.sum()


def _dynamic_routing_weights(
    probabilities: np.ndarray,
    history_quality: np.ndarray,
) -> np.ndarray:
    """Build causal row-wise soft-routing weights from current predictions and prior-only quality."""
    p = np.asarray(probabilities, dtype=float)
    q = safe_probability(np.asarray(history_quality, dtype=float))
    if p.ndim != 2 or p.shape[1] != len(q):
        raise ValueError("invalid routing inputs")
    mean_p = p.mean(axis=1, keepdims=True)
    disagreement = np.abs(p - mean_p)
    scale = np.maximum(p.std(axis=1, keepdims=True), 0.01)
    consensus = np.exp(-disagreement / scale)
    raw = consensus * q[None, :]
    denom = np.sum(raw, axis=1, keepdims=True)
    return raw / np.clip(denom, EPS, None)



def _failure_aware_routing_weights(
    current_weights: np.ndarray,
    failure_risks: Mapping[str, float],
    models: list[str],
) -> np.ndarray:
    """Downweight models by prior-only estimated future failure risk."""
    w = np.asarray(current_weights, dtype=float)
    if w.ndim != 2 or w.shape[1] != len(models):
        raise ValueError("invalid current routing weights")
    survival = np.asarray(
        [
            1.0 - float(np.clip(failure_risks.get(m, 0.5), 0.0, 1.0))
            for m in models
        ],
        dtype=float,
    )
    # Keep a non-zero floor: the failure predictor is uncertain and must not
    # collapse the portfolio to one model.
    survival = np.maximum(survival, 0.15)
    adjusted = w * survival[None, :]
    return adjusted / np.clip(adjusted.sum(axis=1, keepdims=True), EPS, None)


def _fit_failure_predictor(
    x_hist: list[np.ndarray],
    y_hist: list[int],
    x_current: np.ndarray,
) -> float:
    if not y_hist:
        return 0.5
    X = np.asarray(x_hist, dtype=float)
    y = np.asarray(y_hist, dtype=int)
    current = np.asarray(x_current, dtype=float).reshape(1, -1)
    if len(y) < 4 or len(np.unique(y)) < 2 or X.ndim != 2:
        return float(np.mean(y))
    if X.shape[1] != current.shape[1] or not np.isfinite(X).all() or not np.isfinite(current).all():
        return float(np.mean(y))
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=0.35, max_iter=2000, random_state=13013)),
    ])
    model.fit(X, y)
    return float(np.clip(model.predict_proba(current)[:, 1][0], 0.0, 1.0))



def _fit_meta_label_predictor(
    x_hist: list[np.ndarray],
    y_hist: list[np.ndarray],
    x_current: np.ndarray,
) -> np.ndarray:
    """Predict individual prediction correctness using only prior-fold outcomes."""
    current = np.asarray(x_current, dtype=float)
    if current.ndim != 2:
        raise ValueError("invalid meta-label current features")
    if not y_hist:
        return np.full(len(current), 0.5, dtype=float)
    X = np.vstack([np.asarray(x, dtype=float) for x in x_hist if len(x)])
    y = np.concatenate([np.asarray(v, dtype=int) for v in y_hist if len(v)])
    if len(y) < 20 or len(np.unique(y)) < 2 or X.ndim != 2 or X.shape[1] != current.shape[1]:
        return np.full(len(current), float(np.mean(y)) if len(y) else 0.5, dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(current).all():
        return np.full(len(current), float(np.mean(y)), dtype=float)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=0.5, max_iter=2000, random_state=13013)),
    ])
    model.fit(X, y)
    return np.clip(model.predict_proba(current)[:, 1], 0.0, 1.0)


def _failure_label(cur_m: dict[str, float], nxt_m: dict[str, float]) -> int:
    return int(
        (nxt_m["accuracy"] < cur_m["accuracy"] - 0.03)
        or (nxt_m["logloss"] > cur_m["logloss"] + 0.02)
        or (nxt_m["brier"] > cur_m["brier"] + 0.008)
        or (nxt_m["ece"] > cur_m["ece"] + 0.02)
    )


def _severity(risk: float) -> str:
    if risk >= 0.90:
        return "critical"
    if risk >= 0.75:
        return "failure"
    if risk >= 0.50:
        return "warning"
    return "normal"


def _regime_transition(
    current: str,
    historical_modes: list[tuple[str, str]],
    regimes: list[str],
) -> dict[str, float]:
    counts = {r: 1.0 for r in regimes}
    for left, right in historical_modes:
        if left == current:
            counts[right] = counts.get(right, 1.0) + 1.0
    total = sum(counts.values())
    return {k: float(v / total) for k, v in counts.items()}


def _retrieval_success(
    current: pd.DataFrame,
    history_frames: list[pd.DataFrame],
    history_success: list[np.ndarray],
    k: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    if not history_frames:
        return np.full(len(current), 0.5), np.full(len(current), 0.5)
    cols = [c for c in STATE_COLS if c in current.columns]
    hist = pd.concat(history_frames, ignore_index=True)[cols].to_numpy(dtype=float)
    success = np.concatenate(history_success).astype(float)
    if len(hist) != len(success) or len(hist) == 0:
        return np.full(len(current), 0.5), np.full(len(current), 0.5)
    med = np.nanmedian(hist, axis=0)
    sd = np.nanstd(hist, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    h = (np.nan_to_num(hist, nan=0.0) - med) / sd
    c = (np.nan_to_num(current[cols].to_numpy(dtype=float), nan=0.0) - med) / sd

    out_success = np.empty(len(c), dtype=float)
    out_failure = np.empty(len(c), dtype=float)
    chunk = 256
    for start in range(0, len(c), chunk):
        block = c[start : start + chunk]
        d = np.sum((block[:, None, :] - h[None, :, :]) ** 2, axis=2)
        kk = min(k, d.shape[1])
        idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        dist = d[np.arange(len(block))[:, None], idx]
        w = 1.0 / np.clip(dist + 1e-6, 1e-6, None)
        vals = success[idx]
        s = np.sum(w * vals, axis=1) / np.sum(w, axis=1)
        out_success[start : start + len(block)] = s
        out_failure[start : start + len(block)] = 1.0 - s
    return np.clip(out_success, 0.0, 1.0), np.clip(out_failure, 0.0, 1.0)


def _policy(
    predictability: np.ndarray,
    ood: np.ndarray,
    failure: np.ndarray,
    retrieval_success: np.ndarray,
) -> np.ndarray:
    out = np.empty(len(predictability), dtype=object)
    for i in range(len(out)):
        if (
            predictability[i] < 0.25
            or ood[i] >= 0.85
            or failure[i] >= 0.82
        ):
            out[i] = "abstain"
        elif (
            predictability[i] < 0.45
            or ood[i] >= 0.55
            or failure[i] >= 0.60
        ):
            out[i] = "risk_control"
        elif retrieval_success[i] >= 0.70:
            out[i] = "retrieval"
        elif predictability[i] >= 0.70 and ood[i] < 0.30:
            out[i] = "ensemble"
        else:
            out[i] = "standard"
    return out


def _select_probability(
    strategy: str,
    p_matrix: np.ndarray,
    quality: np.ndarray,
    retrieval_success: np.ndarray,
) -> np.ndarray:
    quality_arr = np.asarray(quality, dtype=float)
    if quality_arr.ndim == 1:
        base = np.sum(p_matrix * quality_arr[None, :], axis=1)
    elif quality_arr.ndim == 2:
        if quality_arr.shape != p_matrix.shape:
            raise ValueError("invalid row-wise routing weight shape")
        base = np.sum(p_matrix * quality_arr, axis=1)
    else:
        raise ValueError("invalid routing weight rank")
    if strategy == "standard":
        return safe_probability(p_matrix[:, 0])
    if strategy == "ensemble":
        return safe_probability(base)
    if strategy == "retrieval":
        return safe_probability(0.80 * base + 0.20 * retrieval_success)
    if strategy == "risk_control":
        return safe_probability(0.70 * base + 0.30 * 0.5)
    if strategy == "abstain":
        return safe_probability(base)
    raise ValueError(f"unknown strategy: {strategy}")




def _error_correlation_diagnostics(
    fold_banks: list[dict],
    models: list[str],
) -> dict[str, object]:
    """Retrospective error-diversity diagnostics; never used for current-fold routing."""
    residuals: dict[str, list[float]] = {m: [] for m in models}
    errors: dict[str, list[int]] = {m: [] for m in models}
    for fold in fold_banks:
        y = np.asarray(fold["y"], dtype=int)
        for name in models:
            p = safe_probability(np.asarray(fold["predictions"][name], dtype=float))
            residuals[name].extend((p - y).tolist())
            errors[name].extend(((p >= 0.5).astype(int) != y).astype(int).tolist())

    corr: dict[str, dict[str, float]] = {m: {} for m in models}
    overlap: dict[str, dict[str, float]] = {m: {} for m in models}
    pairs: list[float] = []
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if a == b:
                corr[a][b] = 1.0
                overlap[a][b] = 1.0
                continue
            av = np.asarray(residuals[a], dtype=float)
            bv = np.asarray(residuals[b], dtype=float)
            if len(av) >= 3 and np.std(av) > 1e-12 and np.std(bv) > 1e-12:
                value = float(np.corrcoef(av, bv)[0, 1])
                corr[a][b] = value if np.isfinite(value) else 0.0
            else:
                corr[a][b] = 0.0
            ea = np.asarray(errors[a], dtype=bool)
            eb = np.asarray(errors[b], dtype=bool)
            union = float(np.sum(ea | eb))
            overlap[a][b] = float(np.sum(ea & eb) / union) if union else 1.0
        for b in models[i + 1:]:
            pairs.append(float(corr[a][b]))
    return {
        "status": "EXECUTED_RETROSPECTIVE_DIAGNOSTIC",
        "current_fold_routing_uses": False,
        "residual_correlation": corr,
        "error_overlap_jaccard": overlap,
        "mean_pairwise_error_correlation": float(np.mean(pairs)) if pairs else 0.0,
    }


def _failure_calibration(
    fold_banks: list[dict],
    models: list[str],
    risk_by_model: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Evaluate one-step-ahead failure risk against the next-fold outcome, retrospectively."""
    out: dict[str, dict[str, float]] = {}
    for name in models:
        labels: list[int] = []
        risks: list[float] = []
        for t in range(len(fold_banks) - 1):
            cur = metrics(
                np.asarray(fold_banks[t]["y"], dtype=int),
                np.asarray(fold_banks[t]["predictions"][name], dtype=float),
            )
            nxt = metrics(
                np.asarray(fold_banks[t + 1]["y"], dtype=int),
                np.asarray(fold_banks[t + 1]["predictions"][name], dtype=float),
            )
            labels.append(_failure_label(cur, nxt))
            risks.append(float(risk_by_model[name][t]))
        if not labels:
            out[name] = {
                "horizon_folds": 1.0,
                "observations": 0.0,
                "brier": float("nan"),
                "ece": float("nan"),
                "positive_rate": float("nan"),
                "mean_risk": float("nan"),
            }
            continue
        y = np.asarray(labels, dtype=int)
        p = safe_probability(np.asarray(risks, dtype=float))
        out[name] = {
            "horizon_folds": 1.0,
            "observations": float(len(y)),
            "brier": float(np.mean((p - y) ** 2)),
            "ece": float(ece(y, p)),
            "positive_rate": float(np.mean(y)),
            "mean_risk": float(np.mean(p)),
        }
    return out


def _time_to_failure_evaluation(
    fold_banks: list[dict],
    models: list[str],
    predicted_ttf: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Compare fold-ahead TTF proxy with first observed future failure; censored cases are excluded from MAE."""
    out: dict[str, dict[str, float]] = {}
    for name in models:
        fold_metrics = [
            metrics(
                np.asarray(fold["y"], dtype=int),
                np.asarray(fold["predictions"][name], dtype=float),
            )
            for fold in fold_banks
        ]
        failures = [
            _failure_label(fold_metrics[i], fold_metrics[i + 1])
            for i in range(len(fold_metrics) - 1)
        ]
        observed: list[float] = []
        predicted: list[float] = []
        censored = 0
        for t in range(len(fold_metrics) - 1):
            next_failure = None
            for k in range(t, len(failures)):
                if failures[k]:
                    next_failure = k + 1 - t
                    break
            if next_failure is None:
                censored += 1
                continue
            observed.append(float(next_failure))
            predicted.append(float(predicted_ttf[name][t]))
        if observed:
            errors = np.abs(np.asarray(predicted) - np.asarray(observed))
            out[name] = {
                "status": "EXECUTED_RETROSPECTIVE_PROXY_EVAL",
                "event_cases": float(len(observed)),
                "censored_cases": float(censored),
                "event_rate": float(len(observed) / max(len(observed) + censored, 1)),
                "predicted_mean": float(np.mean(predicted)),
                "observed_mean": float(np.mean(observed)),
                "mae_on_event_cases": float(np.mean(errors)),
                "median_abs_error_on_event_cases": float(np.median(errors)),
            }
        else:
            out[name] = {
                "status": "INSUFFICIENT_FAILURE_EVENTS",
                "event_cases": 0.0,
                "censored_cases": float(censored),
                "event_rate": 0.0,
                "predicted_mean": float(np.mean(predicted_ttf[name])) if predicted_ttf[name] else float("nan"),
                "observed_mean": float("nan"),
                "mae_on_event_cases": float("nan"),
                "median_abs_error_on_event_cases": float("nan"),
            }
    return out


@dataclass
class FoldResult:
    fold: int
    metrics: dict[str, dict[str, float]]
    chosen_strategy: dict[str, int]
    chosen_action: dict[str, int]
    coverage: float
    revision_accuracy: float
    false_revision: float
    predictability_mean: float
    future_predictability_mean: float
    ood_mean: float
    max_failure_risk: float
    mean_time_to_failure: float
    regime_transition: dict[str, float]
    uncertainty_means: dict[str, float]


def evaluate_v13(
    fold_banks: list[dict],
    *,
    locked_folds: int = 2,
    min_folds: int = 5,
) -> dict:
    if len(fold_banks) < min_folds:
        return {
            "schema_version": "v13.0",
            "status": "BLOCKED",
            "reason": f"need_at_least_{min_folds}_chronological_folds",
            "production_changed": False,
            "promotion": "HOLD",
        }

    ordered = list(fold_banks)
    common = set(ordered[0]["predictions"])
    for fold in ordered[1:]:
        common &= set(fold["predictions"])
    models = sorted(common)
    if len(models) < 2:
        return {
            "schema_version": "v13.0",
            "status": "BLOCKED",
            "reason": "need_at_least_2_common_models",
            "production_changed": False,
            "promotion": "HOLD",
        }

    locked_folds = min(max(1, int(locked_folds)), len(ordered) - 1)
    dev_end = len(ordered) - locked_folds

    history_states: list[pd.DataFrame] = []
    history_success: list[np.ndarray] = []
    history_modes: list[tuple[str, str]] = []
    metrics_by_strategy = {s: [] for s in STRATEGIES}
    chosen_rows: list[pd.DataFrame] = []
    fold_results: list[dict] = []
    per_model_failure_risk: dict[str, list[float]] = {m: [] for m in models}
    per_model_ttf: dict[str, list[float]] = {m: [] for m in models}
    failure_labels_history: dict[str, list[int]] = {m: [] for m in models}
    failure_features_history: dict[str, list[np.ndarray]] = {m: [] for m in models}
    meta_features_history: list[np.ndarray] = []
    meta_labels_history: list[np.ndarray] = []
    previous_selected: dict[str, float] = {}
    revision_rows = []

    for t, fold in enumerate(ordered):
        y = np.asarray(fold["y"], dtype=int)
        p_matrix = np.column_stack([
            np.asarray(fold["predictions"][m], dtype=float) for m in models
        ])
        risk_cols = sorted([c for c in fold["frame"].columns if str(c).startswith("risk_")])
        risk = (
            fold["frame"][risk_cols].to_numpy(dtype=float)
            if risk_cols else np.zeros((len(y), 0), dtype=float)
        )
        if len(y) != len(p_matrix) or not np.isfinite(p_matrix).all():
            raise ValueError(f"invalid OOS bank fold {t}")

        previous_risk = (
            np.vstack([x["risk_matrix"] for x in ordered[:t]])
            if t and all("risk_matrix" in x for x in ordered[:t])
            else None
        )
        state = _state_from_probabilities(p_matrix, risk, previous_risk)
        ood = _row_ood(state, history_states)

        # Predictability is target-free at the current fold.
        predictability = np.clip(
            0.28 * state["agreement"].to_numpy()
            + 0.22 * (1.0 - state["prediction_entropy"].to_numpy() / math.log(2.0))
            + 0.20 * state["data_completeness"].to_numpy()
            + 0.20 * state["feature_reliability"].to_numpy()
            + 0.10 * (1.0 - ood),
            0.0, 1.0,
        )
        if history_states:
            hist_pred = [
                float(np.mean(df["_predictability"]))
                for df in history_states
                if "_predictability" in df
            ]
            velocity = hist_pred[-1] - hist_pred[-2] if len(hist_pred) >= 2 else 0.0
        else:
            velocity = 0.0
        future_predictability = np.clip(
            predictability + float(np.clip(velocity, -0.15, 0.15)) * 0.75,
            0.0,
            1.0,
        )

        current_regime = str(pd.Series(fold["frame"]["regime"]).mode().iloc[0]) if "regime" in fold["frame"] else "unknown"
        regimes = sorted({
            str(x)
            for b in ordered
            if "regime" in b["frame"]
            for x in b["frame"]["regime"].astype(str).tolist()
        }) or ["unknown"]
        transition = _regime_transition(current_regime, history_modes, regimes)
        future_regime = max(transition.items(), key=lambda x: x[1])[0]

        quality = _history_quality_weights(ordered[:t], models)
        routing_weights = _dynamic_routing_weights(p_matrix, quality)
        dynamic_ensemble = safe_probability(np.sum(p_matrix * routing_weights, axis=1))
        equal_weight = safe_probability(np.mean(p_matrix, axis=1))
        retrieval_success, retrieval_failure = _retrieval_success(
            state,
            history_states,
            history_success,
        )

        state["_predictability"] = predictability
        state["ood"] = ood
        failure_feature = state[
            [
                "mean_probability",
                "std_probability",
                "prediction_entropy",
                "data_completeness",
                "feature_reliability",
                "feature_drift",
                "information_shock",
                "_predictability",
                "ood",
            ]
        ].to_numpy(dtype=float)

        # Persist compact target-free summaries before they can be used by
        # later failure-prediction folds. These summaries never include current outcomes.
        state["summary_state"] = float(np.mean(predictability))
        state["summary_pred"] = float(np.mean(p_matrix))
        state["summary_drift"] = float(np.mean(state["feature_drift"]))

        failure_risks: dict[str, float] = {}
        ttf: dict[str, float] = {}
        for name in models:
            cur_metrics = metrics(y, p_matrix[:, models.index(name)])
            next_ready = []
            next_labels = []
            for k in range(max(0, t - 8), t - 1):
                if k + 1 >= len(ordered):
                    continue
                km = metrics(
                    np.asarray(ordered[k]["y"], dtype=int),
                    np.asarray(ordered[k]["predictions"][name], dtype=float),
                )
                nm = metrics(
                    np.asarray(ordered[k + 1]["y"], dtype=int),
                    np.asarray(ordered[k + 1]["predictions"][name], dtype=float),
                )
                # history_states[k] is the state snapshot produced at fold k.
                # Current/next-fold outcomes are never part of these features.
                if k >= len(history_states):
                    continue
                prior_state = history_states[k]
                next_ready.append(np.asarray([
                    float(prior_state["summary_state"].iloc[0]
                          if isinstance(prior_state["summary_state"], pd.Series)
                          else prior_state["summary_state"]),
                    float(prior_state["summary_pred"].iloc[0]
                          if isinstance(prior_state["summary_pred"], pd.Series)
                          else prior_state["summary_pred"]),
                    float(prior_state["summary_drift"].iloc[0]
                          if isinstance(prior_state["summary_drift"], pd.Series)
                          else prior_state["summary_drift"]),
                ]))
                next_labels.append(_failure_label(km, nm))
            x_base = np.asarray([
                float(np.mean(predictability)),
                float(state["mean_probability"].mean()),
                float(state["feature_drift"].mean()),
            ])
            risk_value = _fit_failure_predictor(
                next_ready,
                next_labels,
                x_base,
            ) if next_ready else 0.5
            if next_labels:
                empirical = float(np.average(
                    next_labels,
                    weights=np.linspace(0.5, 1.0, len(next_labels)),
                ))
                risk_value = float(0.5 * risk_value + 0.5 * empirical)
            failure_risks[name] = float(np.clip(risk_value, 0.0, 1.0))
            ttf[name] = float(1.0 / max(failure_risks[name], 0.05))
            per_model_failure_risk[name].append(failure_risks[name])
            per_model_ttf[name].append(ttf[name])
            failure_labels_history[name].extend(next_labels)
            failure_features_history[name].extend(next_ready)

        max_failure = max(failure_risks.values(), default=0.5)
        failure_aware_weights = _failure_aware_routing_weights(
            routing_weights, failure_risks, models
        )
        failure_aware_ensemble = safe_probability(
            np.sum(p_matrix * failure_aware_weights, axis=1)
        )
        meta_features = np.column_stack([
            dynamic_ensemble,
            state["std_probability"].to_numpy(dtype=float),
            np.abs(dynamic_ensemble - 0.5),
            predictability,
            ood,
            state["data_completeness"].to_numpy(dtype=float),
            state["feature_reliability"].to_numpy(dtype=float),
            np.full(len(y), max_failure, dtype=float),
            retrieval_success,
        ])
        meta_score = _fit_meta_label_predictor(
            meta_features_history,
            meta_labels_history,
            meta_features,
        )
        meta_active = (meta_score >= 0.60) & (ood < 0.85)
        future_failure_aware_ensemble = safe_probability(
            np.sum(p_matrix * failure_aware_weights, axis=1)
        )
        strategy = _policy(
            predictability,
            ood,
            np.full(len(y), max_failure),
            retrieval_success,
        )

        # Evaluate all policies without using current outcomes for policy selection.
        per_strategy_preds = {}
        for s in STRATEGIES:
            per_strategy_preds[s] = _select_probability(
                s, p_matrix, routing_weights, retrieval_success
            )
            met = metrics(y, per_strategy_preds[s])
            if s == "abstain":
                active = (predictability >= 0.35) & (ood < 0.80)
            else:
                active = np.ones(len(y), dtype=bool)
            active_metrics = metrics(y[active], per_strategy_preds[s][active]) if active.any() else {
                "accuracy": float("nan"), "logloss": float("nan"), "brier": float("nan"), "ece": float("nan")
            }
            metrics_by_strategy[s].append({
                **met,
                "coverage": float(np.mean(active)),
                "active_accuracy": active_metrics["accuracy"],
                "active_logloss": active_metrics["logloss"],
                "active_brier": active_metrics["brier"],
                "active_ece": active_metrics["ece"],
            })

        selected_p = np.empty(len(y), dtype=float)
        active = np.ones(len(y), dtype=bool)
        for s in STRATEGIES:
            mask = strategy == s
            selected_p[mask] = per_strategy_preds[s][mask]
        active[strategy == "abstain"] = False

        # Prediction update/revision policy is state-only plus previous prediction.
        symbols = fold["frame"].get("symbol", pd.Series([""] * len(y))).astype(str).to_numpy()
        chosen_action = np.empty(len(y), dtype=object)
        output_p = selected_p.copy()
        update_need = np.clip(
            0.35 * state["probability_range"].to_numpy()
            + 0.25 * state["information_shock"].to_numpy()
            + 0.20 * (1.0 - state["feature_reliability"].to_numpy())
            + 0.20 * np.abs(selected_p - 0.5) * 0.0,
            0.0, 1.0,
        )
        for i, sym in enumerate(symbols):
            prev = previous_selected.get(sym, float(selected_p[i]))
            movement = abs(float(selected_p[i]) - prev)
            need = float(np.clip(update_need[i] + movement, 0.0, 1.0))
            if not active[i]:
                chosen_action[i] = "abstain"
            elif need < 0.25:
                chosen_action[i] = "maintain"
                output_p[i] = prev
            elif need < 0.45:
                chosen_action[i] = "micro_revision"
                output_p[i] = 0.75 * prev + 0.25 * selected_p[i]
            elif need < 0.72:
                chosen_action[i] = "recompute"
                output_p[i] = selected_p[i]
            else:
                chosen_action[i] = "deep_recompute"
                output_p[i] = 0.85 * selected_p[i] + 0.15 * 0.5

        changed = np.abs(output_p - selected_p) > 1e-9
        revised = np.asarray(changed & active, dtype=bool)
        baseline = equal_weight
        base_correct = (baseline >= 0.5) == y
        revised_correct = (output_p >= 0.5) == y
        revision_accuracy = float(np.mean(revised_correct[revised])) if revised.any() else float("nan")
        false_revision = (
            float(np.mean(revised & base_correct & ~revised_correct) / np.mean(revised))
            if revised.any() else float("nan")
        )
        for i, sym in enumerate(symbols):
            previous_selected[sym] = float(output_p[i])

        meta_metrics = metrics(y[meta_active], failure_aware_ensemble[meta_active]) if meta_active.any() else {
            "accuracy": float("nan"), "logloss": float("nan"), "brier": float("nan"), "ece": float("nan")
        }
        meta_label_rows = {
            "mean": float(np.mean(meta_score)),
            "min": float(np.min(meta_score)) if len(meta_score) else float("nan"),
            "max": float(np.max(meta_score)) if len(meta_score) else float("nan"),
            "coverage": float(np.mean(meta_active)),
            "active_metrics": meta_metrics,
        }

        # Persist row-level state for history-only retrieval/failure modeling.
        history_states.append(state.copy())
        history_success.append(((baseline >= 0.5) == y).astype(int))
        meta_features_history.append(meta_features.copy())
        meta_labels_history.append(((dynamic_ensemble >= 0.5) == y).astype(int))
        if t > 0:
            prev_mode = str(pd.Series(ordered[t - 1]["frame"]["regime"]).mode().iloc[0])
            history_modes.append((prev_mode, current_regime))

        fold_metric = {
            "standard": metrics(y, per_strategy_preds["standard"]),
            "ensemble": metrics(y, per_strategy_preds["ensemble"]),
            "retrieval": metrics(y, per_strategy_preds["retrieval"]),
            "risk_control": metrics(y, per_strategy_preds["risk_control"]),
            "selected_policy": metrics(y[active], output_p[active]) if active.any() else {
                "accuracy": float("nan"), "logloss": float("nan"), "brier": float("nan"), "ece": float("nan")
            },
        }
        chosen_counts = {
            s: int(np.sum(strategy == s)) for s in STRATEGIES
        }
        action_counts = {
            a: int(np.sum(chosen_action == a)) for a in ACTIONS
        }

        fold_results.append({
            "fold": int(t),
            "is_locked": bool(t >= dev_end),
            "current_regime": current_regime,
            "future_regime": future_regime,
            "future_regime_probabilities": transition,
            "metrics": fold_metric,
            "routing": {
                "weight_means": {
                    m: float(np.mean(routing_weights[:, i])) for i, m in enumerate(models)
                },
                "weight_stds": {
                    m: float(np.std(routing_weights[:, i])) for i, m in enumerate(models)
                },
                "weight_concentration": float(np.mean(np.max(routing_weights, axis=1))),
                "weight_entropy": float(np.mean(-np.sum(routing_weights * np.log(np.clip(routing_weights, EPS, 1.0)), axis=1))),
                "dynamic_vs_equal": metrics(y, dynamic_ensemble),
                "equal_weight": metrics(y, equal_weight),
                "future_failure_aware": metrics(y, future_failure_aware_ensemble),
                "future_failure_weight_means": {
                    m: float(np.mean(failure_aware_weights[:, i])) for i, m in enumerate(models)
                },
            },
            "disagreement": {
                "probability_mean": float(np.mean(p_matrix)),
                "probability_std_mean": float(np.mean(np.std(p_matrix, axis=1))),
                "probability_range_mean": float(np.mean(np.ptp(p_matrix, axis=1))),
                "probability_entropy_mean": float(np.mean(state["prediction_entropy"])),
                "agreement_mean": float(np.mean(state["agreement"])),
                "max_row_std": float(np.max(np.std(p_matrix, axis=1))),
                "max_row_range": float(np.max(np.ptp(p_matrix, axis=1))),
            },
            "chosen_strategy_counts": chosen_counts,
            "chosen_action_counts": action_counts,
            "chosen_strategy": strategy.astype(str).tolist(),
            "chosen_action": chosen_action.astype(str).tolist(),
            "coverage": float(np.mean(active)),
            "revision_accuracy": revision_accuracy,
            "false_revision": false_revision,
            "predictability_mean": float(np.mean(predictability)),
            "future_predictability_mean": float(np.mean(future_predictability)),
            "ood_mean": float(np.mean(ood)),
            "max_failure_risk": float(max_failure),
            "meta_label": {**meta_label_rows, "scores": meta_score.astype(float).tolist()},
            "failure_severity": _severity(max_failure),
            "model_failure": {
                m: {
                    "risk": failure_risks[m],
                    "time_to_failure_folds": ttf[m],
                    "severity": _severity(failure_risks[m]),
                }
                for m in models
            },
            "prediction_trajectory": {
                "mean_probability": float(np.mean(output_p)),
                "velocity_proxy": float(np.mean(output_p - selected_p)),
                "future_1": float(np.clip(np.mean(output_p) + (np.mean(output_p) - 0.5) * 0.25, 0.001, 0.999)),
                "future_2": float(np.clip(np.mean(output_p) + (np.mean(output_p) - 0.5) * 0.15, 0.001, 0.999)),
                "revision_risk": float(np.mean(update_need)),
            },
            "uncertainty": {
                "model": float(np.mean(state["std_probability"])),
                "distribution_shift": float(np.mean(ood)),
                "information": float(np.mean(1.0 - state["data_completeness"])),
                "irreducible_proxy": float(np.mean(state["prediction_entropy"])),
            },
            "future_failure_time_proxy": {
                "median_folds": float(np.median(list(ttf.values()))) if ttf else float("nan"),
                "min_folds": float(np.min(list(ttf.values()))) if ttf else float("nan"),
            },
        })



    error_correlation = _error_correlation_diagnostics(ordered, models)
    failure_calibration = _failure_calibration(ordered, models, per_model_failure_risk)
    ttf_evaluation = _time_to_failure_evaluation(ordered, models, per_model_ttf)

    locked = [x for x in fold_results if x["is_locked"]]
    selected_rows = [x["metrics"]["selected_policy"] for x in locked]
    baseline_rows = [x["metrics"]["ensemble"] for x in locked]
    def mean_metric(rows, key):
        vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    selected_summary = {k: mean_metric(selected_rows, k) for k in ("accuracy", "logloss", "brier", "ece")}
    baseline_summary = {k: mean_metric(baseline_rows, k) for k in ("accuracy", "logloss", "brier", "ece")}
    delta = {
        k: float(selected_summary[k] - baseline_summary[k])
        for k in selected_summary
    }

    routing_dynamic_rows = [x["routing"]["dynamic_vs_equal"] for x in locked]
    routing_equal_rows = [x["routing"]["equal_weight"] for x in locked]
    routing_dynamic = {
        k: mean_metric(routing_dynamic_rows, k)
        for k in ("accuracy", "logloss", "brier", "ece")
    }
    routing_equal = {
        k: mean_metric(routing_equal_rows, k)
        for k in ("accuracy", "logloss", "brier", "ece")
    }
    routing_failure_aware_rows = [x["routing"]["future_failure_aware"] for x in locked]
    routing_failure_aware = {
        k: mean_metric(routing_failure_aware_rows, k)
        for k in ("accuracy", "logloss", "brier", "ece")
    }
    relative_logloss_improvement = float(
        (routing_equal["logloss"] - routing_dynamic["logloss"])
        / max(abs(routing_equal["logloss"]), EPS)
    )

    strategy_summary = {}
    for s, rows in metrics_by_strategy.items():
        # Fold order is chronological, so the locked suffix is the deterministic locked sample.
        locked_rows = rows[-locked_folds:]
        strategy_summary[s] = {
            "accuracy": mean_metric(locked_rows, "accuracy"),
            "logloss": mean_metric(locked_rows, "logloss"),
            "brier": mean_metric(locked_rows, "brier"),
            "ece": mean_metric(locked_rows, "ece"),
            "coverage": mean_metric(locked_rows, "coverage"),
            "active_accuracy": mean_metric(locked_rows, "active_accuracy"),
        }

    acc_deltas = [
        x["metrics"]["selected_policy"]["accuracy"] - x["metrics"]["ensemble"]["accuracy"]
        for x in locked
    ]
    ll_deltas = [
        x["metrics"]["selected_policy"]["logloss"] - x["metrics"]["ensemble"]["logloss"]
        for x in locked
    ]
    br_deltas = [
        x["metrics"]["selected_policy"]["brier"] - x["metrics"]["ensemble"]["brier"]
        for x in locked
    ]
    ece_deltas = [
        x["metrics"]["selected_policy"]["ece"] - x["metrics"]["ensemble"]["ece"]
        for x in locked
    ]

    meta_locked = [x.get("meta_label", {}) for x in locked]
    meta_summary = {
        "coverage": mean_metric(meta_locked, "coverage"),
        "active_accuracy": mean_metric(
            [{"active_accuracy": r.get("active_metrics", {}).get("accuracy", float("nan"))} for r in meta_locked],
            "active_accuracy",
        ),
        "active_logloss": mean_metric(
            [{"active_logloss": r.get("active_metrics", {}).get("logloss", float("nan"))} for r in meta_locked],
            "active_logloss",
        ),
        "active_brier": mean_metric(
            [{"active_brier": r.get("active_metrics", {}).get("brier", float("nan"))} for r in meta_locked],
            "active_brier",
        ),
        "active_ece": mean_metric(
            [{"active_ece": r.get("active_metrics", {}).get("ece", float("nan"))} for r in meta_locked],
            "active_ece",
        ),
    }

    # Provenance is intentionally strict: this bank does not carry full event/publication/
    # retrieval timestamps, so v13 does not upgrade PIT status on its own.
    pit_status = "BLOCKED_NO_FULL_TIMESTAMP_LINEAGE"
    leakage_status = "BLOCKED_V13_LAYER_REQUIRES_UPSTREAM_INDEPENDENT_AUDIT"
    meta_status = "BLOCKED_INDEPENDENT_META_LEAKAGE_AUDIT_REQUIRED"

    return {
        "schema_version": "v13.0",
        "status": "OOS_COMPLETE",
        "evaluation_mode": "CHRONOLOGICAL_OOS_CONTROL_PLANE",
        "production_changed": False,
        "promotion": "HOLD",
        "models": models,
        "development_folds": int(dev_end),
        "locked_folds": int(locked_folds),
        "production_isolation": True,
        "core_three_layers": {
            "model_disagreement": "IMPLEMENTED_EXECUTED",
            "predictability": "IMPLEMENTED_EXECUTED",
            "future_model_failure": "IMPLEMENTED_EXECUTED",
            "time_to_failure": "RESEARCH_PROXY_EXECUTED",
        },
        "predictability": {
            "status": "EXECUTED",
            "future_proxy": True,
            "calibration_status": "PENDING_OUTCOME_STREAM",
        },
        "future_failure": {
            "status": "EXECUTED",
            "per_model": per_model_failure_risk,
            "severity_scale": ["normal", "warning", "failure", "critical"],
            "calibration_1step": failure_calibration,
        },
        "time_to_failure": {
            "status": "EXECUTED_PROXY",
            "unit": "chronological_OOS_folds",
            "per_model": per_model_ttf,
            "retrospective_evaluation": ttf_evaluation,
            "interpretation": "1/risk remains a proxy; evaluation reports event-censored agreement with the observed first future failure",
        },
        "regime_transition": {
            "status": "EXECUTED",
            "future_regime_is_predictive_proxy": True,
        },
        "error_correlation": error_correlation,
        "retrieval": {
            "status": "EXECUTED_HISTORY_ONLY",
            "success_and_failure": True,
            "index_contains_current_fold": False,
        },
        "uncertainty": {
            "status": "EXECUTED",
            "components": ["model", "distribution_shift", "information", "irreducible_proxy"],
        },
        "ood": {
            "status": "EXECUTED",
            "method": "history_standardized_state_distance",
            "fail_closed_thresholds": {"warning": 0.55, "abstain": 0.85},
        },
        "prediction_strategy": {
            "status": "EXECUTED",
            "pool": list(STRATEGIES),
            "selection_is_outcome_free_current_fold": True,
        },
        "prediction_output": {
            "status": "EXECUTED",
            "actions": list(ACTIONS),
            "trajectory": True,
            "revision_policy": True,
            "validity_decay_proxy": True,
        },
        "adaptive_compute": {
            "status": "EXECUTED_PROXY",
            "levels": ["standard", "risk_control", "deep_recompute", "abstain"],
        },
        "active_information": {
            "status": "BLOCKED_NO_SOURCE_VALUE_OF_INFORMATION_METADATA",
            "policy": "FAIL_CLOSED",
        },
        "shadow": "NOT_EXECUTED",
        "challenger": "RESEARCH_CANDIDATE_ONLY",
        "metrics_locked": {
            "baseline_ensemble": baseline_summary,
            "selected_policy": selected_summary,
            "delta_selected_minus_baseline": delta,
        },
        "strategy_summary_locked": strategy_summary,
        "meta_label": {
            "status": "EXECUTED_PRIOR_ONLY_INDIVIDUAL_PREDICTION_META_LABEL",
            "threshold": 0.60,
            "locked_summary": meta_summary,
        },
        "statistical_validation": {
            "method": "moving_block_bootstrap_on_locked_fold_deltas",
            "delta_accuracy_ci95": block_bootstrap_ci(acc_deltas),
            "delta_logloss_ci95": block_bootstrap_ci(ll_deltas),
            "delta_brier_ci95": block_bootstrap_ci(br_deltas),
            "delta_ece_ci95": block_bootstrap_ci(ece_deltas),
            "fold_count": len(locked),
        },
        "revision_metrics": {
            "revision_accuracy": float(np.nanmean([x["revision_accuracy"] for x in locked])) if locked else float("nan"),
            "false_revision": float(np.nanmean([x["false_revision"] for x in locked])) if locked else float("nan"),
        },
        "worst_case_locked": {
            "worst_accuracy_delta": float(np.min(acc_deltas)) if acc_deltas else float("nan"),
            "worst_logloss_delta": float(np.max(ll_deltas)) if ll_deltas else float("nan"),
            "worst_brier_delta": float(np.max(br_deltas)) if br_deltas else float("nan"),
            "worst_ece_delta": float(np.max(ece_deltas)) if ece_deltas else float("nan"),
        },
        "routing": {
            "status": "EXECUTED_ROW_WISE_SOFT_ROUTING",
            "folds": fold_results,
            "aggregate": {
                "dynamic": routing_dynamic,
                "equal_weight": routing_equal,
                "delta_dynamic_minus_equal": {
                    k: float(routing_dynamic[k] - routing_equal[k])
                    for k in routing_dynamic
                },
                "relative_logloss_improvement": relative_logloss_improvement,
                "future_failure_aware": routing_failure_aware,
                "delta_failure_aware_minus_dynamic": {
                    k: float(routing_failure_aware[k] - routing_dynamic[k])
                    for k in routing_failure_aware
                },
            },
            "locked_fold_weight_concentration_mean": float(np.mean([x["routing"]["weight_concentration"] for x in locked])),
            "locked_fold_weight_entropy_mean": float(np.mean([x["routing"]["weight_entropy"] for x in locked])),
        },
        "performance_success": False,
        "performance_success_reason": "promotion-blocked until full PIT/meta-leakage/nested-OOS/robustness/statistical evidence is independently verified",
        "drift_monitor": {
            "feature_drift_mean": float(np.mean([x["uncertainty"]["distribution_shift"] for x in locked])) if locked else float("nan"),
            "ood_mean": float(np.mean([x["ood_mean"] for x in locked])) if locked else float("nan"),
        },
        "robustness": {
            "status": "EXECUTED_DESCRIPTIVE_NOT_STRESS_TESTED",
            "stress_dimensions": [
                "model_disagreement",
                "feature_missingness",
                "distribution_shift",
                "information_shock",
                "ood",
                "prediction_revision",
            ],
        },
        "fold_results": fold_results,
        "audits": {
            "PIT": pit_status,
            "Leakage": leakage_status,
            "Meta-Leakage": meta_status,
            "OOS": "PASS",
            "Nested_OOS": "PENDING_INDEPENDENT_NESTED_SELECTION",
            "Reproducibility": "PARTIAL_DETERMINISTIC_SEED_ONLY",
            "Artifact_Integrity": "PENDING_ARTIFACT_STEP",
            "Fallback": "IMPLEMENTED_RESEARCH_ONLY",
            "Rollback": "NOT_MUTATED_PRODUCTION",
        },
        "decision": {
            "candidate": True,
            "promotion_allowed": False,
            "reason": "v13 is research-only; full PIT lineage, shadow, and production challenger evidence are still required",
        },
    }
