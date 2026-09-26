from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-12
UNIFORM_LOGLOSS = float(np.log(2.0))
COVERAGE_LEVELS = (1.00, 0.95, 0.90, 0.80, 0.70)

STATE_COLUMNS = [
    "mean_probability",
    "median_probability",
    "std_probability",
    "min_probability",
    "max_probability",
    "probability_range",
    "prediction_entropy",
    "top_class_agreement_rate",
    "majority_margin",
    "rank_disagreement",
    "pairwise_class_disagreement",
    "js_divergence",
    "l1_disagreement",
    "l2_disagreement",
    "cosine_disagreement",
    "disagreement_velocity",
    "disagreement_acceleration",
    "disagreement_spike",
    "prediction_flip_count",
    "prediction_velocity",
    "prediction_acceleration",
    "data_completeness",
    "feature_reliability",
    "feature_drift",
    "information_shock",
    "model_uncertainty",
    "distribution_shift_uncertainty",
    "information_uncertainty",
    "regime_transition_uncertainty",
]


def safe_probability(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("non-finite probability")
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def entropy_binary(p: np.ndarray) -> np.ndarray:
    q = safe_probability(p)
    return -(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))


def multiclass_js_binary(p: np.ndarray) -> np.ndarray:
    q = np.stack([1.0 - p, p], axis=-1)
    mean = np.mean(q, axis=1)
    # Vectorized stable JS:
    m = mean
    kl = np.sum(q * (np.log(np.clip(q, EPS, 1.0)) - np.log(np.clip(m[:, None, :], EPS, 1.0))), axis=2)
    return 0.5 * np.mean(kl, axis=1)


def disagreement_features(probabilities: np.ndarray) -> pd.DataFrame:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("need >=2 model probabilities")
    if not np.isfinite(p).all():
        raise ValueError("probability matrix contains NaN/Inf")
    p = np.clip(p, 0.0, 1.0)

    mean_p = p.mean(axis=1)
    median_p = np.median(p, axis=1)
    std_p = p.std(axis=1)
    min_p = p.min(axis=1)
    max_p = p.max(axis=1)
    range_p = max_p - min_p
    cls = p >= 0.5
    agreement_frac = cls.mean(axis=1)
    majority_margin = np.abs(2.0 * agreement_frac - 1.0)

    ranks = np.argsort(np.argsort(p, axis=1), axis=1)
    rank_disagreement = ranks.std(axis=1) / max(float(p.shape[1] - 1), 1.0)

    pair_sum = np.zeros(len(p), dtype=float)
    pair_count = 0
    class_pair = np.zeros(len(p), dtype=float)
    for i in range(p.shape[1]):
        for j in range(i + 1, p.shape[1]):
            pair_sum += np.abs(p[:, i] - p[:, j])
            class_pair += (cls[:, i] != cls[:, j]).astype(float)
            pair_count += 1
    pairwise = pair_sum / max(pair_count, 1)
    pairwise_class = class_pair / max(pair_count, 1)

    mean_dist = 2.0 * np.stack([1.0 - mean_p, mean_p], axis=1)
    q = np.stack([1.0 - p, p], axis=2)
    m = np.mean(q, axis=1)
    kl = np.sum(
        q * (
            np.log(np.clip(q, EPS, 1.0))
            - np.log(np.clip(m[:, None, :], EPS, 1.0))
        ),
        axis=2,
    )
    js = 0.5 * np.mean(kl, axis=1)

    l1 = pairwise
    l2 = np.sqrt(np.mean((p - mean_p[:, None]) ** 2, axis=1))
    dot = np.sum(p * mean_p[:, None], axis=1)
    norm_p = np.sqrt(np.sum(p * p, axis=1))
    norm_m = np.sqrt(np.sum(mean_p * mean_p, axis=1))
    cosine = 1.0 - dot / np.clip(norm_p * norm_m, 1e-9, None)

    return pd.DataFrame(
        {
            "mean_probability": mean_p,
            "median_probability": median_p,
            "std_probability": std_p,
            "min_probability": min_p,
            "max_probability": max_p,
            "probability_range": range_p,
            "prediction_entropy": entropy_binary(mean_p),
            "top_class_agreement_rate": np.maximum(agreement_frac, 1.0 - agreement_frac),
            "majority_margin": majority_margin,
            "rank_disagreement": rank_disagreement,
            "pairwise_class_disagreement": pairwise_class,
            "js_divergence": js,
            "l1_disagreement": l1,
            "l2_disagreement": l2,
            "cosine_disagreement": cosine,
            "prediction_flip_count": np.zeros(len(p), dtype=float),
            "prediction_velocity": np.zeros(len(p), dtype=float),
            "prediction_acceleration": np.zeros(len(p), dtype=float),
            "disagreement_velocity": np.zeros(len(p), dtype=float),
            "disagreement_acceleration": np.zeros(len(p), dtype=float),
            "disagreement_spike": np.zeros(len(p), dtype=float),
        }
    )


def add_prediction_dynamics(
    current: pd.DataFrame,
    probabilities: np.ndarray,
    symbols: np.ndarray,
    previous_by_symbol: Mapping[str, Mapping[str, float]],
    previous_velocity_by_symbol: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    p = np.asarray(probabilities, dtype=float)
    out = current.copy()
    mean_p = p.mean(axis=1)
    prev_mean = np.empty(len(p), dtype=float)
    prev_velocity = np.zeros(len(p), dtype=float)
    flips = np.zeros(len(p), dtype=float)
    next_prev: dict[str, dict[str, float]] = {}
    next_vel: dict[str, dict[str, float]] = {}
    flip_state: dict[str, dict[str, int]] = {}

    for i, sym in enumerate(symbols.astype(str)):
        hist = previous_by_symbol.get(sym, {})
        last = float(hist.get("mean_probability", mean_p[i]))
        vel = float(previous_velocity_by_symbol.get(sym, {}).get("velocity", 0.0))
        v = mean_p[i] - last
        a = v - vel
        flip = int((last - 0.5) * (mean_p[i] - 0.5) < 0.0)
        prev_mean[i] = last
        prev_velocity[i] = vel
        flips[i] = float(flip)
        next_prev[sym] = {"mean_probability": float(mean_p[i])}
        next_vel[sym] = {"velocity": float(v)}
        flip_state[sym] = {"flip": flip}

    out["prediction_flip_count"] = flips
    out["prediction_velocity"] = mean_p - prev_mean
    out["prediction_acceleration"] = out["prediction_velocity"].to_numpy() - prev_velocity

    disagreement = out["std_probability"].to_numpy(dtype=float)
    # Current fold is evaluated in chronological order, so previous row is PIT-safe.
    if len(out) > 1:
        dv = np.diff(disagreement, prepend=disagreement[0])
        da = np.diff(dv, prepend=dv[0])
    else:
        dv = np.zeros(len(out), dtype=float)
        da = np.zeros(len(out), dtype=float)
    out["disagreement_velocity"] = dv
    out["disagreement_acceleration"] = da

    ref_mean = float(np.mean(disagreement)) if len(disagreement) else 0.0
    ref_std = float(np.std(disagreement)) if len(disagreement) > 1 else 1.0
    out["disagreement_spike"] = np.clip((disagreement - ref_mean) / max(ref_std, 1e-6), 0.0, 8.0)

    return out, next_prev, next_vel, flip_state


def compute_feature_reliability(
    prior_folds: list[Mapping[str, object]],
    current_risk: np.ndarray,
) -> tuple[float, float]:
    if not prior_folds:
        return 1.0, 0.0
    prior_arrays = [
        np.asarray(b.get("risk_context"), dtype=float)
        for b in prior_folds
        if b.get("risk_context") is not None
    ]
    if not prior_arrays:
        return 1.0, 0.0
    ref = np.concatenate(prior_arrays, axis=0)
    cur = np.asarray(current_risk, dtype=float)
    if ref.ndim != 2 or cur.ndim != 2 or ref.shape[1] != cur.shape[1]:
        return 1.0, 0.0
    missing = float(np.mean(np.isfinite(cur)))
    med = np.nanmedian(ref, axis=0)
    scale = np.nanstd(ref, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    drift = float(np.nanmean(np.clip(np.abs(np.nanmean(cur, axis=0) - med) / scale, 0.0, 8.0) / 8.0))
    reliability = float(np.clip(missing * (1.0 - 0.5 * drift), 0.0, 1.0))
    return reliability, drift


def information_shock_score(risk_context: np.ndarray) -> np.ndarray:
    rc = np.asarray(risk_context, dtype=float)
    if rc.ndim != 2 or rc.shape[1] < 3:
        return np.zeros(len(rc), dtype=float)
    gap = np.abs(np.nan_to_num(rc[:, 2], nan=0.0))
    volume = np.nan_to_num(rc[:, 1], nan=0.0)
    shock = np.maximum((gap - 0.03) / 0.05, 0.0)
    shock = np.maximum(shock, np.maximum((volume - 3.0) / 6.0, 0.0))
    return np.clip(shock, 0.0, 1.0)


def regime_transition_probabilities(
    current_regime: str,
    prior_folds: list[Mapping[str, object]],
    regimes: list[str],
) -> dict[str, float]:
    counts = {r: 1.0 for r in regimes}
    for i in range(len(prior_folds) - 1):
        left = np.asarray(prior_folds[i].get("regimes"), dtype=str)
        right = np.asarray(prior_folds[i + 1].get("regimes"), dtype=str)
        if len(left) == 0 or len(right) == 0:
            continue
        # Fold-level modal regime; transition labels are formed only from mature history.
        left_mode = pd.Series(left).mode().iloc[0]
        right_mode = pd.Series(right).mode().iloc[0]
        if str(left_mode) == current_regime:
            counts[str(right_mode)] = counts.get(str(right_mode), 1.0) + 1.0
    total = float(sum(counts.values()))
    return {k: float(v / total) for k, v in counts.items()}


def _fit_meta_model(
    frames: list[pd.DataFrame],
    labels: list[np.ndarray],
    current: pd.DataFrame,
) -> np.ndarray | None:
    if not frames:
        return None
    X = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.concatenate(labels).astype(int)
    if len(y) < 150 or np.unique(y).size < 2:
        return None
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=1000, C=0.25, random_state=42)),
    ])
    model.fit(X, y)
    return np.asarray(model.predict_proba(current.reindex(columns=X.columns, fill_value=0.0))[:, 1], dtype=float)


def _quality_weights(
    prior_folds: list[Mapping[str, object]],
    models: list[str],
) -> np.ndarray:
    if not prior_folds:
        return np.full(len(models), 1.0 / len(models))
    losses = []
    for name in models:
        vals = []
        for fold in prior_folds:
            p = np.asarray((fold.get("predictions") or {})[name], dtype=float)
            y = np.asarray(fold["y"], dtype=int)
            if len(y):
                vals.append(log_loss(y, safe_probability(p), labels=[0, 1]))
        losses.append(float(np.mean(vals)) if vals else UNIFORM_LOGLOSS)
    z = np.exp(-(np.asarray(losses) - np.min(losses)) / 0.10)
    z = np.maximum(z, 0.03)
    return z / np.sum(z)


def _past_error_correlation(
    prior_folds: list[Mapping[str, object]],
    models: list[str],
) -> np.ndarray:
    if len(prior_folds) < 2:
        return np.eye(len(models), dtype=float)
    errors = []
    for name in models:
        vals = []
        for fold in prior_folds:
            p = np.asarray((fold.get("predictions") or {})[name], dtype=float)
            y = np.asarray(fold["y"], dtype=int)
            vals.extend(((p >= 0.5).astype(int) != y).astype(float).tolist())
        errors.append(np.asarray(vals, dtype=float))
    lengths = min(len(x) for x in errors)
    if lengths < 10:
        return np.eye(len(models), dtype=float)
    corr = np.corrcoef(np.vstack([x[:lengths] for x in errors]))
    if not np.isfinite(corr).all():
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
    return corr


def _nearest_history(
    state: pd.DataFrame,
    history_states: list[pd.DataFrame],
    history_labels: list[np.ndarray],
    current: pd.DataFrame,
    *,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    if not history_states:
        return np.full(len(current), 0.5), np.full(len(current), 0.5)
    cols = [c for c in STATE_COLUMNS if c in state.columns]
    hist = pd.concat(history_states, ignore_index=True)[cols].to_numpy(dtype=float)
    hist_y = np.concatenate([np.asarray(x).reshape(-1) for x in history_labels]).astype(float)
    # Deterministic cap prevents quadratic explosion on large universes.
    if len(hist) > 5000:
        idx = np.linspace(0, len(hist) - 1, 5000, dtype=int)
        hist = hist[idx]
        hist_y = hist_y[idx]
    cur = current[cols].to_numpy(dtype=float)
    # Robust scaling based on prior history only.
    med = np.nanmedian(hist, axis=0)
    scale = np.nanstd(hist, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    hist_z = (np.nan_to_num(hist, nan=0.0) - med) / scale
    cur_z = (np.nan_to_num(cur, nan=0.0) - med) / scale
    failure_score = np.zeros(len(cur), dtype=float)
    success_score = np.zeros(len(cur), dtype=float)
    failure_mass = np.zeros(len(cur), dtype=float)
    success_mass = np.zeros(len(cur), dtype=float)

    # Chunked exact nearest neighbors keeps this dependency-free and deterministic.
    chunk = 256
    for start in range(0, len(cur), chunk):
        block = cur_z[start : start + chunk]
        dist = np.sum((block[:, None, :] - hist_z[None, :, :]) ** 2, axis=2)
        kk = min(k, dist.shape[1])
        idx = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
        weights = 1.0 / np.clip(dist[np.arange(len(block))[:, None], idx] + 1e-6, 1e-6, None)
        vals = hist_y[idx]
        failure_score[start : start + len(block)] = np.sum(weights * vals, axis=1) / np.sum(weights, axis=1)
        success_mass[start : start + len(block)] = np.sum(weights * (1.0 - vals), axis=1)
        failure_mass[start : start + len(block)] = np.sum(weights * vals, axis=1)
        success_score[start : start + len(block)] = success_mass[start : start + len(block)] / np.clip(
            success_mass[start : start + len(block)] + failure_mass[start : start + len(block)], 1e-6, None
        )
    return np.clip(failure_score, 0.0, 1.0), np.clip(success_score, 0.0, 1.0)


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=int)
    p = safe_probability(p)
    value = 0.0
    for lo, hi in zip(np.linspace(0.0, 1.0, bins + 1)[:-1], np.linspace(0.0, 1.0, bins + 1)[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        value += float(mask.mean()) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return value


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = safe_probability(p)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(_ece(y, p)),
    }


def _selective(y: np.ndarray, p: np.ndarray, predictability: np.ndarray) -> dict[str, dict[str, float]]:
    conf = np.abs(p - 0.5) * (0.5 + 0.5 * np.asarray(predictability, dtype=float))
    order = np.argsort(-conf)
    out: dict[str, dict[str, float]] = {}
    for coverage in COVERAGE_LEVELS:
        n = max(1, int(np.ceil(len(y) * coverage)))
        idx = order[:n]
        out[f"{int(coverage * 100)}%"] = {
            "coverage": float(len(idx) / len(y)),
            **_metrics(y[idx], p[idx]),
        }
    n_hi = max(1, int(np.ceil(len(y) * 0.30)))
    idx_hi = order[:n_hi]
    out["high_confidence_accuracy"] = {
        "coverage": float(len(idx_hi) / len(y)),
        "accuracy": float(np.mean((p[idx_hi] >= 0.5) == y[idx_hi])),
    }
    return out


def block_bootstrap_ci(values: Iterable[float], block_length: int = 3, draws: int = 1000, seed: int = 42) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x[0]), float(x[0])) if len(x) else (float("nan"), float("nan"))
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


@dataclass
class ModeResult:
    fold_metrics: list[dict[str, float]]
    selective: list[dict[str, object]]
    stress: list[dict[str, float]]


def evaluate_v6(
    fold_banks: Mapping[int, Mapping[str, object]],
    *,
    locked_folds: int = 2,
    min_folds: int = 5,
) -> dict[str, object]:
    ordered = [fold_banks[k] for k in sorted(fold_banks)]
    if len(ordered) < min_folds:
        return {
            "status": "BLOCKED",
            "reason": f"need_at_least_{min_folds}_chronological_folds",
            "production_changed": False,
            "promotion": "HOLD",
        }

    common = None
    for bank in ordered:
        names = set((bank.get("predictions") or {}).keys())
        common = names if common is None else common & names
    models = sorted(common or [])
    if len(models) < 2:
        return {
            "status": "BLOCKED",
            "reason": "need_at_least_2_common_base_models",
            "production_changed": False,
            "promotion": "HOLD",
        }

    locked_folds = min(max(1, int(locked_folds)), len(ordered) - 1)
    dev_end = len(ordered) - locked_folds
    modes = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
             "K_error_corr", "L_regime_transition", "M_retrieval",
             "N_tta", "O_adaptive_compute", "P_full_v6")
    results = {m: ModeResult([], [], []) for m in modes}
    past_state_frames: list[pd.DataFrame] = []
    past_predictability_labels: list[np.ndarray] = []
    failure_metrics = ("accuracy", "logloss", "brier", "ece")
    past_failure_frames: dict[str, dict[str, list[pd.DataFrame]]] = {
        name: {metric: [] for metric in failure_metrics} for name in models
    }
    past_failure_labels: dict[str, dict[str, list[np.ndarray]]] = {
        name: {metric: [] for metric in failure_metrics} for name in models
    }
    calibration_p_history: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    calibration_y_history: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    failure_component_history: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in failure_metrics} for name in models
    }
    past_perf: dict[str, list[dict[str, float]]] = {name: [] for name in models}
    previous_by_symbol: dict[str, dict[str, float]] = {}
    previous_velocity_by_symbol: dict[str, dict[str, float]] = {}
    per_fold_audit: list[dict[str, object]] = []

    # First pass records current fold states. Meta models at t are always trained
    # on information whose future evaluation window is already complete.
    for t, bank in enumerate(ordered):
        probs = np.column_stack(
            [np.asarray((bank.get("predictions") or {})[name], dtype=float) for name in models]
        )
        y = np.asarray(bank["y"], dtype=int)
        if len(y) != len(probs) or not np.isfinite(probs).all():
            raise ValueError(f"invalid OOS bank at fold {t}")
        symbols = np.asarray(bank.get("symbols", [""] * len(y)), dtype=str)
        risk_raw = bank.get("risk_context")
        risk = (
            np.asarray(risk_raw, dtype=float)
            if risk_raw is not None
            else np.zeros((len(y), 0), dtype=float)
        )
        if risk.ndim != 2 or risk.shape[0] != len(y):
            raise ValueError(f"invalid risk context at fold {t}")
        state = disagreement_features(probs)
        state, next_prev, next_vel, _ = add_prediction_dynamics(
            state, probs, symbols, previous_by_symbol, previous_velocity_by_symbol
        )
        previous_by_symbol.update(next_prev)
        previous_velocity_by_symbol.update(next_vel)

        feature_rel, feature_drift = compute_feature_reliability(ordered[:t], risk)
        shock = information_shock_score(risk)
        state["data_completeness"] = np.isfinite(risk).mean(axis=1) if risk.shape[1] else 1.0
        state["feature_reliability"] = feature_rel
        state["feature_drift"] = feature_drift
        state["information_shock"] = shock
        state["model_uncertainty"] = state["std_probability"].to_numpy(dtype=float)
        state["distribution_shift_uncertainty"] = feature_drift
        state["information_uncertainty"] = state["prediction_entropy"].to_numpy(dtype=float)
        current_regime = str(pd.Series(np.asarray(bank.get("regimes", ["normal"]), dtype=str)).mode().iloc[0])
        regimes = sorted(set(str(x) for b in ordered for x in np.asarray(b.get("regimes", []), dtype=str))) or ["normal"]
        trans = regime_transition_probabilities(current_regime, ordered[:t], regimes)
        transition_entropy = -sum(
            p * np.log(max(p, EPS)) for p in trans.values()
        ) / max(np.log(len(regimes)), 1.0)
        state["regime_transition_uncertainty"] = float(np.clip(transition_entropy, 0.0, 1.0))

        # Dynamical history is target-free. Meta labels below are generated only
        # from prior fold outcomes and become usable after their future fold matures.
        history_predictability = past_state_frames[:-1] if t >= 2 else []
        history_predictability_labels = past_predictability_labels[:-1] if t >= 2 else []
        predictability = _fit_meta_model(history_predictability, history_predictability_labels, state)
        if predictability is None:
            predictability = np.clip(
                0.45
                + 0.35 * (1.0 - state["std_probability"].to_numpy())
                + 0.20 * state["feature_reliability"].to_numpy(),
                0.0,
                1.0,
            )

        # Failure predictors are trained only from k -> k+1 pairs with k <= t-2.
        # Each quality dimension has its own target; routing uses their mean risk.
        failure_risk = np.full((len(y), len(models)), 0.5, dtype=float)
        failure_component_risk: dict[str, dict[str, np.ndarray]] = {
            name: {metric: np.full(len(y), 0.5, dtype=float) for metric in failure_metrics}
            for name in models
        }
        for mi, name in enumerate(models):
            for metric in failure_metrics:
                matured_frames = past_failure_frames[name][metric][:-1] if t >= 2 else []
                matured_labels = past_failure_labels[name][metric][:-1] if t >= 2 else []
                if matured_frames:
                    fr = _fit_meta_model(
                        matured_frames,
                        matured_labels,
                        state,
                    )
                    if fr is not None:
                        failure_component_risk[name][metric] = fr
                failure_component_history[name][metric].append(
                    float(np.mean(failure_component_risk[name][metric]))
                )
            failure_risk[:, mi] = np.mean(
                np.column_stack([
                    failure_component_risk[name][metric]
                    for metric in failure_metrics
                ]),
                axis=1,
            )

        # Retrieval uses only already matured state/outcome history.
        matured_state_frames = past_state_frames[:-1] if t >= 2 else []
        matured_predictability_labels = (
            past_predictability_labels[:-1] if t >= 2 else []
        )
        retrieval_failure, retrieval_success = _nearest_history(
            state,
            matured_state_frames,
            [x for x in matured_predictability_labels],
            state,
            k=12,
        )

        quality = _quality_weights(ordered[:t], models)
        error_corr = _past_error_correlation(ordered[:t], models)
        corr_penalty = np.mean(np.abs(error_corr - np.eye(len(models))), axis=1)

        base_probs = probs.copy()
        # K/L/M/N/O/P are explicit extensions of the core A-J matrix.
        for mode in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                     "K_error_corr", "L_regime_transition", "M_retrieval",
                     "N_tta", "O_adaptive_compute", "P_full_v6"):
            w = np.tile(quality, (len(y), 1))
            if mode != "A":
                logits = np.log(np.clip(w, EPS, 1.0))
                if mode in {"B", "F", "G", "J", "K_error_corr", "P_full_v6"}:
                    deviation = np.abs(base_probs - base_probs.mean(axis=1, keepdims=True))
                    logits -= 1.0 * deviation / np.maximum(deviation.mean(axis=1, keepdims=True), 0.02)
                if mode in {"C", "F", "H", "J", "P_full_v6"}:
                    concentration = (predictability - 0.5)[:, None]
                    logits += 0.70 * concentration * (logits - logits.mean(axis=1, keepdims=True))
                if mode in {"D", "G", "H", "I", "J", "P_full_v6"}:
                    logits -= 2.0 * failure_risk
                if mode in {"E", "I", "J", "P_full_v6"}:
                    flatten = np.clip(feature_drift, 0.0, 1.0)
                    logits = (1.0 - 0.60 * flatten) * logits + (0.60 * flatten) * np.log(
                        np.clip(1.0 / len(models), EPS, 1.0)
                    )
                if mode in {"K_error_corr", "P_full_v6"}:
                    logits -= 0.80 * corr_penalty[None, :]
                if mode in {"L_regime_transition", "P_full_v6"}:
                    # Higher transition uncertainty -> less concentrated routing.
                    logits = (1.0 - 0.50 * state["regime_transition_uncertainty"].to_numpy()[:, None]) * logits
                if mode in {"M_retrieval", "P_full_v6"}:
                    logits -= 0.60 * retrieval_failure[:, None]
                    logits += 0.30 * retrieval_success[:, None]
                if mode == "N_tta":
                    # TTA is calibration-layer-only and uses prior OOS labels.
                    pass
                if mode == "O_adaptive_compute":
                    hard = (
                        (predictability < 0.45)
                        | (state["information_shock"].to_numpy() > 0.5)
                        | (failure_risk.max(axis=1) > 0.65)
                    )
                    easy = ~hard
                    logits[easy] = np.log(np.clip(w[easy], EPS, 1.0))
                logits -= logits.max(axis=1, keepdims=True)
                w = np.exp(logits)
                w /= np.clip(w.sum(axis=1, keepdims=True), EPS, None)

            p_raw = np.sum(base_probs * w, axis=1)

            if mode == "N_tta":
                # Recenter toward the last matured base rate only.
                prior_y = (
                    np.concatenate([np.asarray(b["y"], dtype=int) for b in ordered[max(0, t-1):t]])
                    if t
                    else np.array([], dtype=int)
                )
                if len(prior_y) >= 50:
                    target_rate = float(np.mean(prior_y))
                    p_raw = 0.85 * p_raw + 0.15 * target_rate

            # Calibration is strictly historical and mode-specific.
            prior_p = calibration_p_history[mode]
            prior_y = calibration_y_history[mode]
            if prior_p and sum(map(len, prior_y)) >= 100:
                hp = np.concatenate(prior_p)
                hy = np.concatenate(prior_y)
                cal_model = Pipeline([
                    ("scale", StandardScaler()),
                    ("logistic", LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
                ])
                cal_model.fit(np.log(safe_probability(hp)).reshape(-1, 1), hy)
                p = np.asarray(
                    cal_model.predict_proba(
                        np.log(safe_probability(p_raw)).reshape(-1, 1)
                    )[:, 1],
                    dtype=float,
                )
            else:
                p = safe_probability(p_raw)
            calibration_p_history[mode].append(np.asarray(p_raw, dtype=float))
            calibration_y_history[mode].append(np.asarray(y, dtype=int))

            # Safety: severe shift / non-finite / extreme collapse => verified baseline.
            unsafe = (
                ~np.isfinite(p)
                | ~np.isfinite(w).all(axis=1)
                | (np.max(w, axis=1) > 0.98)
                | (state["information_shock"].to_numpy() > 0.95)
                | (state["feature_reliability"].to_numpy() < 0.20)
            )
            if np.any(unsafe):
                baseline_p = np.sum(base_probs * np.tile(quality, (len(y), 1)), axis=1)
                p = np.where(unsafe, safe_probability(baseline_p), safe_probability(p))

            met = _metrics(y, p)
            results[mode].fold_metrics.append(
                {
                    "fold": float(t),
                    "is_locked": float(t >= dev_end),
                    **met,
                    "predictability_mean": float(np.mean(predictability)),
                    "failure_risk_mean": float(np.mean(failure_risk)),
                    "feature_drift": float(feature_drift),
                    "information_shock_mean": float(np.mean(shock)),
                    "weight_entropy": float(-np.mean(np.sum(w * np.log(np.clip(w, EPS, 1.0)), axis=1))),
                }
            )
            if t >= dev_end:
                results[mode].selective.append({
                    "fold": t,
                    **_selective(y, p, predictability),
                })

            # Counterfactual/output-space stress: no tuning, descriptive only.
            p_flat = 0.90 * p + 0.10 * 0.5
            rng = np.random.default_rng(42 + t)
            z = np.log(safe_probability(p)) + rng.normal(0.0, 0.05, size=len(p))
            p_noise = safe_probability(1.0 / (1.0 + np.exp(-z)))
            results[mode].stress.append({
                "fold": float(t),
                "normal_logloss": float(met["logloss"]),
                "flattened_logloss": float(log_loss(y, p_flat, labels=[0, 1])),
                "noise_logloss": float(log_loss(y, p_noise, labels=[0, 1])),
                "prediction_instability": float(np.mean(np.abs(p_flat - p) + np.abs(p_noise - p))),
            })

        # Create matured future labels only for use after the next fold has completed.
        if t < len(ordered) - 1:
            next_bank = ordered[t + 1]
            for name in models:
                cur_p = np.asarray((bank.get("predictions") or {})[name], dtype=float)
                nxt_p = np.asarray((next_bank.get("predictions") or {})[name], dtype=float)
                nxt_y = np.asarray(next_bank["y"], dtype=int)
                cur_loss = log_loss(y, safe_probability(cur_p), labels=[0, 1])
                nxt_loss = log_loss(nxt_y, safe_probability(nxt_p), labels=[0, 1])
                cur_brier = brier_score_loss(y, safe_probability(cur_p))
                nxt_brier = brier_score_loss(nxt_y, safe_probability(nxt_p))
                cur_acc = accuracy_score(y, cur_p >= 0.5)
                nxt_acc = accuracy_score(nxt_y, nxt_p >= 0.5)
                cur_ece = _ece(y, safe_probability(cur_p))
                nxt_ece = _ece(nxt_y, safe_probability(nxt_p))
                labels_by_metric = {
                    "accuracy": np.full(len(state), int(cur_acc - nxt_acc > 0.03), dtype=int),
                    "logloss": np.full(len(state), int(nxt_loss - cur_loss > 0.02), dtype=int),
                    "brier": np.full(len(state), int(nxt_brier - cur_brier > 0.008), dtype=int),
                    "ece": np.full(len(state), int(nxt_ece - cur_ece > 0.02), dtype=int),
                }
                for metric, label in labels_by_metric.items():
                    past_failure_frames[name][metric].append(state.copy())
                    past_failure_labels[name][metric].append(label)

            blended = np.sum(probs * np.tile(_quality_weights(ordered[:t], models), (len(y), 1)), axis=1)
            difficulty = (
                y * np.log(safe_probability(blended))
                + (1 - y) * np.log(1.0 - safe_probability(blended))
            ) > -UNIFORM_LOGLOSS
            past_state_frames.append(state.copy())
            past_predictability_labels.append(difficulty.astype(int))

        for name in models:
            p0 = np.asarray((bank.get("predictions") or {})[name], dtype=float)
            met0 = _metrics(y, p0)
            past_perf[name].append(met0)

        per_fold_audit.append({
            "fold": t,
            "prior_folds_used": t,
            "locked": bool(t >= dev_end),
            "current_labels_used_for_meta_training": False,
            "failure_future_window_matured": bool(t >= 2),
            "production_changed": False,
        })

    summary: dict[str, object] = {}
    for mode, r in results.items():
        if not r.fold_metrics:
            continue
        frame = pd.DataFrame(r.fold_metrics)
        locked = frame[frame["is_locked"] >= 0.5]
        summary[mode] = {
            "metrics": {k: float(frame[k].mean()) for k in ("accuracy", "logloss", "brier", "ece")},
            "development_metrics": {
                k: float(frame[frame["is_locked"] < 0.5][k].mean())
                for k in ("accuracy", "logloss", "brier", "ece")
            } if (frame["is_locked"] < 0.5).any() else {},
            "locked_metrics": {
                k: float(locked[k].mean()) for k in ("accuracy", "logloss", "brier", "ece")
            } if not locked.empty else {},
            "fold_metrics": r.fold_metrics,
            "selective": r.selective,
            "stress": r.stress,
        }

    baseline = summary["A"]["locked_metrics"]
    full = summary["P_full_v6"]["locked_metrics"]
    delta = {
        "accuracy": float(full["accuracy"] - baseline["accuracy"]),
        "logloss": float(full["logloss"] - baseline["logloss"]),
        "brier": float(full["brier"] - baseline["brier"]),
        "ece": float(full["ece"] - baseline["ece"]),
    }
    acc_delta_folds = [
        float(a["accuracy"] - b["accuracy"])
        for a, b in zip(
            summary["P_full_v6"]["fold_metrics"][-locked_folds:],
            summary["A"]["fold_metrics"][-locked_folds:],
        )
    ]
    ll_delta_folds = [
        float(a["logloss"] - b["logloss"])
        for a, b in zip(
            summary["P_full_v6"]["fold_metrics"][-locked_folds:],
            summary["A"]["fold_metrics"][-locked_folds:],
        )
    ]
    br_delta_folds = [
        float(a["brier"] - b["brier"])
        for a, b in zip(
            summary["P_full_v6"]["fold_metrics"][-locked_folds:],
            summary["A"]["fold_metrics"][-locked_folds:],
        )
    ]

    promotion = "HOLD"
    # Research candidate only. This gate deliberately requires non-degradation
    # across every locked probability-quality metric before considering promotion.
    if (
        delta["accuracy"] >= 0.03
        and delta["logloss"] <= 0.0
        and delta["brier"] <= 0.0
        and delta["ece"] <= 0.0
    ):
        promotion = "CANDIDATE"

    return {
        "schema_version": "v6.0",
        "status": "OOS_COMPLETE",
        "evaluation_mode": "FULL_OOS_BANK",
        "production_changed": False,
        "promotion": promotion,
        "models": models,
        "locked_folds": locked_folds,
        "development_folds": dev_end,
        "locked_oos_untouched_for_tuning": True,
        "architectures": summary,
        "full_vs_baseline_locked": delta,
        "statistical_validation": {
            "delta_accuracy_ci95": block_bootstrap_ci(acc_delta_folds),
            "delta_logloss_ci95": block_bootstrap_ci(ll_delta_folds),
            "delta_brier_ci95": block_bootstrap_ci(br_delta_folds),
            "delta_ece_ci95": (
                block_bootstrap_ci([
                    float(a["ece"] - b["ece"])
                    for a, b in zip(
                        summary["P_full_v6"]["fold_metrics"][-locked_folds:],
                        summary["A"]["fold_metrics"][-locked_folds:],
                    )
                ])
            ),
            "method": "moving_block_bootstrap",
            "block_length": 3,
            "draws": 1000,
        },
        "per_fold_meta_audit": per_fold_audit,
        "future_failure": {
            "status": "RESEARCH_COMPLETE",
            "models": models,
            "future_window": "next_chronological_fold",
            "matured_history_only": True,
        },
        "predictability": {
            "status": "RESEARCH_COMPLETE",
            "dimensions": ["Data", "Model", "Information", "Regime", "Temporal", "Local"],
            "calibration_status": "DESCRIPTIVE_ONLY",
        },
        "error_correlation": {
            "status": "RESEARCH_COMPLETE",
            "historical_only": True,
        },
        "regime_transition": {
            "status": "RESEARCH_COMPLETE",
            "historical_transition_matrix_only": True,
        },
        "retrieval": {
            "status": "RESEARCH_COMPLETE",
            "history_only": True,
        },
        "uncertainty": {
            "status": "RESEARCH_COMPLETE",
            "decomposition": ["Model", "DistributionShift", "Information", "IrreducibleProxy"],
        },
        "counterfactual_stability": "DESCRIPTIVE_STRESS",
        "robustness": "DESCRIPTIVE_STRESS",
        "shadow": "PENDING",
        "challenger": "PENDING",
        "fallback": {
            "status": "IMPLEMENTED_RESEARCH_ONLY",
            "baseline": "quality_weighted_verified_candidate",
        },
        "production_artifact": "UNCHANGED",
    }
