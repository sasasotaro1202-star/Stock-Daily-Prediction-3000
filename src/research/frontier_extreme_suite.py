from __future__ import annotations

"""Large, deterministic, research-only pattern ecology for stock OOS probabilities.

The suite deliberately explores mechanism diversity, not arbitrary hyperparameter
sweeps.  Learned quantities are fitted only from strictly prior chronological
folds.  Locked OOS observations are shared by every candidate and are never
used for tuning.
"""

import math
from typing import Any, Mapping, Sequence

try:
    from scipy.special import ndtr, ndtri
except Exception:  # pragma: no cover - deterministic logit fallback
    ndtr = None
    ndtri = None

import numpy as np

from src.research.frontier_pattern_suite import (
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

EPS = 1e-6
TEMPERATURES = (0.50, 0.75, 1.00, 1.25, 1.50)
SHRINKS = (0.05, 0.10, 0.20, 0.30, 0.40)
WEIGHT_TEMPS = (0.02, 0.05, 0.10)
RECENCY_HALFLIVES = (1.0, 2.0, 4.0, 8.0)


def _logit(p: np.ndarray) -> np.ndarray:
    x = _safe_probability(p)
    return np.log(x) - np.log1p(-x)


def _entropy(p: np.ndarray) -> np.ndarray:
    x = _safe_probability(p)
    return -(x * np.log(x) + (1.0 - x) * np.log(1.0 - x))


def _probit_mean(p_matrix: np.ndarray) -> np.ndarray:
    """Aggregate probabilities in normal-score space."""
    x = _safe_probability(p_matrix)
    if ndtri is None or ndtr is None:
        z = _logit(x)
        return _safe_probability(_sigmoid(np.mean(z, axis=1)))
    z = ndtri(x)
    return _safe_probability(ndtr(np.mean(z, axis=1)))


def _power_mean(p_matrix: np.ndarray, power: float) -> np.ndarray:
    x = _safe_probability(p_matrix)
    if abs(power) < EPS:
        return _safe_probability(np.exp(np.mean(np.log(x), axis=1)))
    return _safe_probability(np.mean(np.power(x, power), axis=1) ** (1.0 / power))


def _odds_mean(p_matrix: np.ndarray, power: float) -> np.ndarray:
    odds = np.exp(np.clip(_logit(p_matrix), -30.0, 30.0))
    if abs(power) < EPS:
        mean_odds = np.exp(np.mean(np.log(odds), axis=1))
    else:
        mean_odds = np.mean(np.power(odds, power), axis=1) ** (1.0 / power)
    return _safe_probability(mean_odds / (1.0 + mean_odds))


def _trimmed_mean(p_matrix: np.ndarray, trim: int) -> np.ndarray:
    x = np.sort(_safe_probability(p_matrix), axis=1)
    if x.shape[1] <= 2 * trim:
        return x.mean(axis=1)
    return x[:, trim:-trim].mean(axis=1)


def _rank_mean(p_matrix: np.ndarray) -> np.ndarray:
    x = _safe_probability(p_matrix)
    ranks = np.argsort(np.argsort(x, axis=1), axis=1)
    return _safe_probability(((ranks + 1.0) / (x.shape[1] + 2.0)).mean(axis=1))


def _confidence_weighted(p_matrix: np.ndarray, power: float, inverse: bool = False) -> np.ndarray:
    x = _safe_probability(p_matrix)
    signal = np.maximum(np.abs(x - 0.5), EPS) ** power
    if inverse:
        signal = 1.0 / signal
    signal /= np.clip(signal.sum(axis=1, keepdims=True), EPS, None)
    return _safe_probability(np.sum(signal * x, axis=1))


def _prior_base_rate(prior_folds: Sequence[Mapping[str, Any]]) -> float:
    ys = []
    for fold in prior_folds:
        y = np.asarray(fold.get("y", []), dtype=int)
        if len(y):
            ys.append(y)
    return float(np.mean(np.concatenate(ys))) if ys else 0.5


def _prior_mean_prediction(prior_folds: Sequence[Mapping[str, Any]], models: Sequence[str]) -> float:
    vals = []
    for fold in prior_folds:
        y = np.asarray(fold.get("y", []), dtype=int)
        if len(y):
            vals.extend(_fold_probabilities(fold, models).mean(axis=1).tolist())
    return float(np.mean(vals)) if vals else 0.5


def _quality_weights(prior_folds: Sequence[Mapping[str, Any]], models: Sequence[str], temperature: float) -> np.ndarray:
    losses = _prior_model_losses(prior_folds, models)
    return _softmax(-losses, temperature) if prior_folds else np.full(len(models), 1.0 / len(models))


def _recency_weights(prior_folds: Sequence[Mapping[str, Any]], models: Sequence[str], half_life: float) -> np.ndarray:
    losses = _prior_recency_losses(prior_folds, models, half_life=half_life)
    return _softmax(-losses, 0.05) if prior_folds else np.full(len(models), 1.0 / len(models))


def _minimax_weights(prior_folds: Sequence[Mapping[str, Any]], models: Sequence[str], temperature: float) -> np.ndarray:
    if not prior_folds:
        return np.full(len(models), 1.0 / len(models))
    fold_losses = []
    for fold in prior_folds:
        y = np.asarray(fold.get("y", []), dtype=int)
        if len(y):
            p = _fold_probabilities(fold, models)
            fold_losses.append([float(_metrics(y, p[:, j])["logloss"]) for j in range(len(models))])
    if not fold_losses:
        return np.full(len(models), 1.0 / len(models))
    worst = np.max(np.asarray(fold_losses, dtype=float), axis=0)
    return _softmax(-worst, temperature)


def _diversity_weights(prior_folds: Sequence[Mapping[str, Any]], models: Sequence[str], temperature: float, penalty: float) -> np.ndarray:
    if not prior_folds:
        return np.full(len(models), 1.0 / len(models))
    losses = _prior_model_losses(prior_folds, models)
    residuals = []
    for j in range(len(models)):
        vals = []
        for fold in prior_folds:
            y = np.asarray(fold.get("y", []), dtype=int)
            if len(y):
                p = _fold_probabilities(fold, models)[:, j]
                vals.extend((p - y).tolist())
        residuals.append(np.asarray(vals, dtype=float))
    redundancy = np.zeros(len(models), dtype=float)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = residuals[i]
            b = residuals[j]
            if len(a) >= 3 and np.std(a) > EPS and np.std(b) > EPS:
                corr = float(np.corrcoef(a, b)[0, 1])
                if np.isfinite(corr):
                    redundancy[i] += abs(corr)
                    redundancy[j] += abs(corr)
    raw = np.exp(-(losses - np.min(losses)) / max(temperature, EPS))
    raw /= 1.0 + penalty * redundancy
    raw = np.maximum(raw, 0.02)
    return raw / np.clip(raw.sum(), EPS, None)


def _individual_calibration(
    prior_folds: Sequence[Mapping[str, Any]],
    current: np.ndarray,
    models: Sequence[str],
    method: str,
) -> np.ndarray:
    prior_p = []
    prior_y = []
    for fold in prior_folds:
        y = np.asarray(fold.get("y", []), dtype=int)
        if len(y):
            prior_p.append(_fold_probabilities(fold, models))
            prior_y.append(y)
    if not prior_p:
        return current.copy()
    pooled = np.vstack(prior_p)
    y = np.concatenate(prior_y)
    out = np.empty_like(current, dtype=float)
    for j in range(current.shape[1]):
        pp = pooled[:, j]
        cp = current[:, j]
        if method == "platt":
            out[:, j] = _fit_platt(pp, y, cp)
        elif method == "beta":
            out[:, j] = _fit_beta(pp, y, cp)
        elif method == "isotonic":
            out[:, j] = _fit_isotonic(pp, y, cp)
        elif method == "temperature":
            best_t = 1.0
            best_loss = float("inf")
            for temp in TEMPERATURES:
                pred = _sigmoid(_logit(pp) / temp)
                loss = float(_metrics(y, pred)["logloss"])
                if loss < best_loss:
                    best_loss = loss
                    best_t = temp
            out[:, j] = _safe_probability(_sigmoid(_logit(cp) / best_t))
        else:
            raise ValueError(f"unknown calibration method: {method}")
    return out


def _prior_calibrated_blend(
    prior_folds: Sequence[Mapping[str, Any]],
    current: np.ndarray,
    models: Sequence[str],
    method: str,
    weighting: str,
) -> np.ndarray:
    calibrated_current = _individual_calibration(prior_folds, current, models, method)
    if weighting == "equal":
        return _safe_probability(calibrated_current.mean(axis=1))
    w = _quality_weights(prior_folds, models, 0.05)
    return _safe_probability(calibrated_current @ w)


def _case_shrink(current: np.ndarray, base: np.ndarray, strength: float, direction: str) -> np.ndarray:
    x = _safe_probability(current)
    b = _safe_probability(base)
    if direction == "high_uncertainty":
        uncertainty = np.clip(np.std(x, axis=1) * 4.0, 0.0, 1.0)
        return _safe_probability((1.0 - strength * uncertainty) * x.mean(axis=1) + strength * uncertainty * b)
    if direction == "low_margin":
        margin = np.clip(1.0 - 2.0 * np.abs(x.mean(axis=1) - 0.5), 0.0, 1.0)
        return _safe_probability((1.0 - strength * margin) * x.mean(axis=1) + strength * margin * b)
    if direction == "entropy":
        ent = _entropy(x.mean(axis=1))
        ent = ent / max(math.log(2.0), EPS)
        return _safe_probability((1.0 - strength * ent) * x.mean(axis=1) + strength * ent * b)
    return _safe_probability((1.0 - strength) * x.mean(axis=1) + strength * b)


def _candidate_family(
    prior_folds: Sequence[Mapping[str, Any]],
    current: np.ndarray,
    models: Sequence[str],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    mean = current.mean(axis=1)
    out["mean"] = _safe_probability(mean)
    out["median"] = _safe_probability(np.median(current, axis=1))
    out["trimmed1"] = _safe_probability(_trimmed_mean(current, 1))
    out["trimmed2"] = _safe_probability(_trimmed_mean(current, 2))
    out["logit_mean"] = _safe_probability(_sigmoid(_logit(current).mean(axis=1)))
    out["logit_median"] = _safe_probability(_sigmoid(np.median(_logit(current), axis=1)))
    out["probit_mean"] = _probit_mean(current)
    out["rank_mean"] = _rank_mean(current)
    out["power_m05"] = _power_mean(current, -0.5)
    out["power_p05"] = _power_mean(current, 0.5)
    out["power_p2"] = _power_mean(current, 2.0)
    out["odds_p05"] = _odds_mean(current, 0.5)
    out["odds_p2"] = _odds_mean(current, 2.0)
    out["confidence_p05"] = _confidence_weighted(current, 0.5)
    out["confidence_p1"] = _confidence_weighted(current, 1.0)
    out["inverse_confidence_p05"] = _confidence_weighted(current, 0.5, inverse=True)

    if prior_folds:
        for temp in WEIGHT_TEMPS:
            w = _quality_weights(prior_folds, models, temp)
            out[f"quality_t{temp}"] = _safe_probability(current @ w)
        for half_life in RECENCY_HALFLIVES:
            w = _recency_weights(prior_folds, models, half_life)
            out[f"recency_h{half_life}"] = _safe_probability(current @ w)
        for temp in WEIGHT_TEMPS:
            out[f"minimax_t{temp}"] = _safe_probability(current @ _minimax_weights(prior_folds, models, temp))
        for penalty in (0.5, 1.0, 2.0):
            out[f"diversity_p{penalty}"] = _safe_probability(
                current @ _diversity_weights(prior_folds, models, 0.05, penalty)
            )

        quality = _quality_weights(prior_folds, models, 0.05)
        recency = _recency_weights(prior_folds, models, 2.0)
        minimax = _minimax_weights(prior_folds, models, 0.05)
        diversity = _diversity_weights(prior_folds, models, 0.05, 1.0)
        for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
            w_qr = alpha * quality + (1.0 - alpha) * recency
            out[f"weight_mix_quality_recency_{int(alpha * 100)}"] = _safe_probability(current @ w_qr)
            w_qm = alpha * quality + (1.0 - alpha) * minimax
            out[f"weight_mix_quality_minimax_{int(alpha * 100)}"] = _safe_probability(current @ w_qm)
            w_qd = alpha * quality + (1.0 - alpha) * diversity
            out[f"weight_mix_quality_diversity_{int(alpha * 100)}"] = _safe_probability(current @ w_qd)

        for method in ("platt", "beta", "isotonic", "temperature"):
            out[f"cal_{method}_equal"] = _prior_calibrated_blend(
                prior_folds, current, models, method, "equal"
            )
            out[f"cal_{method}_quality"] = _prior_calibrated_blend(
                prior_folds, current, models, method, "quality"
            )
    return out


def run_extreme_pattern_suite(
    bank: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    locked_folds: int = 2,
    min_folds: int = 5,
    minimum_patterns: int = 100,
) -> dict[str, Any]:
    ordered = (
        [dict(bank[k]) for k in sorted(bank, key=lambda value: int(value))]
        if isinstance(bank, Mapping)
        else [dict(value) for value in bank]
    )
    base = {
        "schema_version": 1,
        "research_contract_id": "extreme-frontier-pattern-ecology-v1",
        "status": "BLOCKED",
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
    }
    if len(ordered) < min_folds:
        return {**base, "reason": f"need_at_least_{min_folds}_chronological_folds"}

    common = set(ordered[0].get("predictions", {}) or {})
    for fold in ordered[1:]:
        common &= set(fold.get("predictions", {}) or {})
    models = sorted(common)
    if len(models) < 2:
        return {**base, "reason": "need_at_least_2_common_models"}

    locked_count = min(max(1, int(locked_folds)), len(ordered) - 1)
    locked_start = len(ordered) - locked_count

    pattern_rows: dict[str, list[dict[str, Any]]] = {}
    execution_failures: list[dict[str, Any]] = []
    locked_predictions: dict[str, list[float]] = {}
    locked_y: list[int] = []
    locked_sessions: list[str] = []

    for t, fold in enumerate(ordered):
        y = np.asarray(fold.get("y", []), dtype=int)
        current = _fold_probabilities(fold, models)
        equal = current.mean(axis=1)
        prior = ordered[:t]
        base_rate = _prior_base_rate(prior)
        prior_mean = _prior_mean_prediction(prior, models)
        try:
            family = _candidate_family(prior, current, models)
        except Exception as exc:
            execution_failures.append({
                "fold": int(t),
                "candidate_scope": "candidate_family",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })
            family = {
                "mean": _safe_probability(current.mean(axis=1)),
                "median": _safe_probability(np.median(current, axis=1)),
            }

        # Shrinkage around two prior-only anchors.
        for name, anchor in (("base_rate", base_rate), ("prior_mean", prior_mean), ("neutral", 0.5)):
            anchor_arr = np.full(len(y), anchor, dtype=float)
            for strength in SHRINKS:
                family[f"shrink_{name}_{int(strength * 100)}"] = _safe_probability(
                    (1.0 - strength) * equal + strength * anchor_arr
                )

        # Case-adaptive shrinkage uses current predictive uncertainty only.
        for direction in ("high_uncertainty", "low_margin", "entropy"):
            for strength in (0.10, 0.20, 0.30, 0.40, 0.50):
                family[f"case_{direction}_{int(strength * 100)}"] = _case_shrink(
                    current, np.full(len(y), base_rate), strength, direction
                )

        # Rank/probability hybrids.
        rank = _rank_mean(current)
        for alpha in (0.10, 0.25, 0.50, 0.75, 0.90):
            family[f"rank_hybrid_{int(alpha * 100)}"] = _safe_probability(
                alpha * equal + (1.0 - alpha) * rank
            )

        # Direction vote strength variants.
        vote = (current >= 0.5).mean(axis=1)
        for strength in (0.20, 0.40, 0.60, 0.80, 1.00):
            family[f"vote_{int(strength * 100)}"] = _safe_probability(
                0.5 + (vote - 0.5) * strength
            )

        # Tail-aware aggregation.
        sorted_current = np.sort(current, axis=1)
        if current.shape[1] >= 3:
            family["tail_low2"] = _safe_probability(sorted_current[:, :2].mean(axis=1))
            family["tail_high2"] = _safe_probability(sorted_current[:, -2:].mean(axis=1))
        family["lower_quantile_mix"] = _safe_probability(
            0.25 * sorted_current[:, 0] + 0.75 * equal
        )
        family["upper_quantile_mix"] = _safe_probability(
            0.25 * sorted_current[:, -1] + 0.75 * equal
        )

        # Additional orthogonal anchor mixtures: prediction geometry × prior-only anchors.
        anchors = {
            "neutral": np.full(len(y), 0.5, dtype=float),
            "base_rate": np.full(len(y), base_rate, dtype=float),
            "prior_mean": np.full(len(y), prior_mean, dtype=float),
        }
        geometric = {
            "mean": equal,
            "median": np.median(current, axis=1),
            "logit": _safe_probability(_sigmoid(_logit(current).mean(axis=1))),
            "rank": rank,
        }
        for gname, gpred in geometric.items():
            for aname, anchor in anchors.items():
                for alpha in (0.25, 0.50, 0.75):
                    family[f"anchor_mix_{gname}_{aname}_{int(alpha * 100)}"] = _safe_probability(
                        alpha * gpred + (1.0 - alpha) * anchor
                    )

        # Agreement-weighted transforms: let cross-model consensus alter the strength of a geometry.
        disagreement = np.std(current, axis=1)
        agreement = np.clip(1.0 - 4.0 * disagreement, 0.0, 1.0)
        for base_name, gpred in geometric.items():
            for strength in (0.25, 0.50, 0.75):
                family[f"agreement_strength_{base_name}_{int(strength * 100)}"] = _safe_probability(
                    0.5 + (gpred - 0.5) * (
                        (1.0 - strength) + strength * agreement
                    )
                )

        # Register current-fold predictions.  Any fitted object above used only prior folds.
        is_locked = t >= locked_start
        if is_locked:
            locked_y.extend(y.tolist())
            frame = fold.get("frame")
            if frame is not None and "session_date" in frame:
                locked_sessions.extend(frame["session_date"].astype(str).tolist())
            else:
                locked_sessions.extend([str(t)] * len(y))

        for name, pred in family.items():
            p = _safe_probability(np.asarray(pred, dtype=float))
            row = {
                "name": name,
                "fold": int(t),
                "category": name.split("_", 1)[0],
                "metrics": _metrics(y, p),
                "is_locked": bool(is_locked),
                "research_only": True,
                "production_changed": False,
                "promotion_allowed": False,
            }
            pattern_rows.setdefault(name, []).append(row)
            if is_locked:
                locked_predictions.setdefault(name, []).extend(p.tolist())

    y_locked = np.asarray(locked_y, dtype=int)
    summaries = []
    baseline_pred = np.asarray(
        locked_predictions.get("mean", [0.5] * len(y_locked)),
        dtype=float,
    )
    baseline_metrics = _metrics(y_locked, baseline_pred) if len(y_locked) else {}
    baseline_ll = float(baseline_metrics.get("logloss", float("nan")))

    for name, rows in sorted(pattern_rows.items()):
        pred = np.asarray(locked_predictions.get(name, []), dtype=float)
        if len(pred) != len(y_locked):
            continue
        metrics = _metrics(y_locked, pred)
        delta = (
            (metrics["logloss"] - baseline_ll) / max(abs(baseline_ll), EPS)
            if np.isfinite(baseline_ll) and np.isfinite(metrics["logloss"])
            else float("nan")
        )
        summaries.append({
            "name": name,
            "category": rows[0]["category"],
            "locked_metrics": metrics,
            "relative_logloss_delta_vs_mean": float(delta),
            "locked_cases": int(len(pred)),
            "locked_fold_count": int(locked_count),
            "parameters": {"family": rows[0]["category"]},
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
        })

    summaries.sort(
        key=lambda row: (
            float(row["locked_metrics"].get("logloss", float("inf"))),
            float(row["locked_metrics"].get("brier", float("inf"))),
            float(row["locked_metrics"].get("ece", float("inf"))),
        )
    )

    # Prequential development selection. Each development fold is scored only
    # after selecting a candidate from strictly earlier development folds.
    preq_scores = {}
    preq_decisions = []
    for eval_t in range(1, locked_start):
        eligible = {}
        for name, rows in pattern_rows.items():
            prior_rows = [
                r for r in rows
                if (not r["is_locked"]) and int(r["fold"]) < eval_t
            ]
            vals = [
                float(r["metrics"]["logloss"])
                for r in prior_rows
                if np.isfinite(r["metrics"].get("logloss", np.nan))
            ]
            if vals:
                eligible[name] = float(np.mean(vals))
        if not eligible:
            continue
        chosen = min(eligible, key=lambda name: (eligible[name], name))
        target = next(
            (r for r in pattern_rows[chosen] if int(r["fold"]) == eval_t),
            None,
        )
        if target is None:
            continue
        ll = float(target["metrics"]["logloss"])
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
    # The candidate chosen for the final development decision is the candidate
    # selected at the last prequential development fold. That fold is scored only
    # after the decision, so its outcome cannot influence the final selection.
    final_prequential_choice = preq_decisions[-1]["selected_name"] if preq_decisions else None
    final_prequential_fold = int(preq_decisions[-1]["fold"]) if preq_decisions else None
    prequential_selected_name = final_prequential_choice
    selection_trace = [d["selected_name"] for d in preq_decisions]
    selection_stability = {
        "decision_count": 0,
        "unique_selected_patterns": 0,
        "top_selection_share": 0.0,
        "switch_count": 0,
        "selection_counts": {},
    }
    if selection_trace:
        counts = {}
        for selected in selection_trace:
            counts[selected] = counts.get(selected, 0) + 1
        selection_stability = {
            "decision_count": int(len(selection_trace)),
            "unique_selected_patterns": int(len(counts)),
            "top_selection_share": float(max(counts.values()) / len(selection_trace)),
            "switch_count": int(sum(
                1 for i in range(1, len(selection_trace))
                if selection_trace[i] != selection_trace[i - 1]
            )),
            "selection_counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        }

    development_candidates = []
    for name, rows in sorted(pattern_rows.items()):
        dev = [r["metrics"] for r in rows if not r["is_locked"]]
        ll = [float(m["logloss"]) for m in dev if np.isfinite(m.get("logloss", np.nan))]
        br = [float(m["brier"]) for m in dev if np.isfinite(m.get("brier", np.nan))]
        ec = [float(m["ece"]) for m in dev if np.isfinite(m.get("ece", np.nan))]
        if not ll:
            continue
        development_candidates.append({
            "name": name,
            "development_metrics": {
                "logloss": float(np.mean(ll)),
                "brier": float(np.mean(br)) if br else float("nan"),
                "ece": float(np.mean(ec)) if ec else float("nan"),
                "folds": int(len(dev)),
            },
        })
    development_sorted = sorted(
        development_candidates,
        key=lambda row: (
            row["development_metrics"]["logloss"],
            row["development_metrics"]["brier"] if np.isfinite(row["development_metrics"]["brier"]) else float("inf"),
            row["development_metrics"]["ece"] if np.isfinite(row["development_metrics"]["ece"]) else float("inf"),
            row["name"],
        ),
    )
    selected_name = final_prequential_choice
    selected_locked = next(
        (row for row in summaries if row["name"] == selected_name),
        None,
    )
    best_locked_diagnostic = summaries[0] if summaries else None

    pattern_count = len(summaries)
    if prequential_selected_name is None:
        status = "BLOCKED_INSUFFICIENT_PREQUENTIAL_DEVELOPMENT_SELECTION"
    elif pattern_count < int(minimum_patterns):
        status = "BLOCKED_INSUFFICIENT_PATTERN_BREADTH"
    elif execution_failures:
        status = "EXECUTED_EXTREME_PATTERN_MATRIX_WITH_FAILURES"
    else:
        status = "EXECUTED_EXTREME_PATTERN_MATRIX"

    return {
        **base,
        "status": status,
        "evaluation_mode": "chronological_oos_same_locked_cases",
        "models": models,
        "fold_count": len(ordered),
        "development_folds": int(locked_start),
        "locked_folds": int(locked_count),
        "pattern_count": int(pattern_count),
        "minimum_pattern_count": int(minimum_patterns),
        "patterns": summaries,
        "execution_failures": execution_failures,
        "best_research_pattern": selected_locked,
        "best_locked_diagnostic": best_locked_diagnostic,
        "selection": {
            "source": "prequential_development_only",
            "selected_name": selected_name,
            "final_prequential_selection": {
                "source": "last_prequential_development_decision",
                "decision_fold": final_prequential_fold,
                "decision_outcome_used_for_selection": False,
                "selected_before_scoring_decision_fold": True,
            },
            "development_ranking": development_sorted,
            "prequential_ranking": preq_rank,
            "prequential_decisions": preq_decisions,
            "prequential_selection_stability": selection_stability,
        },
        "baseline": {
            "name": "mean",
            "locked_metrics": baseline_metrics,
        },
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
        "session_cluster": {
            "cluster_unit": "session_date",
            "n_clusters": int(len(np.unique(np.asarray(locked_sessions, dtype=str))))
            if locked_sessions
            else 0,
        },
    }


__all__ = ["run_extreme_pattern_suite"]
