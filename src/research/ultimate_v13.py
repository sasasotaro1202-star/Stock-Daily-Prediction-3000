"""Research-only Ultimate v13 intelligence layer.

This module consumes the chronological OOS prediction bank already produced by
run_daily_research.py. It never imports production state and never promotes a
candidate. All future-looking labels are retrospective research labels; all
routing decisions for a fold use only information from strictly earlier folds.
"""

from __future__ import annotations

from collections import defaultdict
from math import log, exp, sqrt
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EPS = 1e-6
DEFAULT_FAILURE_LOGLOSS_FLOOR = 0.75
DEFAULT_FAILURE_LOGLOSS_MARGIN = 0.10
DEFAULT_FAILURE_BRIER_FLOOR = 0.30
DEFAULT_FAILURE_BRIER_MARGIN = 0.05


def _clip_p(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = _clip_p(p)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))
    brier = float(np.mean((p - y) ** 2))
    acc = float(np.mean((p >= 0.5) == y))
    return {"logloss": ll, "brier": brier, "accuracy": acc, "n": float(len(y))}


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=int)
    p = _clip_p(p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = max(len(y), 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(mask):
            continue
        total += float(mask.sum()) / n * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def _normalized_binary_entropy(p: np.ndarray) -> np.ndarray:
    p = _clip_p(p)
    h = -(p * np.log(p) + (1.0 - p) * np.log1p(-p)) / log(2.0)
    return np.clip(h, 0.0, 1.0)


def _softmax_negative(scores: np.ndarray, temperature: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    temperature = max(float(temperature), 1e-3)
    z = -scores / temperature
    z -= np.max(z)
    w = np.exp(z)
    denom = float(w.sum())
    return w / denom if denom > 0.0 and np.isfinite(denom) else np.full_like(w, 1.0 / len(w))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def _fold_order(bank: Mapping[int, Mapping[str, Any]]) -> list[int]:
    return sorted(int(k) for k in bank.keys())


def _model_names(bank: Mapping[int, Mapping[str, Any]]) -> list[str]:
    names: set[str] = set()
    for fold in bank.values():
        names.update(str(k) for k in (fold.get("predictions") or {}).keys())
    return sorted(names)


def _disagreement_fold(
    y: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    names = sorted(predictions)
    matrix = np.column_stack([_clip_p(predictions[name]) for name in names])
    mean_p = matrix.mean(axis=1)
    std_p = matrix.std(axis=1)
    range_p = matrix.max(axis=1) - matrix.min(axis=1)
    classes = matrix >= 0.5
    majority = np.mean(classes, axis=1)
    class_agreement = np.maximum(majority, 1.0 - majority)
    entropy = _normalized_binary_entropy(mean_p)

    pairwise_js: list[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p = matrix[:, i]
            q = matrix[:, j]
            m = 0.5 * (p + q)
            js = 0.5 * (
                p * np.log(p / m)
                + (1.0 - p) * np.log((1.0 - p) / (1.0 - m))
                + q * np.log(q / m)
                + (1.0 - q) * np.log((1.0 - q) / (1.0 - m))
            )
            pairwise_js.append(float(np.mean(js)))

    return {
        "models": names,
        "row_mean_probability": float(np.mean(mean_p)),
        "row_probability_std": float(np.mean(std_p)),
        "row_probability_range": float(np.mean(range_p)),
        "row_class_agreement": float(np.mean(class_agreement)),
        "row_mean_entropy": float(np.mean(entropy)),
        "pairwise_js_mean": float(np.mean(pairwise_js)) if pairwise_js else 0.0,
        "per_row": {
            "mean_probability": mean_p.tolist(),
            "std_probability": std_p.tolist(),
            "range_probability": range_p.tolist(),
            "class_agreement": class_agreement.tolist(),
            "entropy": entropy.tolist(),
        },
        "y_checksum_count": int(len(y)),
    }


def _risk_context_matrix(fold: Mapping[str, Any]) -> np.ndarray:
    context = fold.get("risk_context")
    if context is None:
        return np.empty((0, 0), dtype=float)
    a = np.asarray(context, dtype=float)
    if a.ndim != 2:
        return np.empty((0, 0), dtype=float)
    return a


def _compute_ood(
    current: np.ndarray,
    history: list[np.ndarray],
) -> tuple[float, float]:
    if current.size == 0 or not history:
        return 0.0, 0.0
    prior = np.vstack([x for x in history if x.size and x.shape[1] == current.shape[1]])
    if prior.size == 0:
        return 0.0, 0.0
    mu = np.nanmedian(prior, axis=0)
    sigma = np.nanmedian(np.abs(prior - mu), axis=0) * 1.4826
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-6), sigma, 1.0)
    z = np.abs((current - mu) / sigma)
    finite = np.isfinite(z)
    score = float(np.nanmean(z[finite])) if np.any(finite) else 0.0
    completeness = float(np.isfinite(current).mean()) if current.size else 0.0
    return float(np.clip(score / 3.0, 0.0, 1.0)), completeness


def _future_failure_labels(
    fold_metrics: dict[str, list[dict[str, float]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    labels: dict[str, list[dict[str, Any]]] = {}
    risks: dict[str, list[dict[str, Any]]] = {}
    for model, rows in fold_metrics.items():
        labels[model] = []
        risks[model] = []
        prior_ll: list[float] = []
        prior_br: list[float] = []
        for idx, row in enumerate(rows):
            ll = float(row["logloss"])
            br = float(row["brier"])
            if len(prior_ll) < 2:
                risk = 0.5
                risk_basis = "insufficient_prior_folds"
            else:
                base_ll = float(np.mean(prior_ll))
                trend_ll = float(prior_ll[-1] - prior_ll[-2])
                base_br = float(np.mean(prior_br))
                raw = (
                    (ll - base_ll) / max(DEFAULT_FAILURE_LOGLOSS_MARGIN, 1e-6)
                    + 0.5 * trend_ll / max(DEFAULT_FAILURE_LOGLOSS_MARGIN, 1e-6)
                    + (br - base_br) / max(DEFAULT_FAILURE_BRIER_MARGIN, 1e-6)
                )
                risk = float(1.0 / (1.0 + np.exp(-0.5 * raw)))
                risk_basis = "prior_folds_only"
            risks[model].append(
                {"fold": idx, "future_failure_risk": float(np.clip(risk, 0.0, 1.0)), "risk_basis": risk_basis}
            )
            # This label is intentionally defined using the next fold(s). It is
            # never fed into same-fold routing; it is the retrospective target
            # for evaluating the early-warning layer.
            next_ll = float(rows[idx + 1]["logloss"]) if idx + 1 < len(rows) else float("nan")
            next_br = float(rows[idx + 1]["brier"]) if idx + 1 < len(rows) else float("nan")
            ll_limit = max(DEFAULT_FAILURE_LOGLOSS_FLOOR, ll + DEFAULT_FAILURE_LOGLOSS_MARGIN)
            br_limit = max(DEFAULT_FAILURE_BRIER_FLOOR, br + DEFAULT_FAILURE_BRIER_MARGIN)
            failure = bool(
                np.isfinite(next_ll)
                and np.isfinite(next_br)
                and (next_ll >= ll_limit or next_br >= br_limit)
            )
            labels[model].append({
                "fold": idx,
                "next_fold_failure": failure,
                "next_logloss": next_ll,
                "next_brier": next_br,
                "label_definition": "next_fold LogLoss/Brier deterioration with fixed safety floors",
            })
            prior_ll.append(ll)
            prior_br.append(br)
    return labels, risks


def _time_to_failure(labels: Mapping[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for model, rows in labels.items():
        model_out: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            hit = next((j for j in range(i + 1, len(rows)) if rows[j]["next_fold_failure"]), None)
            model_out.append({
                "fold": int(row["fold"]),
                "time_to_failure_folds": None if hit is None else int(hit - i),
                "failure_event_within_history": hit is not None,
            })
        out[model] = model_out
    return out


def _dynamic_routing(
    fold_order: list[int],
    bank: Mapping[int, Mapping[str, Any]],
    fold_disagreement: Mapping[int, dict[str, Any]],
    models: list[str],
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    prior_errors: dict[str, list[dict[str, float]]] = defaultdict(list)
    trajectory: list[dict[str, Any]] = []
    prev_mean_p: float | None = None

    for pos, fold_idx in enumerate(fold_order):
        fold = bank[fold_idx]
        y = np.asarray(fold["y"], dtype=int)
        available = [m for m in models if m in (fold.get("predictions") or {})]
        if not available:
            continue

        hist_scores = []
        for model in available:
            hist = prior_errors[model]
            if len(hist) >= 2:
                vals = np.asarray([x["logloss"] for x in hist], dtype=float)
                score = float(np.mean(vals) + 0.25 * np.std(vals, ddof=1))
                trend = float(vals[-1] - vals[-2])
                score += 0.15 * trend
            elif hist:
                score = float(hist[-1]["logloss"])
            else:
                score = 0.0
            hist_scores.append(score)

        disagreement = float(fold_disagreement[fold_idx]["row_probability_std"])
        temperature = 0.75 + 2.0 * disagreement
        weights = (
            np.full(len(available), 1.0 / len(available))
            if not any(prior_errors[m] for m in available)
            else _softmax_negative(np.asarray(hist_scores, dtype=float), temperature)
        )
        matrix = np.column_stack([_clip_p(fold["predictions"][m]) for m in available])
        p = np.clip(matrix @ weights, EPS, 1.0 - EPS)

        metrics = _binary_metrics(y, p)
        metrics["ece"] = _ece(y, p)
        equal_p = np.clip(matrix.mean(axis=1), EPS, 1.0 - EPS)
        equal_metrics = _binary_metrics(y, equal_p)
        equal_metrics["ece"] = _ece(y, equal_p)

        revision = None if prev_mean_p is None else float(np.mean(np.abs(p.mean() - prev_mean_p)))
        fold_rows.append({
            "fold": int(fold_idx),
            "models": available,
            "weights": {m: float(w) for m, w in zip(available, weights)},
            "disagreement_temperature": float(temperature),
            "dynamic_metrics": metrics,
            "equal_weight_metrics": equal_metrics,
            "relative_logloss_improvement": float(
                (equal_metrics["logloss"] - metrics["logloss"])
                / max(abs(equal_metrics["logloss"]), 1e-9)
            ),
            "revision_magnitude_proxy": revision,
        })
        trajectory.append({
            "fold": int(fold_idx),
            "mean_probability": float(np.mean(p)),
            "predictability": float(np.clip(
                (1.0 - fold_disagreement[fold_idx]["row_mean_entropy"])
                * fold_disagreement[fold_idx]["row_class_agreement"], 0.0, 1.0
            )),
            "disagreement": disagreement,
        })

        for model, pred in zip(available, matrix.T):
            prior_errors[model].append(_binary_metrics(y, pred))
        prev_mean_p = float(np.mean(p))

    if fold_rows:
        dyn = [r["dynamic_metrics"] for r in fold_rows]
        eq = [r["equal_weight_metrics"] for r in fold_rows]
        dynamic_agg = {
            k: float(np.mean([x[k] for x in dyn]))
            for k in ("logloss", "brier", "accuracy", "ece")
        }
        equal_agg = {
            k: float(np.mean([x[k] for x in eq]))
            for k in ("logloss", "brier", "accuracy", "ece")
        }
    else:
        dynamic_agg = {}
        equal_agg = {}

    return {
        "status": "EVALUATED" if fold_rows else "DEFERRED",
        "research_only": True,
        "production_changed": False,
        "folds": fold_rows,
        "aggregate": {
            "dynamic": dynamic_agg,
            "equal_weight": equal_agg,
            "relative_logloss_improvement": float(
                (equal_agg["logloss"] - dynamic_agg["logloss"])
                / max(abs(equal_agg["logloss"]), 1e-9)
            ) if dynamic_agg and equal_agg else None,
        },
        "prediction_trajectory": trajectory,
        "causality_rule": "router weight for fold T uses only outcomes from folds < T",
    }


def _error_correlation(
    fold_order: list[int],
    bank: Mapping[int, Mapping[str, Any]],
    models: list[str],
) -> dict[str, Any]:
    errors: dict[str, list[float]] = defaultdict(list)
    for fold_idx in fold_order:
        fold = bank[fold_idx]
        y = np.asarray(fold["y"], dtype=int)
        for model in models:
            pred = (fold.get("predictions") or {}).get(model)
            if pred is None:
                continue
            errors[model].extend((_clip_p(pred) - y).tolist())

    matrix: dict[str, dict[str, float]] = {m: {} for m in models}
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            matrix[a][b] = 1.0 if a == b else _safe_corr(
                np.asarray(errors.get(a, []), dtype=float),
                np.asarray(errors.get(b, []), dtype=float),
            ) if errors.get(a) and errors.get(b) else 0.0
    pairs = [
        matrix[a][b] for i, a in enumerate(models) for b in models[i + 1:]
    ]
    return {
        "residual_correlation": matrix,
        "mean_pairwise_error_correlation": float(np.mean(pairs)) if pairs else 0.0,
        "interpretation": "high correlation means limited ensemble error diversity; low/negative values indicate more diverse residuals",
    }


def build_ultimate_intelligence(
    bank: Mapping[int, Mapping[str, Any]],
    out_dir: str | Path = "data/research/ultimate_v13",
) -> dict[str, Any]:
    folds = _fold_order(bank)
    models = _model_names(bank)
    base = {
        "version": "ultimate_v13_research_v1",
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "input_contract": {
            "source": "chronological OOS prediction bank from run_daily_research",
            "future_labels_allowed_only_for_retrospective_evaluation": True,
            "upstream_pit_and_leakage_gates_required": True,
        },
        "status": "DEFERRED",
        "folds": len(folds),
        "models": models,
    }
    if len(folds) < 2 or len(models) < 2:
        base["reason"] = "need_at_least_two_oos_folds_and_two_models"
        return base

    fold_disagreement: dict[int, dict[str, Any]] = {}
    fold_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    predictability_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    context_history: list[np.ndarray] = []

    for fold_idx in folds:
        fold = bank[fold_idx]
        y = np.asarray(fold["y"], dtype=int)
        preds = {str(k): np.asarray(v, dtype=float) for k, v in (fold.get("predictions") or {}).items()}
        fold_disagreement[fold_idx] = _disagreement_fold(y, preds)

        mean_entropy = fold_disagreement[fold_idx]["row_mean_entropy"]
        agreement = fold_disagreement[fold_idx]["row_class_agreement"]
        context = _risk_context_matrix(fold)
        ood, completeness = _compute_ood(context, context_history)
        predictability = float(np.clip(
            (1.0 - mean_entropy)
            * agreement
            * (0.5 + 0.5 * completeness)
            * (1.0 - 0.5 * ood),
            0.0, 1.0
        ))
        predictability_rows.append({
            "fold": int(fold_idx),
            "predictability_proxy": predictability,
            "entropy_component": float(1.0 - mean_entropy),
            "agreement_component": float(agreement),
            "feature_completeness": completeness,
            "ood_score": ood,
            "label_free": True,
        })
        ood_rows.append({
            "fold": int(fold_idx),
            "ood_score": ood,
            "feature_completeness": completeness,
            "method": "prior-fold robust-median/MAD risk-context distance",
        })

        for model, pred in preds.items():
            metrics = _binary_metrics(y, pred)
            metrics["ece"] = _ece(y, pred)
            metrics["fold"] = float(fold_idx)
            fold_metrics[model].append(metrics)

        if context.size:
            context_history.append(context)

    failure_labels, failure_risks = _future_failure_labels(fold_metrics)
    ttf = _time_to_failure(failure_labels)
    routing = _dynamic_routing(folds, bank, fold_disagreement, models)
    errors = _error_correlation(folds, bank, models)

    regime_counts: dict[str, dict[str, int]] = {}
    dominant: list[str] = []
    for fold_idx in folds:
        values = [str(x) for x in (bank[fold_idx].get("situations") or [])]
        counts: dict[str, int] = {}
        for x in values:
            counts[x] = counts.get(x, 0) + 1
        regime_counts[str(fold_idx)] = counts
        dominant.append(max(counts, key=counts.get) if counts else "unknown")
    transition_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for a, b in zip(dominant, dominant[1:]):
        transition_counts.setdefault(a, {})
        transition_counts[a][b] = transition_counts[a].get(b, 0) + 1

    all_dyn = routing.get("folds") or []
    revisions = [x["revision_magnitude_proxy"] for x in all_dyn if x["revision_magnitude_proxy"] is not None]
    selective_rows = []
    for row in predictability_rows:
        selective_rows.append({
            "fold": row["fold"],
            "coverage_proxy_ge_0_60": float(row["predictability_proxy"] >= 0.60),
            "threshold": 0.60,
        })

    aggregate_selective = {
        "coverage": float(np.mean([r["coverage_proxy_ge_0_60"] for r in selective_rows])) if selective_rows else 0.0,
        "threshold": 0.60,
        "note": "coverage is a research routing diagnostic; it is not a production abstention gate",
    }

    summary = {
        **base,
        "status": "EVALUATED",
        "core_layers": {
            "model_disagreement": "IMPLEMENTED_RESEARCH",
            "predictability": "IMPLEMENTED_PROXY",
            "future_model_failure": "IMPLEMENTED_RETROSPECTIVE_BASELINE",
            "time_to_failure": "IMPLEMENTED_RETROSPECTIVE_BASELINE",
        },
        "additional_layers": {
            "error_correlation": "IMPLEMENTED_RESEARCH",
            "future_regime_transition": "IMPLEMENTED_PROXY",
            "ood": "IMPLEMENTED_RESEARCH",
            "dynamic_soft_routing": "IMPLEMENTED_RESEARCH_ONLY",
            "selective_prediction": "IMPLEMENTED_DIAGNOSTIC",
            "prediction_trajectory": "IMPLEMENTED_FOLD_LEVEL",
            "active_information_acquisition": "DEFERRED_NO_PIT_SAFE_ABLATION_SOURCE_IN_EXISTING_BANK",
            "retrieval": "DEFERRED",
            "tta": "DEFERRED",
            "counterfactual_stability": "DEFERRED",
            "online_production_learning": "BLOCKED_RESEARCH_ONLY",
        },
        "model_disagreement": {
            str(k): v for k, v in fold_disagreement.items()
        },
        "predictability": {
            "rows": predictability_rows,
            "mean": float(np.mean([r["predictability_proxy"] for r in predictability_rows])),
            "std": float(np.std([r["predictability_proxy"] for r in predictability_rows], ddof=1)) if len(predictability_rows) >= 2 else 0.0,
        },
        "future_model_failure": {
            "labels": failure_labels,
            "risk_predictions_prior_only": failure_risks,
            "evaluation_note": "risk_predictions use only prior-fold outcomes; labels use the next fold for retrospective early-warning evaluation",
        },
        "time_to_failure": ttf,
        "error_correlation": errors,
        "regime_transition": {
            "dominant_by_fold": dominant,
            "transition_counts": {a: dict(b) for a, b in transition_counts.items()},
        },
        "ood": {"rows": ood_rows},
        "routing": routing,
        "prediction_update": {
            "revision_count_proxy": len(revisions),
            "mean_revision_magnitude_proxy": float(np.mean(revisions)) if revisions else 0.0,
            "action_definition": "fold-level proxy: revise when mean probability changes from previous fold; no production update is performed",
        },
        "selective_prediction": {
            "diagnostic": selective_rows,
            "aggregate": aggregate_selective,
        },
        "information_acquisition": {
            "status": "DEFERRED",
            "reason": "existing OOS prediction bank does not contain independent PIT-safe source-group ablations required to measure incremental information value",
        },
        "prediction_contract": {
            "prediction_time_source": "upstream OOS fold timestamp context",
            "valid_until": "not implemented at fold-level research granularity",
            "pit_status": "UPSTREAM_GATE_REQUIRED",
            "model_version": "candidate-model names from existing OOS pipeline",
            "strategy": "research_dynamic_soft_routing",
        },
        "promotion_gate": {
            "allowed": False,
            "reasons": [
                "research-only implementation",
                "predictability is a proxy, not calibrated",
                "failure/time-to-failure require larger historical sample for statistical validation",
                "active information/retrieval/TTA are deferred",
            ],
        },
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "ultimate_summary.json": summary,
        "model_disagreement.json": summary["model_disagreement"],
        "predictability.json": summary["predictability"],
        "future_failure.json": summary["future_model_failure"],
        "time_to_failure.json": summary["time_to_failure"],
        "error_correlation.json": summary["error_correlation"],
        "routing.json": summary["routing"],
        "prediction_trajectory.json": summary["routing"].get("prediction_trajectory", []),
        "selective.json": summary["selective_prediction"],
    }
    for name, payload in files.items():
        (out / name).write_text(
            __import__("json").dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    manifest = {
        "version": "ultimate_v13_research_v1",
        "artifacts": sorted(files),
        "folds": folds,
        "models": models,
        "production_changed": False,
    }
    (out / "artifact_manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary
