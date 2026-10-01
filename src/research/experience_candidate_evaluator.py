from __future__ import annotations

import math
from typing import Any, Mapping

SCHEMA_VERSION = 1
DEFAULT_MIN_FOLDS = 3
DEFAULT_MIN_ROWS_PER_FOLD = 30
DEFAULT_MIN_RELATIVE_IMPROVEMENT = 0.03
STABILITY_PENALTY = 0.25

_SCOPE_MAP = {
    "by_regime": ("regime_metrics", "regime_selected_models", "regime"),
    "by_market_situation": ("situation_metrics", "situation_selected_models", "situation"),
    "by_asset_class": ("asset_class_metrics", "asset_class_selected_models", "asset_class"),
}

def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def _score(metric: Mapping[str, Any]) -> float:
    ll = metric.get("selection_logloss", metric.get("logloss"))
    std = metric.get("selection_logloss_std", metric.get("logloss_std", 0.0))
    if not _finite(ll) or not _finite(std):
        return float("inf")
    return float(ll) + STABILITY_PENALTY * float(std)

def _eligible(metric: Mapping[str, Any], *, min_folds: int, min_rows_per_fold: int) -> bool:
    folds = int(metric.get("folds", 0) or 0)
    rows = metric.get("n_test_min", min_rows_per_fold)
    return (
        folds >= min_folds
        and _finite(metric.get("logloss"))
        and _finite(rows)
        and float(rows) >= min_rows_per_fold
    )

def _scope_info(
    candidate: Mapping[str, Any],
    latest_metrics: Mapping[str, Any],
) -> tuple[str, str, str, str | None] | None:
    dimension = str(candidate.get("source_dimension", "")).strip()
    segment = str(candidate.get("source_segment", "")).strip()
    spec = _SCOPE_MAP.get(dimension)
    if not spec or not segment:
        return None
    metrics_key, selected_key, label = spec
    scope_metrics = latest_metrics.get(metrics_key)
    if not isinstance(scope_metrics, Mapping):
        return None
    scoped = scope_metrics.get(segment)
    if not isinstance(scoped, Mapping):
        return None
    selected_map = latest_metrics.get(selected_key)
    parent = None
    if isinstance(selected_map, Mapping):
        value = selected_map.get(segment)
        if value:
            parent = str(value)
    return metrics_key, segment, label, parent

def evaluate_experience_candidates(
    candidate_plan: Mapping[str, Any],
    latest_metrics: Mapping[str, Any],
    *,
    min_folds: int = DEFAULT_MIN_FOLDS,
    min_rows_per_fold: int = DEFAULT_MIN_ROWS_PER_FOLD,
    min_relative_improvement: float = DEFAULT_MIN_RELATIVE_IMPROVEMENT,
) -> dict[str, Any]:
    safety = candidate_plan.get("safety_contract")
    if not isinstance(safety, Mapping):
        raise ValueError("candidate plan safety contract missing")
    if safety.get("research_only") is not True:
        raise ValueError("candidate plan must remain research_only")
    if safety.get("production_changed") is not False:
        raise ValueError("candidate plan claims production mutation")
    if safety.get("frozen_holdout_allowed") is not False:
        raise ValueError("candidate plan cannot use frozen holdout")
    if min_folds < 3:
        raise ValueError("experience candidate evaluation requires >=3 OOS folds")
    if min_rows_per_fold < 10:
        raise ValueError("experience candidate evaluation requires >=10 rows/fold")
    if not 0.0 <= min_relative_improvement <= 1.0:
        raise ValueError("invalid minimum relative improvement")

    rows: list[dict[str, Any]] = []
    for candidate in candidate_plan.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        scope = _scope_info(candidate, latest_metrics)
        base = {
            "candidate_id": str(candidate.get("candidate_id", "")),
            "action": str(candidate.get("action", "")),
            "source_dimension": str(candidate.get("source_dimension", "")),
            "source_segment": str(candidate.get("source_segment", "")),
            "evidence_n": int(candidate.get("evidence_n", 0) or 0),
            "experience_signal_logloss_impact": float(
                candidate.get("impact_vs_global_logloss", 0.0) or 0.0
            ),
            "research_only": True,
            "promotion_allowed": False,
            "fresh_oos_required": True,
        }

        if scope is None:
            base.update({
                "status": "NO_DIRECT_OOS_SCOPE",
                "reason": "candidate requires specialized outcome-context evaluation not represented by a direct latest_metrics scope",
            })
            rows.append(base)
            continue

        metrics_key, segment, label, selected_parent = scope
        scoped_metrics = latest_metrics[metrics_key][segment]
        eligible = {
            str(name): metric
            for name, metric in scoped_metrics.items()
            if isinstance(metric, Mapping)
            and _eligible(metric, min_folds=min_folds, min_rows_per_fold=min_rows_per_fold)
        }
        if not eligible:
            base.update({
                "status": "INSUFFICIENT_OOS_EVIDENCE",
                "scope_type": label,
                "scope": segment,
                "reason": "no candidate model has sufficient comparable chronological OOS evidence",
            })
            rows.append(base)
            continue

        ranked = sorted(eligible.items(), key=lambda item: (_score(item[1]), item[0]))
        best_model, best_metric = ranked[0]
        parent_model = selected_parent or str(latest_metrics.get("selected_model", ""))
        parent_metric = eligible.get(parent_model)
        if parent_metric is None:
            base.update({
                "status": "NO_COMPARABLE_PARENT",
                "scope_type": label,
                "scope": segment,
                "best_model": best_model,
                "best_logloss": float(best_metric["logloss"]),
                "parent_model": parent_model,
                "reason": "current locked parent is not measured on the same scoped OOS slice",
            })
            rows.append(base)
            continue

        parent_score = _score(parent_metric)
        best_score = _score(best_metric)
        improvement = parent_score - best_score
        relative = improvement / max(abs(parent_score), 1e-9)
        row = dict(base)
        row.update({
            "status": "EVALUATED",
            "scope_type": label,
            "scope": segment,
            "best_model": best_model,
            "parent_model": parent_model,
            "parent_score": float(parent_score),
            "best_score": float(best_score),
            "score_improvement": float(improvement),
            "relative_score_improvement": float(relative),
            "best_logloss": float(best_metric.get("logloss")),
            "parent_logloss": float(parent_metric.get("logloss")),
            "logloss_delta_best_minus_parent": float(best_metric.get("logloss") - parent_metric.get("logloss")),
            "accuracy_delta_best_minus_parent": (
                float(best_metric.get("accuracy") - parent_metric.get("accuracy"))
                if _finite(best_metric.get("accuracy")) and _finite(parent_metric.get("accuracy"))
                else None
            ),
            "brier_delta_best_minus_parent": (
                float(best_metric.get("brier") - parent_metric.get("brier"))
                if _finite(best_metric.get("brier")) and _finite(parent_metric.get("brier"))
                else None
            ),
            "ece_delta_best_minus_parent": (
                float(best_metric.get("ece") - parent_metric.get("ece"))
                if _finite(best_metric.get("ece")) and _finite(parent_metric.get("ece"))
                else None
            ),
            "candidate_passes_screen": bool(
                best_model != parent_model
                and improvement >= 0.0
                and relative >= min_relative_improvement
            ),
            "selection_note": (
                "This is a diagnostic reuse of an existing chronological OOS artifact. "
                "Experience-derived selection must be re-evaluated on a fresh chronological period "
                "strictly after candidate creation before any promotion."
            ),
        })
        rows.append(row)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "EVALUATED" if rows else "NO_CANDIDATES",
        "candidate_count": len(rows),
        "evaluations": rows,
        "policy": {
            "research_only": True,
            "promotion_allowed": False,
            "fresh_oos_required": True,
            "frozen_holdout_used": False,
            "min_folds": int(min_folds),
            "min_rows_per_fold": int(min_rows_per_fold),
            "min_relative_improvement": float(min_relative_improvement),
            "stability_penalty": STABILITY_PENALTY,
        },
    }
