"""Research-only v13 extensions: prior-only TTA, scenarios, contracts and audit summaries.

All computations are causal with respect to the supplied chronological bank:
current-fold outcomes are used only for evaluation, never for current prediction
adaptation. Missing timestamp/provenance metadata remains explicitly blocked.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.research.ultimate_control_v13 import (
    _dynamic_routing_weights,
    _history_quality_weights,
    metrics,
    safe_probability,
)


def _logit(p: np.ndarray) -> np.ndarray:
    q = safe_probability(p)
    return np.log(q / (1.0 - q))


def _prior_only_tta(
    prior_predictions: list[np.ndarray],
    prior_y: list[np.ndarray],
    current_prediction: np.ndarray,
    *,
    max_folds: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not prior_predictions:
        return safe_probability(current_prediction), {
            "status": "BLOCKED_NO_PRIOR_OUTCOME_HISTORY",
            "train_rows": 0,
            "train_folds": 0,
        }
    ps = prior_predictions[-max_folds:]
    ys = prior_y[-max_folds:]
    p = np.concatenate(ps).astype(float)
    y = np.concatenate(ys).astype(int)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return safe_probability(current_prediction), {
            "status": "BLOCKED_INSUFFICIENT_PRIOR_OUTCOMES",
            "train_rows": int(len(y)),
            "train_folds": int(len(ps)),
        }
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=13013)
    model.fit(_logit(p).reshape(-1, 1), y)
    calibrated = safe_probability(model.predict_proba(_logit(current_prediction).reshape(-1, 1))[:, 1])
    return calibrated, {
        "status": "EXECUTED_PRIOR_ONLY_TTA",
        "train_rows": int(len(y)),
        "train_folds": int(len(ps)),
    }


def _prediction_snapshot_digest(fold: Mapping[str, Any]) -> str:
    """Hash prediction-time inputs only; never include realized outcome y."""
    payload = {
        "session_dates": np.asarray(fold.get("session_dates", [])).astype(str).tolist(),
        "symbols": np.asarray(fold.get("symbols", [])).astype(str).tolist(),
        "asset_classes": np.asarray(fold.get("asset_classes", [])).astype(str).tolist(),
        "regimes": np.asarray(fold.get("regimes", fold.get("situations", []))).astype(str).tolist(),
        "predictions": {
            str(k): np.asarray(v, dtype=float).tolist()
            for k, v in sorted((fold.get("predictions") or {}).items())
        },
        "risk_context": np.asarray(
            fold.get("risk_context", []), dtype=float
        ).tolist(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _router_stability(fold_results: list[Mapping[str, Any]], models: list[str]) -> dict[str, Any]:
    if len(fold_results) < 2:
        return {
            "status": "INSUFFICIENT_FOLDS",
            "mean_l1_weight_change": float("nan"),
            "max_l1_weight_change": float("nan"),
            "weight_concentration_rate_ge_0_90": float("nan"),
        }
    rows = []
    concentrations = []
    for row in fold_results:
        wm = row.get("routing", {}).get("weight_means", {})
        rows.append(np.asarray([float(wm.get(m, 0.0)) for m in models], dtype=float))
        concentrations.append(float(row.get("routing", {}).get("weight_concentration", 0.0)))
    mat = np.vstack(rows)
    diffs = np.sum(np.abs(np.diff(mat, axis=0)), axis=1)
    return {
        "status": "EXECUTED_DESCRIPTIVE_ROUTER_MONITOR",
        "mean_l1_weight_change": float(np.mean(diffs)),
        "max_l1_weight_change": float(np.max(diffs)),
        "weight_concentration_rate_ge_0_90": float(np.mean(np.asarray(concentrations) >= 0.90)),
        "weight_collapse_flag": bool(np.any(np.asarray(concentrations) >= 0.90)),
    }


def _scenario_row(mean_p: float, dispersion: float, ood: float) -> dict[str, Any]:
    spread = float(np.clip(0.03 + 0.45 * dispersion + 0.20 * ood, 0.03, 0.25))
    direction = float(mean_p - 0.5)
    tail = float(np.clip(0.10 + 0.55 * dispersion + 0.35 * ood, 0.10, 0.60))
    directional = float(np.clip(0.5 + direction, 0.15, 0.85))
    normal = 1.0 - tail
    bull_w = tail * directional
    bear_w = tail - bull_w
    return {
        "status": "EXECUTED_HEURISTIC_SCENARIO_PROXY",
        "probability_proxy": {
            "normal": float(normal),
            "bullish": float(bull_w),
            "bearish": float(bear_w),
        },
        "level_proxy": {
            "normal": float(np.clip(mean_p, 0.001, 0.999)),
            "bullish": float(np.clip(mean_p + spread, 0.001, 0.999)),
            "bearish": float(np.clip(mean_p - spread, 0.001, 0.999)),
        },
        "spread_proxy": spread,
        "calibrated": False,
    }


def _active_information_contract(bank: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    candidates_seen = 0
    accepted = []
    for key in sorted(bank, key=lambda x: int(x)):
        candidates = bank[key].get("information_candidates")
        if not candidates:
            continue
        for c in candidates:
            candidates_seen += 1
            if not isinstance(c, Mapping) or c.get("pit_safe") is not True:
                continue
            try:
                benefit = float(c["expected_logloss_reduction"])
                cost = float(c["cost"])
                risk = float(c["failure_risk"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(np.isfinite(v) for v in (benefit, cost, risk)):
                continue
            accepted.append({
                "name": str(c.get("name", "unknown")),
                "net_value_proxy": float(benefit - cost - risk),
            })
    if not candidates_seen:
        return {
            "status": "BLOCKED_NO_SOURCE_VALUE_OF_INFORMATION_METADATA",
            "policy": "FAIL_CLOSED",
            "candidates_seen": 0,
            "accepted": [],
        }
    return {
        "status": "EXECUTED_PIT_SAFE_METADATA_CONTRACT" if accepted else "BLOCKED_NO_PIT_SAFE_CANDIDATES",
        "policy": "PIT_SAFE_ONLY",
        "candidates_seen": candidates_seen,
        "accepted": sorted(accepted, key=lambda x: x["net_value_proxy"], reverse=True),
    }


def _attribute_failure(
    *,
    prediction: float,
    outcome: int,
    predictability: float,
    ood: float,
    failure_risk: float,
    disagreement: float,
) -> str:
    """Post-outcome attribution only; never used to form the same-fold prediction."""
    wrong = (prediction >= 0.5) != bool(outcome)
    if not wrong:
        return "correct"
    if prediction >= 0.75 or prediction <= 0.25:
        return "high_confidence_wrong"
    if np.isfinite(ood) and ood >= 0.85:
        return "ood_extrapolation"
    if np.isfinite(failure_risk) and failure_risk >= 0.82:
        return "future_failure_warning"
    if np.isfinite(predictability) and predictability < 0.25:
        return "low_predictability"
    if np.isfinite(disagreement) and disagreement >= 0.06:
        return "high_model_disagreement"
    return "other_prediction_error"


def _prior_failure_rate(
    failures: list[Mapping[str, Any]],
    *,
    regime: str,
    strategy: str,
) -> float:
    matched = [
        row for row in failures
        if row.get("regime") == regime and row.get("strategy") == strategy
    ]
    if not matched:
        return float("nan")
    return float(np.mean([float(row.get("failed", 0)) for row in matched]))


def _historical_prediction_retrieval(
    rows: list[Mapping[str, Any]],
    *,
    regime: str,
    prediction: float,
    predictability: float,
    ood: float,
    failure_risk: float,
    disagreement: float,
    top_k: int = 10,
) -> dict[str, Any]:
    """Retrieve prior prediction cases only; current-fold rows are absent by construction."""
    if not rows:
        return {
            "status": "BLOCKED_NO_PRIOR_PREDICTION_HISTORY",
            "hits": 0,
            "failure_rate": float("nan"),
            "success_probability": float("nan"),
        }
    candidates = []
    for row in rows:
        rp = np.asarray([
            float(row.get("prediction", 0.5)),
            float(row.get("predictability", 0.5)),
            float(row.get("ood", 0.0)),
            float(row.get("failure_risk", 0.0)),
            float(row.get("disagreement", 0.0)),
        ], dtype=float)
        cp = np.asarray([prediction, predictability, ood, failure_risk, disagreement], dtype=float)
        if not np.isfinite(rp).all() or not np.isfinite(cp).all():
            continue
        scale = np.asarray([0.20, 0.50, 0.50, 0.50, 0.10], dtype=float)
        distance = float(np.mean(np.abs(rp - cp) / scale))
        if str(row.get("regime")) == regime:
            distance *= 0.80
        candidates.append((distance, int(row.get("failed", 0))))
    if not candidates:
        return {
            "status": "BLOCKED_NO_FINITE_PRIOR_MATCHES",
            "hits": 0,
            "failure_rate": float("nan"),
            "success_probability": float("nan"),
        }
    candidates.sort(key=lambda x: x[0])
    selected = candidates[:max(1, int(top_k))]
    failure_rate = float(np.mean([x[1] for x in selected]))
    return {
        "status": "EXECUTED_PRIOR_ONLY_RETRIEVAL",
        "hits": int(len(selected)),
        "failure_rate": failure_rate,
        "success_probability": float(1.0 - failure_rate),
        "best_distance": float(selected[0][0]),
    }


def _strategy_failure_rate(
    failures: list[Mapping[str, Any]],
    *,
    regime: str,
    strategy: str,
) -> float:
    """Smoothed historical failure rate using only completed prior rows."""
    matched = [
        row for row in failures
        if str(row.get("regime")) == regime and str(row.get("strategy")) == strategy
    ]
    if not matched:
        return float("nan")
    failed = float(np.sum([int(row.get("failed", 0)) for row in matched]))
    n = float(len(matched))
    # Jeffreys-style smoothing prevents an empty/small history from becoming an extreme 0/1 rate.
    return float((failed + 0.5) / (n + 1.0))


def augment_v13_result(
    bank: Mapping[int, Mapping[str, Any]],
    result: dict[str, Any],
) -> dict[str, Any]:
    ordered = [bank[k] for k in sorted(bank, key=lambda x: int(x))]
    models = list(result.get("models") or [])
    locked_folds = int(result.get("locked_folds") or 0)

    history_dynamic: list[np.ndarray] = []
    history_y: list[np.ndarray] = []
    tta_rows = []
    scenario_rows = []
    contracts = []
    ledger_rows = []
    failure_memory_rows: list[dict[str, Any]] = []
    error_attribution_counts: dict[str, int] = {}
    prediction_history_rows: list[dict[str, Any]] = []
    strategy_failure_rows: list[dict[str, Any]] = []

    for t, fold in enumerate(ordered):
        y = np.asarray(fold.get("y", []), dtype=int)
        p_matrix = np.column_stack([
            np.asarray(fold["predictions"][m], dtype=float) for m in models
        ])
        quality = _history_quality_weights(ordered[:t], models)
        routing = _dynamic_routing_weights(p_matrix, quality)
        dynamic = safe_probability(np.sum(p_matrix * routing, axis=1))
        tta_p, tta_status = _prior_only_tta(history_dynamic, history_y, dynamic)

        met_tta = metrics(y, tta_p)
        met_dynamic = metrics(y, dynamic)
        tta_rows.append({
            "fold": int(t),
            "is_locked": bool(t >= len(ordered) - locked_folds),
            "tta": met_tta,
            "dynamic": met_dynamic,
            "delta_tta_minus_dynamic": {
                k: float(met_tta[k] - met_dynamic[k])
                for k in ("accuracy", "logloss", "brier", "ece")
            },
            "adaptation": tta_status,
        })

        fold_result = result.get("fold_results", [])[t] if t < len(result.get("fold_results", [])) else {}
        mean_p = float(np.mean(dynamic))
        dispersion = float(np.mean(np.std(p_matrix, axis=1)))
        ood = float(fold_result.get("ood_mean", 0.0))
        scenario_rows.append({
            "fold": int(t),
            "is_locked": bool(t >= len(ordered) - locked_folds),
            "mean_probability": mean_p,
            "dispersion": dispersion,
            "ood_mean": ood,
            "scenario": _scenario_row(mean_p, dispersion, ood),
        })

        symbols = np.asarray(fold.get("symbols", [""] * len(y))).astype(str)
        regimes = np.asarray(
            fold.get("regimes", fold.get("situations", ["unknown"] * len(y)))
        ).astype(str)
        meta_scores = np.asarray(
            fold_result.get("meta_label", {}).get("scores", [float("nan")] * len(y)),
            dtype=float,
        )
        actions = fold_result.get("chosen_action_counts", {})
        strategies = fold_result.get("chosen_strategy_counts", {})
        weight_means = fold_result.get("routing", {}).get("weight_means", {})
        chosen_strategy = np.asarray(
            fold_result.get("chosen_strategy", ["unknown"] * len(y)), dtype=object
        )
        chosen_action = np.asarray(
            fold_result.get("chosen_action", ["unknown"] * len(y)), dtype=object
        )
        fold_predictability = float(
            fold_result.get("predictability_mean", float("nan"))
        )
        fold_ood = float(fold_result.get("ood_mean", float("nan")))
        fold_failure_risk = float(
            fold_result.get("max_failure_risk", float("nan"))
        )
        fold_disagreement = float(
            fold_result.get("disagreement", {}).get("probability_std_mean", float("nan"))
        )
        # Freeze prior-state views at fold start. Current-fold outcomes must never
        # enter retrieval or strategy-failure estimates for another row in this fold.
        prior_failure_memory = list(failure_memory_rows)
        prior_ledger_rows = list(ledger_rows)
        for i, symbol in enumerate(symbols):
            strategy_i = (
                str(chosen_strategy[i]) if i < len(chosen_strategy) else "unknown"
            )
            action_i = str(chosen_action[i]) if i < len(chosen_action) else "unknown"
            failure_type = _attribute_failure(
                prediction=float(tta_p[i]),
                outcome=int(y[i]),
                predictability=fold_predictability,
                ood=fold_ood,
                failure_risk=fold_failure_risk,
                disagreement=fold_disagreement,
            )
            failed = int(failure_type != "correct")
            error_attribution_counts[failure_type] = (
                error_attribution_counts.get(failure_type, 0) + 1
            )
            prior_failure_rate = _prior_failure_rate(
                prior_failure_memory,
                regime=str(regimes[i]),
                strategy=strategy_i,
            )
            smoothed_strategy_failure_rate = _strategy_failure_rate(
                prior_failure_memory,
                regime=str(regimes[i]),
                strategy=strategy_i,
            )
            retrieval = _historical_prediction_retrieval(
                prior_ledger_rows,
                regime=str(regimes[i]),
                prediction=float(tta_p[i]),
                predictability=fold_predictability,
                ood=fold_ood,
                failure_risk=fold_failure_risk,
                disagreement=fold_disagreement,
            )
            prediction_history_rows.append({
                "fold": int(t),
                "row": int(i),
                "regime": str(regimes[i]),
                "strategy": strategy_i,
                **retrieval,
            })
            strategy_failure_rows.append({
                "fold": int(t),
                "row": int(i),
                "regime": str(regimes[i]),
                "strategy": strategy_i,
                "prior_failure_rate_raw": prior_failure_rate,
                "prior_failure_rate_smoothed": smoothed_strategy_failure_rate,
            })
            ledger_rows.append({
                "fold": int(t),
                "row": int(i),
                "is_locked": bool(t >= len(ordered) - locked_folds),
                "symbol": str(symbol),
                "prediction_time": None,
                "prediction": float(tta_p[i]),
                "dynamic_prediction": float(dynamic[i]),
                "strategy": strategy_i,
                "action": action_i,
                "model_weights": {str(k): float(v) for k, v in weight_means.items()},
                "predictability": fold_predictability,
                "ood": fold_ood,
                "failure_risk": fold_failure_risk,
                "meta_label_probability": (
                    float(meta_scores[i]) if i < len(meta_scores) else float("nan")
                ),
                "pit_status": "BLOCKED_NO_FULL_TIMESTAMP_LINEAGE",
                "result": int(y[i]),
                "failed": failed,
                "failure_type": failure_type,
                "prior_similar_failure_rate": prior_failure_rate,
                "prior_strategy_failure_rate": smoothed_strategy_failure_rate,
                "historical_retrieval": retrieval,
            })
            failure_memory_rows.append({
                "fold": int(t),
                "row": int(i),
                "symbol": str(symbol),
                "regime": str(regimes[i]),
                "strategy": strategy_i,
                "action": action_i,
                "prediction": float(tta_p[i]),
                "predictability": fold_predictability,
                "ood": fold_ood,
                "failure_risk": fold_failure_risk,
                "disagreement": fold_disagreement,
                "failed": failed,
                "failure_type": failure_type,
            })
        contracts.append({
            "fold": int(t),
            "prediction_time": None,
            "valid_until": None,
            "prediction_timestamp_status": "BLOCKED_NO_EXACT_PREDICTION_TIME",
            "prediction_input_snapshot_sha256": _prediction_snapshot_digest(fold),
            "models": models,
            "strategy_counts": strategies,
            "action_counts": actions,
            "symbol_count": int(len(symbols)),
            "confidence_proxy": float(abs(mean_p - 0.5) * 2.0),
            "predictability": float(fold_result.get("predictability_mean", float("nan"))),
            "ood": ood,
            "uncertainty_model": float(
                fold_result.get("uncertainty", {}).get("model", float("nan"))
            ),
            "pit_status": "BLOCKED_NO_FULL_TIMESTAMP_LINEAGE",
        })

        # Current outcomes enter history only after current-fold evaluation.
        history_dynamic.append(dynamic)
        history_y.append(y.copy())

    locked_tta = [r for r in tta_rows if r["is_locked"]]
    def _mean(rows: list[Mapping[str, Any]], metric: str) -> float:
        vals = [float(r[metric]) for r in rows if np.isfinite(float(r[metric]))]
        return float(np.mean(vals)) if vals else float("nan")

    tta_summary = {
        "status": "EXECUTED_PRIOR_ONLY_RESEARCH_ABLATION",
        "locked_folds": len(locked_tta),
        "tta": {m: _mean([r["tta"] for r in locked_tta], m) for m in ("accuracy", "logloss", "brier", "ece")},
        "dynamic": {m: _mean([r["dynamic"] for r in locked_tta], m) for m in ("accuracy", "logloss", "brier", "ece")},
        "delta_tta_minus_dynamic": {
            m: _mean([r["delta_tta_minus_dynamic"] for r in locked_tta], m)
            for m in ("accuracy", "logloss", "brier", "ece")
        },
        "folds": tta_rows,
        "production_changed": False,
        "promotion_allowed": False,
    }

    selected = result.get("metrics_locked", {}).get("selected_policy", {})
    baseline = result.get("metrics_locked", {}).get("baseline_ensemble", {})

    ledger_actions = {}
    ledger_strategies = {}
    total_predictions = 0
    for row, contract in zip(result.get("fold_results", []), contracts):
        total_predictions += int(contract["symbol_count"])
        for k, v in row.get("chosen_action_counts", {}).items():
            ledger_actions[k] = ledger_actions.get(k, 0) + int(v)
        for k, v in row.get("chosen_strategy_counts", {}).items():
            ledger_strategies[k] = ledger_strategies.get(k, 0) + int(v)

    result["tta"] = tta_summary
    result["scenarios"] = {
        "status": "EXECUTED_HEURISTIC_PROXY",
        "calibrated": False,
        "folds": scenario_rows,
        "locked_mean_normal_probability_proxy": _mean(
            [{"normal": r["scenario"]["probability_proxy"]["normal"]} for r in scenario_rows if r["is_locked"]],
            "normal",
        ) if any(r["is_locked"] for r in scenario_rows) else float("nan"),
    }
    result["prediction_contracts"] = {
        "status": "EXECUTED_RESEARCH_CONTRACT_WITH_PIT_BLOCK",
        "folds": contracts,
        "exact_timestamp_lineage": False,
        "production_changed": False,
    }
    locked_memory = [
        row for row in failure_memory_rows
        if int(row["fold"]) >= len(ordered) - locked_folds
    ]
    result["error_attribution"] = {
        "status": "EXECUTED_POST_OUTCOME_ATTRIBUTION",
        "counts_all_folds": error_attribution_counts,
        "locked_failure_rate": (
            float(np.mean([r["failed"] for r in locked_memory]))
            if locked_memory else float("nan")
        ),
    }
    result["prediction_history"] = {
        "status": "EXECUTED_PRIOR_ONLY_HISTORY_RETRIEVAL",
        "rows": prediction_history_rows,
        "total_rows": len(prediction_history_rows),
        "causality_contract": "Only ledger rows from folds strictly before the current fold are retrievable.",
        "current_fold_outcomes_excluded": True,
    }
    result["strategy_failure"] = {
        "status": "EXECUTED_PRIOR_ONLY_STRATEGY_FAILURE_MEMORY",
        "rows": strategy_failure_rows,
        "total_rows": len(strategy_failure_rows),
        "smoothing": "Jeffreys-style (failed+0.5)/(n+1.0)",
        "current_fold_outcomes_excluded": True,
    }

    result["failure_memory"] = {
        "status": "EXECUTED_CAUSAL_POST_OUTCOME_MEMORY",
        "rows": failure_memory_rows,
        "total_rows": len(failure_memory_rows),
        "total_failures": int(sum(r["failed"] for r in failure_memory_rows)),
        "locked_failures": int(sum(r["failed"] for r in locked_memory)),
        "future_use_contract": (
            "Only rows from folds strictly before the current fold may be "
            "used for subsequent prediction-time adaptation."
        ),
    }

    result["prediction_ledger"] = {
        "status": "EXECUTED_ROW_LEVEL_LEDGER_WITH_PIT_BLOCK",
        "scope": "v13_control_plane_summary",
        "total_predictions": total_predictions,
        "action_counts": ledger_actions,
        "strategy_counts": ledger_strategies,
        "row_level": ledger_rows,
        "selected_policy_metrics_snapshot": selected,
        "baseline_ensemble_metrics_snapshot": baseline,
        "production_changed": False,
    }
    result["router_stability"] = _router_stability(
        result.get("fold_results", []), models
    )
    result["active_information"] = _active_information_contract(bank)
    result.setdefault("prediction_output", {})["scenario_proxy"] = True
    result.setdefault("prediction_output", {})["forecast_contract"] = True
    result.setdefault("prediction_output", {})["ledger"] = True
    result.setdefault("prediction_output", {})["tta_ablation"] = True
    return result
