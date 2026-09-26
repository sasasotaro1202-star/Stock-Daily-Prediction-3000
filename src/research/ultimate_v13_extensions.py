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
        meta_scores = np.asarray(fold_result.get("meta_label", {}).get("scores", [float("nan")] * len(y)), dtype=float)
        actions = fold_result.get("chosen_action_counts", {})
        strategies = fold_result.get("chosen_strategy_counts", {})
        weight_means = fold_result.get("routing", {}).get("weight_means", {})
        chosen_strategy = np.asarray(fold_result.get("chosen_strategy", ["unknown"] * len(y)), dtype=object)
        chosen_action = np.asarray(fold_result.get("chosen_action", ["unknown"] * len(y)), dtype=object)
        for i, symbol in enumerate(symbols):
            ledger_rows.append({
                "fold": int(t),
                "row": int(i),
                "is_locked": bool(t >= len(ordered) - locked_folds),
                "symbol": str(symbol),
                "prediction_time": None,
                "prediction": float(tta_p[i]),
                "dynamic_prediction": float(dynamic[i]),
                "strategy": str(chosen_strategy[i]) if i < len(chosen_strategy) else "unknown",
                "action": str(chosen_action[i]) if i < len(chosen_action) else "unknown",
                "model_weights": {str(k): float(v) for k, v in weight_means.items()},
                "predictability": float(fold_result.get("predictability_mean", float("nan"))),
                "ood": float(fold_result.get("ood_mean", float("nan"))),
                "failure_risk": float(fold_result.get("max_failure_risk", float("nan"))),
                "meta_label_probability": float(meta_scores[i]) if i < len(meta_scores) else float("nan"),
                "pit_status": "BLOCKED_NO_FULL_TIMESTAMP_LINEAGE",
                "result": None,
                "failure_type": None,
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
