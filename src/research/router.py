from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class Regime(str, Enum):
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    TREND = "trend"
    EVENT = "event"
    DATA_STRESSED = "data_stressed"


@dataclass(frozen=True)
class ModelPlan:
    names: tuple[str, ...]
    weights: tuple[float, ...]
    reason: str
    scope: str = "global"


CANDIDATES = {
    Regime.NORMAL: ("blend_hgb_lgbm_regularized_recent_25_75", "blend_hgb_lgbm_regularized_recent", "blend_hgb_lgbm_regularized_recent_75_25", "lightgbm_conservative_recent", "lightgbm_regularized_recent", "lightgbm_recent", "lightgbm", "lightgbm_regularized", "hgb_conservative_recent", "hgb", "extra_trees", "logistic"),
    Regime.HIGH_VOL: ("blend_hgb_lgbm_regularized_recent_25_75", "blend_hgb_lgbm_regularized_recent", "blend_hgb_lgbm_regularized_recent_75_25", "lightgbm_conservative_recent", "lightgbm_regularized_recent", "lightgbm_recent", "lightgbm", "lightgbm_regularized", "hgb", "extra_trees", "logistic"),
    Regime.TREND: ("blend_hgb_lgbm_regularized_recent_25_75", "blend_hgb_lgbm_regularized_recent", "blend_hgb_lgbm_regularized_recent_75_25", "lightgbm_conservative_recent", "lightgbm_regularized_recent", "lightgbm_recent", "lightgbm", "lightgbm_regularized", "hgb", "extra_trees", "logistic"),
    Regime.EVENT: ("blend_hgb_lgbm_regularized_recent_25_75", "blend_hgb_lgbm_regularized_recent", "blend_hgb_lgbm_regularized_recent_75_25", "lightgbm_conservative_recent", "lightgbm_recent", "lightgbm_regularized_recent", "lightgbm", "lightgbm_regularized", "hgb", "extra_trees"),
    Regime.DATA_STRESSED: ("hgb",),
}

ASSET_CANDIDATES = {
    "jp_stock": ("blend_hgb_lgbm_regularized_recent_25_75", "blend_hgb_lgbm_regularized_recent", "blend_hgb_lgbm_regularized_recent_75_25", "lightgbm_conservative_recent", "lightgbm_recent", "lightgbm_regularized_recent", "lightgbm", "lightgbm_regularized", "extra_trees", "hgb", "logistic"),
    "jp_etf": ("blend_hgb_lgbm_regularized_recent_25_75", "blend_hgb_lgbm_regularized_recent", "blend_hgb_lgbm_regularized_recent_75_25", "lightgbm_conservative_recent", "lightgbm_regularized_recent", "lightgbm_recent", "lightgbm", "lightgbm_regularized", "hgb", "extra_trees", "logistic"),
    "jp_reit": ("blend_hgb_lgbm_regularized_recent_25_75", "blend_hgb_lgbm_regularized_recent", "blend_hgb_lgbm_regularized_recent_75_25", "lightgbm_conservative_recent", "lightgbm_regularized_recent", "lightgbm_recent", "lightgbm", "lightgbm_regularized", "hgb", "extra_trees", "logistic"),
    "us_stock": ("blend_hgb_lgbm_regularized_recent_25_75", "blend_hgb_lgbm_regularized_recent", "blend_hgb_lgbm_regularized_recent_75_25", "lightgbm_conservative_recent", "lightgbm_regularized_recent", "lightgbm_recent", "lightgbm", "lightgbm_regularized", "hgb", "extra_trees", "logistic"),
    "us_etf": ("blend_hgb_lgbm_regularized_recent_25_75", "blend_hgb_lgbm_regularized_recent", "blend_hgb_lgbm_regularized_recent_75_25", "lightgbm_conservative_recent", "lightgbm_regularized_recent", "lightgbm_recent", "lightgbm", "lightgbm_regularized", "hgb", "extra_trees", "logistic"),
}


def regime_for_row(
    volatility: float | None,
    price_vs_sma60: float | None,
    vol_threshold: float,
    gap_pct: float | None = None,
    volume_ratio_20: float | None = None,
    vix_level: float | None = None,
    breadth_up: float | None = None,
) -> Regime:
    if (
        volatility is None
        or price_vs_sma60 is None
        or not math.isfinite(float(volatility))
        or not math.isfinite(float(price_vs_sma60))
    ):
        return Regime.DATA_STRESSED
    if (
        gap_pct is not None
        and math.isfinite(float(gap_pct))
        and abs(float(gap_pct)) >= 0.03
    ) or (
        volume_ratio_20 is not None
        and math.isfinite(float(volume_ratio_20))
        and float(volume_ratio_20) >= 3.0
    ):
        return Regime.EVENT
    if (
        volatility >= vol_threshold
        or (
            vix_level is not None
            and math.isfinite(float(vix_level))
            and float(vix_level) >= 30.0
        )
    ):
        return Regime.HIGH_VOL
    if (
        abs(price_vs_sma60) >= 0.02
        or (
            breadth_up is not None
            and math.isfinite(float(breadth_up))
            and (
                float(breadth_up) <= 0.25
                or float(breadth_up) >= 0.75
            )
        )
    ):
        return Regime.TREND
    return Regime.NORMAL


def situation_for_row(
    regime: str,
    *,
    gap_pct: float | None = None,
    volume_ratio_20: float | None = None,
    vix_level: float | None = None,
    breadth_up: float | None = None,
    price_vs_sma60: float | None = None,
) -> str:
    """Compact human-readable market situation label for production telemetry."""
    try:
        if regime == Regime.EVENT.value:
            if gap_pct is not None and math.isfinite(float(gap_pct)) and abs(float(gap_pct)) >= 0.03:
                return "event_gap"
            if volume_ratio_20 is not None and math.isfinite(float(volume_ratio_20)) and float(volume_ratio_20) >= 3.0:
                return "event_volume"
            return "event"
        if regime == Regime.HIGH_VOL.value:
            if vix_level is not None and math.isfinite(float(vix_level)) and float(vix_level) >= 30.0:
                return "high_vol_vix"
            return "high_vol"
        if regime == Regime.TREND.value:
            # Preserve the actual trigger for trend classification. A market
            # can enter TREND because breadth is extreme while price_vs_sma60
            # remains near zero; labeling that case as trend_up/down would
            # erase useful breadth-driven situation information.
            if (
                price_vs_sma60 is not None
                and math.isfinite(float(price_vs_sma60))
                and abs(float(price_vs_sma60)) >= 0.02
            ):
                return "trend_up" if float(price_vs_sma60) > 0 else "trend_down"
            if breadth_up is not None and math.isfinite(float(breadth_up)):
                if float(breadth_up) >= 0.75:
                    return "breadth_up"
                if float(breadth_up) <= 0.25:
                    return "breadth_down"
            return "trend"
        if regime == Regime.DATA_STRESSED.value:
            return "data_stressed"
        return "range"
    except (TypeError, ValueError):
        return "data_stressed"


def _score(metric: dict[str, float]) -> float:
    # Penalize unstable OOS results without allowing dispersion to dominate.
    mean = float(metric.get("selection_logloss", metric.get("logloss", float("inf"))))
    std = float(
        metric.get("selection_logloss_std", metric.get("logloss_std", 0.0))
    )
    return mean + 0.25 * std


def materially_better_than_parent(
    candidate_metrics: dict[str, dict[str, float]],
    candidate_model: str,
    parent_model: str,
    *,
    min_improvement_logloss: float = 0.0,
) -> bool:
    """Return True only when a scoped model clears a stable OOS LogLoss edge over its parent."""
    if candidate_model == parent_model:
        return False
    candidate = candidate_metrics.get(candidate_model)
    parent = candidate_metrics.get(parent_model)
    if not candidate or not parent:
        # If the parent model is not measured in the scoped slice, do not
        # promote a specialist based on an incomparable objective.
        return False
    candidate_score = _score(candidate)
    parent_score = _score(parent)
    if not math.isfinite(candidate_score) or not math.isfinite(parent_score):
        return False
    margin = max(0.0, float(min_improvement_logloss))
    return (parent_score - candidate_score) >= margin


def rebalance_global_oos_candidates(
    global_candidates: dict[str, dict[str, float]],
    asset_metrics: dict[str, dict[str, dict[str, float]]],
    *,
    blend_weight: float = 0.50,
    min_folds: int = 3,
) -> dict[str, dict[str, float]]:
    """
    Blend raw row-weighted OOS LogLoss with macro-averaged asset-class OOS
    LogLoss. This prevents a very large asset class from dominating global
    model selection while retaining half of the raw objective.
    """
    weight = float(np.clip(blend_weight, 0.0, 1.0))
    out: dict[str, dict[str, float]] = {}
    for name, raw in global_candidates.items():
        per_asset: list[float] = []
        for candidates in asset_metrics.values():
            metric = candidates.get(name)
            if not metric:
                continue
            folds = int(metric.get("folds", min_folds))
            value = metric.get(
                "selection_logloss",
                metric.get("logloss"),
            )
            if folds < min_folds or value is None:
                continue
            value = float(value)
            if math.isfinite(value):
                per_asset.append(value)

        candidate = dict(raw)
        if len(per_asset) >= 2:
            macro = float(np.mean(per_asset))
            macro_std = (
                float(np.std(per_asset, ddof=1))
                if len(per_asset) >= 2
                else 0.0
            )
            raw_mean = float(
                raw.get("selection_logloss", raw.get("logloss", float("inf")))
            )
            raw_std = float(raw.get("logloss_std", 0.0))
            candidate["asset_class_macro_logloss"] = macro
            candidate["asset_class_macro_logloss_std"] = macro_std
            candidate["asset_class_macro_count"] = float(len(per_asset))
            candidate["selection_logloss"] = (
                (1.0 - weight) * raw_mean + weight * macro
            )
            candidate["selection_logloss_std"] = (
                (1.0 - weight) * raw_std + weight * macro_std
            )
        else:
            candidate["asset_class_macro_count"] = float(len(per_asset))
        out[name] = candidate
    return out


def choose_from_oos(
    regime: str,
    candidate_metrics: dict[str, dict[str, float]],
    *,
    candidates: tuple[str, ...] | None = None,
    scope: str = "global",
    min_folds: int = 3,
    rank_ic_tiebreak_tolerance: float = 0.002,
) -> ModelPlan:
    try:
        reg = Regime(regime)
    except ValueError:
        reg = Regime.NORMAL
    allowed = candidates or CANDIDATES.get(reg, CANDIDATES[Regime.NORMAL])
    usable = []
    for name in allowed:
        metric = candidate_metrics.get(name)
        if not metric or "logloss" not in metric:
            continue
        folds = int(metric.get("folds", min_folds))
        if folds < min_folds:
            continue
        usable.append((name, _score(metric)))

    if not usable:
        return ModelPlan(
            ("hgb",),
            (1.0,),
            f"{regime}:oos_unavailable_fallback",
            scope,
        )

    usable.sort(key=lambda x: x[1])
    best_score = usable[0][1]
    close_names = {
        name for name, score in usable
        if score <= best_score + max(0.0, float(rank_ic_tiebreak_tolerance))
    }
    if len(close_names) > 1:
        name = max(
            close_names,
            key=lambda candidate: float(
                candidate_metrics[candidate].get("rank_ic", float("-inf"))
            ),
        )
        reason = f"{regime}:minimum_stable_oos_logloss_rank_ic_tiebreak"
    else:
        name = usable[0][0]
        reason = f"{regime}:minimum_stable_oos_logloss"
    return ModelPlan(
        (name,),
        (1.0,),
        reason,
        scope,
    )


def asset_plan(
    asset_class: str,
    candidate_metrics: dict[str, dict[str, float]],
    *,
    min_folds: int = 3,
    rank_ic_tiebreak_tolerance: float = 0.002,
) -> ModelPlan:
    allowed = ASSET_CANDIDATES.get(asset_class, CANDIDATES[Regime.NORMAL])
    return choose_from_oos(
        "normal",
        candidate_metrics,
        candidates=allowed,
        scope=f"asset:{asset_class}",
        min_folds=min_folds,
        rank_ic_tiebreak_tolerance=rank_ic_tiebreak_tolerance,
    )



def regime_for_situation(situation: str | None) -> str | None:
    """Map a production situation label to its parent regime for OOS gating."""
    if not situation:
        return None
    value = str(situation)
    if value.startswith("event"):
        return Regime.EVENT.value
    if value.startswith("high_vol"):
        return Regime.HIGH_VOL.value
    if value in {"trend", "trend_up", "trend_down", "breadth_up", "breadth_down"}:
        return Regime.TREND.value
    if value == "range":
        return Regime.NORMAL.value
    return None


def route_plan(
    asset_class: str,
    regime: str,
    *,
    asset_regime_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    asset_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    regime_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    situation: str | None = None,
    global_selected: str = "hgb",
    symbol: str | None = None,
    locked_symbol_regime: dict[str, str] | None = None,
    locked_symbol: dict[str, str] | None = None,
    locked_asset_situation: dict[str, str] | None = None,
    locked_situation: dict[str, str] | None = None,
    locked_asset_regime: dict[str, str] | None = None,
    locked_asset: dict[str, str] | None = None,
    locked_regime: dict[str, str] | None = None,
    locked_global: str | None = None,
    min_folds_asset_regime: int = 2,
    min_folds_situation: int = 3,
    min_folds_asset: int = 3,
    min_folds_regime: int = 3,
) -> ModelPlan:
    if regime == Regime.DATA_STRESSED.value:
        return ModelPlan(("hgb",), (1.0,), "data_stressed:fail_closed_fallback", "fallback")

    key = f"{asset_class}::{regime}"
    symbol_key = (
        f"{asset_class}::{str(symbol).strip()}"
        if symbol is not None and str(symbol).strip()
        else None
    )

    # Exact security routes are allowed only when they were selected by
    # chronological OOS and frozen into the production release.
    if symbol_key and locked_symbol_regime:
        symbol_regime_key = f"{symbol_key}::{regime}"
        if symbol_regime_key in locked_symbol_regime:
            return ModelPlan(
                (locked_symbol_regime[symbol_regime_key],),
                (1.0,),
                f"locked:symbol_regime:{symbol_regime_key}",
                symbol_regime_key,
            )
    if symbol_key and locked_symbol and symbol_key in locked_symbol:
        return ModelPlan(
            (locked_symbol[symbol_key],),
            (1.0,),
            f"locked:symbol:{symbol_key}",
            symbol_key,
        )

    # Situation specialists are intentionally below exact security routes.
    # They may only activate from a frozen OOS-selected map; otherwise the
    # ordinary asset/regime hierarchy remains the fallback.
    parent_regime = regime_for_situation(situation)
    if situation and parent_regime != Regime.DATA_STRESSED.value:
        asset_situation_key = f"{asset_class}::{situation}"
        if locked_asset_situation and asset_situation_key in locked_asset_situation:
            return ModelPlan(
                (locked_asset_situation[asset_situation_key],),
                (1.0,),
                f"locked:asset_situation:{asset_situation_key}",
                f"asset_situation:{asset_situation_key}",
            )
        if locked_situation and str(situation) in locked_situation:
            return ModelPlan(
                (locked_situation[str(situation)],),
                (1.0,),
                f"locked:situation:{situation}",
                f"situation:{situation}",
            )

    # Once a model release is frozen, the selected route cannot silently
    # change on the next scheduled run.
    if locked_asset_regime and key in locked_asset_regime:
        return ModelPlan(
            (locked_asset_regime[key],),
            (1.0,),
            f"locked:{key}",
            key,
        )
    if locked_asset and asset_class in locked_asset:
        return ModelPlan(
            (locked_asset[asset_class],),
            (1.0,),
            f"locked:asset:{asset_class}",
            f"asset:{asset_class}",
        )
    if locked_regime and regime in locked_regime:
        return ModelPlan(
            (locked_regime[regime],),
            (1.0,),
            f"locked:regime:{regime}",
            f"regime:{regime}",
        )
    if locked_global:
        return ModelPlan(
            (locked_global,),
            (1.0,),
            "locked:global",
            "global",
        )

    if asset_regime_metrics and key in asset_regime_metrics:
        plan = choose_from_oos(
            regime,
            asset_regime_metrics[key],
            candidates=ASSET_CANDIDATES.get(asset_class),
            scope=key,
            min_folds=min_folds_asset_regime,
        )
        if not plan.reason.endswith("fallback"):
            return plan

    if asset_metrics and asset_class in asset_metrics:
        plan = asset_plan(
            asset_class,
            asset_metrics[asset_class],
            min_folds=min_folds_asset,
        )
        if not plan.reason.endswith("fallback"):
            return plan

    if regime_metrics and regime in regime_metrics:
        plan = choose_from_oos(
            regime,
            regime_metrics[regime],
            scope=f"regime:{regime}",
            min_folds=min_folds_regime,
        )
        if not plan.reason.endswith("fallback"):
            return plan

    return ModelPlan(
        (global_selected,),
        (1.0,),
        "global_oos_fallback",
        "global",
    )
