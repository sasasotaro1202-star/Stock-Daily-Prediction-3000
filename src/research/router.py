from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
    Regime.NORMAL: ("logistic", "extra_trees", "hgb"),
    Regime.HIGH_VOL: ("hgb", "extra_trees", "logistic"),
    Regime.TREND: ("hgb", "extra_trees", "logistic"),
    Regime.EVENT: ("hgb", "extra_trees"),
    Regime.DATA_STRESSED: ("hgb",),
}

ASSET_CANDIDATES = {
    "jp_stock": ("extra_trees", "hgb", "logistic"),
    "jp_etf": ("hgb", "extra_trees", "logistic"),
    "jp_reit": ("hgb", "extra_trees", "logistic"),
    "us_stock": ("hgb", "extra_trees", "logistic"),
    "us_etf": ("hgb", "extra_trees", "logistic"),
}


def regime_for_row(
    volatility: float | None,
    price_vs_sma60: float | None,
    vol_threshold: float,
) -> Regime:
    if volatility is None or price_vs_sma60 is None:
        return Regime.DATA_STRESSED
    if volatility >= vol_threshold:
        return Regime.HIGH_VOL
    if abs(price_vs_sma60) >= 0.02:
        return Regime.TREND
    return Regime.NORMAL


def _score(metric: dict[str, float]) -> float:
    # Penalize unstable OOS results without allowing dispersion to dominate.
    mean = float(metric.get("logloss", float("inf")))
    std = float(metric.get("logloss_std", 0.0))
    return mean + 0.25 * std


def choose_from_oos(
    regime: str,
    candidate_metrics: dict[str, dict[str, float]],
    *,
    candidates: tuple[str, ...] | None = None,
    scope: str = "global",
    min_folds: int = 3,
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
    name, score = usable[0]
    return ModelPlan(
        (name,),
        (1.0,),
        f"{regime}:minimum_stable_oos_logloss",
        scope,
    )


def asset_plan(
    asset_class: str,
    candidate_metrics: dict[str, dict[str, float]],
    *,
    min_folds: int = 3,
) -> ModelPlan:
    allowed = ASSET_CANDIDATES.get(asset_class, CANDIDATES[Regime.NORMAL])
    return choose_from_oos(
        "normal",
        candidate_metrics,
        candidates=allowed,
        scope=f"asset:{asset_class}",
        min_folds=min_folds,
    )


def route_plan(
    asset_class: str,
    regime: str,
    *,
    asset_regime_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    asset_metrics: dict[str, dict[str, float]] | None = None,
    regime_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    global_selected: str = "hgb",
) -> ModelPlan:
    if regime == Regime.DATA_STRESSED.value:
        return ModelPlan(("hgb",), (1.0,), "data_stressed:fail_closed_fallback", "fallback")

    key = f"{asset_class}::{regime}"
    if asset_regime_metrics and key in asset_regime_metrics:
        plan = choose_from_oos(
            regime,
            asset_regime_metrics[key],
            candidates=ASSET_CANDIDATES.get(asset_class),
            scope=key,
        )
        if not plan.reason.endswith("fallback"):
            return plan

    if asset_metrics and asset_class in asset_metrics:
        plan = asset_plan(asset_class, asset_metrics[asset_class])
        if not plan.reason.endswith("fallback"):
            return plan

    if regime_metrics and regime in regime_metrics:
        plan = choose_from_oos(regime, regime_metrics[regime], scope=f"regime:{regime}")
        if not plan.reason.endswith("fallback"):
            return plan

    return ModelPlan(
        (global_selected,),
        (1.0,),
        "global_oos_fallback",
        "global",
    )
