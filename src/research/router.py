from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


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
    gap_pct: float | None = None,
    volume_ratio_20: float | None = None,
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
    name, _ = usable[0]
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
    asset_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    regime_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    global_selected: str = "hgb",
    locked_asset_regime: dict[str, str] | None = None,
    locked_asset: dict[str, str] | None = None,
    locked_regime: dict[str, str] | None = None,
    locked_global: str | None = None,
) -> ModelPlan:
    if regime == Regime.DATA_STRESSED.value:
        return ModelPlan(("hgb",), (1.0,), "data_stressed:fail_closed_fallback", "fallback")

    key = f"{asset_class}::{regime}"

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
        )
        if not plan.reason.endswith("fallback"):
            return plan

    if asset_metrics and asset_class in asset_metrics:
        plan = asset_plan(asset_class, asset_metrics[asset_class])
        if not plan.reason.endswith("fallback"):
            return plan

    if regime_metrics and regime in regime_metrics:
        plan = choose_from_oos(
            regime,
            regime_metrics[regime],
            scope=f"regime:{regime}",
        )
        if not plan.reason.endswith("fallback"):
            return plan

    return ModelPlan(
        (global_selected,),
        (1.0,),
        "global_oos_fallback",
        "global",
    )
