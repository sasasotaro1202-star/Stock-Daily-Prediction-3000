from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class Regime(str, Enum):
    NORMAL="normal"; HIGH_VOL="high_vol"; TREND="trend"; EVENT="event"; DATA_STRESSED="data_stressed"

@dataclass(frozen=True)
class ModelPlan:
    names: tuple[str,...]
    weights: tuple[float,...]
    reason: str

def choose_model_plan(*, realized_vol: float|None, market_trend: float|None,
                      event_intensity: float=0.0, data_coverage: float=1.0) -> ModelPlan:
    if data_coverage < 0.98: return ModelPlan(("conservative",),(1.0,),"data_stressed")
    if event_intensity >= 0.7: return ModelPlan(("event","global_hgb"),(0.5,0.5),"event")
    if realized_vol is not None and realized_vol >= 0.04:
        return ModelPlan(("global_hgb","volatility"),(0.6,0.4),"high_vol")
    if market_trend is not None and abs(market_trend) >= 0.02:
        return ModelPlan(("global_hgb","trend"),(0.7,0.3),"trend")
    return ModelPlan(("logistic","global_hgb"),(0.5,0.5),"normal")
