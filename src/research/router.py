from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class Regime(str,Enum):
    NORMAL="normal"; HIGH_VOL="high_vol"; TREND="trend"; EVENT="event"; DATA_STRESSED="data_stressed"

@dataclass(frozen=True)
class ModelPlan:
    names:tuple[str,...]
    weights:tuple[float,...]
    reason:str

CANDIDATES={
    Regime.NORMAL:("logistic","extra_trees","hgb"),
    Regime.HIGH_VOL:("hgb","extra_trees","logistic"),
    Regime.TREND:("hgb","extra_trees","logistic"),
    Regime.EVENT:("hgb","extra_trees"),
    Regime.DATA_STRESSED:("hgb",),
}

def regime_for_row(volatility:float|None,price_vs_sma60:float|None,vol_threshold:float)->Regime:
    if volatility is None or price_vs_sma60 is None:return Regime.DATA_STRESSED
    if volatility>=vol_threshold:return Regime.HIGH_VOL
    if abs(price_vs_sma60)>=0.02:return Regime.TREND
    return Regime.NORMAL

def choose_from_oos(regime:str,candidate_metrics:dict[str,dict[str,float]])->ModelPlan:
    candidates=CANDIDATES.get(Regime(regime),CANDIDATES[Regime.NORMAL])
    usable=[(name,candidate_metrics[name]["logloss"]) for name in candidates if name in candidate_metrics and "logloss" in candidate_metrics[name]]
    if not usable:return ModelPlan(("hgb",),(1.0,),f"{regime}:oos_unavailable_fallback")
    name,_=min(usable,key=lambda x:x[1])
    return ModelPlan((name,),(1.0,),f"{regime}:minimum_oos_logloss")
