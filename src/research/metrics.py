from __future__ import annotations
from numbers import Real

import numpy as np
from sklearn.metrics import accuracy_score,brier_score_loss,log_loss,roc_auc_score

def expected_calibration_error(y_true,p,bins:int=10)->float:
    y=np.asarray(y_true,dtype=float); prob=np.asarray(p,dtype=float)
    if len(y)==0: return float("nan")
    edges=np.linspace(0,1,bins+1); ece=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(prob>=lo)&((prob<hi) if hi<1 else (prob<=hi))
        if mask.any(): ece+=mask.mean()*abs(y[mask].mean()-prob[mask].mean())
    return float(ece)

def classification_metrics(y_true,p)->dict[str,float]:
    y=np.asarray(y_true,dtype=int)
    raw_prob=np.asarray(p,dtype=float)
    if raw_prob.ndim != 1 or raw_prob.shape[0] != y.shape[0]:
        raise ValueError("probability vector shape mismatch")
    if not np.isfinite(raw_prob).all():
        raise ValueError("probabilities must be finite")
    if np.any(raw_prob < -1e-8) or np.any(raw_prob > 1.0 + 1e-8):
        raise ValueError("probabilities must be within [0,1]")
    # Allow only numerical noise at the bounds; never silently repair a
    # materially invalid upstream probability.
    prob=np.clip(raw_prob,1e-6,1-1e-6)
    pred=(prob>=0.5).astype(int)
    out={"logloss":float(log_loss(y,np.column_stack([1-prob,prob]),labels=[0,1])),
         "brier":float(brier_score_loss(y,prob)),
         "ece":expected_calibration_error(y,prob),
         "accuracy":float(accuracy_score(y,pred))}
    out["roc_auc"]=float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else float("nan")
    return out

def aggregate_metric_rows(rows:list[dict[str,object]])->dict[str,float]:
    """Average numeric metrics while ignoring row-level metadata.

    OOS research rows may carry categorical fields such as calibration
    method/model/regime alongside numeric metrics. Aggregation must never
    attempt numeric reduction over those descriptive fields.
    """
    if not rows:
        return {}

    numeric_values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                continue
            numeric_values.setdefault(key, []).append(float(value))

    aggregated: dict[str, float] = {}
    for key in sorted(numeric_values):
        values = np.asarray(numeric_values[key], dtype=float)
        finite = values[np.isfinite(values)]
        aggregated[key] = (
            float(np.mean(finite))
            if finite.size
            else float("nan")
        )
    return aggregated


def cross_sectional_rank_ic(
    y_true,
    signal,
    group_keys,
) -> float:
    """Mean Spearman rank correlation within prediction groups."""
    import pandas as pd

    frame=pd.DataFrame({
        "y":pd.to_numeric(y_true,errors="coerce"),
        "signal":pd.to_numeric(signal,errors="coerce"),
        "group":list(group_keys),
    }).dropna(subset=["y","signal"])
    if frame.empty:
        return float("nan")
    values=[]
    for _, group in frame.groupby("group",sort=False):
        if len(group) < 5 or group["y"].nunique() < 2 or group["signal"].nunique() < 2:
            continue
        corr=group["y"].corr(group["signal"],method="spearman")
        if pd.notna(corr):
            values.append(float(corr))
    return float(pd.Series(values).mean()) if values else float("nan")
