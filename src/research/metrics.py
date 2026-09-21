from __future__ import annotations
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
    y=np.asarray(y_true,dtype=int); prob=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    pred=(prob>=0.5).astype(int)
    out={"logloss":float(log_loss(y,np.column_stack([1-prob,prob]),labels=[0,1])),
         "brier":float(brier_score_loss(y,prob)),
         "ece":expected_calibration_error(y,prob),
         "accuracy":float(accuracy_score(y,pred))}
    out["roc_auc"]=float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else float("nan")
    return out

def aggregate_metric_rows(rows:list[dict[str,float]])->dict[str,float]:
    if not rows:return {}
    return {k:float(np.nanmean([row.get(k,np.nan) for row in rows])) for k in sorted({k for row in rows for k in row})}
