from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.fit import fit_classifier
from src.prediction.model_factories import models
from src.prediction.targets import add_targets
from src.research.metrics import classification_metrics
from src.research.treasury_features import TREASURY_FEATURE_COLUMNS, add_treasury_context
from src.validation.walk_forward import make_date_folds

PRICE=Path("data/prices/canonical.parquet")
CONTEXT=Path("data/market_context.parquet")
TREASURY=Path("data/research/treasury_curve.parquet")
OUT=Path("data/research/treasury_ablation.json")

def _run(frame: pd.DataFrame, columns: list[str], folds) -> list[dict]:
    factory=models().get("logistic")
    if factory is None:
        raise SystemExit("FAIL: logistic model factory unavailable")
    dates=sorted(pd.to_datetime(frame["session_date"]).dt.date.unique())
    rows=[]
    for fold in folds:
        train_dates=dates[:fold.train_end]
        cal_n=max(20,int(len(train_dates)*0.2))
        core=frame[frame.session_date.isin(set(train_dates[:-cal_n]))].dropna(subset=columns+["target_up_1d","session_date"])
        cal=frame[frame.session_date.isin(set(train_dates[-cal_n:]))].dropna(subset=columns+["target_up_1d","session_date"])
        test=frame[frame.session_date.isin(set(dates[fold.test_start:fold.test_end]))].dropna(subset=columns+["target_up_1d","session_date"])
        if min(len(core),len(cal),len(test))<100 or min(core.target_up_1d.nunique(),cal.target_up_1d.nunique(),test.target_up_1d.nunique())<2:
            continue
        model=factory()
        fit_classifier(model,"logistic",core[columns],core.target_up_1d.astype(int),core.session_date,half_life_sessions=252)
        raw_cal=model.predict_proba(cal[columns])[:,1]
        raw_test=model.predict_proba(test[columns])[:,1]
        from src.validation.calibration import make_calibrator
        p=make_calibrator("platt").fit(raw_cal,cal.target_up_1d.astype(int)).predict(raw_test)
        metrics=classification_metrics(test.target_up_1d.astype(int),p)
        metrics["n_test"]=float(len(test))
        rows.append(metrics)
    return rows

def main() -> None:
    for path,message in [(PRICE,"canonical price dataset missing"),(CONTEXT,"market context missing"),(TREASURY,"Treasury artifact missing")]:
        if not path.exists():
            raise SystemExit(f"DEFERRED: {message}")
    prices=pd.read_parquet(PRICE)
    context=pd.read_parquet(CONTEXT)
    treasury=pd.read_parquet(TREASURY)
    frame=add_technical_features(prices)
    frame=add_market_context(frame,context)
    frame=add_cross_sectional_context(frame)
    frame=add_treasury_context(frame,treasury)
    frame=add_targets(frame)
    dates=sorted(pd.to_datetime(frame["session_date"]).dt.date.unique())
    folds=make_date_folds(dates,min_train=252,test_size=21,step=21,embargo=1,purge=1)
    if len(folds)<5:
        raise SystemExit(f"DEFERRED: only {len(folds)} folds")
    base=_run(frame,list(FEATURE_COLUMNS),folds)
    aug=_run(frame,list(FEATURE_COLUMNS)+list(TREASURY_FEATURE_COLUMNS),folds)
    if len(base)<3 or len(aug)<3:
        raise SystemExit("DEFERRED: both variants need >=3 valid OOS folds")
    bm={k:float(np.mean([row[k] for row in base])) for k in ("accuracy","logloss","brier","ece","roc_auc")}
    am={k:float(np.mean([row[k] for row in aug])) for k in bm}
    delta={k:float(am[k]-bm[k]) for k in bm}
    payload={"status":"OOS_COMPLETE","research_only":True,"production_changed":False,"model":"logistic","folds_total":len(folds),"folds_valid_baseline":len(base),"folds_valid_augmented":len(aug),"baseline":{"metrics":bm,"fold_rows":base},"treasury_augmented":{"metrics":am,"fold_rows":aug},"delta_augmented_minus_baseline":delta,"feature_columns_added":list(TREASURY_FEATURE_COLUMNS)}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    main()
