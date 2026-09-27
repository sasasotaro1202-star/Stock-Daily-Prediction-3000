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
from src.research.finra_features import FINRA_FEATURE_COLUMNS, add_finra_context
from src.research.metrics import classification_metrics
from src.validation.calibration import make_calibrator
from src.validation.walk_forward import make_date_folds

PRICE=Path("data/prices/canonical.parquet"); CONTEXT=Path("data/market_context.parquet")
FINRA=Path("data/research/finra_short_sale.parquet"); OUT=Path("data/research/finra_short_sale_ablation.json")

def _run(frame,columns,folds):
    factory=models().get("logistic")
    if factory is None: raise SystemExit("FAIL: logistic model factory unavailable")
    dates=sorted(pd.to_datetime(frame.session_date).dt.date.unique()); rows=[]
    for fold in folds:
        train=dates[:fold.train_end]; cal_n=max(20,int(len(train)*.2))
        core=frame[frame.session_date.isin(set(train[:-cal_n]))].dropna(subset=columns+["target_up_1d","session_date"])
        cal=frame[frame.session_date.isin(set(train[-cal_n:]))].dropna(subset=columns+["target_up_1d","session_date"])
        test=frame[frame.session_date.isin(set(dates[fold.test_start:fold.test_end]))].dropna(subset=columns+["target_up_1d","session_date"])
        if min(len(core),len(cal),len(test))<100 or min(core.target_up_1d.nunique(),cal.target_up_1d.nunique(),test.target_up_1d.nunique())<2: continue
        model=factory(); fit_classifier(model,"logistic",core[columns],core.target_up_1d.astype(int),core.session_date,half_life_sessions=252)
        pcal=model.predict_proba(cal[columns])[:,1]; ptest=model.predict_proba(test[columns])[:,1]
        p=make_calibrator("platt").fit(pcal,cal.target_up_1d.astype(int)).predict(ptest)
        m=classification_metrics(test.target_up_1d.astype(int),p); m["n_test"]=float(len(test)); rows.append(m)
    return rows

def main():
    for p,msg in ((PRICE,"canonical price dataset missing"),(CONTEXT,"market context missing"),(FINRA,"FINRA research artifact missing")):
        if not p.exists(): raise SystemExit("DEFERRED: "+msg)
    frame=add_technical_features(pd.read_parquet(PRICE))
    frame=add_market_context(frame,pd.read_parquet(CONTEXT))
    frame=add_cross_sectional_context(frame)
    frame=add_finra_context(frame,pd.read_parquet(FINRA)); frame=add_targets(frame)
    dates=sorted(pd.to_datetime(frame.session_date).dt.date.unique())
    folds=make_date_folds(dates,min_train=126,test_size=21,step=21,embargo=1,purge=1)
    if len(folds)<5: raise SystemExit(f"DEFERRED: only {len(folds)} folds")
    base=_run(frame,list(FEATURE_COLUMNS),folds); aug=_run(frame,list(FEATURE_COLUMNS)+list(FINRA_FEATURE_COLUMNS),folds)
    if len(base)<3 or len(aug)<3: raise SystemExit("DEFERRED: both variants need >=3 valid OOS folds")
    names=("accuracy","logloss","brier","ece","roc_auc")
    bm={k:float(np.mean([r[k] for r in base])) for k in names}; am={k:float(np.mean([r[k] for r in aug])) for k in names}
    payload={"status":"OOS_COMPLETE","research_only":True,"production_changed":False,"model":"logistic",
             "folds_total":len(folds),"folds_valid_baseline":len(base),"folds_valid_augmented":len(aug),
             "baseline":{"metrics":bm,"fold_rows":base},"finra_augmented":{"metrics":am,"fold_rows":aug},
             "delta_augmented_minus_baseline":{k:float(am[k]-bm[k]) for k in names},
             "feature_columns_added":list(FINRA_FEATURE_COLUMNS)}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
