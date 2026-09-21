from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src.features.technical import FEATURE_COLUMNS,add_technical_features
from src.prediction.targets import add_targets
from src.research.metrics import classification_metrics
from src.validation.calibration import PlattCalibrator

def factories():
    return {
        "logistic":lambda:make_pipeline(SimpleImputer(strategy="median"),LogisticRegression(max_iter=1000,C=0.5)),
        "extra_trees":lambda:make_pipeline(SimpleImputer(strategy="median"),ExtraTreesClassifier(n_estimators=300,min_samples_leaf=20,n_jobs=-1,random_state=42)),
        "hgb":lambda:make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(max_iter=300,learning_rate=0.04,max_leaf_nodes=31,l2_regularization=1.0,random_state=42)),
    }

def main():
    result=Path("data/research/frozen_holdout_result.json")
    if result.exists(): raise SystemExit("FAIL: frozen holdout has already been evaluated; refusing to overwrite")
    lock=Path("config/frozen_holdout.json")
    metrics=Path("data/research/latest_metrics.json")
    if not lock.exists() or not metrics.exists(): raise SystemExit("DEFERRED: frozen configuration and OOS metrics required")
    frozen=json.loads(lock.read_text(encoding="utf-8")); selected=frozen["selected_model"]
    if selected not in factories(): raise SystemExit("FAIL: frozen model is unavailable")
    df=add_targets(add_technical_features(pd.read_parquet("data/prices")))
    cutoff=pd.Timestamp(frozen["cutoff_date"]).date()
    df["date"]=pd.to_datetime(df["session_date"]).dt.date
    train=df[(df["date"]<=cutoff)].dropna(subset=FEATURE_COLUMNS+["target_up_1d"])
    test=df[(df["date"]>cutoff)].dropna(subset=FEATURE_COLUMNS+["target_up_1d"])
    dates=sorted(train["date"].unique()); cal_n=max(20,int(len(dates)*0.2))
    core=train[train["date"].isin(set(dates[:-cal_n]))]; cal=train[train["date"].isin(set(dates[-cal_n:]))]
    model=factories()[selected](); model.fit(core[FEATURE_COLUMNS],core.target_up_1d.astype(int))
    cal_p=model.predict_proba(cal[FEATURE_COLUMNS])[:,1]; calibrator=PlattCalibrator().fit(cal_p,cal.target_up_1d.astype(int))
    p=calibrator.predict(model.predict_proba(test[FEATURE_COLUMNS])[:,1])
    model_metrics=classification_metrics(test.target_up_1d.astype(int),p)
    base=float(core.target_up_1d.mean())
    baseline_metrics=classification_metrics(test.target_up_1d.astype(int),np.full(len(test),base))
    max_ece=0.20
    config=Path("config/pipeline.yml").read_text(encoding="utf-8")
    import re
    m=re.search(r"max_holdout_ece:s*([0-9.]+)",config); max_ece=float(m.group(1)) if m else max_ece
    payload={"status":"EVALUATED_ONCE","frozen_model":selected,"holdout_rows":int(len(test)),
             "model_metrics":model_metrics,"baseline_metrics":baseline_metrics,
             "beats_baseline":bool(model_metrics["logloss"]<baseline_metrics["logloss"]),
             "calibration_within_limit":bool(model_metrics["ece"]<=max_ece),"max_ece":max_ece}
    result.parent.mkdir(parents=True,exist_ok=True); result.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
