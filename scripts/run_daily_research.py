from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src.features.technical import FEATURE_COLUMNS,add_technical_features
from src.prediction.targets import add_targets
from src.research.metrics import aggregate_metric_rows,classification_metrics
from src.validation.calibration import PlattCalibrator
from src.validation.leakage import audit_feature_columns,audit_target_separation
from src.validation.walk_forward import make_date_folds

PRICE_DIR=Path("data/prices")
OUT=Path("data/research/latest_metrics.json")
AUDIT=Path("data/research/leakage_audit.json")

def make_models():
    return {
        "logistic":make_pipeline(SimpleImputer(strategy="median"),LogisticRegression(max_iter=1000,C=0.5)),
        "extra_trees":make_pipeline(SimpleImputer(strategy="median"),ExtraTreesClassifier(
            n_estimators=300,min_samples_leaf=20,n_jobs=-1,random_state=42)),
        "hgb":make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(
            max_iter=250,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=1.0,random_state=42)),
    }

def main():
    if not PRICE_DIR.exists(): raise SystemExit("DEFERRED: price dataset is absent")
    df=pd.read_parquet(PRICE_DIR)
    if len(df)<2000: raise SystemExit(f"DEFERRED: insufficient price rows ({len(df)})")
    df=add_targets(add_technical_features(df))
    feature_audit=audit_feature_columns(FEATURE_COLUMNS)
    targets=[c for c in df.columns if c.startswith("target_")]
    separation=audit_target_separation(FEATURE_COLUMNS,targets)
    audit={"feature_columns":list(FEATURE_COLUMNS),"ok":feature_audit.ok and separation.ok,
           "violations":list(feature_audit.violations+separation.violations)}
    AUDIT.parent.mkdir(parents=True,exist_ok=True)
    AUDIT.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    if not audit["ok"]: raise SystemExit(f"FAIL: leakage audit {audit['violations']}")
    df=df.dropna(subset=FEATURE_COLUMNS+["target_up_1d"]).copy()
    dates=sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    folds=make_date_folds(dates)
    if len(folds)<3: raise SystemExit(f"DEFERRED: only {len(folds)} OOS folds available")
    results={}
    for name,model_factory in ((n,lambda m=m:m) for n,m in make_models().items()):
        fold_rows=[]
        for fold in folds:
            train_dates=set(dates[fold.train_end-len(dates[:fold.train_end]):fold.train_end])
            cal_n=max(20,int(len(train_dates)*0.2))
            core_dates=set(dates[fold.train_end-cal_n and 0:fold.train_end-cal_n])
            test_dates=set(dates[fold.test_start:fold.test_end])
            train=df[df.session_date.isin(train_dates)]
            test=df[df.session_date.isin(test_dates)]
            if len(train)<1000 or len(test)<50 or train.target_up_1d.nunique()<2 or test.target_up_1d.nunique()<2: continue
            cal_dates=set(dates[fold.train_end-cal_n:fold.train_end])
            core=df[df.session_date.isin(train_dates-cal_dates)]
            cal=df[df.session_date.isin(cal_dates)]
            model=model_factory
            model.fit(core[FEATURE_COLUMNS],core.target_up_1d.astype(int))
            cal_p=model.predict_proba(cal[FEATURE_COLUMNS])[:,1]
            calibrator=PlattCalibrator().fit(cal_p,cal.target_up_1d.astype(int))
            p=calibrator.predict(model.predict_proba(test[FEATURE_COLUMNS])[:,1])
            row=classification_metrics(test.target_up_1d.astype(int),p); row["n_test"]=float(len(test)); fold_rows.append(row)
        results[name]={"folds":len(fold_rows),"metrics":aggregate_metric_rows(fold_rows)}
    usable={k:v for k,v in results.items() if v["folds"]>=3 and "logloss" in v["metrics"]}
    if not usable: raise SystemExit("DEFERRED: no model has >=3 valid OOS folds")
    selected=min(usable,key=lambda k:usable[k]["metrics"]["logloss"])
    payload={"results":results,"selected_model":selected,
             "selection_basis":"minimum mean chronological OOS logloss; no frozen holdout used",
             "status":"OOS_COMPLETE"}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
