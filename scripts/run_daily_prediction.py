from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.features.technical import FEATURE_COLUMNS,add_technical_features
from src.prediction.regression import make_return_model
from src.prediction.targets import add_targets
from src.ranking.cross_sectional import cross_sectional_rank
from src.research.router import Regime,regime_for_row
from src.validation.calibration import PlattCalibrator

PRICE=Path("data/prices"); METRICS=Path("data/research/latest_metrics.json"); GATE=Path("data/research/release_gate.json"); OUT=Path("data/predictions/latest.parquet")

def models():
    return {
        "logistic":lambda:make_pipeline(SimpleImputer(strategy="median"),LogisticRegression(max_iter=1000,C=0.5)),
        "extra_trees":lambda:make_pipeline(SimpleImputer(strategy="median"),ExtraTreesClassifier(n_estimators=300,min_samples_leaf=20,n_jobs=-1,random_state=42)),
        "hgb":lambda:make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(max_iter=300,learning_rate=0.04,max_leaf_nodes=31,l2_regularization=1.0,random_state=42)),
    }

def main():
    if not PRICE.exists() or not METRICS.exists() or not GATE.exists(): raise SystemExit("DEFERRED: research/release artifacts are missing")
    gate=json.loads(GATE.read_text(encoding="utf-8"))
    if not gate.get("approved",False): raise SystemExit(f"DEFERRED: release gate not approved: {gate.get('reasons',[])}")
    payload=json.loads(METRICS.read_text(encoding="utf-8")); route=payload.get("regime_selected_models",{})
    df=pd.read_parquet(PRICE); df=add_technical_features(df)
    labeled=add_targets(df).dropna(subset=FEATURE_COLUMNS+["target_up_1d","target_ret_1d"]).copy()
    if len(labeled)<5000: raise SystemExit("DEFERRED: insufficient training data")
    latest_date=max(pd.to_datetime(df["session_date"]).dt.date); latest=df[df["session_date"].eq(latest_date)].copy()
    if latest.empty: raise SystemExit("DEFERRED: no latest session rows")
    threshold=float(labeled["volatility_20"].dropna().quantile(0.75)) if labeled["volatility_20"].notna().any() else 0.02
    fitted={}; calibrated={}
    model_factories=models()
    train_dates=sorted(pd.to_datetime(labeled["session_date"]).dt.date.unique()); cal_n=max(20,int(len(train_dates)*0.2))
    core=labeled[labeled.session_date.isin(set(train_dates[:-cal_n]))]; cal=labeled[labeled.session_date.isin(set(train_dates[-cal_n:]))]
    for name,factory in model_factories.items():
        m=factory(); m.fit(core[FEATURE_COLUMNS],core.target_up_1d.astype(int))
        cal_p=m.predict_proba(cal[FEATURE_COLUMNS])[:,1]; fitted[name]=m; calibrated[name]=PlattCalibrator().fit(cal_p,cal.target_up_1d.astype(int))
    probs={}
    for name,m in fitted.items(): probs[name]=calibrated[name].predict(m.predict_proba(latest[FEATURE_COLUMNS])[:,1])
    selected=[]
    for _,row in latest.iterrows():
        reg=regime_for_row(row["volatility_20"],row["price_vs_sma60"],threshold).value
        selected.append(route.get(reg,payload.get("selected_model","hgb")))
    latest["p_up_1d"]=[float(np.clip(probs[name][i],1e-5,1-1e-5)) for i,name in enumerate(selected)]
    ret_model=make_return_model(); ret_model.fit(core[FEATURE_COLUMNS],core.target_ret_1d)
    expected=ret_model.predict(latest[FEATURE_COLUMNS])
    vol=latest["volatility_20"].fillna(core["target_ret_1d"].std()).clip(lower=0.0)
    latest["expected_return_1d"]=expected; latest["expected_close_1d"]=latest["close"]*(1+expected)
    latest["range_low_1d"]=latest["close"]*np.exp(-1.96*vol); latest["range_high_1d"]=latest["close"]*np.exp(1.96*vol)
    latest["prediction_time"]=pd.Timestamp(datetime.now(timezone.utc)); latest["prediction_date"]=pd.Timestamp.now(tz="UTC").date(); latest["model_id"]=pd.Series(selected,index=latest.index)
    cols=["symbol","asset_class","session_date","close","prediction_time","prediction_date","p_up_1d","expected_return_1d","expected_close_1d","range_low_1d","range_high_1d","model_id"]
    out=cross_sectional_rank(latest[cols]); OUT.parent.mkdir(parents=True,exist_ok=True); out.to_parquet(OUT,index=False)
    print(f"prediction_rows={len(out)}")

if __name__=="__main__": main()
