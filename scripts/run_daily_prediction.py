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

PRICE=Path("data/prices")
METRICS=Path("data/research/latest_metrics.json")
GATE=Path("data/research/release_gate.json")
OUT=Path("data/predictions/latest.parquet")

def models():
    return {
        "logistic":lambda:make_pipeline(SimpleImputer(strategy="median"),LogisticRegression(max_iter=1000,C=0.5)),
        "extra_trees":lambda:make_pipeline(SimpleImputer(strategy="median"),ExtraTreesClassifier(n_estimators=300,min_samples_leaf=20,n_jobs=-1,random_state=42)),
        "hgb":lambda:make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(max_iter=300,learning_rate=0.04,max_leaf_nodes=31,l2_regularization=1.0,random_state=42)),
    }

def main():
    if not PRICE.exists() or not METRICS.exists() or not GATE.exists():
        raise SystemExit("DEFERRED: research/release artifacts are missing")
    gate=json.loads(GATE.read_text(encoding="utf-8"))
    if not gate.get("approved",False):
        raise SystemExit(f"DEFERRED: release gate not approved: {gate.get('reasons',[])}")
    payload=json.loads(METRICS.read_text(encoding="utf-8"))
    selected=payload.get("selected_model")
    if selected not in models():
        raise SystemExit("DEFERRED: selected model unavailable")
    df=pd.read_parquet(PRICE)
    df=add_technical_features(df)
    labeled=add_targets(df).dropna(subset=FEATURE_COLUMNS+["target_up_1d","target_ret_1d"])
    if len(labeled)<5000:
        raise SystemExit("DEFERRED: insufficient training data")
    latest_date=max(pd.to_datetime(df["session_date"]).dt.date)
    latest=df[df["session_date"].eq(latest_date)].copy()
    if latest.empty:
        raise SystemExit("DEFERRED: no latest session rows")
    clf=models()[selected]()
    clf.fit(labeled[FEATURE_COLUMNS],labeled["target_up_1d"].astype(int))
    ret_model=make_return_model()
    ret_model.fit(labeled[FEATURE_COLUMNS],labeled["target_ret_1d"])
    p=np.clip(clf.predict_proba(latest[FEATURE_COLUMNS])[:,1],1e-5,1-1e-5)
    expected=ret_model.predict(latest[FEATURE_COLUMNS])
    vol=latest["volatility_20"].fillna(labeled["target_ret_1d"].std()).clip(lower=0.0)
    latest["p_up_1d"]=p
    latest["expected_return_1d"]=expected
    latest["expected_close_1d"]=latest["close"]*(1+expected)
    latest["range_low_1d"]=latest["close"]*np.exp(-1.96*vol)
    latest["range_high_1d"]=latest["close"]*np.exp(1.96*vol)
    latest["prediction_time"]=pd.Timestamp(datetime.now(timezone.utc))
    latest["prediction_date"]=latest["prediction_time"].dt.date if hasattr(latest["prediction_time"],"dt") else pd.Timestamp.now(tz="UTC").date()
    latest["model_id"]=f"{selected}-oos-v1"
    cols=["symbol","asset_class","session_date","close","prediction_time","prediction_date",
          "p_up_1d","expected_return_1d","expected_close_1d","range_low_1d","range_high_1d","model_id"]
    out=cross_sectional_rank(latest[cols])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    out.to_parquet(OUT,index=False)
    print(f"prediction_rows={len(out)} model={selected}")

if __name__=="__main__":
    main()
