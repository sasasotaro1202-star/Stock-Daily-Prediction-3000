from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.features.technical import FEATURE_COLUMNS,add_technical_features
from src.features.context import add_cross_sectional_context
from src.prediction.targets import add_targets
from src.research.metrics import aggregate_metric_rows,classification_metrics
from src.research.router import Regime,choose_from_oos,regime_for_row
from src.validation.calibration import PlattCalibrator
from src.validation.leakage import audit_feature_columns,audit_target_separation
from src.validation.walk_forward import make_date_folds

PRICE_DIR=Path("data/prices"); OUT=Path("data/research/latest_metrics.json"); AUDIT=Path("data/research/leakage_audit.json")

def make_models():
    return {
        "logistic":lambda:make_pipeline(SimpleImputer(strategy="median"),LogisticRegression(max_iter=1000,C=0.5)),
        "extra_trees":lambda:make_pipeline(SimpleImputer(strategy="median"),ExtraTreesClassifier(n_estimators=300,min_samples_leaf=20,n_jobs=-1,random_state=42)),
        "hgb":lambda:make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(max_iter=250,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=1.0,random_state=42)),
    }

def main():
    if not PRICE_DIR.exists(): raise SystemExit("DEFERRED: price dataset is absent")
    df=pd.read_parquet(PRICE_DIR)
    if len(df)<2000: raise SystemExit(f"DEFERRED: insufficient price rows ({len(df)})")
    df=add_targets(add_cross_sectional_context(add_technical_features(df)))
    fa=audit_feature_columns(FEATURE_COLUMNS)
    targets=[c for c in df.columns if c.startswith("target_")]
    ts=audit_target_separation(FEATURE_COLUMNS,targets)
    audit={"feature_columns":list(FEATURE_COLUMNS),"ok":fa.ok and ts.ok,"violations":list(fa.violations+ts.violations)}
    AUDIT.parent.mkdir(parents=True,exist_ok=True); AUDIT.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    if not audit["ok"]: raise SystemExit(f"FAIL: leakage audit {audit['violations']}")
    df=df.dropna(subset=FEATURE_COLUMNS+["target_up_1d"]).copy()
    frozen_path=Path("config/frozen_holdout.json")
    if frozen_path.exists():
        frozen=json.loads(frozen_path.read_text(encoding="utf-8"))
        cutoff=pd.Timestamp(frozen["cutoff_date"]).date()
        df=df[pd.to_datetime(df["session_date"]).dt.date<=cutoff].copy()
    dates=sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    folds=make_date_folds(dates,min_train=252,test_size=21,step=21,embargo=1)
    if len(folds)<3: raise SystemExit(f"DEFERRED: only {len(folds)} OOS folds available")
    model_results={}; regime_rows={reg.value:[] for reg in Regime if reg is not Regime.DATA_STRESSED}
    for name,factory in make_models().items():
        fold_rows=[]
        for fold in folds:
            train_dates=dates[:fold.train_end]
            cal_n=max(20,int(len(train_dates)*0.2))
            core_dates=set(train_dates[:-cal_n]); cal_dates=set(train_dates[-cal_n:]); test_dates=set(dates[fold.test_start:fold.test_end])
            core=df[df.session_date.isin(core_dates)]; cal=df[df.session_date.isin(cal_dates)]; test=df[df.session_date.isin(test_dates)]
            if min(len(core),len(cal),len(test))<50: continue
            if core.target_up_1d.nunique()<2 or cal.target_up_1d.nunique()<2 or test.target_up_1d.nunique()<2: continue
            threshold=float(core["volatility_20"].dropna().quantile(0.75)) if core["volatility_20"].notna().any() else 0.02
            model=factory(); model.fit(core[FEATURE_COLUMNS],core.target_up_1d.astype(int))
            cal_p=model.predict_proba(cal[FEATURE_COLUMNS])[:,1]; calibrator=PlattCalibrator().fit(cal_p,cal.target_up_1d.astype(int))
            p=calibrator.predict(model.predict_proba(test[FEATURE_COLUMNS])[:,1])
            row=classification_metrics(test.target_up_1d.astype(int),p); row["n_test"]=float(len(test)); fold_rows.append(row)
            regime=test.apply(lambda x:regime_for_row(x["volatility_20"],x["price_vs_sma60"],threshold).value,axis=1)
            for reg_name in regime.unique():
                if reg_name not in regime_rows: continue
                mask=regime.eq(reg_name)
                if mask.sum()==0 or test.loc[mask,"target_up_1d"].nunique()<2: continue
                rr=classification_metrics(test.loc[mask,"target_up_1d"].astype(int),p[mask]); rr["n_test"]=float(mask.sum()); regime_rows[reg_name].append((name,rr))
        model_results[name]={"folds":len(fold_rows),"metrics":aggregate_metric_rows(fold_rows)}
    usable={k:v for k,v in model_results.items() if v["folds"]>=3 and "logloss" in v["metrics"]}
    if not usable: raise SystemExit("DEFERRED: no model has >=3 valid OOS folds")
    regime_metrics={}
    selected={}
    for reg_name,rows in regime_rows.items():
        grouped={}
        for model_name,row in rows: grouped.setdefault(model_name,[]).append(row)
        aggregated={m:aggregate_metric_rows(rs) for m,rs in grouped.items()}
        regime_metrics[reg_name]=aggregated
        if aggregated: selected[reg_name]=choose_from_oos(reg_name,aggregated).names[0]
    global_selected=min(usable,key=lambda k:usable[k]["metrics"]["logloss"])
    payload={"results":model_results,"regime_metrics":regime_metrics,"regime_selected_models":selected,
             "selected_model":global_selected,
             "selection_basis":"minimum mean chronological OOS logloss; regime routing also uses only OOS regime metrics; calibration is inside each training fold; frozen holdout unused",
             "status":"OOS_COMPLETE"}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
