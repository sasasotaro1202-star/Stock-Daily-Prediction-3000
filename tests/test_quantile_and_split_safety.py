from __future__ import annotations

import numpy as np

from src.prediction.regression import make_quantile_model, make_quantile_models
from src.prediction.targets import add_targets
import pandas as pd


def test_quantile_factory_keys_are_stable():
    models=make_quantile_models()
    assert set(models)=={"q10","q50","q90"}


def test_quantile_model_constructs_and_predicts():
    model=make_quantile_model(0.10)
    x=np.arange(200,dtype=float).reshape(-1,1)
    y=np.linspace(-0.1,0.1,200)
    model.fit(x,y)
    pred=model.predict(x[-5:])
    assert np.isfinite(pred).all()


def test_split_event_is_excluded_from_raw_return_target():
    df=pd.DataFrame({
        "symbol":["AAA","AAA","AAA"],
        "session_date":pd.to_datetime(
            ["2026-01-02","2026-01-03","2026-01-06"]
        ),
        "close":[100.0,50.0,51.0],
        "stock_splits":[0.0,2.0,0.0],
    })
    out=add_targets(df)
    assert pd.isna(out.loc[0,"target_ret_1d"])
