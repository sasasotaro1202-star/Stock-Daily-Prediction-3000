from __future__ import annotations

import numpy as np
import pandas as pd

from src.prediction.regression import make_quantile_model
from src.research.router import Regime, regime_for_row


def test_quantile_model_constructs():
    model=make_quantile_model(0.10)
    x=np.arange(200,dtype=float).reshape(-1,1)
    y=np.linspace(-0.1,0.1,200)
    model.fit(x,y)
    pred=model.predict(x[-5:])
    assert np.isfinite(pred).all()


def test_event_and_nan_routes_are_fail_closed():
    assert regime_for_row(0.10,0.0,0.20,gap_pct=0.05)==Regime.EVENT
    assert regime_for_row(float("nan"),0.0,0.20)==Regime.DATA_STRESSED
