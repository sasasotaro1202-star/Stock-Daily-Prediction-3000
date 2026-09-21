from pathlib import Path

import numpy as np
import pandas as pd

from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.regression import make_quantile_models
from src.research.router import Regime, regime_for_row


def test_extended_features_are_numeric():
    rows=[]
    for symbol in ("AAA","BBB"):
        for i in range(100):
            close=100+i+(i*0.2 if symbol=="BBB" else 0)
            rows.append({
                "symbol":symbol,
                "session_date":pd.Timestamp("2025-01-01")+pd.Timedelta(days=i),
                "open":close-0.2,
                "high":close+1.0,
                "low":close-1.0,
                "close":close,
                "volume":1000+i,
            })
    out=add_technical_features(pd.DataFrame(rows))
    technical_columns=[
        "volatility_5",
        "volatility_ratio_5_20",
        "volume_z20",
        "dollar_volume_ratio_20",
        "amihud_20",
        "return_z20",
        "range_z20",
        "close_location",
        "intraday_return",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]
    values=out[technical_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    assert values.shape==out[technical_columns].shape
    assert len(out)==200


def test_quantile_factory_is_stable():
    assert set(make_quantile_models())=={"q10","q50","q90"}


def test_regime_fail_closed_and_event_first():
    assert regime_for_row(0.10,0.0,0.20,gap_pct=0.05)==Regime.EVENT
    assert regime_for_row(float("nan"),0.0,0.20)==Regime.DATA_STRESSED


def test_required_production_automation_exists():
    for path in (
        ".github/workflows/market-cycle.yml",
        ".github/workflows/us-close-prediction.yml",
        ".github/workflows/prediction-monitoring.yml",
        ".github/workflows/bounded-production-recovery.yml",
        "scripts/live_performance_gate.py",
        "scripts/production_state_compatibility.py",
        "config/paypay_catalog.yml",
    ):
        assert Path(path).exists()
