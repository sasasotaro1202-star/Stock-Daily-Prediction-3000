from pathlib import Path

import numpy as np
import pandas as pd

from src.features.technical import FEATURE_COLUMNS, add_technical_features


def _sample():
    rows=[]
    for symbol in ("AAA","BBB"):
        for i in range(100):
            close=100.0+i+(0.2*i if symbol=="BBB" else 0)
            rows.append({
                "symbol":symbol,
                "session_date":pd.Timestamp("2025-01-01")+pd.Timedelta(days=i),
                "open":close-0.2,
                "high":close+1.0,
                "low":close-1.0,
                "close":close,
                "volume":1000+i,
            })
    return pd.DataFrame(rows)


def test_extended_technical_features_are_numeric_and_present():
    out=add_technical_features(_sample())
    assert set(FEATURE_COLUMNS).issuperset({
        "volatility_5",
        "volatility_ratio_5_20",
        "volume_z20",
        "dollar_volume_ratio_20",
        "amihud_20",
        "return_z20",
        "range_z20",
        "dow_sin",
        "month_cos",
    })
    numeric=out[FEATURE_COLUMNS].apply(pd.to_numeric,errors="coerce")
    assert numeric.shape==out[FEATURE_COLUMNS].shape


def test_feature_row_count_is_preserved():
    out=add_technical_features(_sample())
    assert len(out)==200
