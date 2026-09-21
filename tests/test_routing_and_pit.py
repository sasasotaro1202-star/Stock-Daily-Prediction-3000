from __future__ import annotations

import pandas as pd

from src.ranking.cross_sectional import cross_sectional_rank
from src.research.router import route_plan


def test_route_prefers_asset_regime_oos_evidence():
    metrics = {
        "jp_stock::high_vol": {
            "logistic": {"logloss": 0.70, "logloss_std": 0.01, "folds": 2},
            "hgb": {"logloss": 0.60, "logloss_std": 0.01, "folds": 2},
        }
    }
    plan = route_plan(
        "jp_stock",
        "high_vol",
        asset_regime_metrics=metrics,
        asset_metrics={},
        regime_metrics={},
        global_selected="logistic",
    )
    assert plan.names == ("hgb",)
    assert plan.scope == "jp_stock::high_vol"


def test_cross_sectional_rank_does_not_mix_market_families():
    df = pd.DataFrame(
        {
            "prediction_date": [pd.Timestamp("2026-09-22").date()] * 4,
            "asset_class": ["jp_stock", "jp_stock", "us_stock", "us_stock"],
            "p_up_1d": [0.80, 0.60, 0.90, 0.50],
            "expected_return_1d": [0.04, 0.01, 0.05, 0.00],
        }
    )
    out = cross_sectional_rank(df)
    jp = out[out.asset_class.eq("jp_stock")]
    us = out[out.asset_class.eq("us_stock")]
    assert jp["rank_probability"].max() <= 1.0
    assert us["rank_probability"].max() <= 1.0
    assert jp["rank_probability"].min() >= 0.5
    assert us["rank_probability"].min() >= 0.5
