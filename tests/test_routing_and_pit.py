from __future__ import annotations

import pandas as pd

from src.ranking.cross_sectional import cross_sectional_rank
from src.research.router import route_plan


def test_route_prefers_asset_regime_oos_evidence():
    metrics = {
        "jp_stock::high_vol": {
            "logistic": {"logloss": 0.70, "logloss_std": 0.01, "folds": 3},
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


def test_cross_sectional_context_requires_market_family_separation():
    from src.features.context import add_cross_sectional_context

    df = pd.DataFrame({
        "symbol": ["JP1", "US1"],
        "asset_class": ["jp_stock", "us_stock"],
        "session_date": [pd.Timestamp("2026-09-21").date()] * 2,
        "ret_1d": [0.01, 0.99],
        "volatility_20": [0.10, 0.90],
    })
    out = add_cross_sectional_context(df)
    assert out.loc[out.symbol.eq("JP1"), "cs_ret_1d_rank"].iloc[0] == 1.0
    assert out.loc[out.symbol.eq("US1"), "cs_ret_1d_rank"].iloc[0] == 1.0


def test_event_regime_is_detected_before_volatility():
    from src.research.router import Regime, regime_for_row

    assert regime_for_row(0.10, 0.00, 0.20, gap_pct=0.04) == Regime.EVENT
    assert regime_for_row(0.10, 0.00, 0.20, volume_ratio_20=3.5) == Regime.EVENT


def test_nan_market_state_is_data_stressed():
    from src.research.router import Regime, regime_for_row

    assert regime_for_row(float("nan"), 0.01, 0.20) == Regime.DATA_STRESSED
    assert regime_for_row(0.10, float("nan"), 0.20) == Regime.DATA_STRESSED


def test_global_selection_balances_asset_classes():
    from src.research.router import rebalance_global_oos_candidates

    global_metrics = {
        "logistic": {"logloss": 0.60, "logloss_std": 0.01, "folds": 4},
        "hgb": {"logloss": 0.62, "logloss_std": 0.01, "folds": 4},
    }
    asset_metrics = {
        "jp_stock": {
            "logistic": {"logloss": 0.55, "folds": 4},
            "hgb": {"logloss": 0.60, "folds": 4},
        },
        "us_stock": {
            "logistic": {"logloss": 0.80, "folds": 4},
            "hgb": {"logloss": 0.65, "folds": 4},
        },
    }
    balanced = rebalance_global_oos_candidates(
        global_metrics,
        asset_metrics,
        blend_weight=0.50,
        min_folds=3,
    )
    assert balanced["logistic"]["asset_class_macro_logloss"] == 0.675
    assert balanced["hgb"]["asset_class_macro_logloss"] == 0.625
    assert balanced["logistic"]["selection_logloss"] == 0.6375
    assert balanced["hgb"]["selection_logloss"] == 0.6225


def test_rank_ic_tiebreak_prefers_rank_quality_within_tolerance():
    from src.research.router import choose_from_oos

    metrics = {
        "logistic": {"logloss": 0.6000, "logloss_std": 0.0100, "rank_ic": 0.010, "folds": 4},
        "hgb": {"logloss": 0.6010, "logloss_std": 0.0100, "rank_ic": 0.080, "folds": 4},
    }
    plan = choose_from_oos(
        "normal",
        metrics,
        candidates=("logistic", "hgb"),
        min_folds=3,
        rank_ic_tiebreak_tolerance=0.002,
    )
    assert plan.names == ("hgb",)
    assert "rank_ic_tiebreak" in plan.reason
