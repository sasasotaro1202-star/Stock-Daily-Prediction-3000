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


def test_backtest_ranks_within_asset_class():
    from src.backtest.cross_sectional import evaluate_predictions

    dates = pd.to_datetime(["2026-09-21", "2026-09-22"])
    symbols = [f"JP{i}" for i in range(20)] + [f"US{i}" for i in range(20)]
    asset = ["jp_stock"] * 20 + ["us_stock"] * 20
    rows = []
    predictions = []
    for i, (symbol, asset_class) in enumerate(zip(symbols, asset)):
        first_close = 100.0
        second_close = 110.0 if i % 10 < 2 else 100.0
        rows.extend([
            {
                "symbol": symbol,
                "asset_class": asset_class,
                "session_date": dates[0],
                "close": first_close,
                "available_at": pd.Timestamp("2026-09-21T10:00:00Z"),
            },
            {
                "symbol": symbol,
                "asset_class": asset_class,
                "session_date": dates[1],
                "close": second_close,
                "available_at": pd.Timestamp("2026-09-22T10:00:00Z"),
            },
        ])
        predictions.append({
            "symbol": symbol,
            "asset_class": asset_class,
            "session_date": dates[0],
            "prediction_date": dates[0],
            "prediction_time": pd.Timestamp("2026-09-21T12:00:00Z"),
            "expected_return_1d": float((i % 20) + (1000 if asset_class == "us_stock" else 0)),
        })

    result = evaluate_predictions(
        pd.DataFrame(predictions),
        pd.DataFrame(rows),
        top_quantile=0.10,
        cost_bps=0.0,
    )
    assert result["status"] == "PASS"
    assert result["rows"] == 2
    assert result["days"] == 1
    assert set(result["asset_classes"]) == {"jp_stock", "us_stock"}


def test_backtest_uses_next_session_outcome_availability_for_pit():
    from src.backtest.cross_sectional import evaluate_predictions

    base = pd.Timestamp("2026-09-21")
    bars = []
    preds = []
    for i in range(20):
        symbol = f"A{i}"
        bars.extend([
            {
                "symbol": symbol,
                "asset_class": "jp_stock",
                "session_date": base,
                "close": 100.0,
                "available_at": pd.Timestamp("2026-09-21T10:00:00Z"),
            },
            {
                "symbol": symbol,
                "asset_class": "jp_stock",
                "session_date": base + pd.Timedelta(days=1),
                "close": 101.0 + i,
                "available_at": pd.Timestamp("2026-09-22T10:00:00Z"),
            },
        ])
        preds.append({
            "symbol": symbol,
            "asset_class": "jp_stock",
            "session_date": base,
            "prediction_date": base,
            "prediction_time": pd.Timestamp("2026-09-21T12:00:00Z"),
            "expected_return_1d": float(i),
        })

    result = evaluate_predictions(
        pd.DataFrame(preds),
        pd.DataFrame(bars),
        top_quantile=0.10,
        cost_bps=0.0,
    )
    assert result["status"] == "PASS"
    assert result["rows"] == 1
    assert result["days"] == 1
