import numpy as np
import pandas as pd


def test_rd_style_factor_pack_is_numeric_after_warmup():
    from src.features.technical import add_technical_features

    rows=[]
    for i in range(90):
        d=pd.Timestamp("2026-01-01")+pd.Timedelta(days=i)
        rows.append({
            "symbol":"A",
            "asset_class":"jp_stock",
            "session_date":d,
            "open":100+i*0.2,
            "high":101+i*0.2,
            "low":99+i*0.2,
            "close":100+i*0.2,
            "volume":1000+i*5,
        })
    out=add_technical_features(pd.DataFrame(rows))
    cols=[
        "ret_60d","return_log_volume_corr_5","return_log_volume_corr_10",
        "trend_slope_60","trend_r2_60","resi_5_pct","resi_10_pct",
        "wvma_5","volume_std_ratio_5",
    ]
    warm=out.iloc[-1]
    assert all(np.isfinite(float(warm[c])) for c in cols)


def test_market_relative_momentum_features_are_causal_and_market_scoped():
    from src.features.context import add_cross_sectional_context
    from src.features.technical import add_technical_features

    rows = []
    for i in range(25):
        d = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
        for symbol, asset_class, drift in (
            ("JP1", "jp_stock", 0.001),
            ("JP2", "jp_stock", -0.0005),
            ("US1", "us_stock", 0.01),
            ("US2", "us_stock", 0.009),
        ):
            close = 100.0 * (1.0 + drift) ** i
            rows.append(
                {
                    "symbol": symbol,
                    "asset_class": asset_class,
                    "session_date": d,
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close,
                    "volume": 1000 + i,
                }
            )
    out = add_cross_sectional_context(add_technical_features(pd.DataFrame(rows)))

    assert {
        "market_median_ret_5d",
        "market_median_ret_20d",
        "residual_momentum_20d",
        "market_median_vol_20d",
        "residual_volatility_20d",
    }.issubset(out.columns)

    early = out[
        pd.to_datetime(out["session_date"]) < pd.Timestamp("2026-01-20")
    ]
    assert early["market_median_ret_20d"].isna().all()

    jp = out[out["asset_class"].eq("jp_stock")].iloc[-1]
    us = out[out["asset_class"].eq("us_stock")].iloc[-1]
    assert float(jp["market_median_ret_20d"]) != float(us["market_median_ret_20d"])
    assert set(out["market_family"].unique()) == {"jp", "us"}


def test_cross_sectional_dispersion_context_is_available_and_causal():
    frame = pd.DataFrame({
        "session_date": pd.to_datetime(["2026-01-02"] * 12),
        "market_family": ["jp"] * 12,
        "asset_class": ["jp_stock"] * 12,
        "symbol": [f"S{i}" for i in range(12)],
        "ret_1d": np.linspace(-0.055, 0.055, 12),
        "volatility_20": np.linspace(0.01, 0.04, 12),
        "price_vs_sma20": np.zeros(12),
        "volume_ratio_20": np.ones(12),
        "range_pct": np.full(12, 0.02),
        "ret_20d": np.zeros(12),
    })
    out = add_cross_sectional_context(frame)
    assert float(out["market_dispersion_1d"].iloc[0]) > 0.0
    assert float(out["market_ret_iqr_1d"].iloc[0]) > 0.0
