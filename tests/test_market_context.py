from __future__ import annotations

import pandas as pd

from src.features.context import add_market_context


def test_market_context_uses_latest_pit_observation():
    prices=pd.DataFrame({
        "symbol":["AAA","AAA"],
        "session_date":pd.to_datetime(
            ["2026-09-22","2026-09-22"]
        ).date,
        "available_at":pd.to_datetime([
            "2026-09-22T07:00:00Z",
            "2026-09-22T09:00:00Z",
        ]),
        "ret_1d":[0.01,0.02],
        "volatility_20":[0.10,0.11],
    })
    context=pd.DataFrame({
        "session_date":pd.to_datetime([
            "2026-09-21","2026-09-22"
        ]).date,
        "family":["sp500","sp500"],
        "available_at":pd.to_datetime([
            "2026-09-22T05:30:00Z",
            "2026-09-23T05:30:00Z",
        ]),
        "ret_1d":[0.10,0.20],
        "volatility_20":[0.40,0.50],
        "close":[100.0,110.0],
    })
    out=add_market_context(prices,context)
    assert out.loc[0,"sp500_ret_1d_lag1"]==0.10
    assert out.loc[0,"sp500_ret_1d_lag1"] != 0.20
    assert out.loc[1,"sp500_ret_1d_lag1"]==0.10


def test_market_context_keeps_price_row_count():
    prices=pd.DataFrame({
        "symbol":["AAA","BBB"],
        "session_date":pd.to_datetime(
            ["2026-09-22","2026-09-22"]
        ).date,
        "available_at":pd.to_datetime([
            "2026-09-22T07:00:00Z",
            "2026-09-22T07:00:00Z",
        ]),
        "ret_1d":[0.01,-0.01],
        "volatility_20":[0.10,0.20],
    })
    context=pd.DataFrame({
        "session_date":pd.to_datetime(
            ["2026-09-21","2026-09-22"]
        ).date,
        "family":["nikkei","nikkei"],
        "available_at":pd.to_datetime([
            "2026-09-21T07:30:00Z",
            "2026-09-22T07:30:00Z",
        ]),
        "ret_1d":[0.01,0.02],
        "volatility_20":[0.30,0.31],
        "close":[100.0,101.0],
    })
    out=add_market_context(prices,context)
    assert len(out)==len(prices)


def test_market_context_requires_available_at():
    prices=pd.DataFrame({
        "symbol":["AAA"],
        "session_date":[pd.Timestamp("2026-09-22").date()],
        "ret_1d":[0.01],
        "volatility_20":[0.10],
    })
    context=pd.DataFrame({
        "session_date":[pd.Timestamp("2026-09-21").date()],
        "family":["sp500"],
        "available_at":[pd.Timestamp("2026-09-22T05:30:00Z")],
        "ret_1d":[0.10],
        "volatility_20":[0.40],
        "close":[100.0],
    })
    try:
        add_market_context(prices,context)
    except ValueError as exc:
        assert "available_at" in str(exc)
    else:
        raise AssertionError("expected missing available_at to fail")


def test_cross_sectional_robust_zscores_are_market_scoped_and_clipped():
    from src.features.context import add_cross_sectional_context

    n=10
    rows=[]
    for i in range(n):
        rows.append({
            "symbol":f"A{i}",
            "asset_class":"jp_stock",
            "session_date":pd.Timestamp("2026-09-22"),
            "ret_1d":float(i),
            "volatility_20":float(10-i),
            "volume_ratio_20":float(i+1),
            "range_pct":float(i)/100.0,
            "price_vs_sma20":float(i)/100.0,
        })
    out=add_cross_sectional_context(pd.DataFrame(rows))
    cols=[
        "cs_ret_1d_robust_z",
        "cs_volatility_20_robust_z",
        "cs_volume_ratio_20_robust_z",
        "cs_range_pct_robust_z",
        "cs_price_vs_sma20_robust_z",
    ]
    for col in cols:
        assert out[col].notna().all()
        assert float(out[col].abs().max()) <= 5.0


def test_macro_context_fields_flow_into_features():
    from src.features.technical import FEATURE_COLUMNS

    assert {
        "us10y_level_lag1",
        "dxy_ret_1d_lag1",
        "gold_ret_1d_lag1",
        "oil_ret_1d_lag1",
        "hyg_ret_1d_lag1",
    }.issubset(FEATURE_COLUMNS)


def test_topix_provider_symbol_is_supported_yahoo_index_code():
    from src.data.market_context import CONTEXT_SYMBOLS

    assert CONTEXT_SYMBOLS["topix"] == "^TOPX"


def test_market_context_quality_requires_extended_macro_families(tmp_path, monkeypatch):
    import scripts.update_market_context as updater

    rows = []
    families = sorted({
        "nikkei", "topix", "sp500", "nasdaq", "vix", "usd_jpy",
        "us10y", "dxy", "gold", "oil", "hyg",
    })
    for family in families:
        rows.append({
            "family": family,
            "session_date": pd.Timestamp("2026-09-22").date(),
            "available_at": pd.Timestamp("2026-09-23T00:00:00Z"),
            "ret_1d": 0.0,
            "volatility_20": 0.1,
            "close": 100.0,
        })
    monkeypatch.chdir(tmp_path)
    updater.OUT = tmp_path / "data" / "market_context.parquet"
    updater.RESULT = tmp_path / "data" / "research" / "market_context_quality.json"


def test_market_context_download_records_retrieval_provenance(monkeypatch):
    import src.data.market_context as market_context

    idx=pd.to_datetime(["2026-09-22"], name="Date")
    columns=pd.MultiIndex.from_product(
        [["^TOPX"],["Close"]]
    )
    raw=pd.DataFrame([[2500.0]],index=idx,columns=columns)
    monkeypatch.setenv("GITHUB_RUN_ID","123456")
    monkeypatch.setattr(market_context.yf,"download",lambda *args,**kwargs: raw)
    out=market_context.download_market_context(period="1mo")
    assert out["source"].tolist()==["yfinance"]
    assert out["provider_symbol"].tolist()==["^TOPX"]
    assert out["retrieved_at"].notna().all()
    assert out["retrieval_run_id"].tolist()==["123456"]
    assert pd.api.types.is_datetime64tz_dtype(out["retrieved_at"])
