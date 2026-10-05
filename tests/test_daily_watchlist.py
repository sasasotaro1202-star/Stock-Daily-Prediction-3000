from pathlib import Path
import json

import pandas as pd
import pytest

from scripts.generate_daily_watchlist import (
    build_market_rows,
    load_config,
    next_session_date,
    validate_items,
)


def test_watchlist_has_all_requested_instruments():
    cfg = load_config(Path("config/daily_watchlist.yml"))
    jp = cfg["equities"]["items"]
    us = cfg["us_equities"]["items"]
    markets = cfg["market_instruments"]["instruments"]

    assert len(jp) == 25
    assert len(us) == 13
    assert {x["display_name"] for x in markets} == {"日経平均株価", "S&P500", "ドル/円"}
    assert {x["symbol"] for x in jp} == {
        "8802","8801","9409","9404","4676","9401","285A","1333","2802","2897",
        "7267","7201","4689","4385","215A","2379","9843","9432","6752","6758",
        "8035","6857","7974","9434","7203",
    }
    assert {x["symbol"] for x in us} == {
        "DIS","SBUX","MCD","NFLX","KO","TSLA","NKE","META","AMZN","GOOGL","MSFT","NVDA","AAPL",
    }


def test_validate_items_rejects_duplicate():
    with pytest.raises(SystemExit, match="duplicate watchlist key"):
        validate_items(
            [
                {"display_name": "A", "symbol": "7203", "asset_class": "jp_stock"},
                {"display_name": "B", "symbol": "7203", "asset_class": "jp_stock"},
            ],
            "jp_stock",
        )


def test_market_baseline_has_direction_and_pit_cutoff():
    n = 160
    dates = pd.date_range("2026-01-01", periods=n, freq="B").date
    context = pd.DataFrame(
        {
            "session_date": dates,
            "family": ["nikkei"] * n,
            "provider_symbol": ["^N225"] * n,
            "close": [100.0 + 0.1 * i + (i % 4) for i in range(n)],
            "ret_1d": [0.003 if i % 2 else -0.002 for i in range(n)],
            "volatility_20": [0.01 + i * 0.00002 for i in range(n)],
            "available_at": pd.date_range("2026-01-01", periods=n, freq="B", tz="UTC")
            + pd.Timedelta(hours=8),
        }
    )
    rows = build_market_rows(
        context,
        [{
            "display_name": "日経平均株価",
            "instrument_id": "JP-NIKKEI-225",
            "provider_symbol": "^N225",
            "context_family": "nikkei",
            "calendar": "XTKS",
        }],
        cutoff=pd.Timestamp("2026-07-01", tz="UTC"),
        threshold=0.50,
    )
    assert rows[0]["prediction_status"] == "READY"
    assert rows[0]["direction"] in {"UP", "DOWN"}
    assert rows[0]["target_date"] > rows[0]["session_date"]
    assert rows[0]["production_status"] == "RESEARCH_ONLY"
    assert rows[0]["target_date"] == "2026-07-01"


def test_watchlist_workflow_contract():
    workflow = Path(".github/workflows/daily-watchlist.yml").read_text(encoding="utf-8")
    assert 'cron: "30 8 * * 1-5"' in workflow
    assert "restore_prediction_history.py" in workflow
    assert "restore_latest_universe_state.py" in workflow
    assert "update_market_context.py" in workflow
    assert "generate_daily_watchlist.py" in workflow
    assert "contents: read" in workflow


def test_fx_24_5_next_session_skips_weekend():
    assert next_session_date("2026-10-02", "24/5") == "2026-10-05"


def test_equity_prediction_cutoff_is_never_from_the_future(tmp_path, monkeypatch):
    import scripts.generate_daily_watchlist as module

    prediction_path = tmp_path / "prediction_20990101T000000Z.parquet"
    frame = pd.DataFrame(
        {
            "asset_class": ["jp_stock"],
            "symbol": ["7203"],
            "p_up_1d": [0.90],
            "prediction_time": ["2099-01-01T00:00:00Z"],
            "prediction_status": ["READY"],
        }
    )
    frame.to_parquet(prediction_path, index=False)

    row, filename = module.latest_equity_prediction(
        [prediction_path],
        "jp_stock",
        "7203",
        cutoff=pd.Timestamp("2026-10-05T08:00:00Z"),
    )
    assert row is None
    assert filename is None


def test_stale_market_context_is_deferred():
    n = 160
    dates = pd.date_range("2026-01-01", periods=n, freq="B").date
    context = pd.DataFrame(
        {
            "session_date": dates,
            "family": ["nikkei"] * n,
            "provider_symbol": ["^N225"] * n,
            "close": [100.0 + 0.1 * i for i in range(n)],
            "ret_1d": [0.001 if i % 2 else -0.001 for i in range(n)],
            "volatility_20": [0.01] * n,
            "available_at": pd.date_range("2026-01-01", periods=n, freq="B", tz="UTC")
            + pd.Timedelta(hours=8),
        }
    )
    rows = build_market_rows(
        context,
        [{
            "display_name": "日経平均株価",
            "instrument_id": "JP-NIKKEI-225",
            "provider_symbol": "^N225",
            "context_family": "nikkei",
            "calendar": "XTKS",
        }],
        cutoff=pd.Timestamp("2026-08-30", tz="UTC"),
        threshold=0.50,
        expected_target_dates={"XTKS": "2026-09-01"},
    )
    assert rows[0]["prediction_status"] == "DEFERRED_STALE_CONTEXT"
