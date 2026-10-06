from pathlib import Path
import json

import pandas as pd
import pytest

from scripts.generate_daily_watchlist import (
    build_market_rows,
    load_config,
    next_session_date,
    latest_equity_prediction,
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
    assert len(jp) + len(us) + len(markets) == 41
    assert cfg["priority_watchlist"]["mandatory_daily_prediction"] is True
    assert int(cfg["priority_watchlist"]["expected_count"]) == 41


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
    n = 190
    dates = pd.bdate_range(end="2026-06-30", periods=n).date
    context = pd.DataFrame(
        {
            "session_date": dates,
            "family": ["nikkei"] * n,
            "provider_symbol": ["^N225"] * n,
            "close": [100.0 + 0.1 * i + (i % 4) for i in range(n)],
            "ret_1d": [0.003 if i % 2 else -0.002 for i in range(n)],
            "volatility_20": [0.01 + i * 0.00002 for i in range(n)],
            "available_at": pd.DatetimeIndex(pd.bdate_range(end="2026-06-30", periods=n)).tz_localize("UTC")
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
    assert 'cron: "50 18 * * 1-5"' in workflow
    assert 'timezone: "Asia/Tokyo"' in workflow
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
            "close": [100.0 + 0.1 * i + (i % 4) for i in range(n)],
            "ret_1d": [0.001 if i % 2 else -0.001 for i in range(n)],
            "volatility_20": [0.01] * n,
            "available_at": pd.to_datetime(dates).tz_localize("UTC") + pd.Timedelta(hours=8),
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


def test_main_binds_watchlist_items_from_config():
    source = Path("scripts/generate_daily_watchlist.py").read_text(encoding="utf-8")
    assert 'jp_items = cfg["equities"]["items"]' in source
    assert 'us_items = cfg["us_equities"]["items"]' in source


def test_workflow_runs_output_integrity_gate():
    workflow = Path(".github/workflows/daily-watchlist.yml").read_text(encoding="utf-8")
    assert "validate_daily_watchlist.py" in workflow


def test_equity_target_date_is_derived_from_session_date():
    path = Path("/tmp/unused.parquet")
    frame = pd.DataFrame(
        {
            "asset_class": ["us_stock"],
            "symbol": ["AAPL"],
            "p_up_1d": [0.60],
            "prediction_time": ["2026-10-05T07:17:00Z"],
            "available_at": ["2026-10-05T07:16:00Z"],
            "prediction_status": ["READY"],
            "session_date": ["2026-10-02"],
        }
    )
    temp = Path(__import__("tempfile").mkstemp(suffix=".parquet")[1])
    try:
        frame.to_parquet(temp, index=False)
        row, filename = latest_equity_prediction(
            [temp],
            "us_stock",
            "AAPL",
            cutoff=pd.Timestamp("2026-10-05T08:00:00Z"),
            expected_target_date="2026-10-05",
            calendar_code="XNYS",
        )
        assert row is not None
        assert filename == temp.name
    finally:
        temp.unlink(missing_ok=True)


def test_equity_prediction_rejects_unavailable_before_cutoff(tmp_path):
    import scripts.generate_daily_watchlist as module

    path = tmp_path / "prediction_pit_invalid.parquet"
    frame = pd.DataFrame(
        {
            "asset_class": ["jp_stock"],
            "symbol": ["7203"],
            "p_up_1d": [0.80],
            "prediction_time": ["2026-10-05T07:17:00Z"],
            "available_at": ["2026-10-05T07:18:00Z"],
            "prediction_status": ["READY"],
            "session_date": ["2026-10-03"],
        }
    )
    frame.to_parquet(path, index=False)

    row, filename = module.latest_equity_prediction(
        [path],
        "jp_stock",
        "7203",
        cutoff=pd.Timestamp("2026-10-05T08:00:00Z"),
    )
    assert row is None
    assert filename is None


def test_watchlist_validator_rejects_non_monotone_quantiles(tmp_path, monkeypatch):
    import scripts.validate_daily_watchlist as validator

    config = {
        "priority_watchlist": {"mandatory_daily_prediction": True, "expected_count": 1},
        "equities": {"items": [{"display_name": "A", "symbol": "7203", "asset_class": "jp_stock"}]},
        "us_equities": {"items": []},
        "market_instruments": {"instruments": []},
    }
    rows = [{
        "instrument_type": "equity",
        "display_name": "A",
        "symbol": "7203",
        "prediction_status": "READY",
        "p_up_1d": 0.6,
        "direction": "UP",
        "prediction_time": "2026-10-05T08:00:00Z",
        "available_at": "2026-10-05T07:55:00Z",
        "target_date": "2026-10-06",
        "q10_1d": 102.0,
        "q50_1d": 101.0,
        "q90_1d": 103.0,
    }]
    cfg_path = tmp_path / "config.yml"
    out_path = tmp_path / "watch.json"
    cfg_path.write_text(__import__("yaml").safe_dump(config, allow_unicode=True), encoding="utf-8")
    out_path.write_text(__import__("json").dumps({
        "cutoff": "2026-10-05T08:30:00+00:00",
        "rows": rows,
        "coverage": {"total": 1, "ready": 1, "deferred": 0, "production_ready": 1},
    }), encoding="utf-8")
    monkeypatch.setattr(validator, "CONFIG", cfg_path)
    monkeypatch.setattr(validator, "OUTPUT", out_path)
    with pytest.raises(SystemExit, match="non-monotone q10/q50/q90"):
        validator.main()


def test_near_production_prediction_writes_timestamped_snapshot_and_quantiles():
    source = Path("scripts/run_now_prediction.py").read_text(encoding="utf-8")
    assert 'f"prediction_{stamp}.parquet"' in source
    assert 'q50_v = qmodels["q50"].predict(frame)' in source
    assert 'out["q10_1d"] = out["range_low_1d"]' in source
    assert 'out["q50_1d"] = out["close"] * (1 + q50_return)' in source
    assert 'out["q90_1d"] = out["range_high_1d"]' in source
    production = Path("scripts/run_daily_prediction.py").read_text(encoding="utf-8")
    assert 'q50 = qmodels["q50"].predict(group[FEATURE_COLUMNS])' in production
    assert 'latest["q50_1d"] = np.where(' in production


def test_validator_rejects_inconsistent_production_status(tmp_path, monkeypatch):
    import scripts.validate_daily_watchlist as validator

    config = {
        "priority_watchlist": {"mandatory_daily_prediction": True, "expected_count": 1},
        "equities": {
            "items": [
                {"display_name": "A", "symbol": "7203", "asset_class": "jp_stock"}
            ]
        },
        "us_equities": {"items": []},
        "market_instruments": {"instruments": []},
    }
    rows = [{
        "instrument_type": "equity",
        "display_name": "A",
        "symbol": "7203",
        "prediction_status": "READY",
        "production_status": "NEAR_PRODUCTION_PREDICTION_REFERENCE",
        "p_up_1d": 0.6,
        "direction": "UP",
        "prediction_time": "2026-10-05T08:00:00Z",
        "available_at": "2026-10-05T07:55:00Z",
        "target_date": "2026-10-06",
        "q10_1d": 100.0,
        "q50_1d": 101.0,
        "q90_1d": 102.0,
    }]
    cfg_path = tmp_path / "config.yml"
    out_path = tmp_path / "watch.json"
    cfg_path.write_text(__import__("yaml").safe_dump(config, allow_unicode=True), encoding="utf-8")
    out_path.write_text(__import__("json").dumps({
        "cutoff": "2026-10-05T08:30:00+00:00",
        "rows": rows,
        "coverage": {"total": 1, "ready": 1, "production_ready": 0, "deferred": 0},
    }), encoding="utf-8")
    monkeypatch.setattr(validator, "CONFIG", cfg_path)
    monkeypatch.setattr(validator, "OUTPUT", out_path)
    with pytest.raises(SystemExit, match="not marked PRODUCTION_PREDICTION_REFERENCE"):
        validator.main()


def test_validator_excludes_research_only_market_rows_from_production_ready(tmp_path, monkeypatch):
    import scripts.validate_daily_watchlist as validator

    config = {
        "priority_watchlist": {"mandatory_daily_prediction": True, "expected_count": 2},
        "equities": {"items": [{"display_name": "A", "symbol": "7203", "asset_class": "jp_stock"}]},
        "us_equities": {"items": []},
        "market_instruments": {
            "instruments": [{
                "display_name": "Nikkei",
                "provider_symbol": "^N225",
                "instrument_id": "JP-NIKKEI-225",
                "calendar": "XTKS",
            }]
        },
    }
    rows = [
        {
            "instrument_type": "equity",
            "display_name": "A",
            "symbol": "7203",
            "prediction_status": "READY",
            "production_status": "PRODUCTION_PREDICTION_REFERENCE",
            "p_up_1d": 0.6,
            "direction": "UP",
            "prediction_time": "2026-10-05T08:00:00Z",
            "available_at": "2026-10-05T07:55:00Z",
            "target_date": "2026-10-06",
            "q10_1d": 100.0,
            "q50_1d": 101.0,
            "q90_1d": 102.0,
        },
        {
            "instrument_type": "index_or_fx",
            "display_name": "Nikkei",
            "symbol": "^N225",
            "prediction_status": "READY",
            "production_status": "RESEARCH_ONLY",
            "p_up_1d": 0.55,
            "direction": "UP",
            "prediction_time": "2026-10-05T08:00:00Z",
            "available_at": "2026-10-05T07:55:00Z",
            "target_date": "2026-10-06",
            "q10_1d": 100.0,
            "q50_1d": 101.0,
            "q90_1d": 102.0,
        },
    ]
    cfg_path = tmp_path / "config.yml"
    out_path = tmp_path / "watch.json"
    cfg_path.write_text(__import__("yaml").safe_dump(config, allow_unicode=True), encoding="utf-8")
    out_path.write_text(
        __import__("json").dumps({
            "cutoff": "2026-10-05T08:30:00+00:00",
            "rows": rows,
            "coverage": {"total": 2, "ready": 2, "production_ready": 1, "deferred": 0},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(validator, "CONFIG", cfg_path)
    monkeypatch.setattr(validator, "OUTPUT", out_path)
    validator.main()


def test_near_production_output_contract_preserves_pit_timestamp():
    source = Path("scripts/run_now_prediction.py").read_text(encoding="utf-8")
    assert '"available_at"' in source
    assert '"retrieved_at"' in source
    assert '"prediction_time"' in source


def test_daily_watchlist_validator_accepts_schema_timestamp(tmp_path, monkeypatch):
    import scripts.validate_daily_watchlist as validator

    config = {
        "priority_watchlist": {"mandatory_daily_prediction": True, "expected_count": 3},
        "equities": {"items": [{"display_name": "A", "symbol": "7203", "asset_class": "jp_stock"}]},
        "us_equities": {"items": [{"display_name": "B", "symbol": "AAPL", "asset_class": "us_stock"}]},
        "market_instruments": {"instruments": [{"display_name": "FX", "provider_symbol": "USDJPY=X", "instrument_id": "FX", "calendar": "24/5"}]},
    }
    rows = [
        {
            "instrument_type": "equity",
            "display_name": "A",
            "symbol": "7203",
            "prediction_status": "READY",
            "p_up_1d": 0.6,
            "direction": "UP",
            "prediction_time": "2026-10-05T08:00:00Z",
            "target_date": "2026-10-06",
            "available_at": "2026-10-05T07:55:00Z",
            "q10_1d": 100.0,
            "q50_1d": 101.0,
            "q90_1d": 102.0,
        },
        {
            "instrument_type": "equity",
            "display_name": "B",
            "symbol": "AAPL",
            "prediction_status": "DEFERRED_PREDICTION_HISTORY_MISSING",
        },
        {
            "instrument_type": "index_or_fx",
            "display_name": "FX",
            "symbol": "USDJPY=X",
            "prediction_status": "DEFERRED_STALE_CONTEXT",
        },
    ]
    cfg_path = tmp_path / "config.yml"
    out_path = tmp_path / "watch.json"
    cfg_path.write_text(__import__("yaml").safe_dump(config, allow_unicode=True), encoding="utf-8")
    out_path.write_text(
        __import__("json").dumps(
            {
                "cutoff": "2026-10-05T08:30:00+00:00",
                "rows": rows,
                "coverage": {"total": 3, "ready": 1, "production_ready": 1, "deferred": 2},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(validator, "CONFIG", cfg_path)
    monkeypatch.setattr(validator, "OUTPUT", out_path)
    validator.main()