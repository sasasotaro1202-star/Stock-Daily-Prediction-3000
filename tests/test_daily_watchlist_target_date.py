from __future__ import annotations

import pandas as pd

from scripts.generate_daily_watchlist import build_market_rows, latest_equity_prediction


def test_latest_equity_prediction_persists_derived_target_date(tmp_path):
    path = tmp_path / "prediction_20261008T132416Z.parquet"
    frame = pd.DataFrame(
        [
            {
                "asset_class": "jp_stock",
                "symbol": "8802",
                "p_up_1d": 0.61,
                "prediction_time": "2026-10-08T13:24:16Z",
                "available_at": "2026-10-08T13:00:00Z",
                "session_date": "2026-10-08",
                "prediction_status": "READY_NEAR_PRODUCTION",
            }
        ]
    )
    frame.to_parquet(path, index=False)

    row, filename = latest_equity_prediction(
        [path],
        "jp_stock",
        "8802",
        cutoff=pd.Timestamp("2026-10-08T14:00:00Z"),
        expected_target_date="2026-10-09",
        calendar_code="XTKS",
    )

    assert filename == path.name
    assert row is not None
    assert str(row["target_date"]) == "2026-10-09"

def test_market_watchlist_ready_row_has_pit_safe_prediction_time():
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
    cutoff = pd.Timestamp("2026-07-01T00:00:00Z")
    rows = build_market_rows(
        context,
        [{
            "display_name": "日経平均株価",
            "instrument_id": "JP-NIKKEI-225",
            "provider_symbol": "^N225",
            "context_family": "nikkei",
            "calendar": "XTKS",
        }],
        cutoff=cutoff,
        threshold=0.50,
    )
    row = rows[0]
    assert row["prediction_status"] == "READY"
    assert pd.to_datetime(row["prediction_time"], utc=True) == cutoff
    assert pd.to_datetime(row["available_at"], utc=True) <= pd.to_datetime(row["prediction_time"], utc=True)
