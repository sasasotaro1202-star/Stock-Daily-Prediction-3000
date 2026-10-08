from __future__ import annotations

import pandas as pd

from scripts.generate_daily_watchlist import latest_equity_prediction


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
