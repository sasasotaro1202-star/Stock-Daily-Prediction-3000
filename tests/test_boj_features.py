import pandas as pd
import pytest

from src.research.boj_features import BOJ_FEATURE_COLUMNS, add_boj_context


def _boj():
    return pd.DataFrame(
        {
            "available_at": pd.to_datetime(
                ["2026-09-28T06:00:00Z", "2026-09-29T06:00:00Z"]
            ),
            "observation_date": [
                pd.Timestamp("2026-09-25").date(),
                pd.Timestamp("2026-09-28").date(),
            ],
            "series_code": [
                "STRDCLUCON",
                "STRDCLUCON",
            ],
            "value": [0.70, 0.72],
            "source": ["boj_timeseries", "boj_timeseries"],
        }
    ).join(
        pd.DataFrame(
            {
                "series_code": ["STRDCLUCONH", "STRDCLUCONH"],
                "value": [0.71, 0.73],
            }
        ).set_axis([0, 1])
    )


def test_boj_context_does_not_use_future_available_observation():
    boj = pd.DataFrame(
        [
            ["2026-09-28T06:00:00Z", "2026-09-25", "STRDCLUCON", 0.70],
            ["2026-09-28T06:00:00Z", "2026-09-25", "STRDCLUCONH", 0.71],
            ["2026-09-28T06:00:00Z", "2026-09-25", "STRDCLUCONL", 0.69],
        ],
        columns=["available_at", "observation_date", "series_code", "value"],
    )
    boj["available_at"] = pd.to_datetime(boj["available_at"], utc=True)
    boj["observation_date"] = pd.to_datetime(boj["observation_date"]).dt.date
    boj["source"] = "boj_timeseries"

    price = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "available_at": pd.to_datetime(
                ["2026-09-28T05:00:00Z", "2026-09-28T07:00:00Z"]
            ),
            "close": [100.0, 101.0],
        }
    )
    out = add_boj_context(price, boj, min_history=1)
    assert pd.isna(out.loc[0, "boj_call_rate_level"])
    assert out.loc[1, "boj_call_rate_level"] == pytest.approx(0.70)
    assert set(BOJ_FEATURE_COLUMNS).issubset(out.columns)


def test_boj_context_rejects_non_boj_source():
    boj = pd.DataFrame(
        [
            ["2026-09-28T06:00:00Z", "2026-09-25", "STRDCLUCON", 0.70, "other"],
            ["2026-09-28T06:00:00Z", "2026-09-25", "STRDCLUCONH", 0.71, "other"],
            ["2026-09-28T06:00:00Z", "2026-09-25", "STRDCLUCONL", 0.69, "other"],
        ],
        columns=[
            "available_at", "observation_date", "series_code", "value", "source"
        ],
    )
    boj["available_at"] = pd.to_datetime(boj["available_at"], utc=True)
    boj["observation_date"] = pd.to_datetime(boj["observation_date"]).dt.date
    with pytest.raises(ValueError, match="non-BOJ"):
        add_boj_context(
            pd.DataFrame(
                {"available_at": pd.to_datetime(["2026-09-28T07:00:00Z"])}
            ),
            boj,
        )
