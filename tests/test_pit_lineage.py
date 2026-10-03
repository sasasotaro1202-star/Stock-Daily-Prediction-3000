from __future__ import annotations

import pandas as pd

from src.research.pit_lineage import build_research_pit_lineage, scheduled_prediction_time


CFG = {
    "asia_prediction_time_jst": "18:17",
    "us_prediction_time_jst": "07:17",
}


def _frame(*, session_date="2026-10-01", asset_class="jp_stock", available_at="2026-10-01T07:00:00+00:00", published_at=None, retrieved_at="2026-10-02T00:00:00+00:00"):
    return pd.DataFrame(
        {
            "session_date": [session_date],
            "asset_class": [asset_class],
            "available_at": [available_at],
            "retrieved_at": [retrieved_at],
            "available_at_method": ["conservative_post_close_inferred"],
            "source": ["yfinance"],
            "provider_symbol": ["TEST"],
            "published_at": [published_at],
            "retrieval_run_id": ["unit-test"],
        }
    )


def test_scheduled_prediction_time_is_exactly_derived_from_declared_schedule():
    prediction, source = scheduled_prediction_time(
        "2026-10-01",
        "jp_stock",
        CFG,
    )
    assert prediction is not None
    assert prediction.isoformat() == "2026-10-01T09:17:00+00:00"
    assert source == "DECLARED_CONFIG_SCHEDULE:asia_prediction_time_jst"


def test_us_session_uses_next_configured_weekday_prediction_clock():
    prediction, source = scheduled_prediction_time(
        "2026-10-02",  # Friday U.S. session
        "us_stock",
        CFG,
    )
    assert prediction is not None
    assert prediction.isoformat() == "2026-10-04T22:17:00+00:00"
    assert source == "DECLARED_CONFIG_SCHEDULE:us_prediction_time_jst"


def test_us_session_skips_exchange_holiday():
    prediction, source = scheduled_prediction_time(
        "2026-11-25",  # Wednesday before U.S. Thanksgiving
        "us_stock",
        CFG,
    )
    assert prediction is not None
    # XNYS is closed on 2026-11-26; next session is Friday 2026-11-27.
    assert prediction.isoformat() == "2026-11-26T22:17:00+00:00"
    assert source == "DECLARED_CONFIG_SCHEDULE:us_prediction_time_jst"


def test_jp_session_skips_exchange_holiday():
    prediction, source = scheduled_prediction_time(
        "2026-11-20",  # Friday before Labor Thanksgiving Day
        "jp_stock",
        CFG,
    )
    assert prediction is not None
    # XTKS is closed on Monday 2026-11-23; next session is Tuesday 2026-11-24.
    # Prediction remains on the JP session date itself.
    assert prediction.isoformat() == "2026-11-20T09:17:00+00:00"
    assert source == "DECLARED_CONFIG_SCHEDULE:asia_prediction_time_jst"


def test_invalid_non_session_date_fails_closed():
    import pytest
    with pytest.raises(ValueError, match="invalid_market_session:us_stock:2026-11-26"):
        scheduled_prediction_time("2026-11-26", "us_stock", CFG)


def test_row_lineage_passes_pit_even_when_research_acquisition_happens_later():
    lineage = build_research_pit_lineage(_frame(), CFG)
    assert lineage["pit_status"] == ["PASS"]
    assert lineage["prediction_time"] == ["2026-10-01T09:17:00+00:00"]
    assert lineage["prediction_cutoff"] == ["2026-10-01T09:17:00+00:00"]
    assert lineage["available_at"] == ["2026-10-01T07:00:00+00:00"]
    assert lineage["retrieved_at"] == ["2026-10-02T00:00:00+00:00"]
    assert lineage["prediction_time_observed"] == [False]
    assert len(lineage["lineage_sha256"][0]) == 64


def test_row_lineage_accepts_canonical_nested_pipeline_config():
    pipeline_cfg = {
        "automation": {
            "asia_prediction_time_jst": "18:17",
            "us_prediction_time_jst": "07:17",
        }
    }
    lineage = build_research_pit_lineage(_frame(), pipeline_cfg)
    assert lineage["pit_status"] == ["PASS"]
    assert lineage["prediction_time"] == ["2026-10-01T09:17:00+00:00"]
    assert lineage["prediction_time_source"] == [
        "DECLARED_CONFIG_SCHEDULE:asia_prediction_time_jst"
    ]


def test_row_lineage_fails_closed_when_available_after_prediction():
    lineage = build_research_pit_lineage(
        _frame(available_at="2026-10-01T10:00:00+00:00"),
        CFG,
    )
    assert lineage["pit_status"] == ["BLOCKED_AVAILABLE_AFTER_PREDICTION"]


def test_row_lineage_fails_closed_for_unknown_asset_class():
    lineage = build_research_pit_lineage(
        _frame(asset_class="unknown_asset"),
        CFG,
    )
    assert lineage["pit_status"] == ["BLOCKED_UNKNOWN_ASSET_CLASS"]

def test_row_lineage_reuses_shared_contract_for_publication_ordering():
    lineage = build_research_pit_lineage(
        _frame(published_at="2026-10-01T08:00:00+00:00"),
        CFG,
    )
    assert lineage["pit_status"] == ["BLOCKED_PUBLISHED_AFTER_AVAILABLE"]
    assert lineage["published_at"] == ["2026-10-01T08:00:00+00:00"]


def test_row_lineage_reuses_shared_contract_for_retrieval_ordering():
    lineage = build_research_pit_lineage(
        _frame(retrieved_at="2026-10-01T06:59:00+00:00"),
        CFG,
    )
    assert lineage["pit_status"] == ["BLOCKED_RETRIEVED_BEFORE_AVAILABLE"]


def test_row_lineage_blocks_malformed_publication_timestamp():
    lineage = build_research_pit_lineage(
        _frame(published_at="not-a-timestamp"),
        CFG,
    )
    assert lineage["pit_status"] == ["BLOCKED_INVALID_PUBLISHED_AT"]
