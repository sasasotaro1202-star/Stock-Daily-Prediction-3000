from datetime import date

import pandas as pd

from src.data.finra_api import _available_at
from src.research.finra_features import FINRA_FEATURE_COLUMNS, add_finra_context

def test_finra_availability_is_conservative():
    ts=_available_at(date(2026,9,25)).tz_convert("Asia/Tokyo")
    assert ts.date().isoformat()=="2026-09-26" and ts.hour==12

def test_finra_uses_only_prior_available_observation():
    prices=pd.DataFrame([
        {"symbol":"ABC","asset_class":"us_stock","available_at":"2026-09-28T07:17:00Z"},
        {"symbol":"ABC","asset_class":"us_stock","available_at":"2026-09-29T07:17:00Z"}])
    finra=pd.DataFrame([{"symbol":"ABC","available_at":"2026-09-27T03:00:00Z",
        "finra_short_ratio":.4,"finra_short_exempt_ratio":.01,"finra_total_volume_log":12.,
        "finra_venue_disagreement":.1,"finra_short_ratio_change":.02,"finra_short_ratio_z20":1.5}])
    out=add_finra_context(prices,finra)
    assert out.loc[0,"finra_data_available"]==1.0 and out.loc[1,"finra_data_available"]==1.0
    assert float(out.loc[0,"finra_short_ratio"])==.4
    assert set(FINRA_FEATURE_COLUMNS)<=set(out.columns)

def test_finra_missing_history_is_explicit():
    prices=pd.DataFrame([{"symbol":"XYZ","asset_class":"us_stock","available_at":"2026-09-28T07:17:00Z"}])
    out=add_finra_context(prices,pd.DataFrame())
    assert out.loc[0,"finra_data_available"]==0.0 and out.loc[0,"finra_short_ratio"]==0.0

def test_finra_rejects_missing_or_impossible_volumes(monkeypatch):
    import src.data.finra_api as api

    def fake_submit(*args, **kwargs):
        return [{
            "tradeReportDate":"2026-09-25",
            "securitiesInformationProcessorSymbolIdentifier":"ABC",
            "shortParQuantity":None,
            "shortExemptParQuantity":1,
            "totalParQuantity":10,
            "reportingFacilityCode":"NQ",
            "marketCode":"Q",
        }]

    monkeypatch.setattr(api, "_submit", fake_submit)
    try:
        api.collect_finra_short_sale(
            date(2026,9,25), date(2026,9,25), ["ABC"]
        )
    except ValueError as exc:
        assert "missing numeric volume" in str(exc)
    else:
        raise AssertionError("expected missing volume to fail closed")

def test_finra_collection_uses_bounded_request_batches(monkeypatch):
    import src.data.finra_api as api

    calls = []

    def fake_submit(start, end, symbols):
        calls.append((start, end, tuple(symbols)))
        return []

    monkeypatch.setattr(api, "_submit", fake_submit)
    api.collect_finra_short_sale(
        date(2026,1,1),
        date(2026,3,31),
        [f"S{i}" for i in range(1001)],
    )
    assert len(calls) == 9
    assert all(len(symbols) <= 500 for _, _, symbols in calls)
    assert all((end - start).days <= 29 for start, end, _ in calls)
