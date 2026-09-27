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
