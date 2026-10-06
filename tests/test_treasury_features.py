import pandas as pd
from src.research.treasury_features import TREASURY_FEATURE_COLUMNS, add_treasury_context

def test_treasury_merge_is_point_in_time():
    prices=pd.DataFrame({"symbol":["AAA","AAA"],"available_at":pd.to_datetime(["2026-09-28T05:59:00Z","2026-09-28T06:01:00Z"],utc=True)})
    curve=pd.DataFrame({"session_date":[pd.Timestamp("2026-09-28").date()],"available_at":pd.to_datetime(["2026-09-28T06:00:00Z"],utc=True),"source":["treasury"],"treasury_10y":[4.0],"treasury_2s10s":[1.0]})
    out=add_treasury_context(prices,curve)
    assert pd.isna(out.loc[0,"treasury_10y"])
    assert out.loc[1,"treasury_10y"]==4.0

def test_treasury_feature_columns_are_explicit():
    assert "treasury_2s10s" in TREASURY_FEATURE_COLUMNS
    assert "treasury_curvature_2y5y10y" in TREASURY_FEATURE_COLUMNS
