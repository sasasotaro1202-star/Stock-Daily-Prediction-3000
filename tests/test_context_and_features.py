import numpy as np
import pandas as pd


def test_rd_style_factor_pack_is_numeric_after_warmup():
    from src.features.technical import add_technical_features

    rows=[]
    for i in range(90):
        d=pd.Timestamp("2026-01-01")+pd.Timedelta(days=i)
        rows.append({
            "symbol":"A",
            "asset_class":"jp_stock",
            "session_date":d,
            "open":100+i*0.2,
            "high":101+i*0.2,
            "low":99+i*0.2,
            "close":100+i*0.2,
            "volume":1000+i*5,
        })
    out=add_technical_features(pd.DataFrame(rows))
    cols=[
        "ret_60d","return_log_volume_corr_5","return_log_volume_corr_10",
        "trend_slope_60","trend_r2_60","resi_5_pct","resi_10_pct",
        "wvma_5","volume_std_ratio_5",
    ]
    warm=out.iloc[-1]
    assert all(np.isfinite(float(warm[c])) for c in cols)
