from __future__ import annotations

import numpy as np
import pandas as pd

FINRA_FEATURE_COLUMNS=("finra_data_available","finra_short_ratio","finra_short_exempt_ratio",
                       "finra_total_volume_log","finra_venue_disagreement","finra_short_ratio_change",
                       "finra_short_ratio_z20")

def add_finra_context(prices: pd.DataFrame,finra: pd.DataFrame) -> pd.DataFrame:
    out=prices.copy()
    for c in FINRA_FEATURE_COLUMNS: out[c]=0.0
    if finra is None or finra.empty: return out
    required={"symbol","available_at"}|set(FINRA_FEATURE_COLUMNS[1:])
    if not required.issubset(finra.columns): raise ValueError("FINRA frame missing required columns")
    out["available_at"]=pd.to_datetime(out["available_at"],utc=True,errors="coerce")
    ev=finra.copy()
    ev["available_at"]=pd.to_datetime(ev["available_at"],utc=True,errors="coerce")
    ev["symbol"]=ev["symbol"].astype(str).str.upper()
    ev=ev.dropna(subset=["available_at","symbol"]).sort_values(["symbol","available_at"])
    if not {"symbol","asset_class"}.issubset(out.columns): raise ValueError("price frame requires symbol and asset_class")
    mask=out.asset_class.astype(str).eq("us_stock")
    for symbol,idx in out.loc[mask].groupby(out.loc[mask].symbol.astype(str).str.upper()).groups.items():
        q=out.loc[list(idx),["available_at","symbol"]].sort_values("available_at").reset_index(names="original_index")
        e=ev[ev.symbol.eq(symbol)]
        if q.empty or e.empty: continue
        m=pd.merge_asof(q,e[["available_at"]+list(FINRA_FEATURE_COLUMNS[1:])].sort_values("available_at"),
                        on="available_at",direction="backward",allow_exact_matches=True)
        target=m.original_index.to_numpy()
        available=m.finra_short_ratio.notna().to_numpy()
        for c in FINRA_FEATURE_COLUMNS[1:]:
            v=pd.to_numeric(m[c],errors="coerce").to_numpy(float)
            out.loc[target,c]=np.where(np.isfinite(v),v,0.0)
        out.loc[target,"finra_data_available"]=available.astype(float)
    return out
