from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

FINRA_URL="https://api.finra.org/data/group/otcMarket/name/regShoDaily"
USER_AGENT="Stock-Daily-Prediction-3000/0.1"
FIELDS=("tradeReportDate","securitiesInformationProcessorSymbolIdentifier","shortParQuantity",
        "shortExemptParQuantity","totalParQuantity","reportingFacilityCode","marketCode")

def _json(payload: bytes) -> list[dict]:
    text=payload.decode("utf-8",errors="replace")
    return [] if not text.strip() else json.loads(text)

def _poll(location: str, timeout_seconds: int=180) -> list[dict]:
    deadline=time.time()+timeout_seconds
    while time.time()<deadline:
        with urlopen(Request(location,headers={"User-Agent":USER_AGENT,"Accept":"application/json"}),timeout=60) as r:
            body=json.loads(r.read().decode("utf-8",errors="replace") or "{}")
        status=str(body.get("status","")).lower()
        if status=="complete":
            link=body.get("resultLink")
            if not link: raise RuntimeError("FINRA async request completed without resultLink")
            with urlopen(Request(str(link),headers={"User-Agent":USER_AGENT,"Accept":"application/json"}),timeout=120) as r:
                return _json(r.read())
        if status not in {"","pending","running"}:
            raise RuntimeError(f"FINRA async request failed: {status}")
        time.sleep(3)
    raise TimeoutError("FINRA async request timed out")

def _submit(start: date,end: date,symbols: list[str]) -> list[dict]:
    body={"async":True,"limit":100000,"offset":0,"fields":list(FIELDS),
          "dateRangeFilters":[{"fieldName":"tradeReportDate","startDate":start.isoformat(),"endDate":end.isoformat()}],
          "domainFilters":[{"fieldName":"securitiesInformationProcessorSymbolIdentifier","values":symbols}]}
    req=Request(FINRA_URL,data=json.dumps(body).encode(),method="POST",
                headers={"User-Agent":USER_AGENT,"Accept":"application/json","Content-Type":"application/json"})
    with urlopen(req,timeout=90) as r:
        payload=r.read(); headers={str(k).lower():str(v) for k,v in r.headers.items()}; status=getattr(r,"status",200)
    if status==200: return _json(payload)
    if status!=202: raise RuntimeError(f"FINRA unexpected HTTP status={status}")
    if "location" not in headers: raise RuntimeError("FINRA async response missing Location header")
    return _poll(headers["location"])

def _available_at(d: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d+timedelta(days=1),datetime.min.time()),tz="Asia/Tokyo")+pd.Timedelta(hours=12)

def collect_finra_short_sale(start: date,end: date,symbols: list[str],*,symbol_chunk_size: int=250,window_days: int=15) -> pd.DataFrame:
    symbols=sorted({str(x).strip().upper() for x in symbols if str(x).strip()})
    if not symbols or start>end: return pd.DataFrame()
    rows=[]
    cur=start
    while cur<=end:
        wend=min(end,cur+timedelta(days=window_days-1))
        for i in range(0,len(symbols),symbol_chunk_size):
            rows.extend(_submit(cur,wend,symbols[i:i+symbol_chunk_size]))
        cur=wend+timedelta(days=1)
    if not rows: return pd.DataFrame()
    f=pd.DataFrame(rows).rename(columns={
        "tradeReportDate":"session_date","securitiesInformationProcessorSymbolIdentifier":"symbol",
        "shortParQuantity":"short_volume","shortExemptParQuantity":"short_exempt_volume",
        "totalParQuantity":"total_volume","reportingFacilityCode":"reporting_facility","marketCode":"market_code"})
    need={"session_date","symbol","short_volume","short_exempt_volume","total_volume","reporting_facility","market_code"}
    if not need.issubset(f.columns): raise ValueError("FINRA response missing fields")
    f["session_date"]=pd.to_datetime(f["session_date"],errors="coerce").dt.date
    f["symbol"]=f["symbol"].astype(str).str.upper().str.strip()
    for c in ("short_volume","short_exempt_volume","total_volume"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f=f.dropna(subset=["session_date","symbol"])
    required_numeric=("short_volume","short_exempt_volume","total_volume")
    if f[list(required_numeric)].isna().any().any():
        raise ValueError("FINRA response contains missing numeric volume fields")
    if (f[list(required_numeric)] < 0).any().any():
        raise ValueError("FINRA response contains negative volume fields")
    if (f["short_volume"] > f["total_volume"]).any():
        raise ValueError("FINRA response has short volume greater than total volume")
    if (f["short_exempt_volume"] > f["total_volume"]).any():
        raise ValueError("FINRA response has short-exempt volume greater than total volume")
    g=f.groupby(["session_date","symbol"],as_index=False).agg(
        short_volume=("short_volume","sum"),short_exempt_volume=("short_exempt_volume","sum"),total_volume=("total_volume","sum"))
    den=g["total_volume"].replace(0,np.nan)
    g["finra_short_ratio"]=(g.short_volume/den).astype(float)
    g["finra_short_exempt_ratio"]=(g.short_exempt_volume/den).astype(float)
    g["finra_total_volume_log"]=np.log1p(g.total_volume).astype(float)
    q=f.assign(facility_ratio=f.short_volume/f.total_volume.replace(0,np.nan))
    d=q.groupby(["session_date","symbol"],as_index=False).facility_ratio.agg(
        lambda s: float(s.max()-s.min()) if s.notna().any() else 0.0).rename(columns={"facility_ratio":"finra_venue_disagreement"})
    g=g.merge(d,on=["session_date","symbol"],how="left").fillna({"finra_venue_disagreement":0.0})
    g=g.sort_values(["symbol","session_date"]).reset_index(drop=True)
    by=g.groupby("symbol",sort=False)
    g["finra_short_ratio_change"]=by.finra_short_ratio.diff()
    m=by.finra_short_ratio.transform(lambda s:s.rolling(20,min_periods=10).mean())
    sd=by.finra_short_ratio.transform(lambda s:s.rolling(20,min_periods=10).std())
    g["finra_short_ratio_z20"]=(g.finra_short_ratio-m)/sd.replace(0,np.nan)
    g["available_at"]=g.session_date.map(_available_at)
    g["source"]="finra_reg_sho_daily"
    g["available_at_method"]="next_calendar_day_1200_jst_conservative"
    return g
