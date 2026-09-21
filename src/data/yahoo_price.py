from __future__ import annotations
import os
from datetime import datetime,time
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

def yahoo_symbol(symbol:str,asset_class:str)->str:
    return f"{symbol}.T" if asset_class.startswith("jp_") else symbol

def available_at_for(session_date,asset_class:str)->pd.Timestamp:
    if asset_class.startswith("jp_"):
        dt=datetime.combine(session_date,time(15,35),tzinfo=ZoneInfo("Asia/Tokyo"))
    else:
        dt=datetime.combine(session_date,time(16,10),tzinfo=ZoneInfo("America/New_York"))
    return pd.Timestamp(dt).tz_convert("UTC")

def download_batch(records:list[dict],period:str="5y")->pd.DataFrame:
    if not records:return pd.DataFrame()
    symbols=[yahoo_symbol(r["symbol"],r["asset_class"]) for r in records]
    mapping={yahoo_symbol(r["symbol"],r["asset_class"]):r for r in records}
    raw=yf.download(symbols,period=period,auto_adjust=False,progress=False,group_by="ticker",threads=False)
    frames=[]
    if isinstance(raw.columns,pd.MultiIndex):
        for ysym in symbols:
            if ysym not in raw.columns.get_level_values(0):continue
            part=raw[ysym].reset_index().rename(columns=str.lower)
            if "date" not in part:continue
            rec=mapping[ysym]; part["symbol"]=rec["symbol"]; part["asset_class"]=rec["asset_class"]
            part["session_date"]=pd.to_datetime(part["date"]).dt.date
            part["available_at"]=part["session_date"].map(lambda d:available_at_for(d,rec["asset_class"]))
            part["source"]="yfinance"; part["provider_symbol"]=ysym
            frames.append(part[["symbol","asset_class","session_date","available_at","source","provider_symbol","open","high","low","close","volume"]])
    else:
        part=raw.reset_index().rename(columns=str.lower)
        if not part.empty and "date" in part:
            rec=records[0]; part["symbol"]=rec["symbol"]; part["asset_class"]=rec["asset_class"]
            part["session_date"]=pd.to_datetime(part["date"]).dt.date
            part["available_at"]=part["session_date"].map(lambda d:available_at_for(d,rec["asset_class"]))
            part["source"]="yfinance"; part["provider_symbol"]=ysym
            frames.append(part[["symbol","asset_class","session_date","available_at","source","provider_symbol","open","high","low","close","volume"]])
    if not frames:return pd.DataFrame()
    out=pd.concat(frames,ignore_index=True)
    for c in ["open","high","low","close","volume"]:out[c]=pd.to_numeric(out[c],errors="coerce")
    out=out.dropna(subset=["open","high","low","close"]); out["volume"]=out["volume"].fillna(0)
    return out

def upsert_batch_parquet(new_data:pd.DataFrame,path:str)->int:
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    old=pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    combined=pd.concat([old,new_data],ignore_index=True).drop_duplicates(["symbol","session_date"],keep="last")
    combined=combined.sort_values(["symbol","session_date"])
    combined.to_parquet(path,index=False)
    return len(combined)
