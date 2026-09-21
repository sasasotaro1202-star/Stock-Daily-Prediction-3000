from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS=[
    "ret_1d","ret_5d","ret_20d","sma_5","sma_20","sma_60","ema_20",
    "macd","macd_signal","macd_hist","rsi_14","stoch_k","stoch_d",
    "bb_mid","bb_upper","bb_lower","bb_width","atr_14","adx_14",
    "obv","mfi_14","volatility_20","volume_ratio_20","range_pct",
    "gap_pct","price_vs_sma20","price_vs_sma60",
]

def add_technical_features(df:pd.DataFrame,group_col:str="symbol")->pd.DataFrame:
    out=df.copy()
    required={"open","high","low","close","volume",group_col,"session_date"}
    missing=required-set(out.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    out=out.sort_values([group_col,"session_date"]).reset_index(drop=True)
    g=out.groupby(group_col,sort=False)
    close=g["close"]; volume=g["volume"]
    prev=out.groupby(group_col)["close"].shift(1)
    out["ret_1d"]=close.pct_change(); out["ret_5d"]=close.pct_change(5); out["ret_20d"]=close.pct_change(20)
    out["sma_5"]=close.transform(lambda s:s.rolling(5,min_periods=5).mean())
    out["sma_20"]=close.transform(lambda s:s.rolling(20,min_periods=20).mean())
    out["sma_60"]=close.transform(lambda s:s.rolling(60,min_periods=60).mean())
    ema12=close.transform(lambda s:s.ewm(span=12,adjust=False,min_periods=12).mean())
    ema26=close.transform(lambda s:s.ewm(span=26,adjust=False,min_periods=26).mean())
    out["macd"]=ema12-ema26
    out["macd_signal"]=out["macd"].groupby(out[group_col]).transform(lambda s:s.ewm(span=9,adjust=False,min_periods=9).mean())
    out["macd_hist"]=out["macd"]-out["macd_signal"]
    out["ema_20"]=close.transform(lambda s:s.ewm(span=20,adjust=False,min_periods=20).mean())
    delta=close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    avg_gain=gain.transform(lambda s:s.rolling(14,min_periods=14).mean())
    avg_loss=loss.transform(lambda s:s.rolling(14,min_periods=14).mean())
    rs=avg_gain/avg_loss.replace(0,np.nan); out["rsi_14"]=100-(100/(1+rs))
    low14=g["low"].transform(lambda s:s.rolling(14,min_periods=14).min())
    high14=g["high"].transform(lambda s:s.rolling(14,min_periods=14).max())
    out["stoch_k"]=100*(out["close"]-low14)/(high14-low14).replace(0,np.nan)
    out["stoch_d"]=out["stoch_k"].groupby(out[group_col]).transform(lambda s:s.rolling(3,min_periods=3).mean())
    out["bb_mid"]=out["sma_20"]; bb_std=close.transform(lambda s:s.rolling(20,min_periods=20).std())
    out["bb_upper"]=out["bb_mid"]+2*bb_std; out["bb_lower"]=out["bb_mid"]-2*bb_std
    out["bb_width"]=(out["bb_upper"]-out["bb_lower"])/out["bb_mid"].replace(0,np.nan)
    tr=pd.concat([(out["high"]-out["low"]),(out["high"]-prev).abs(),(out["low"]-prev).abs()],axis=1).max(axis=1)
    out["atr_14"]=tr.groupby(out[group_col]).transform(lambda s:s.rolling(14,min_periods=14).mean())
    up=out["high"]-out.groupby(group_col)["high"].shift(1)
    down=out.groupby(group_col)["low"].shift(1)-out["low"]
    plus_dm=pd.Series(np.where((up>down)&(up>0),up,0.0),index=out.index)
    minus_dm=pd.Series(np.where((down>up)&(down>0),down,0.0),index=out.index)
    tr14=tr.groupby(out[group_col]).transform(lambda s:s.rolling(14,min_periods=14).sum())
    plus14=plus_dm.groupby(out[group_col]).transform(lambda s:s.rolling(14,min_periods=14).sum())
    minus14=minus_dm.groupby(out[group_col]).transform(lambda s:s.rolling(14,min_periods=14).sum())
    plus_di=100*plus14/tr14.replace(0,np.nan); minus_di=100*minus14/tr14.replace(0,np.nan)
    dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)
    out["adx_14"]=dx.groupby(out[group_col]).transform(lambda s:s.rolling(14,min_periods=14).mean())
    direction=np.sign(delta.fillna(0))
    out["obv"]=(direction*out["volume"]).groupby(out[group_col]).cumsum()
    tp=(out["high"]+out["low"]+out["close"])/3
    raw_flow=tp*out["volume"]; tp_prev=tp.groupby(out[group_col]).shift(1)
    pos=raw_flow.where(tp>tp_prev,0.0); neg=raw_flow.where(tp<tp_prev,0.0).abs()
    pos14=pos.groupby(out[group_col]).transform(lambda s:s.rolling(14,min_periods=14).sum())
    neg14=neg.groupby(out[group_col]).transform(lambda s:s.rolling(14,min_periods=14).sum())
    money_ratio=pos14/neg14.replace(0,np.nan); out["mfi_14"]=100-(100/(1+money_ratio))
    out["volatility_20"]=out["ret_1d"].groupby(out[group_col]).transform(lambda s:s.rolling(20,min_periods=20).std())
    out["volume_ratio_20"]=volume.transform(lambda s:s/s.rolling(20,min_periods=20).mean())
    out["range_pct"]=(out["high"]-out["low"])/out["close"].replace(0,np.nan)
    out["gap_pct"]=(out["open"]/prev)-1.0
    out["price_vs_sma20"]=out["close"]/out["sma_20"]-1.0
    out["price_vs_sma60"]=out["close"]/out["sma_60"]-1.0
    return out
