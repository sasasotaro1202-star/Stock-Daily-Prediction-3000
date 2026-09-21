from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS=[
    "ret_1d","ret_5d","ret_20d",
    "close_vs_sma5","close_vs_sma20","close_vs_sma60","close_vs_ema20",
    "macd_pct","macd_signal_pct","macd_hist_pct",
    "rsi_14","stoch_k","stoch_d","bb_width","atr_pct","adx_14","obv_z20","mfi_14",
    "volatility_20","volume_ratio_20","range_pct","gap_pct",
    "price_vs_sma20","price_vs_sma60",
    "cs_ret_1d_rank","cs_vol_rank","median_vol","median_ret","breadth_up",
    "nikkei_ret_1d_lag1","topix_ret_1d_lag1",
    "sp500_ret_1d_lag1","nasdaq_ret_1d_lag1",
    "vix_level_lag1","usd_jpy_ret_1d_lag1",
    "jp_market_volatility_lag1","us_market_volatility_lag1",
    "asset_is_jp","asset_is_us","asset_is_stock",
    "asset_is_etf","asset_is_reit",
    "ret_vs_market_median","vol_vs_market_median",
    "volatility_5","volatility_ratio_5_20",
    "volume_z20","dollar_volume_ratio_20",
    "amihud_20","return_z20","range_z20",
    "close_location","intraday_return",
    "dow_sin","dow_cos","month_sin","month_cos",
]

def _rolling_z(s:pd.Series,window:int)->pd.Series:
    mean=s.rolling(window,min_periods=window).mean()
    std=s.rolling(window,min_periods=window).std().replace(0,np.nan)
    return (s-mean)/std

def add_technical_features(df:pd.DataFrame,group_col:str="symbol")->pd.DataFrame:
    out=df.copy()
    required={"open","high","low","close","volume",group_col,"session_date"}
    missing=required-set(out.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    out=out.sort_values([group_col,"session_date"]).reset_index(drop=True)
    g=out.groupby(group_col,sort=False)
    close=g["close"]; volume=g["volume"]; prev=out.groupby(group_col)["close"].shift(1)
    out["ret_1d"]=close.pct_change()
    out["ret_5d"]=close.pct_change(5)
    out["ret_20d"]=close.pct_change(20)
    sma5=close.transform(lambda s:s.rolling(5,min_periods=5).mean())
    sma20=close.transform(lambda s:s.rolling(20,min_periods=20).mean())
    sma60=close.transform(lambda s:s.rolling(60,min_periods=60).mean())
    ema20=close.transform(lambda s:s.ewm(span=20,adjust=False,min_periods=20).mean())
    out["close_vs_sma5"]=out["_model_close"]/sma5-1.0
    out["close_vs_sma20"]=out["close"]/sma20-1.0
    out["close_vs_sma60"]=out["close"]/sma60-1.0
    out["close_vs_ema20"]=out["_model_close"]/ema20-1.0
    ema12=close.transform(lambda s:s.ewm(span=12,adjust=False,min_periods=12).mean())
    ema26=close.transform(lambda s:s.ewm(span=26,adjust=False,min_periods=26).mean())
    macd=ema12-ema26
    macd_signal=macd.groupby(out[group_col]).transform(lambda s:s.ewm(span=9,adjust=False,min_periods=9).mean())
    out["macd_pct"]=macd/out["close"].replace(0,np.nan)
    out["macd_signal_pct"]=macd_signal/out["close"].replace(0,np.nan)
    out["macd_hist_pct"]=(macd-macd_signal)/out["close"].replace(0,np.nan)
    delta=close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    avg_gain=gain.transform(lambda s:s.rolling(14,min_periods=14).mean())
    avg_loss=loss.transform(lambda s:s.rolling(14,min_periods=14).mean())
    rs=avg_gain/avg_loss.replace(0,np.nan)
    out["rsi_14"]=100-(100/(1+rs))
    low14=g["low"].transform(lambda s:s.rolling(14,min_periods=14).min())
    high14=g["high"].transform(lambda s:s.rolling(14,min_periods=14).max())
    out["stoch_k"]=100*(out["close"]-low14)/(high14-low14).replace(0,np.nan)
    out["stoch_d"]=out["stoch_k"].groupby(out[group_col]).transform(lambda s:s.rolling(3,min_periods=3).mean())
    bb_std=close.transform(lambda s:s.rolling(20,min_periods=20).std())
    out["bb_width"]=4*bb_std/sma20.replace(0,np.nan)
    tr=pd.concat([(out["high"]-out["low"]),(out["high"]-prev).abs(),(out["low"]-prev).abs()],axis=1).max(axis=1)
    atr=tr.groupby(out[group_col]).transform(lambda s:s.rolling(14,min_periods=14).mean())
    out["atr_pct"]=atr/out["close"].replace(0,np.nan)
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
    obv=(direction*out["volume"]).groupby(out[group_col]).cumsum()
    out["obv_z20"]=obv.groupby(out[group_col]).transform(lambda s:_rolling_z(s,20))
    tp=(out["high"]+out["low"]+out["close"])/3
    raw_flow=tp*out["volume"]; tp_prev=tp.groupby(out[group_col]).shift(1)
    pos=raw_flow.where(tp>tp_prev,0.0); neg=raw_flow.where(tp<tp_prev,0.0).abs()
    pos14=pos.groupby(out[group_col]).transform(lambda s:s.rolling(14,min_periods=14).sum())
    neg14=neg.groupby(out[group_col]).transform(lambda s:s.rolling(14,min_periods=14).sum())
    money_ratio=pos14/neg14.replace(0,np.nan)
    out["mfi_14"]=100-(100/(1+money_ratio))
    out["volatility_20"]=out["ret_1d"].groupby(out[group_col]).transform(
        lambda s:s.rolling(20,min_periods=20).std()
    )
    out["volatility_5"]=out["ret_1d"].groupby(out[group_col]).transform(
        lambda s:s.rolling(5,min_periods=5).std()
    )
    out["volatility_ratio_5_20"]=(
        out["volatility_5"]/out["volatility_20"].replace(0,np.nan)
    )
    out["volume_ratio_20"]=volume.transform(
        lambda s:s/s.rolling(20,min_periods=20).mean()
    )
    out["volume_z20"]=volume.transform(
        lambda s:_rolling_z(s.astype(float),20)
    )
    dollar_volume=(out["close"].abs()*out["volume"].abs()).astype(float)
    dollar_median=dollar_volume.groupby(out[group_col]).transform(
        lambda s:s.rolling(20,min_periods=20).median()
    )
    out["dollar_volume_ratio_20"]=(
        dollar_volume/dollar_median.replace(0,np.nan)
    )
    amihud_raw=out["ret_1d"].abs()/dollar_volume.replace(0,np.nan)
    out["amihud_20"]=amihud_raw.groupby(out[group_col]).transform(
        lambda s:s.rolling(20,min_periods=20).median()
    )
    out["return_z20"]=out["ret_1d"].groupby(out[group_col]).transform(
        lambda s:_rolling_z(s.astype(float),20)
    )    out["range_pct"]=(out["high"]-out["low"])/out["close"].replace(0,np.nan)
    out["range_z20"]=out["range_pct"].groupby(out[group_col]).transform(
        lambda s:_rolling_z(s.astype(float),20)
    )
    out["gap_pct"]=(out["open"]/prev)-1.0
    out["intraday_return"]=out["close"]/out["open"].replace(0,np.nan)-1.0
    out["close_location"]=(
        (out["close"]-out["low"])/(out["high"]-out["low"]).replace(0,np.nan)
    )
    date_series=pd.to_datetime(out["session_date"],errors="coerce")
    dow=date_series.dt.dayofweek.astype(float)
    month=date_series.dt.month.astype(float)-1.0
    out["dow_sin"]=np.sin(2*np.pi*dow/7.0)
    out["dow_cos"]=np.cos(2*np.pi*dow/7.0)
    out["month_sin"]=np.sin(2*np.pi*month/12.0)
    out["month_cos"]=np.cos(2*np.pi*month/12.0)
    out["price_vs_sma20"]=out["close"]/sma20-1.0
    out["price_vs_sma60"]=out["close"]/sma60-1.0
    return out
