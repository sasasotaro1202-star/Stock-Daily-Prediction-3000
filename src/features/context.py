from __future__ import annotations

import numpy as np
import pandas as pd


def _cross_sectional_robust_zscore(s: pd.Series) -> pd.Series:
    """Median/MAD z-score, clipped to reduce single-name outlier dominance."""
    x=pd.to_numeric(s,errors="coerce")
    if x.notna().sum() < 10:
        return pd.Series(np.nan,index=x.index,dtype="float64")
    med=float(x.median())
    mad=float((x-med).abs().median())
    scale=1.4826*mad
    if not np.isfinite(scale) or scale<=1e-12:
        return pd.Series(0.0,index=x.index,dtype="float64")
    return ((x-med)/scale).clip(-5.0,5.0).astype("float64")


def add_market_context(
    df: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    out=df.copy()
    required={"session_date","family","ret_1d","volatility_20","close","available_at"}
    if context.empty or not required.issubset(context.columns):
        raise ValueError("market context is missing required columns")
    if "available_at" not in out.columns:
        raise ValueError("security price data must contain available_at")

    out["available_at"]=pd.to_datetime(
        out["available_at"],utc=True,errors="coerce"
    )
    ctx=context.copy()
    ctx["available_at"]=pd.to_datetime(
        ctx["available_at"],utc=True,errors="coerce"
    )
    ctx["session_date"]=pd.to_datetime(
        ctx["session_date"],errors="coerce"
    ).dt.date
    ctx=ctx.dropna(subset=["available_at"]).sort_values("available_at")

    # Point-in-time as-of joins: for each security bar, use only context
    # observations whose publication/availability timestamp is <= that bar.
    for family in sorted(ctx["family"].dropna().unique()):
        part=ctx[ctx["family"].eq(family)].sort_values("available_at").copy()
        part=part[
            ["available_at","ret_1d","volatility_20","close"]
        ].rename(columns={
            "ret_1d":f"__{family}_ret",
            "volatility_20":f"__{family}_vol",
            "close":f"__{family}_close",
        })
        out=pd.merge_asof(
            out.sort_values("available_at"),
            part,
            on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )

    mapping={
        "nikkei_ret_1d_lag1":"__nikkei_ret",
        "topix_ret_1d_lag1":"__topix_ret",
        "sp500_ret_1d_lag1":"__sp500_ret",
        "nasdaq_ret_1d_lag1":"__nasdaq_ret",
        "vix_level_lag1":"__vix_close",
        "usd_jpy_ret_1d_lag1":"__usd_jpy_ret",
        "jp_market_volatility_lag1":"__nikkei_vol",
        "us_market_volatility_lag1":"__sp500_vol",
    }
    for target,source in mapping.items():
        out[target]=out[source] if source in out.columns else float("nan")

    drop=[c for c in out.columns if c.startswith("__")]
    return out.drop(columns=drop)


def add_cross_sectional_context(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()

    if "asset_class" in out.columns:
        out["market_family"]=out["asset_class"].map(
            lambda x: "jp" if str(x).startswith("jp_")
            else "us" if str(x).startswith("us_")
            else str(x)
        )
        asset=out["asset_class"].astype(str)
        out["asset_is_jp"]=asset.str.startswith("jp_").astype(float)
        out["asset_is_us"]=asset.str.startswith("us_").astype(float)
        out["asset_is_stock"]=asset.str.endswith("_stock").astype(float)
        out["asset_is_etf"]=asset.str.endswith("_etf").astype(float)
        out["asset_is_reit"]=asset.eq("jp_reit").astype(float)
    elif "market_family" not in out.columns:
        raise ValueError("asset_class or market_family is required")

    out=out.sort_values(
        ["session_date","market_family","symbol"]
    ).reset_index(drop=True)

    g=out.groupby(
        ["session_date","market_family"],sort=False
    )
    out["cs_ret_1d_rank"]=g["ret_1d"].rank(
        method="average",pct=True
    )
    out["cs_vol_rank"]=g["volatility_20"].rank(
        method="average",pct=True,ascending=False
    )

    # Qlib-style robust cross-sectional normalization, scoped to each
    # market family and trading date so the transform stays causal at the
    # post-close prediction timestamp.
    robust_sources={
        "ret_1d":"cs_ret_1d_robust_z",
        "volatility_20":"cs_volatility_20_robust_z",
        "volume_ratio_20":"cs_volume_ratio_20_robust_z",
        "range_pct":"cs_range_pct_robust_z",
        "price_vs_sma20":"cs_price_vs_sma20_robust_z",
    }
    for source,target in robust_sources.items():
        if source in out.columns:
            out[target]=g[source].transform(_cross_sectional_robust_zscore)

    daily=g.agg(
        median_vol=("volatility_20","median"),
        median_ret=("ret_1d","median"),
        breadth_up=("ret_1d",lambda s:float((s>0).mean())),
    ).reset_index()

    out=out.merge(
        daily,
        on=["session_date","market_family"],
        how="left",
    )
    out["ret_vs_market_median"]=out["ret_1d"]-out["median_ret"]
    out["vol_vs_market_median"]=(
        out["volatility_20"]
        /out["median_vol"].replace(0,float("nan"))
        -1.0
    )

    if "sector" in out.columns:
        sg=out.groupby(
            ["session_date","market_family","sector"],
            sort=False
        )
        out["sector_ret_1d"]=sg["ret_1d"].transform("mean")
        out["relative_strength_1d"]=(
            out["ret_1d"]-out["sector_ret_1d"]
        )
        out["sector_rank"]=sg["ret_1d"].rank(
            method="average",pct=True
        )

    if "market_ret_1d" in out.columns:
        out["market_relative_ret_1d"]=(
            out["ret_1d"]-out["market_ret_1d"]
        )
    return out
