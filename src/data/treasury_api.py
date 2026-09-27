from __future__ import annotations

from datetime import date, datetime, time, timedelta
from io import StringIO
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

TREASURY_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView"
USER_AGENT = "Stock-Daily-Prediction-3000/0.1"
YIELD_COLUMNS = {
    "1 Mo": "treasury_1m",
    "3 Mo": "treasury_3m",
    "6 Mo": "treasury_6m",
    "1 Yr": "treasury_1y",
    "2 Yr": "treasury_2y",
    "5 Yr": "treasury_5y",
    "10 Yr": "treasury_10y",
    "30 Yr": "treasury_30y",
}

def _next_weekday_0600_jst(session_date: date) -> pd.Timestamp:
    current = session_date + timedelta(days=1)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return pd.Timestamp(datetime.combine(current, time(6, 0), tzinfo=ZoneInfo("Asia/Tokyo"))).tz_convert("UTC")

def _fetch_year(year: int) -> pd.DataFrame:
    url = TREASURY_URL + f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    with urlopen(request, timeout=45) as response:
        html = response.read().decode("utf-8", errors="replace")
    tables = pd.read_html(StringIO(html), flavor="lxml")
    if not tables:
        return pd.DataFrame()
    table = max(tables, key=lambda frame: frame.shape[1]).copy()
    table.columns = [str(column).strip() for column in table.columns]
    date_column = next((c for c in table.columns if c.lower() == "date"), None)
    if date_column is None:
        return pd.DataFrame()
    wanted = [date_column] + [c for c in YIELD_COLUMNS if c in table.columns]
    out = table[wanted].rename(columns={date_column: "session_date", **YIELD_COLUMNS})
    out["session_date"] = pd.to_datetime(out["session_date"], errors="coerce").dt.date
    for column in YIELD_COLUMNS.values():
        if column in out.columns:
            out[column] = pd.to_numeric(out[column].replace({"N/A": pd.NA, "": pd.NA}), errors="coerce")
    return out.dropna(subset=["session_date"])

def collect_treasury_curve(start_year: int, end_year: int) -> pd.DataFrame:
    frames = [_fetch_year(year) for year in range(start_year, end_year + 1)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["session_date"]).sort_values("session_date").reset_index(drop=True)
    if {"treasury_2y", "treasury_10y"} <= set(df.columns):
        df["treasury_2s10s"] = df["treasury_10y"] - df["treasury_2y"]
    if {"treasury_3m", "treasury_10y"} <= set(df.columns):
        df["treasury_3m10y"] = df["treasury_10y"] - df["treasury_3m"]
    if {"treasury_2y", "treasury_5y", "treasury_10y"} <= set(df.columns):
        df["treasury_curvature_2y5y10y"] = 2.0 * df["treasury_5y"] - df["treasury_2y"] - df["treasury_10y"]
    for column in [c for c in df.columns if c.startswith("treasury_")]:
        df[f"{column}_chg1d"] = df[column].diff()
        df[f"{column}_chg5d"] = df[column].diff(5)
    df["available_at"] = df["session_date"].map(_next_weekday_0600_jst)
    df["source"] = "us_treasury_daily_par_yield_curve"
    df["available_at_method"] = "next_weekday_0600_jst_conservative"
    return df
