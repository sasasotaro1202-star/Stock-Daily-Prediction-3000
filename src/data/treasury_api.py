from __future__ import annotations

from datetime import date, datetime, time, timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import pandas as pd

TREASURY_XML_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
USER_AGENT = "Stock-Daily-Prediction-3000/0.1"

YIELD_COLUMNS = {
    "BC_1MONTH": "treasury_1m",
    "BC_3MONTH": "treasury_3m",
    "BC_6MONTH": "treasury_6m",
    "BC_1YEAR": "treasury_1y",
    "BC_2YEAR": "treasury_2y",
    "BC_5YEAR": "treasury_5y",
    "BC_10YEAR": "treasury_10y",
    "BC_30YEAR": "treasury_30y",
}


def _next_weekday_0600_jst(session_date: date) -> pd.Timestamp:
    current = session_date + timedelta(days=1)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return pd.Timestamp(
        datetime.combine(current, time(6, 0), tzinfo=ZoneInfo("Asia/Tokyo"))
    ).tz_convert("UTC")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _fetch_year(year: int) -> pd.DataFrame:
    params = urlencode(
        {
            "data": "daily_treasury_yield_curve",
            "field_tdr_date_value": str(year),
        }
    )
    url = f"{TREASURY_XML_URL}?{params}"
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/xml,text/xml"},
    )
    with urlopen(request, timeout=45) as response:
        payload = response.read()
    root = ET.fromstring(payload)

    rows: list[dict[str, object]] = []
    for entry in root.iter():
        if _local_name(entry.tag) != "entry":
            continue
        properties = next(
            (child for child in entry.iter() if _local_name(child.tag) == "properties"),
            None,
        )
        if properties is None:
            continue
        values = {
            _local_name(child.tag): (child.text or "").strip()
            for child in properties
        }
        raw_date = values.get("NEW_DATE") or values.get("record_date")
        if not raw_date:
            continue
        row: dict[str, object] = {"session_date": str(raw_date)[:10]}
        for xml_key, output_name in YIELD_COLUMNS.items():
            raw = values.get(xml_key, "")
            if raw in {"", "N/A", "NA", "null", "None"}:
                row[output_name] = pd.NA
                continue
            try:
                row[output_name] = float(raw)
            except (TypeError, ValueError):
                row[output_name] = pd.NA
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["session_date"] = pd.to_datetime(
        out["session_date"], errors="coerce"
    ).dt.date
    for column in YIELD_COLUMNS.values():
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["session_date"])


def collect_treasury_curve(start_year: int, end_year: int) -> pd.DataFrame:
    frames = [_fetch_year(year) for year in range(start_year, end_year + 1)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["session_date"])
        .sort_values("session_date")
        .reset_index(drop=True)
    )
    if {"treasury_2y", "treasury_10y"} <= set(df.columns):
        df["treasury_2s10s"] = df["treasury_10y"] - df["treasury_2y"]
    if {"treasury_3m", "treasury_10y"} <= set(df.columns):
        df["treasury_3m10y"] = df["treasury_10y"] - df["treasury_3m"]
    if {"treasury_2y", "treasury_5y", "treasury_10y"} <= set(df.columns):
        df["treasury_curvature_2y5y10y"] = (
            2.0 * df["treasury_5y"] - df["treasury_2y"] - df["treasury_10y"]
        )
    for column in [c for c in df.columns if c.startswith("treasury_")]:
        df[f"{column}_chg1d"] = df[column].diff()
        df[f"{column}_chg5d"] = df[column].diff(5)
    df["available_at"] = df["session_date"].map(_next_weekday_0600_jst)
    df["source"] = "us_treasury_daily_par_yield_curve"
    df["available_at_method"] = "next_weekday_0600_jst_conservative"
    return df
