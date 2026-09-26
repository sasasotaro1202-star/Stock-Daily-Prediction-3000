from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import hashlib
import json
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG = Path("config/boj_series.yml")
JST = timezone(timedelta(hours=9))

@dataclass(frozen=True)
class BOJObservation:
    db: str
    series_code: str
    series_name: str
    unit: str
    frequency: str
    observation_date: date
    value: float
    available_at: datetime

def _next_weekday(day: date) -> date:
    candidate = day + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate

def conservative_available_at(observation_date: date, *, hour: int = 15) -> datetime:
    return datetime.combine(_next_weekday(observation_date), time(hour=hour), tzinfo=JST)

def build_request_url(
    base_url: str,
    *,
    db: str,
    codes: list[str],
    start_date: str,
    end_date: str | None = None,
    language: str = "en",
) -> str:
    params = {
        "format": "json",
        "lang": language,
        "db": db,
        "code": ",".join(codes),
        "startDate": start_date,
    }
    if end_date:
        params["endDate"] = end_date
    return f"{base_url}?{urlencode(params)}"

def fetch_json(url: str, *, timeout_seconds: int = 30) -> tuple[dict[str, Any], bytes]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Stock-Daily-Prediction-3000/0.1 (BOJ research; fail-closed)",
        },
        method="GET",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("FAIL: BOJ response is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("FAIL: BOJ response root is not an object")
    return parsed, raw

def _resultset(payload: dict[str, Any]) -> list[dict[str, Any]]:
    direct = payload.get("RESULTSET")
    if isinstance(direct, list):
        return direct
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("RESULTSET"), list):
        return data["RESULTSET"]
    raise ValueError("FAIL: BOJ RESULTSET missing")

def parse_get_data_code_payload(
    payload: dict[str, Any],
    *,
    db: str,
    available_hour_jst: int = 15,
) -> list[BOJObservation]:
    observations: list[BOJObservation] = []
    for series in _resultset(payload):
        code = str(series.get("SERIES_CODE") or "").strip()
        if not code:
            raise ValueError("FAIL: BOJ series code missing")
        name = str(series.get("NAME_OF_TIME_SERIES") or "").strip()
        unit = str(series.get("UNIT") or "").strip()
        frequency = str(series.get("FREQUENCY") or "").strip()
        values = series.get("VALUES") or {}
        dates = values.get("SURVEY_DATES") if isinstance(values, dict) else None
        numbers = values.get("VALUES") if isinstance(values, dict) else None
        if not isinstance(dates, list) or not isinstance(numbers, list):
            raise ValueError(f"FAIL: BOJ values missing for {db}/{code}")
        if len(dates) != len(numbers):
            raise ValueError(f"FAIL: BOJ date/value length mismatch for {db}/{code}")
        for raw_date, raw_value in zip(dates, numbers):
            if raw_value in (None, "", ".", "-"):
                continue
            try:
                obs_date = pd.to_datetime(str(raw_date), format="%Y%m%d").date()
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"FAIL: invalid BOJ observation {db}/{code}") from exc
            if not pd.notna(value):
                continue
            observations.append(
                BOJObservation(
                    db=db,
                    series_code=code,
                    series_name=name,
                    unit=unit,
                    frequency=frequency,
                    observation_date=obs_date,
                    value=value,
                    available_at=conservative_available_at(
                        obs_date, hour=available_hour_jst
                    ),
                )
            )
    return observations

def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if config.get("research_only") is not True:
        raise ValueError("FAIL: BOJ collector must remain research_only")
    return config

def collect_from_config(
    config: dict[str, Any],
    *,
    start_override: str | None = None,
    end_override: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    policy = config.get("availability_policy") or {}
    if policy.get("type") != "next_weekday":
        raise ValueError("FAIL: unsupported BOJ availability policy")
    hour_jst = int(str(policy.get("time_jst", "15:00")).split(":", 1)[0])
    base_url = str(config["base_url"])
    language = str(config.get("language", "en"))
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    groups = config.get("series_groups") or []
    if not groups:
        raise ValueError("FAIL: no BOJ series groups configured")

    for group in groups:
        db = str(group["db"])
        codes = [str(x) for x in group.get("codes") or []]
        if not codes:
            raise ValueError(f"FAIL: no series codes configured for {db}")
        url = build_request_url(
            base_url,
            db=db,
            codes=codes,
            start_date=start_override or str(group["start_date"]),
            end_date=end_override,
            language=language,
        )
        payload, raw = fetch_json(url)
        hashes[db] = hashlib.sha256(raw).hexdigest()
        for obs in parse_get_data_code_payload(payload, db=db, available_hour_jst=hour_jst):
            rows.append({
                "db": obs.db,
                "series_code": obs.series_code,
                "series_name": obs.series_name,
                "unit": obs.unit,
                "frequency": obs.frequency,
                "observation_date": obs.observation_date,
                "value": obs.value,
                "available_at": obs.available_at,
                "source": "boj_timeseries",
                "source_url": url,
                "retrieved_at": now,
                "raw_sha256": hashes[db],
            })

    if not rows:
        raise RuntimeError("DEFERRED: BOJ API returned no observations")

    frame = pd.DataFrame(rows)
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce").dt.date
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    frame["retrieved_at"] = pd.to_datetime(frame["retrieved_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["observation_date", "available_at", "retrieved_at", "value"])
    frame = frame[frame["available_at"] <= pd.Timestamp(now)].copy()
    frame = (
        frame.drop_duplicates(["db", "series_code", "observation_date"], keep="last")
        .sort_values(["db", "series_code", "observation_date"])
        .reset_index(drop=True)
    )
    metadata = {
        "status": "OOS_READY" if not frame.empty else "DEFERRED",
        "source": "Bank of Japan Time-Series Data Search API",
        "source_id": "boj_timeseries",
        "research_only": True,
        "rows": int(len(frame)),
        "series": sorted(frame["series_code"].unique().tolist()),
        "dbs": sorted(frame["db"].unique().tolist()),
        "raw_sha256_by_db": hashes,
        "retrieved_at": now.isoformat(),
        "pit_policy": {
            "type": policy["type"],
            "time_jst": str(policy.get("time_jst", "15:00")),
            "fail_closed": True,
        },
    }
    return frame, metadata
