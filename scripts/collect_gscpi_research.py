from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import yaml

CONFIG = Path("config/research_data_sources.yml")
OUT_DIR = Path("data/research/gscpi_newyorkfed")
DEFAULT_URL = "https://www.newyorkfed.org/medialibrary/media/research/interactives/gscpi/downloads/gscpi_data.xls"
USER_AGENT = os.getenv("GSCPI_USER_AGENT", "Stock-Daily-Prediction-3000/1.0 gscpi-research")
MAX_ATTEMPTS = max(1, int(os.getenv("GSCPI_MAX_ATTEMPTS", "3")))
TIMEOUT = int(os.getenv("GSCPI_TIMEOUT_SECONDS", "60"))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_bytes(url: str) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/vnd.ms-excel,application/octet-stream,*/*",
            },
        )
        try:
            with urlopen(request, timeout=TIMEOUT) as response:
                raw = response.read()
                headers = {
                    "etag": str(response.headers.get("ETag") or ""),
                    "last_modified": str(response.headers.get("Last-Modified") or ""),
                    "content_type": str(response.headers.get("Content-Type") or ""),
                }
            if not raw:
                raise ValueError("GSCPI response body is empty")
            return raw, headers
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt >= MAX_ATTEMPTS:
                raise
    raise RuntimeError(f"GSCPI request failed: {last_error!r}")


def _parse_frame(raw: bytes, *, sheet_name: str) -> pd.DataFrame:
    frame = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name, engine="xlrd")
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("GSCPI workbook does not contain the expected two-column table")
    frame = frame.iloc[:, :2].copy()
    frame.columns = ["observation_date", "gscpi"]
    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"], errors="coerce", dayfirst=True
    )
    frame["gscpi"] = pd.to_numeric(frame["gscpi"], errors="coerce")
    frame = frame.dropna(subset=["observation_date", "gscpi"]).copy()
    if frame.empty:
        raise ValueError("GSCPI workbook contains no parseable observations")
    frame["observation_date"] = frame["observation_date"].dt.date.astype(str)
    return (
        frame.sort_values("observation_date")
        .drop_duplicates("observation_date", keep="last")
        .reset_index(drop=True)
    )


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    source = cfg.get("gscpi_newyorkfed") or {}
    if source.get("enabled") is False:
        raise SystemExit("DEFERRED: GSCPI source disabled")

    url = str(source.get("source_url") or DEFAULT_URL)
    sheet_name = str(source.get("sheet_name") or "GSCPI Monthly Data")
    raw, headers = _fetch_bytes(url)
    frame = _parse_frame(raw, sheet_name=sheet_name)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / "gscpi_data.csv", index=False)
    (OUT_DIR / "gscpi_data.xls").write_bytes(raw)

    report = {
        "schema_version": 1,
        "status": "EVALUATED",
        "retrieved_at": _utc(),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "source": "Federal Reserve Bank of New York Global Supply Chain Pressure Index",
        "source_url": url,
        "summary": {
            "rows": int(len(frame)),
            "first_observation": str(frame["observation_date"].iloc[0]),
            "last_observation": str(frame["observation_date"].iloc[-1]),
        },
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "headers": headers,
        "pit_status": "UNVERIFIED",
        "pit_policy": {
            "observation_date_preserved": True,
            "published_schedule_recorded": True,
            "row_level_available_at_verified": False,
            "source_available_at_inferred": False,
            "revision_policy_verified": False,
            "fail_closed_for_production": True,
        },
        "release_schedule": {
            "frequency": "monthly",
            "time_local": "10:00 America/New_York",
            "rule": "fourth business day",
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
