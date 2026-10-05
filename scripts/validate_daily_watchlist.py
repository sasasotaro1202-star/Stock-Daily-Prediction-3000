from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CONFIG = Path("config/daily_watchlist.yml")
OUTPUT = Path("data/predictions/daily_watchlist.json")


def main() -> None:
    if not CONFIG.exists():
        raise SystemExit("FAIL: daily watchlist config is missing")
    if not OUTPUT.exists():
        raise SystemExit("FAIL: daily watchlist output is missing")

    cfg = json.loads(json.dumps(__import__("yaml").safe_load(CONFIG.read_text(encoding="utf-8")) or {}))
    expected = (
        len(cfg["equities"]["items"])
        + len(cfg["us_equities"]["items"])
        + len(cfg["market_instruments"]["instruments"])
    )

    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("FAIL: daily watchlist rows is not a list")
    if len(rows) != expected:
        raise SystemExit(
            f"FAIL: daily watchlist row count mismatch: expected={expected} actual={len(rows)}"
        )

    keys = [(str(row.get("instrument_type")), str(row.get("symbol"))) for row in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("FAIL: duplicate daily watchlist instrument keys")

    cutoff = pd.Timestamp(payload.get("cutoff"), tz="UTC")
    if pd.isna(cutoff):
        raise SystemExit("FAIL: invalid watchlist cutoff")

    ready = 0
    for row in rows:
        status = str(row.get("prediction_status", ""))
        if status == "READY":
            ready += 1
            p = pd.to_numeric(row.get("p_up_1d"), errors="coerce")
            if pd.isna(p) or not 0.0 <= float(p) <= 1.0:
                raise SystemExit(
                    f"FAIL: invalid p_up_1d for READY row {row.get('display_name')}"
                )

            if row.get("direction") not in {"UP", "DOWN"}:
                raise SystemExit(
                    f"FAIL: invalid direction for READY row {row.get('display_name')}"
                )

            prediction_time = pd.to_datetime(
                row.get("prediction_time"),
                utc=True,
                errors="coerce",
            )
            if pd.isna(prediction_time):
                raise SystemExit(
                    f"FAIL: invalid prediction_time for {row.get('display_name')}"
                )
            if prediction_time > cutoff:
                raise SystemExit(
                    f"FAIL: future prediction_time for {row.get('display_name')}"
                )

            target_date = str(row.get("target_date", "")).strip()
            if not target_date:
                raise SystemExit(
                    f"FAIL: missing target_date for {row.get('display_name')}"
                )

    coverage = payload.get("coverage") or {}
    if int(coverage.get("total", -1)) != expected:
        raise SystemExit("FAIL: coverage.total mismatch")
    if int(coverage.get("ready", -1)) != ready:
        raise SystemExit("FAIL: coverage.ready mismatch")
    if int(coverage.get("deferred", -1)) != expected - ready:
        raise SystemExit("FAIL: coverage.deferred mismatch")

    print(
        f"daily-watchlist-integrity: PASS total={expected} "
        f"ready={ready} deferred={expected - ready}"
    )


if __name__ == "__main__":
    main()
