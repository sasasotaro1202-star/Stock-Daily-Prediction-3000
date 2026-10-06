from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

CONFIG = Path("config/daily_watchlist.yml")
OUTPUT = Path("data/predictions/daily_watchlist.json")


def main() -> None:
    if not CONFIG.exists():
        raise SystemExit("FAIL: daily watchlist config is missing")
    if not OUTPUT.exists():
        raise SystemExit("FAIL: daily watchlist output is missing")

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    priority = cfg.get("priority_watchlist") or {}
    if not bool(priority.get("mandatory_daily_prediction", False)):
        raise SystemExit("FAIL: mandatory daily prediction is disabled")
    expected = (
        len(cfg["equities"]["items"])
        + len(cfg["us_equities"]["items"])
        + len(cfg["market_instruments"]["instruments"])
    )
    configured_expected = int(priority.get("expected_count", expected))
    if configured_expected != expected:
        raise SystemExit(
            f"FAIL: configured expected_count mismatch: expected={expected} configured={configured_expected}"
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

    cutoff = pd.to_datetime(payload.get("cutoff"), utc=True, errors="coerce")
    if pd.isna(cutoff):
        raise SystemExit("FAIL: invalid watchlist cutoff")

    usable_statuses = {"READY", "READY_NEAR_PRODUCTION"}
    ready = 0
    production_ready = 0
    for row in rows:
        status = str(row.get("prediction_status", ""))
        if status in usable_statuses:
            ready += 1
            if status == "READY":
                production_ready += 1
            p = pd.to_numeric(row.get("p_up_1d"), errors="coerce")
            if pd.isna(p) or not 0.0 <= float(p) <= 1.0:
                raise SystemExit(
                    f"FAIL: invalid p_up_1d for usable row {row.get('display_name')}"
                )

            if row.get("direction") not in {"UP", "DOWN"}:
                raise SystemExit(
                    f"FAIL: invalid direction for usable row {row.get('display_name')}"
                )

            prediction_time = pd.to_datetime(
                row.get("prediction_time"),
                utc=True,
                errors="coerce",
            )
            if pd.isna(prediction_time):
                raise SystemExit(
                    f"FAIL: invalid prediction_time for usable row {row.get('display_name')}"
                )
            if prediction_time > cutoff:
                raise SystemExit(
                    f"FAIL: future prediction_time for usable row {row.get('display_name')}"
                )

            available_at = pd.to_datetime(
                row.get("available_at"),
                utc=True,
                errors="coerce",
            )
            if pd.isna(available_at):
                raise SystemExit(
                    f"FAIL: invalid available_at for usable row {row.get('display_name')}"
                )
            if available_at > prediction_time or available_at > cutoff:
                raise SystemExit(
                    f"FAIL: PIT availability violation for usable row {row.get('display_name')}"
                )

            target_date = str(row.get("target_date", "")).strip()
            if not target_date:
                raise SystemExit(
                    f"FAIL: missing target_date for usable row {row.get('display_name')}"
                )

    coverage = payload.get("coverage") or {}
    if int(coverage.get("total", -1)) != expected:
        raise SystemExit("FAIL: coverage.total mismatch")
    if int(coverage.get("ready", -1)) != ready:
        raise SystemExit("FAIL: coverage.ready mismatch")
    if int(coverage.get("deferred", -1)) != expected - ready:
        raise SystemExit("FAIL: coverage.deferred mismatch")
    if int(coverage.get("production_ready", -1)) != production_ready:
        raise SystemExit("FAIL: coverage.production_ready mismatch")

    print(
        f"daily-watchlist-integrity: PASS total={expected} "
        f"ready={ready} deferred={expected - ready}"
    )


if __name__ == "__main__":
    main()
