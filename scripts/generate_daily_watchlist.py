from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from exchange_calendars import get_calendar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONFIG = Path("config/daily_watchlist.yml")
PREDICTION_DIR = Path("data/predictions")
UNIVERSE_SOURCE = Path("data/universe/latest.json")
CONTEXT_SOURCE = Path("data/market_context.parquet")
OUT_JSON = Path("data/predictions/daily_watchlist.json")
OUT_MD = Path("data/predictions/daily_watchlist.md")

FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "volatility_20",
    "price_vs_sma20",
    "volatility_vs_median20",
]


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if cfg.get("schema_version") != 1:
        raise SystemExit("DEFERRED: unsupported daily watchlist schema_version")
    return cfg


def load_universe(path: Path = UNIVERSE_SOURCE) -> set[tuple[str, str]]:
    if not path.exists():
        raise SystemExit("DEFERRED: official universe snapshot is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise SystemExit("DEFERRED: official universe snapshot has no records")
    return {
        (str(row.get("asset_class", "")), str(row.get("symbol", "")).upper())
        for row in records
        if isinstance(row, dict)
    }


def validate_items(items: list[dict[str, Any]], expected_asset_class: str) -> None:
    seen: set[tuple[str, str]] = set()
    for item in items:
        name = str(item.get("display_name", "")).strip()
        symbol = str(item.get("symbol", "")).strip().upper()
        asset_class = str(item.get("asset_class", "")).strip()
        key = (asset_class, symbol)
        if not name or not symbol or asset_class != expected_asset_class:
            raise SystemExit(f"DEFERRED: invalid watchlist item {item}")
        if key in seen:
            raise SystemExit(f"DEFERRED: duplicate watchlist key {key}")
        seen.add(key)


def prediction_files() -> list[Path]:
    return sorted(PREDICTION_DIR.glob("prediction_*.parquet"))


def latest_equity_prediction(
    files: list[Path],
    asset_class: str,
    symbol: str,
    *,
    cutoff: pd.Timestamp,
    expected_target_date: str | None = None,
) -> tuple[pd.Series | None, str | None]:
    candidates: list[tuple[pd.Timestamp, Path, pd.Series]] = []
    symbol = symbol.upper()
    for path in files:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        required = {"asset_class", "symbol", "p_up_1d", "prediction_time", "prediction_status"}
        if not required.issubset(frame.columns):
            continue
        match = frame[
            frame["asset_class"].astype(str).eq(asset_class)
            & frame["symbol"].astype(str).str.upper().eq(symbol)
        ].copy()
        if len(match) != 1:
            continue
        prediction_time = pd.to_datetime(
            match.iloc[0]["prediction_time"],
            utc=True,
            errors="coerce",
        )
        if pd.isna(prediction_time) or prediction_time > cutoff:
            continue
        target_date = str(match.iloc[0].get("target_date", "")).strip()
        if expected_target_date is not None and target_date != expected_target_date:
            continue
        candidates.append((prediction_time, path, match.iloc[0]))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    _, path, row = candidates[-1]
    return row, path.name


def safe_num(value: Any) -> float | None:
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def build_equity_rows(
    items: list[dict[str, Any]],
    *,
    universe: set[tuple[str, str]],
    threshold: float,
    require_official_universe: bool,
    cutoff: pd.Timestamp,
    expected_target_date: str,
) -> list[dict[str, Any]]:
    expected_class = str(items[0].get("asset_class", "")) if items else "jp_stock"
    validate_items(items, expected_class)
    files = prediction_files()
    rows: list[dict[str, Any]] = []

    for item in items:
        asset_class = str(item["asset_class"])
        symbol = str(item["symbol"]).upper()
        base = {
            "instrument_type": "equity",
            "display_name": str(item["display_name"]),
            "symbol": symbol,
            "asset_class": asset_class,
            "direction": None,
            "p_up_1d": None,
            "expected_return_1d": None,
            "expected_close_1d": None,
            "range_low_1d": None,
            "range_high_1d": None,
            "rank": None,
            "model_id": None,
            "uncertainty": None,
            "prediction_status": None,
            "production_status": "DEFERRED",
            "prediction_file": None,
        }

        key = (asset_class, symbol)
        if require_official_universe and key not in universe:
            base["prediction_status"] = "DEFERRED_NOT_IN_OFFICIAL_UNIVERSE"
            rows.append(base)
            continue

        row, filename = latest_equity_prediction(
            files,
            asset_class,
            symbol,
            cutoff=cutoff,
            expected_target_date=expected_target_date,
        )
        if row is None:
            base["prediction_status"] = "DEFERRED_PREDICTION_HISTORY_MISSING"
            rows.append(base)
            continue

        p = safe_num(row.get("p_up_1d"))
        status = str(row.get("prediction_status", "UNKNOWN"))
        if p is None or not 0.0 <= p <= 1.0:
            base["prediction_status"] = "DEFERRED_INVALID_PROBABILITY"
            rows.append(base)
            continue

        base.update(
            {
                "direction": "UP" if p >= threshold else "DOWN",
                "p_up_1d": p,
                "expected_return_1d": safe_num(row.get("expected_return_1d")),
                "expected_close_1d": safe_num(row.get("expected_close_1d")),
                "range_low_1d": safe_num(row.get("range_low_1d")),
                "range_high_1d": safe_num(row.get("range_high_1d")),
                "rank": safe_num(row.get("rank")),
                "model_id": str(row.get("model_id", "")),
                "uncertainty": safe_num(row.get("model_disagreement")),
                "prediction_status": status,
                "production_status": (
                    "PRODUCTION_PREDICTION_REFERENCE" if status == "READY" else "DEFERRED"
                ),
                "prediction_file": filename,
                "prediction_time": str(row.get("prediction_time")),
                "target_date": str(row.get("target_date", "")),
            }
        )
        if status != "READY":
            base["direction"] = None
        rows.append(base)

    return rows


def prepare_market_frame(context: pd.DataFrame, family: str) -> pd.DataFrame:
    required = {
        "session_date",
        "family",
        "close",
        "ret_1d",
        "volatility_20",
        "available_at",
    }
    if not required.issubset(context.columns):
        raise SystemExit(f"DEFERRED: market context schema incomplete for {family}")

    frame = context[context["family"].astype(str).eq(family)].copy()
    if frame.empty:
        raise SystemExit(f"DEFERRED: market context family unavailable: {family}")

    frame["session_date"] = pd.to_datetime(
        frame["session_date"], errors="coerce"
    ).dt.date
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, errors="coerce"
    )
    for col in ("close", "ret_1d", "volatility_20"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = (
        frame.dropna(
            subset=["session_date", "available_at", "close", "ret_1d", "volatility_20"]
        )
        .drop_duplicates(["session_date"], keep="last")
        .sort_values("session_date")
        .reset_index(drop=True)
    )
    frame["ret_5d"] = (1.0 + frame["ret_1d"]).rolling(5, min_periods=5).apply(
        lambda x: float(np.prod(x)) - 1.0,
        raw=True,
    )
    frame["ret_20d"] = (1.0 + frame["ret_1d"]).rolling(20, min_periods=20).apply(
        lambda x: float(np.prod(x)) - 1.0,
        raw=True,
    )
    frame["price_vs_sma20"] = (
        frame["close"] / frame["close"].rolling(20, min_periods=20).mean() - 1.0
    )
    median20 = frame["volatility_20"].rolling(20, min_periods=20).median()
    frame["volatility_vs_median20"] = (
        frame["volatility_20"] / median20.replace(0, np.nan) - 1.0
    )
    return frame


def next_session_date(date_value: Any, calendar_code: str) -> str:
    start = pd.Timestamp(date_value).normalize()

    # FX is quoted on the global 24/5 market and is not represented by the
    # exchange_calendars stock-exchange calendar registry. Its next session
    # is the next weekday; exchange holidays are not encoded because this
    # pipeline's provider series is itself sampled on available FX sessions.
    if calendar_code == "24/5":
        candidate = start + pd.Timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += pd.Timedelta(days=1)
        return str(candidate.date())

    calendar = get_calendar(calendar_code)
    sessions = calendar.sessions_in_range(start, start + pd.Timedelta(days=14))
    for session in sessions:
        candidate = pd.Timestamp(session)
        if candidate.date() > start.date():
            return str(candidate.date())
    raise SystemExit(f"DEFERRED: next session unavailable for {calendar_code}")


def market_baseline(
    frame: pd.DataFrame,
    *,
    instrument: dict[str, Any],
    cutoff: pd.Timestamp,
    threshold: float,
) -> dict[str, Any]:
    usable = frame[frame["available_at"].le(cutoff)].copy()
    usable = usable.dropna(subset=FEATURES).sort_values("session_date").reset_index(drop=True)

    if len(usable) < 120:
        return {
            "instrument_type": "index_or_fx",
            "display_name": str(instrument["display_name"]),
            "symbol": str(instrument["provider_symbol"]),
            "instrument_id": str(instrument["instrument_id"]),
            "direction": None,
            "p_up_1d": None,
            "expected_return_1d": None,
            "expected_close_1d": None,
            "range_low_1d": None,
            "range_high_1d": None,
            "model_id": "logistic_regression_baseline_v1",
            "uncertainty": None,
            "prediction_status": "DEFERRED_INSUFFICIENT_PIT_HISTORY",
            "production_status": "RESEARCH_ONLY",
            "session_date": str(usable["session_date"].iloc[-1]) if not usable.empty else None,
            "target_date": None,
            "cutoff": str(cutoff),
        }

    work = usable.copy()
    work["next_close"] = work["close"].shift(-1)
    work["target_up"] = work["next_close"].gt(work["close"])
    work["target_return"] = work["next_close"] / work["close"] - 1.0
    train = work.iloc[:-1].dropna(subset=FEATURES + ["target_up", "target_return"])
    latest = usable.iloc[-1]

    if len(train) < 100 or train["target_up"].nunique() < 2:
        raise SystemExit(
            f"DEFERRED: insufficient matured baseline training data for {instrument['display_name']}"
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=0,
        ),
    )
    model.fit(train[FEATURES].astype(float), train["target_up"].astype(int))
    latest_x = pd.DataFrame([latest])[FEATURES].astype(float)
    p_up = float(model.predict_proba(latest_x)[0, 1])

    returns = train["target_return"].astype(float).tail(120)
    expected_return = float(returns.tail(60).mean())
    q10 = float(returns.quantile(0.10))
    q90 = float(returns.quantile(0.90))
    close = float(latest["close"])

    return {
        "instrument_type": "index_or_fx",
        "display_name": str(instrument["display_name"]),
        "symbol": str(instrument["provider_symbol"]),
        "instrument_id": str(instrument["instrument_id"]),
        "direction": "UP" if p_up >= threshold else "DOWN",
        "p_up_1d": p_up,
        "expected_return_1d": expected_return,
        "expected_close_1d": close * (1.0 + expected_return),
        "range_low_1d": close * (1.0 + q10),
        "range_high_1d": close * (1.0 + q90),
        "model_id": "logistic_regression_baseline_v1",
        "uncertainty": None,
        "prediction_status": "READY",
        "production_status": "RESEARCH_ONLY",
        "session_date": str(latest["session_date"]),
        "target_date": next_session_date(latest["session_date"], str(instrument["calendar"])),
        "cutoff": str(cutoff),
        "available_at": str(latest["available_at"]),
        "provider_symbol": str(latest.get("provider_symbol", instrument["provider_symbol"])),
    }


def build_market_rows(
    context: pd.DataFrame,
    instruments: list[dict[str, Any]],
    *,
    cutoff: pd.Timestamp,
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for instrument in instruments:
        frame = prepare_market_frame(context, str(instrument["context_family"]))
        rows.append(
            market_baseline(
                frame,
                instrument=instrument,
                cutoff=cutoff,
                threshold=threshold,
            )
        )
    return rows


def markdown(rows: list[dict[str, Any]], generated_at: str) -> str:
    lines = [
        "# Daily Stock / Index / FX Direction Watchlist",
        "",
        f"generated_at: {generated_at}",
        "",
        "| 対象 | 種別 | 方向 | UP確率 | 期待リターン | 予想終値/レート | 状態 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        p = "-" if row.get("p_up_1d") is None else f"{float(row['p_up_1d']):.1%}"
        ret = "-" if row.get("expected_return_1d") is None else f"{float(row['expected_return_1d']):+.2%}"
        close = "-" if row.get("expected_close_1d") is None else f"{float(row['expected_close_1d']):,.2f}"
        lines.append(
            f"| {row['display_name']} | {row['instrument_type']} | "
            f"{row.get('direction') or '-'} | {p} | {ret} | {close} | "
            f"{row.get('prediction_status') or '-'} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Equity entries reference the latest available PIT-safe production prediction artifact; they are not recomputed here.",
            "- Nikkei 225, S&P 500, and USDJPY use a research-only logistic baseline based on matured market-context data.",
            "- Missing universe, prediction history, or PIT-safe market context is DEFERRED; no value is fabricated.",
            "- Google is represented by Alphabet Class A (GOOGL).",
        ]
    )
    return "\n".join(lines) + "\n"


def expected_daily_target_date(current_date_jst: Any, calendar_code: str) -> str:
    return next_session_date(
        pd.Timestamp(current_date_jst).date() - pd.Timedelta(days=1),
        calendar_code,
    )


def main() -> None:
    cfg = load_config()
    if not bool(cfg.get("enabled", True)):
        raise SystemExit("DEFERRED: daily watchlist disabled")

    universe = load_universe()
    cutoff = now_utc()
    current_jst_date = cutoff.tz_convert("Asia/Tokyo").date()
    expected_jp_target_date = expected_daily_target_date(current_jst_date, "XTKS")
    expected_us_target_date = expected_daily_target_date(current_jst_date, "XNYS")
    jp_rows = build_equity_rows(
        jp_items,
        universe=universe,
        threshold=float(cfg["equities"]["direction_threshold"]),
        require_official_universe=bool(cfg["equities"]["require_official_universe"]),
        cutoff=cutoff,
        expected_target_date=expected_jp_target_date,
    )
    us_rows = build_equity_rows(
        us_items,
        universe=universe,
        threshold=float(cfg["us_equities"]["direction_threshold"]),
        require_official_universe=bool(cfg["us_equities"]["require_official_universe"]),
        cutoff=cutoff,
        expected_target_date=expected_us_target_date,
    )

    if not CONTEXT_SOURCE.exists():
        raise SystemExit("DEFERRED: market context is missing")
    context = pd.read_parquet(CONTEXT_SOURCE)
    markets = cfg["market_instruments"]
    market_rows = build_market_rows(
        context,
        markets["instruments"],
        cutoff=cutoff,
        threshold=float(markets["direction_threshold"]),
    )

    rows = jp_rows + us_rows + market_rows
    generated_at = cutoff.isoformat()
    payload = {
        "schema_version": 1,
        "status": "READY" if any(r.get("prediction_status") == "READY" for r in rows) else "DEFERRED",
        "generated_at": generated_at,
        "cutoff": generated_at,
        "as_of_date_jst": str(current_jst_date),
        "expected_equity_target_dates": {
            "jp_stock": expected_jp_target_date,
            "us_stock": expected_us_target_date,
        },
        "universe_sha256": file_sha256(UNIVERSE_SOURCE),
        "market_context_sha256": file_sha256(CONTEXT_SOURCE),
        "rows": rows,
        "coverage": {
            "total": len(rows),
            "ready": sum(r.get("prediction_status") == "READY" for r in rows),
            "deferred": sum(r.get("prediction_status") != "READY" for r in rows),
        },
        "research_only_market_instruments": True,
        "promotion_allowed": False,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(markdown(rows, generated_at), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "total": payload["coverage"]["total"],
                "ready": payload["coverage"]["ready"],
                "deferred": payload["coverage"]["deferred"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
