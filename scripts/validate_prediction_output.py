from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


OUT = Path("data/predictions/latest.parquet")
ALLOWED_STATUS = {"READY", "DEFERRED_INCOMPLETE_FEATURES"}


def main() -> None:
    path = Path(os.environ.get("PREDICTION_OUTPUT", str(OUT)))
    if not path.exists():
        raise SystemExit(f"FAIL: prediction output is missing: {path}")

    df = pd.read_parquet(path)
    required = {
        "symbol",
        "asset_class",
        "prediction_time",
        "prediction_date",
        "p_up_1d",
        "expected_return_1d",
        "expected_close_1d",
        "range_low_1d",
        "range_high_1d",
        "model_id",
        "training_scope",
        "return_training_scope",
        "route_reason",
        "regime",
        "model_disagreement",
        "prediction_status",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"FAIL: prediction output missing columns: {missing}")
    if df.empty:
        raise SystemExit("FAIL: prediction output contains zero rows")

    keys = ["symbol", "asset_class", "prediction_date"]
    if df.duplicated(keys).any():
        raise SystemExit("FAIL: duplicate prediction security/date rows detected")

    prediction_time = pd.to_datetime(df["prediction_time"], utc=True, errors="coerce")
    if prediction_time.isna().any():
        raise SystemExit("FAIL: prediction_time contains invalid timestamps")
    now = pd.Timestamp.now(tz="UTC")
    if (prediction_time > now).any():
        raise SystemExit("FAIL: prediction_time is in the future")

    if not df["prediction_status"].isin(ALLOWED_STATUS).all():
        bad = sorted(df.loc[~df["prediction_status"].isin(ALLOWED_STATUS), "prediction_status"].astype(str).unique())
        raise SystemExit(f"FAIL: unknown prediction_status values: {bad}")

    ready = df["prediction_status"].eq("READY")
    for col in ("p_up_1d", "expected_return_1d", "expected_close_1d", "range_low_1d", "range_high_1d"):
        values = pd.to_numeric(df.loc[ready, col], errors="coerce")
        if values.isna().any() or not values.map(lambda x: pd.notna(x) and pd.api.types.is_number(x)).all():
            raise SystemExit(f"FAIL: READY rows contain invalid numeric values in {col}")

    p = pd.to_numeric(df.loc[ready, "p_up_1d"], errors="coerce")
    if not p.between(0.0, 1.0).all():
        raise SystemExit("FAIL: p_up_1d outside [0,1]")

    low = pd.to_numeric(df.loc[ready, "range_low_1d"], errors="coerce")
    mid = pd.to_numeric(df.loc[ready, "expected_return_1d"], errors="coerce")
    high = pd.to_numeric(df.loc[ready, "range_high_1d"], errors="coerce")
    if ((low > mid) | (mid > high)).any():
        raise SystemExit("FAIL: return interval ordering is invalid")

    if df.loc[ready, "model_id"].astype(str).str.strip().eq("").any():
        raise SystemExit("FAIL: READY rows contain empty model_id")
    if df.loc[ready, "route_reason"].astype(str).str.strip().eq("").any():
        raise SystemExit("FAIL: READY rows contain empty route_reason")

    for col in ("expected_return_1d", "expected_close_1d", "range_low_1d", "range_high_1d"):
        values = pd.to_numeric(df.loc[ready, col], errors="coerce")
        if not values.map(pd.notna).all():
            raise SystemExit(f"FAIL: READY rows contain non-finite numeric values in {col}")

    print(
        f"prediction-output-integrity: PASS rows={len(df)} "
        f"ready={int(ready.sum())} deferred={int((~ready).sum())}"
    )


if __name__ == "__main__":
    main()
