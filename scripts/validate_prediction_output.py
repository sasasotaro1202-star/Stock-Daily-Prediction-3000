from __future__ import annotations

import os
from pathlib import Path

import numpy as np
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
        "return_q10_1d",
        "return_q90_1d",
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
    if not ready.any():
        raise SystemExit("FAIL: prediction output contains no READY rows")
    expected_prediction_date = prediction_time.dt.tz_convert("Asia/Tokyo").dt.date.astype(str)
    if not df["prediction_date"].astype(str).eq(expected_prediction_date).all():
        raise SystemExit("FAIL: prediction_date does not match prediction_time in JST")

    for col in ("p_up_1d", "expected_return_1d", "expected_close_1d", "range_low_1d", "range_high_1d"):
        values = pd.to_numeric(df.loc[ready, col], errors="coerce").to_numpy(dtype=float)
        if values.size and not np.isfinite(values).all():
            raise SystemExit(f"FAIL: READY rows contain non-finite numeric values in {col}")

    p = pd.to_numeric(df.loc[ready, "p_up_1d"], errors="coerce")
    if not p.between(0.0, 1.0).all():
        raise SystemExit("FAIL: p_up_1d outside [0,1]")

    return_low = pd.to_numeric(df.loc[ready, "return_q10_1d"], errors="coerce")
    return_mid = pd.to_numeric(df.loc[ready, "expected_return_1d"], errors="coerce")
    return_high = pd.to_numeric(df.loc[ready, "return_q90_1d"], errors="coerce")
    if ((return_low > return_mid) | (return_mid > return_high)).any():
        raise SystemExit("FAIL: return interval ordering is invalid")

    close_low = pd.to_numeric(df.loc[ready, "range_low_1d"], errors="coerce")
    close_mid = pd.to_numeric(df.loc[ready, "expected_close_1d"], errors="coerce")
    close_high = pd.to_numeric(df.loc[ready, "range_high_1d"], errors="coerce")
    if ((close_low > close_mid) | (close_mid > close_high)).any():
        raise SystemExit("FAIL: price interval ordering is invalid")

    if df.loc[ready, "model_id"].astype(str).str.strip().eq("").any():
        raise SystemExit("FAIL: READY rows contain empty model_id")
    if df.loc[ready, "route_reason"].astype(str).str.strip().eq("").any():
        raise SystemExit("FAIL: READY rows contain empty route_reason")
    if df.loc[ready, "training_scope"].astype(str).str.strip().eq("").any():
        raise SystemExit("FAIL: READY rows contain empty training_scope")
    if df.loc[ready, "return_training_scope"].astype(str).str.strip().eq("").any():
        raise SystemExit("FAIL: READY rows contain empty return_training_scope")
    disagreement = pd.to_numeric(df.loc[ready, "model_disagreement"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(disagreement).all() or (disagreement < 0.0).any():
        raise SystemExit("FAIL: READY rows contain invalid model_disagreement")

    print(
        f"prediction-output-integrity: PASS rows={len(df)} "
        f"ready={int(ready.sum())} deferred={int((~ready).sum())}"
    )


if __name__ == "__main__":
    main()
