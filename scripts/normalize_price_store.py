from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_DIR = Path("data/prices")
OUT = PRICE_DIR / "canonical.parquet"
AUDIT = Path("data/research/price_normalization.json")
KEYS = ["asset_class", "symbol", "session_date"]
NUMERIC = ["open", "high", "low", "close", "volume"]
REQUIRED = set(KEYS + ["available_at", "retrieved_at", *NUMERIC])


def main() -> None:
    files = sorted(PRICE_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit("DEFERRED: no raw price partitions to normalize")

    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(
            f"FAIL: price normalization missing columns: {sorted(missing)}"
        )

    for col in ("available_at", "retrieved_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df["session_date"] = pd.to_datetime(
        df["session_date"], errors="coerce"
    ).dt.date

    before = len(df)
    invalid_pit = int(
        df["available_at"].gt(df["retrieved_at"]).fillna(False).sum()
    )
    invalid_ohlc_mask = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | df[NUMERIC].isna().any(axis=1)
    )
    invalid_ohlc = int(invalid_ohlc_mask.sum())

    work = df.loc[~(df["available_at"] > df["retrieved_at"])].copy()
    work = work.loc[~invalid_ohlc_mask.loc[work.index]].copy()

    duplicate_rows = int(work.duplicated(KEYS, keep=False).sum())
    conflict_groups = 0
    if duplicate_rows:
        grouped = work.loc[
            work.duplicated(KEYS, keep=False)
        ].groupby(KEYS, sort=False)
        for _, group in grouped:
            base = group[NUMERIC].iloc[0].to_numpy(dtype=float)
            values = group[NUMERIC].to_numpy(dtype=float)
            if not np.allclose(values, base, rtol=0.0, atol=0.0, equal_nan=True):
                conflict_groups += 1

    if conflict_groups:
        raise SystemExit(
            f"FAIL: conflicting duplicate price groups: {conflict_groups}"
        )

    work = work.sort_values(
        KEYS + ["retrieved_at", "available_at"],
        na_position="first",
    )
    work = work.drop_duplicates(KEYS, keep="last")
    work = work.sort_values(KEYS).reset_index(drop=True)

    for path in files:
        if path != OUT:
            path.unlink()
    work.to_parquet(OUT, index=False)

    result = {
        "status": "PASS",
        "input_files": len(files),
        "input_rows": int(before),
        "output_rows": int(len(work)),
        "duplicate_rows_removed": int(max(0, duplicate_rows - (duplicate_rows and len(work) or 0))
        ),
        "duplicate_rows_detected": duplicate_rows,
        "conflicting_duplicate_groups": conflict_groups,
        "future_pit_rows_removed": invalid_pit,
        "invalid_ohlc_rows_removed": invalid_ohlc,
        "output": str(OUT),
    }
    # Keep the exact removed-duplicate count audit-friendly by calculating it
    # from key cardinality rather than assuming one duplicate per group.
    input_unique = int(
        df.loc[~(df["available_at"] > df["retrieved_at"])]
        .drop_duplicates(KEYS)
        .shape[0]
    )
    result["duplicate_key_rows_removed"] = max(0, before - invalid_pit - invalid_ohlc - input_unique)

    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
