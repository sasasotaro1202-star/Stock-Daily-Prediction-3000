from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.validation.independent_audit import audit_raw_inputs


PRICE = Path("data/prices")
CONTEXT = Path("data/market_context.parquet")
OUT = Path("data/research/independent_leakage_audit.json")


def _load_prices() -> pd.DataFrame:
    if not PRICE.exists():
        raise SystemExit("DEFERRED: price directory missing")
    files = sorted(PRICE.glob("*.parquet"))
    if not files:
        raise SystemExit("DEFERRED: no price parquet partitions")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def main() -> int:
    prices = _load_prices()
    if not CONTEXT.exists():
        raise SystemExit("DEFERRED: market context is absent")
    context = pd.read_parquet(CONTEXT)

    result = audit_raw_inputs(prices, context)
    payload = {
        "status": "PASS" if result.ok else "FAIL",
        "ok": result.ok,
        "violations": list(result.violations),
        "checks": result.checks,
        "auditor": "independent_raw_input_temporal_audit_v1",
        "audited_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if not result.ok:
        raise SystemExit("FAIL: independent leakage audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
