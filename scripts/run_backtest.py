from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from src.backtest.cross_sectional import evaluate_predictions

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",default="data/predictions/latest.parquet")
    args=ap.parse_args()
    pred=Path(args.predictions)
    bars=Path("data/prices")
    if not pred.exists() or not bars.exists():
        raise SystemExit("DEFERRED: predictions and price data required")
    result=evaluate_predictions(
        pd.read_parquet(pred),
        pd.read_parquet(bars),
        top_quantile=0.1,
        cost_bps=5.0,
    )
    if result.get("status") == "PASS":
        # Capafy-inspired secondary audit. The backtest remains the source of
        # realized returns; this only measures selection/noise robustness.
        dates = pd.to_datetime(
            [row["date"] for row in result.get("_daily_rows", [])],
            errors="coerce",
        )
        returns = pd.Series(
            [row["long_short_return"] for row in result.get("_daily_rows", [])],
            dtype=float,
        )
        result["anti_overfit"] = anti_overfit_battery(
            dates,
            returns,
            trials=1000,
        )
    else:
        result["anti_overfit"] = {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": "backtest_not_pass",
        }
    result.pop("_daily_rows", None)
    out=Path("data/research/backtest_latest.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
