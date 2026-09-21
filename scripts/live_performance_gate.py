from __future__ import annotations

import json
import math
from pathlib import Path


MONITOR=Path("data/research/monitor_latest.json")
OUT=Path("data/research/live_performance_gate.json")


def main():
    if not MONITOR.exists():
        result={
            "status":"ALLOW",
            "reason":"monitoring_not_warmed",
            "production_action":"ALLOW",
        }
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result,indent=2))
        return

    payload=json.loads(MONITOR.read_text(encoding="utf-8"))
    status=str(payload.get("status","NO_BASELINE"))

    if status in {"NO_BASELINE","WARMUP"}:
        result={
            "status":"ALLOW",
            "reason":status,
            "production_action":"ALLOW",
        }
    elif status=="PASS":
        metrics=payload.get("metrics",{})
        valid=all(
            k in metrics and isinstance(metrics[k],(int,float))
            and math.isfinite(float(metrics[k]))
            for k in ("logloss","brier","ece")
        )
        result={
            "status":"ALLOW" if valid else "DEFERRED",
            "reason":"monitoring_pass" if valid else "monitor_metrics_invalid",
            "production_action":"ALLOW" if valid else "DEFERRED",
        }
    else:
        result={
            "status":"DEFERRED",
            "reason":"live_monitor_failed",
            "production_action":"DEFERRED",
            "monitor_status":status,
        }

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

    if result["production_action"]!="ALLOW":
        raise SystemExit("DEFERRED: live performance gate blocked production prediction")


if __name__=="__main__":
    main()
