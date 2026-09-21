from __future__ import annotations
import json
from pathlib import Path

def main():
    metrics_path=Path("data/research/latest_metrics.json")
    audit_path=Path("data/research/leakage_audit.json")
    quality_path=Path("data/research/data_quality.json")
    universe_path=Path("data/research/universe_quality.json")
    context_path=Path("data/research/market_context_quality.json")
    frozen_path=Path("data/research/frozen_holdout_result.json")
    manifest_path=Path("data/research/reproducibility_manifest.json")
    reasons=[]
    if not metrics_path.exists(): reasons.append("missing_oos_metrics")
    if not audit_path.exists(): reasons.append("missing_leakage_audit")
    if not quality_path.exists(): reasons.append("missing_data_quality")
    if not universe_path.exists(): reasons.append("missing_universe_quality")
    if not context_path.exists(): reasons.append("missing_market_context_quality")
    if not frozen_path.exists(): reasons.append("missing_frozen_holdout_result")
    if not manifest_path.exists(): reasons.append("missing_reproducibility_manifest")
    metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    quality=json.loads(quality_path.read_text()) if quality_path.exists() else {}
    universe=json.loads(universe_path.read_text()) if universe_path.exists() else {}
    context=json.loads(context_path.read_text()) if context_path.exists() else {}
    frozen=json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
    if metrics.get("status")!="OOS_COMPLETE":
        reasons.append("direction_oos_not_complete")
    return_oos=metrics.get("return_oos",{})
    return_metrics=return_oos.get("metrics",{})
    if return_oos.get("status")!="OOS_COMPLETE":
        reasons.append("return_oos_not_complete")
    if not return_metrics or not all(
        key in return_metrics
        and isinstance(return_metrics.get(key),(int,float))
        and __import__("math").isfinite(float(return_metrics.get(key)))
        for key in ("mae","rmse","sign_accuracy","range_80_coverage")
    ):
        reasons.append("return_oos_metrics_missing")
    elif not 0.0 <= float(return_metrics["range_80_coverage"]) <= 1.0:
        reasons.append("return_oos_interval_coverage_invalid")
    if audit.get("ok") is not True: reasons.append("leakage_audit_failed")
    if quality.get("status")!="PASS": reasons.append("data_quality_not_pass")
    if universe.get("status")!="PASS": reasons.append("universe_quality_not_pass")
    if context.get("status")!="PASS": reasons.append("market_context_quality_not_pass")
    if frozen.get("status")!="EVALUATED_ONCE": reasons.append("holdout_not_evaluated_once")
    if frozen.get("beats_baseline") is not True: reasons.append("holdout_does_not_beat_baseline")
    if not frozen.get("calibration_within_limit",False): reasons.append("holdout_calibration_out_of_limit")
    result={"approved":not reasons,"reasons":reasons,"production_status":"APPROVED" if not reasons else "DEFERRED"}
    Path("data/research").mkdir(parents=True,exist_ok=True)
    Path("data/research/release_gate.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
