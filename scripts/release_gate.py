from __future__ import annotations
import json
from pathlib import Path
def main():
    metrics=Path("data/research/latest_metrics.json")
    audit=Path("data/research/leakage_audit.json")
    frozen=Path("data/research/frozen_holdout.json")
    manifest=Path("data/research/reproducibility_manifest.json")
    reasons=[]
    if not metrics.exists(): reasons.append("missing_oos_metrics")
    if not audit.exists(): reasons.append("missing_independent_leakage_audit")
    if not frozen.exists(): reasons.append("missing_frozen_holdout")
    if not manifest.exists(): reasons.append("missing_reproducibility_manifest")
    payload=json.loads(metrics.read_text()) if metrics.exists() else {}
    if payload.get("status")!="OOS_COMPLETE": reasons.append("oos_not_complete")
    if payload.get("selected_model") is None: reasons.append("no_oos_selected_model")
    result={"approved":not reasons,"reasons":reasons,"production_status":"APPROVED" if not reasons else "DEFERRED"}
    Path("data/research").mkdir(parents=True,exist_ok=True)
    Path("data/research/release_gate.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
