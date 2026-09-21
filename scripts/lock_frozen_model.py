from __future__ import annotations
import json
import os
from datetime import datetime,timezone
from pathlib import Path

def main():
    path=Path("config/frozen_holdout.json")
    metrics=Path("data/research/latest_metrics.json")
    if not path.exists() or not metrics.exists(): raise SystemExit("DEFERRED: cutoff freeze and OOS metrics are required")
    lock=json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status")=="FROZEN":
        print("frozen-model: already locked")
        return
    if lock.get("status")!="CUTOFF_FROZEN_PENDING_MODEL":
        raise SystemExit(f"FAIL: invalid freeze status {lock.get('status')}")
    payload=json.loads(metrics.read_text(encoding="utf-8"))
    selected=payload.get("selected_model")
    if not selected: raise SystemExit("FAIL: OOS did not select a model")
    lock["selected_model"]=selected
    lock["regime_selected_models"]=payload.get("regime_selected_models",{})
    lock["selection_locked_at"]=datetime.now(timezone.utc).isoformat()
    lock["selection_locked_git_sha"]=os.getenv("GITHUB_SHA")
    lock["selection_source"]="chronological OOS using only observations <= cutoff_date"
    lock["status"]="FROZEN"
    path.write_text(json.dumps(lock,indent=2),encoding="utf-8")
    print(json.dumps(lock,indent=2))

if __name__=="__main__":
    main()
