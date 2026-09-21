from pathlib import Path
import re

def main():
    cfg=Path("config/universe.yml").read_text()
    pipe=Path("config/pipeline.yml").read_text()
    checks=[
        ("dynamic_universe","target_count: dynamic" in cfg),
        ("pit_required","require_pit: true" in cfg),
        ("frozen_holdout","frozen_holdout: true" in cfg),
        ("causal_features","causal_only: true" in pipe),
        ("oos_selection","selection_source: chronological_oos_only" in pipe),
        ("price_shards","price_shards: 4" in pipe),
    ]
    bad=[name for name,ok in checks if not ok]
    for name,ok in checks: print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if bad: raise SystemExit(f"FAIL: invariants {bad}")
    print("production-invariants: PASS")

if __name__=="__main__": main()
