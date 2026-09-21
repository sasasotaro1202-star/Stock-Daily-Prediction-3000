from pathlib import Path


def main():
    cfg=Path("config/universe.yml").read_text()
    pipe=Path("config/pipeline.yml").read_text()
    router=Path("src/research/router.py").read_text()
    prediction=Path("scripts/run_daily_prediction.py").read_text()
    checks=[
        ("dynamic_universe","target_count: dynamic" in cfg),
        ("unlimited_universe","count_limit: none" in cfg),
        ("all_paypay_scope","scope: all_currently_tradeable_paypay_securities" in cfg),
        ("pit_required","require_pit: true" in cfg),
        ("frozen_holdout","frozen_holdout: true" in cfg),
        ("causal_features","causal_only: true" in pipe),
        ("oos_selection","selection_source: chronological_oos_only" in pipe),
        ("hierarchical_routing","asset_class_and_regime" in pipe and "route_plan" in router),
        ("stability_penalty","stability_penalty: 0.25" in pipe and "0.25 * std" in router),
        ("pit_snapshot_gate","available_at <= prediction_time" in prediction),
        ("market_clock_split","groupby(\"asset_class\"" in prediction),
        ("price_shards","price_shards: 4" in pipe),
    ]
    bad=[name for name,ok in checks if not ok]
    for name,ok in checks: print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if bad: raise SystemExit(f"FAIL: invariants {bad}")
    print("production-invariants: PASS")


if __name__=="__main__":
    main()
