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
        ("raw_pit_price_modeling","price_feature_basis: raw_close" in pipe and "target_return_basis: raw_close" in pipe),
        ("quantile_intervals","interval_method: conditional_quantiles_q10_q50_q90" in pipe and "make_quantile_model" in Path("scripts/run_daily_prediction.py").read_text()),
        ("oos_selection","selection_source: chronological_oos_only" in pipe),
        ("hierarchical_routing","asset_class_and_regime" in pipe and "route_plan" in router),
        ("stability_penalty","stability_penalty: 0.25" in pipe and "0.25 * std" in router),
        ("pit_snapshot_gate","available_at <= prediction_time" in prediction),
        ("market_clock_split","groupby(\"asset_class\"" in prediction),
        ("market_context_pipeline","add_market_context" in Path("scripts/run_daily_research.py").read_text() and "update_market_context.py" in Path(".github/workflows/market-cycle.yml").read_text()),
        ("state_compatibility","production_state_compatibility.py" in Path(".github/workflows/us-close-prediction.yml").read_text() and Path("scripts/production_state_compatibility.py").exists()),
        ("price_shards","price_shards: 4" in pipe),
    ]
    bad=[name for name,ok in checks if not ok]
    for name,ok in checks: print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if bad: raise SystemExit(f"FAIL: invariants {bad}")
    print("production-invariants: PASS")


if __name__=="__main__":
    main()
