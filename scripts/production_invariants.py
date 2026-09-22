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
        ("cross_sectional_robust_zscore","cross_sectional_robust_zscore: true" in pipe and "_cross_sectional_robust_zscore" in Path("src/features/context.py").read_text()),
        ("raw_pit_price_modeling","price_feature_basis: raw_close" in pipe and "target_return_basis: raw_close" in pipe),
        ("quantile_intervals","interval_method: conditional_quantiles_q10_q50_q90" in pipe and "load_production_artifact" in Path("scripts/run_daily_prediction.py").read_text()),
        ("immutable_production_artifact","build_production_artifact.py" in Path(".github/workflows/market-cycle.yml").read_text() and Path("scripts/build_production_artifact.py").exists() and Path("src/prediction/production_artifact.py").exists()),
        ("exact_runtime_restore","install_production_runtime.py" in Path(".github/workflows/us-close-prediction.yml").read_text() and "runtime_dependency_versions" in Path("scripts/build_production_artifact.py").read_text()),
        ("oos_selection","selection_source: chronological_oos_only" in pipe and "rank_ic_tiebreak_tolerance: 0.002" in pipe),
        ("label_aware_purge","purge_sessions: 1" in pipe and "purge=int(model_cfg.get" in Path("scripts/run_daily_research.py").read_text() and "purge" in Path("src/validation/walk_forward.py").read_text()),
        ("rank_ic_oos","cross_sectional_rank_ic" in Path("scripts/run_daily_research.py").read_text() and "rank_ic" in Path("src/research/router.py").read_text()),
        ("oos_optimized_ranking_weight","rank_probability_weight" in Path("scripts/run_daily_research.py").read_text() and "rank_probability_weight" in Path("scripts/lock_frozen_model.py").read_text() and "probability_weight=rank_weight" in Path("scripts/run_daily_prediction.py").read_text()),
        ("oos_optimized_uncertainty_penalty","rank_uncertainty_penalty" in Path("scripts/run_daily_research.py").read_text() and "rank_uncertainty_penalty" in Path("scripts/lock_frozen_model.py").read_text() and "uncertainty_penalty=rank_uncertainty_penalty" in Path("scripts/run_daily_prediction.py").read_text()),
        ("oos_training_window_selection","classifier_training_window_sessions" in Path("scripts/run_daily_research.py").read_text() and "classifier_training_window_sessions" in Path("scripts/lock_frozen_model.py").read_text() and "restrict_to_lookback" in Path("scripts/build_production_artifact.py").read_text() and "restrict_to_lookback" in Path("scripts/evaluate_frozen_holdout.py").read_text()),
        ("ranking_backtest_parity","rank_score" in Path("src/backtest/cross_sectional.py").read_text() and "rank_score" in Path("src/ranking/cross_sectional.py").read_text()),
        ("asset_class_balanced_selection","asset_class_balance_weight: 0.50" in pipe and "rebalance_global_oos_candidates" in Path("scripts/run_daily_research.py").read_text()),
        ("recency_weighted_challenger","recency_weighted_challenger: true" in pipe and "fit_classifier" in Path("scripts/run_daily_research.py").read_text()),
        ("optional_lightgbm_challenger","lightgbm" in pipe and "LGBMClassifier" in Path("src/prediction/model_factories.py").read_text() and "research" in Path("pyproject.toml").read_text()),
        ("hierarchical_routing","asset_class_and_regime" in pipe and "route_plan" in router),
        ("stability_penalty","stability_penalty: 0.25" in pipe and "0.25 * std" in router),
        ("pit_snapshot_gate","available_at <= prediction_time" in prediction),
        ("market_clock_split","groupby(\"asset_class\"" in prediction),
        ("market_context_pipeline","add_market_context" in Path("scripts/run_daily_research.py").read_text() and "update_market_context.py" in Path(".github/workflows/market-cycle.yml").read_text()),
        ("macro_cross_asset_context",all(k in Path("src/features/context.py").read_text() for k in ("us10y_level_lag1","dxy_ret_1d_lag1","gold_ret_1d_lag1","oil_ret_1d_lag1","hyg_ret_1d_lag1")) and all(k in Path("src/data/market_context.py").read_text() for k in ("us10y","dxy","gold","oil","hyg"))),
        ("state_compatibility","production_state_compatibility.py" in Path(".github/workflows/us-close-prediction.yml").read_text() and Path("scripts/production_state_compatibility.py").exists()),
        ("us_artifact_validation","Validate immutable production model artifact" in Path(".github/workflows/us-close-prediction.yml").read_text()),
        ("route_aware_frozen_holdout","frozen_production_routes" in Path("scripts/evaluate_frozen_holdout.py").read_text() and "holdout_not_using_frozen_production_routes" in Path("scripts/release_gate.py").read_text()),
        ("return_estimator_selection","choose_return_estimator" in Path("scripts/run_daily_research.py").read_text() and "return_selected_estimator" in Path("scripts/lock_frozen_model.py").read_text() and "return_selected_estimator" in Path("scripts/build_production_artifact.py").read_text()),
        ("code_fingerprint","fingerprint_sha256" in Path("src/validation/code_fingerprint.py").read_text() and "code_fingerprint_sha256" in Path("scripts/build_manifest.py").read_text()),
        ("paypay_master_catalog",Path("config/paypay_catalog.yml").exists() and Path("scripts/catalog_integrity.py").exists()),
        ("prediction_history_monitoring","restore_prediction_history.py" in Path(".github/workflows/prediction-monitoring.yml").read_text() and Path("scripts/monitor_predictions.py").exists() and 'cron: "27 9 * * 1-5"' in Path(".github/workflows/prediction-monitoring.yml").read_text()),
        ("live_performance_gate","live_performance_gate.py" in Path(".github/workflows/us-close-prediction.yml").read_text() and Path("scripts/live_performance_gate.py").exists()),
        ("regime_thresholds","high_vol_vix: 30.0" in pipe and "event_gap_abs: 0.03" in pipe and "trend_breadth_low: 0.25" in pipe and "regime_vol_threshold" in Path("scripts/lock_frozen_model.py").read_text() and "regime_vol_threshold" in Path("scripts/build_production_artifact.py").read_text()),
        ("price_shards","price_shards: 4" in pipe),
        ("price_retrieval_provenance","retrieved_at" in Path("src/data/yahoo_price.py").read_text() and "available_at_after_retrieved_at" in Path("scripts/data_quality_gate.py").read_text() and "audit_retrieval_provenance" in Path("src/validation/leakage.py").read_text()),
    ]
    bad=[name for name,ok in checks if not ok]
    for name,ok in checks: print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if bad: raise SystemExit(f"FAIL: invariants {bad}")
    print("production-invariants: PASS")


if __name__=="__main__":
    main()
