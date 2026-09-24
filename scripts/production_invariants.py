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
        ("recent_oos_selection","recent_logloss" in Path("scripts/run_daily_research.py").read_text() and "selection_logloss" in Path("src/research/router.py").read_text()),
        ("label_aware_purge","purge_sessions: 1" in pipe and "purge=int(model_cfg.get" in Path("scripts/run_daily_research.py").read_text() and "purge" in Path("src/validation/walk_forward.py").read_text()),
        ("rank_ic_oos","cross_sectional_rank_ic" in Path("scripts/run_daily_research.py").read_text() and "rank_ic" in Path("src/research/router.py").read_text()),
        ("oos_optimized_ranking_weight","rank_probability_weight" in Path("scripts/run_daily_research.py").read_text() and "rank_probability_weight" in Path("scripts/lock_frozen_model.py").read_text() and "probability_weight=rank_weight" in Path("scripts/run_daily_prediction.py").read_text()),
        ("oos_optimized_uncertainty_penalty","rank_uncertainty_penalty" in Path("scripts/run_daily_research.py").read_text() and "rank_uncertainty_penalty" in Path("scripts/lock_frozen_model.py").read_text() and "uncertainty_penalty=rank_uncertainty_penalty" in Path("scripts/run_daily_prediction.py").read_text()),
        ("oos_training_window_selection","classifier_training_window_sessions" in Path("scripts/run_daily_research.py").read_text() and "classifier_training_window_sessions" in Path("scripts/lock_frozen_model.py").read_text() and "restrict_to_lookback" in Path("scripts/build_production_artifact.py").read_text() and "restrict_to_lookback" in Path("scripts/evaluate_frozen_holdout.py").read_text()),
        ("ranking_backtest_parity","rank_score" in Path("src/backtest/cross_sectional.py").read_text() and "rank_score" in Path("src/ranking/cross_sectional.py").read_text()),
        ("asset_class_balanced_selection","asset_class_balance_weight: 0.50" in pipe and "rebalance_global_oos_candidates" in Path("scripts/run_daily_research.py").read_text()),
        ("recency_weighted_challenger","recency_weighted_challenger: true" in pipe and "fit_classifier" in Path("scripts/run_daily_research.py").read_text()),
        ("optional_lightgbm_challenger","lightgbm" in pipe and "LGBMClassifier" in Path("src/prediction/model_factories.py").read_text() and "lightgbm_regularized_recent" in Path("src/prediction/model_factories.py").read_text() and "research" in Path("pyproject.toml").read_text()),
        ("asymmetric_blend_challengers","blend_hgb_lgbm_regularized_recent_25_75" in Path("src/prediction/model_factories.py").read_text() and "blend_hgb_lgbm_regularized_recent_75_25" in Path("src/research/router.py").read_text() and "blend_hgb_lgbm_regularized_recent_25_75" in pipe),
        ("hgb_conservative_challenger","hgb_conservative_recent" in Path("src/prediction/model_factories.py").read_text() and "hgb_conservative_recent" in router),
        ("hierarchical_routing","asset_class_and_regime" in pipe and "route_plan" in router),
        ("frozen_situation_routing",
 "situation_selected_models" in Path("scripts/run_daily_research.py").read_text()
 and "situation_selected_models" in Path("scripts/lock_frozen_model.py").read_text()
 and 'locked_situation=frozen_routes.get("situation_selected_models")' in prediction
 and "locked_situation" in router),
        ("frozen_asset_situation_routing",
 "asset_situation_selected_models" in Path("scripts/run_daily_research.py").read_text()
 and "asset_situation_selected_models" in Path("scripts/lock_frozen_model.py").read_text()
 and 'locked_asset_situation=frozen_routes.get("asset_situation_selected_models")' in prediction
 and "locked_asset_situation" in router),
        ("scoped_route_parent_edge","minimum_scoped_oos_improvement_logloss: 0.002" in pipe and "materially_better_than_parent" in router and "minimum_scoped_oos_improvement_logloss" in Path("scripts/run_daily_research.py").read_text()),
        ("stability_penalty","stability_penalty: 0.25" in pipe and "0.25 * std" in router),
        ("pit_snapshot_gate","available_at <= prediction_time" in prediction),
        ("market_clock_split","groupby(\"asset_class\"" in prediction),
        ("market_context_pipeline","add_market_context" in Path("scripts/run_daily_research.py").read_text() and "update_market_context.py" in Path(".github/workflows/market-cycle.yml").read_text()),        ("market_context_extended_quality","us10y" in Path("scripts/update_market_context.py").read_text() and "MAX_STALENESS_DAYS=10" in Path("scripts/update_market_context.py").read_text() and '"stale_families"' in Path("scripts/update_market_context.py").read_text()),
        ("macro_cross_asset_context",all(k in Path("src/features/context.py").read_text() for k in ("us10y_level_lag1","dxy_ret_1d_lag1","gold_ret_1d_lag1","oil_ret_1d_lag1","hyg_ret_1d_lag1")) and all(k in Path("src/data/market_context.py").read_text() for k in ("us10y","dxy","gold","oil","hyg"))),
        ("state_compatibility","production_state_compatibility.py" in Path(".github/workflows/us-close-prediction.yml").read_text() and Path("scripts/production_state_compatibility.py").exists()),
        ("all_security_prediction",'groupby(["asset_class", "symbol"]' in prediction),
        ("on_demand_production_prediction",Path(".github/workflows/on-demand-production-prediction.yml").exists() and "workflow_dispatch:" in Path(".github/workflows/on-demand-production-prediction.yml").read_text() and "run_now_prediction.py" in Path(".github/workflows/on-demand-production-prediction.yml").read_text() and "load_production_artifact" in Path("scripts/run_now_prediction.py").read_text() and "Production-or-near-production prediction" in Path(".github/workflows/on-demand-production-prediction.yml").read_text() and "Validate production prediction output" in Path(".github/workflows/on-demand-production-prediction.yml").read_text()),
        ("us_artifact_validation","Validate immutable production model artifact" in Path(".github/workflows/us-close-prediction.yml").read_text()),
        ("route_aware_frozen_holdout","frozen_production_routes" in Path("scripts/evaluate_frozen_holdout.py").read_text() and "holdout_not_using_frozen_production_routes" in Path("scripts/release_gate.py").read_text()),
        ("return_estimator_selection","choose_return_estimator" in Path("scripts/run_daily_research.py").read_text() and "return_selected_estimator" in Path("scripts/lock_frozen_model.py").read_text() and "return_selected_estimator" in Path("scripts/build_production_artifact.py").read_text()),
        ("code_fingerprint","fingerprint_sha256" in Path("src/validation/code_fingerprint.py").read_text() and "code_fingerprint_sha256" in Path("scripts/build_manifest.py").read_text() and "research_code_fingerprint_sha256" in Path("scripts/build_production_artifact.py").read_text()),
        ("paypay_master_catalog",Path("config/paypay_catalog.yml").exists() and Path("scripts/catalog_integrity.py").exists()),
        ("prediction_history_monitoring","restore_prediction_history.py" in Path(".github/workflows/prediction-monitoring.yml").read_text() and Path("scripts/monitor_predictions.py").exists() and 'cron: "27 9 * * 1-5"' in Path(".github/workflows/prediction-monitoring.yml").read_text()),
        ("bounded_universe_restore_for_us_paths","restore_latest_universe_state.py" in Path(".github/workflows/us-close-prediction.yml").read_text() and "restore_latest_universe_state.py" in Path(".github/workflows/prediction-monitoring.yml").read_text() and "refresh_universe.py" not in Path(".github/workflows/us-close-prediction.yml").read_text() and "refresh_universe.py" not in Path(".github/workflows/prediction-monitoring.yml").read_text()),
        ("us_asset_scope_minimizes_free_actions","PRICE_ASSET_CLASSES: us_stock,us_etf" in Path(".github/workflows/us-close-prediction.yml").read_text() and "QUALITY_ASSET_CLASSES: us_stock,us_etf" in Path(".github/workflows/us-close-prediction.yml").read_text() and "PRICE_ASSET_CLASSES" in Path("scripts/update_prices.py").read_text() and "QUALITY_ASSET_CLASSES" in Path("scripts/data_quality_gate.py").read_text()),
        ("immutable_holdout_generation","research_fingerprint_sha256" in Path("src/validation/code_fingerprint.py").read_text() and "research_code_fingerprint_sha256" in Path("scripts/lock_frozen_model.py").read_text() and "frozen_holdout_history" in Path("scripts/bootstrap_frozen_holdout.py").read_text() and "holdout_generation" in Path("scripts/evaluate_frozen_holdout.py").read_text()),

        ("actions_reliability_watchdog",Path(".github/workflows/actions-reliability-watchdog.yml").exists() and 'cron: "7,22,37,52 * * * *"' in Path(".github/workflows/actions-reliability-watchdog.yml").read_text() and "bounded recovery" in Path(".github/workflows/actions-reliability-watchdog.yml").read_text()),        ("failure_only_bounded_recovery","conclusion == 'failure'" in Path(".github/workflows/bounded-production-recovery.yml").read_text() and "conclusion == 'cancelled'" not in Path(".github/workflows/bounded-production-recovery.yml").read_text() and "run_attempt == 1" in Path(".github/workflows/bounded-production-recovery.yml").read_text()),        ("automation_heartbeat",Path(".github/workflows/heartbeat.yml").exists() and 'cron: "17 */6 * * *"' in Path(".github/workflows/heartbeat.yml").read_text()),        ("live_performance_gate","live_performance_gate.py" in Path(".github/workflows/us-close-prediction.yml").read_text() and Path("scripts/live_performance_gate.py").exists()),
        ("prediction_output_integrity","validate_prediction_output.py" in Path(".github/workflows/market-cycle.yml").read_text() and "validate_prediction_output.py" in Path(".github/workflows/us-close-prediction.yml").read_text() and Path("scripts/validate_prediction_output.py").exists()),
        ("sec_research_isolation",
         Path("scripts/sec_filings_research.py").exists()
         and Path("scripts/run_sec_filing_ablation.py").exists()
         and Path("src/research/sec_features.py").exists()
         and "research_only" in Path("scripts/sec_filings_research.py").read_text()
         and "production_changed" in Path("scripts/sec_filings_research.py").read_text()
         and "research_only" in Path("scripts/run_sec_filing_ablation.py").read_text()
         and "production_changed" in Path("scripts/run_sec_filing_ablation.py").read_text()
         and "run_sec_filing_ablation.py" in Path(".github/workflows/market-cycle.yml").read_text()
         and "src.research.sec_features" not in Path("scripts/run_daily_prediction.py").read_text()
         and "SEC_FEATURE_COLUMNS" not in Path("scripts/run_daily_prediction.py").read_text()),
        ("selective_probability_research_isolation",
         Path("src/research/selective.py").exists()
         and "selective_probability_research" in Path("scripts/run_daily_research.py").read_text()
         and "production_changed" in Path("scripts/run_daily_research.py").read_text()
         and "apply_confidence_shrinkage" not in prediction),
        ("online_expert_research_isolation",
         "online_expert_average" in Path("scripts/run_daily_research.py").read_text()
         and "fixed_learning_rate_chronological_online_update_after_each_session" in Path("scripts/run_daily_research.py").read_text()
         and Path("src/research/online_ensemble.py").exists()
         and "online_expert_average" not in prediction),
        ("selective_oos_test_not_tuned",
         "select_confidence_shrinkage_parameters" in Path("scripts/run_daily_research.py").read_text()
         and "calibration_slice_selection_then_untouched_chronological_oos_test" in Path("scripts/run_daily_research.py").read_text()
         and "for threshold in (0.03, 0.05, 0.075, 0.10, 0.15)" not in Path("scripts/run_daily_research.py").read_text()),
        ("cpcv_research_audit",
         Path("scripts/run_cpcv_validation_audit.py").exists()
         and "CPCV leakage-boundary research audit" in Path(".github/workflows/market-cycle.yml").read_text()
         and "production_changed" in Path("src/research/cpcv_validation_audit.py").read_text()
         and "validate_split" in Path("src/research/cpcv_validation_audit.py").read_text()),
        ("monitoring_bounded_price_recovery",
         "Bounded price-state recovery after deferred quality" in Path(".github/workflows/prediction-monitoring.yml").read_text()
         and "Re-check data quality after bounded recovery" in Path(".github/workflows/prediction-monitoring.yml").read_text()
         and "PRICE_SHARD_COUNT: 4" in Path(".github/workflows/prediction-monitoring.yml").read_text()
         and "data_quality_deferred_after_bounded_recovery" in Path(".github/workflows/prediction-monitoring.yml").read_text()
         and "timeout-minutes: 60" in Path(".github/workflows/prediction-monitoring.yml").read_text()),

        ("regime_thresholds","high_vol_vix: 30.0" in pipe and "event_gap_abs: 0.03" in pipe and "trend_breadth_low: 0.25" in pipe and "regime_vol_threshold" in Path("scripts/lock_frozen_model.py").read_text() and "regime_vol_threshold" in Path("scripts/build_production_artifact.py").read_text()),
        ("oos_calibration_selection","method: oos_selected" in pipe and "CALIBRATION_METHODS" in Path("src/validation/calibration.py").read_text() and "calibration_method" in Path("scripts/lock_frozen_model.py").read_text() and "calibration_method" in Path("scripts/build_production_artifact.py").read_text()),
        ("asset_balanced_calibration","asset_calibration_rows" in Path("scripts/run_daily_research.py").read_text() and "asset_macro_logloss" in Path("scripts/run_daily_research.py").read_text()),
        ("price_shards","price_shards: 4" in pipe),
        ("price_retrieval_provenance","retrieved_at" in Path("src/data/yahoo_price.py").read_text() and "available_at_after_retrieved_at" in Path("scripts/data_quality_gate.py").read_text() and "audit_retrieval_provenance" in Path("src/validation/leakage.py").read_text()),
        ("price_store_canonicalization",Path("scripts/normalize_price_store.py").exists() and "conflicting_duplicate_groups" in Path("scripts/normalize_price_store.py").read_text()),
        ("market_context_retrieval_provenance","retrieved_at" in Path("src/data/market_context.py").read_text() and "retrieval_run_id" in Path("src/data/market_context.py").read_text() and "\"source\"" in Path("src/data/market_context.py").read_text()),
    ]
    bad=[name for name,ok in checks if not ok]
    for name,ok in checks: print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if bad: raise SystemExit(f"FAIL: invariants {bad}")
    print("production-invariants: PASS")


if __name__=="__main__":
    main()
