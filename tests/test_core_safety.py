import pandas as pd
from src.features.technical import add_technical_features,FEATURE_COLUMNS
from src.prediction.targets import add_targets
from src.validation.leakage import audit_feature_columns,audit_target_separation
from src.research.metrics import expected_calibration_error

def sample():
    rows=[]
    for s in ("AAA","BBB"):
        for i in range(80):
            p=100+i+(0 if s=="AAA" else i*0.2)
            rows.append({"symbol":s,"session_date":pd.Timestamp("2020-01-01")+pd.Timedelta(days=i),
                         "open":p,"high":p+1,"low":p-1,"close":p+0.2,"adj_close":p+0.2,"volume":1000+i})
    return pd.DataFrame(rows)

def test_features_are_causal_and_targets_separate():
    x=add_targets(add_technical_features(sample()))
    assert set(FEATURE_COLUMNS).isdisjoint({c for c in x if c.startswith("target_")})
    assert audit_feature_columns(FEATURE_COLUMNS).ok
    assert audit_target_separation(FEATURE_COLUMNS,[c for c in x if c.startswith("target_")]).ok
    assert x.loc[x.groupby("symbol").tail(1).index,"target_up_1d"].isna().all()

def test_calibration_selection_is_asset_balanced():
    from pathlib import Path

    source = Path("scripts/run_daily_research.py").read_text(encoding="utf-8")
    assert "asset_calibration_rows" in source
    assert "asset_macro_logloss" in source
    assert "asset_balance_weight" in source
    assert "selection_mean + 0.25 * selection_std" in source


def test_oos_calibration_methods_fit_and_bound_outputs():
    import numpy as np
    from src.validation.calibration import CALIBRATION_METHODS, make_calibrator

    p = np.array([0.05, 0.10, 0.25, 0.35, 0.60, 0.75, 0.90, 0.97] * 8)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1] * 8)
    assert set(CALIBRATION_METHODS) == {"platt", "beta", "isotonic"}
    for method in CALIBRATION_METHODS:
        calibrator = make_calibrator(method).fit(p, y)
        out = calibrator.predict(p)
        assert np.isfinite(out).all()
        assert np.all((out >= 0.0) & (out <= 1.0))


def test_ece_bounds():
    v=expected_calibration_error([0,1,0,1],[0.1,0.9,0.2,0.8])
    assert 0<=v<=1


def test_model_factory_matches_production_hgb_contract():
    from src.prediction.model_factories import models

    m = models()["hgb"]()
    params = m.get_params()
    assert params["histgradientboostingclassifier__max_iter"] == 300
    assert params["histgradientboostingclassifier__learning_rate"] == 0.04


def test_conservative_hgb_recent_is_available_in_research_ci():
    from src.prediction.model_factories import models

    available = models()
    assert "hgb_conservative_recent" in available
    params = available["hgb_conservative_recent"]().get_params()
    assert params["histgradientboostingclassifier__max_iter"] == 500
    assert params["histgradientboostingclassifier__learning_rate"] == 0.02
    assert params["histgradientboostingclassifier__max_leaf_nodes"] == 15
    assert params["histgradientboostingclassifier__min_samples_leaf"] == 40
    assert params["histgradientboostingclassifier__l2_regularization"] == 3.0


def test_lightgbm_challenger_is_available_in_research_ci():
    from src.prediction.model_factories import models
    import lightgbm

    available = models()
    assert "lightgbm" in available
    assert "lightgbm_regularized" in available
    assert "lightgbm_regularized_recent" in available
    assert "lightgbm_conservative" in available
    assert "lightgbm_conservative_recent" in available
    assert lightgbm.__version__


def test_current_main_has_shared_model_and_route_holdout_contract():
    from pathlib import Path

    factory = Path("src/prediction/model_factories.py").read_text(encoding="utf-8")
    research = Path("scripts/run_daily_research.py").read_text(encoding="utf-8")
    holdout = Path("scripts/evaluate_frozen_holdout.py").read_text(encoding="utf-8")
    artifact = Path("scripts/build_production_artifact.py").read_text(encoding="utf-8")

    assert "def models()" in factory
    assert "from src.prediction.model_factories import models" in research
    assert "frozen_production_routes" in holdout
    assert "required_classifiers" in artifact


def test_price_update_has_bounded_retry():
    from pathlib import Path

    source = Path("scripts/update_prices.py").read_text(encoding="utf-8")
    assert "for attempt in range(3)" in source
    assert "time.sleep(2 ** attempt)" in source


def test_paypay_visible_text_parser_handles_non_table_layout():
    from src.data.paypay_collector import parse_visible_text

    raw = """<section><h3>日本株 個別銘柄（50音順）</h3>
    <div>7013</div><div>IHI</div><div>成長投資</div>
    <div>trade_on,mini_on,cfd_</div>
    <h3>国内ETF（上場投資信託）（50音順）</h3>
    <div>1306</div><div>TOPIX連動型ETF</div><div>成長投資</div>
    <div>trade_on,mini_on,cfd_</div>
    <h3>REIT（不動産投資信託）（50音順）</h3>
    <div>8951</div><div>日本ビルファンド投資法人</div><div>成長投資</div>
    <div>trade_on,mini_on,cfd_</div></section>""".encode()

    rows = parse_visible_text(
        raw,
        "japan",
        "https://www.paypay-sec.co.jp/stock/list/",
    )
    keys = {(row["asset_class"], row["symbol"]) for row in rows}
    assert ("jp_stock", "7013") in keys
    assert ("jp_etf", "1306") in keys
    assert ("jp_reit", "8951") in keys
    assert all(row["tradeable"] is True for row in rows)


def test_paypay_runtime_dependency_is_declared():
    from pathlib import Path

    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "curl-cffi>=0.16.0" in text


def test_return_estimator_selector_guards_mae_and_uses_rank_ic():
    from src.research.return_selection import choose_return_estimator

    candidates = {
        "mean": {"mae": 0.010, "rank_ic": 0.040, "rank_ic_std": 0.010, "folds": 4},
        "q50": {"mae": 0.011, "rank_ic": 0.070, "rank_ic_std": 0.020, "folds": 4},
        "blend_mean_q50": {"mae": 0.0105, "rank_ic": 0.090, "rank_ic_std": 0.015, "folds": 4},
    }
    assert (
        choose_return_estimator(
            candidates,
            min_folds=3,
            mae_guard=1.10,
            stability_penalty=0.25,
            rank_ic_tolerance=0.005,
        )
        == "blend_mean_q50"
    )

    bad_mae = dict(candidates)
    bad_mae["blend_mean_q50"] = {
        "mae": 0.020,
        "rank_ic": 0.50,
        "rank_ic_std": 0.01,
        "folds": 4,
    }
    assert choose_return_estimator(bad_mae) == "q50"


def test_distribution_and_drawdown_factors_are_causal():
    x = sample()
    out = add_technical_features(x)
    cols = [
        "return_skew_20",
        "return_kurtosis_20",
        "positive_return_fraction_20",
        "downside_volatility_20",
        "upside_volatility_20",
        "drawdown_from_high_20",
        "distance_from_low_20",
        "close_location_mean_20",
    ]
    assert set(cols).issubset(out.columns)
    assert out.loc[out.groupby("symbol").tail(1).index, cols].notna().all().all()




def test_ranking_penalizes_high_uncertainty():
    from src.ranking.cross_sectional import cross_sectional_rank

    df = pd.DataFrame({
        "prediction_date": ["2026-09-22"] * 4,
        "asset_class": ["jp_stock"] * 4,
        "symbol": ["A", "B", "C", "D"],
        "p_up_1d": [0.9, 0.8, 0.7, 0.6],
        "expected_return_1d": [0.04, 0.03, 0.02, 0.01],
        "ranking_uncertainty": [0.01, 0.02, 0.03, 0.40],
    })
    without_penalty = cross_sectional_rank(
        df,
        probability_weight=0.5,
        uncertainty_col="ranking_uncertainty",
        uncertainty_penalty=0.0,
    )
    with_penalty = cross_sectional_rank(
        df,
        probability_weight=0.5,
        uncertainty_col="ranking_uncertainty",
        uncertainty_penalty=0.2,
    )
    d0 = without_penalty.loc[
        without_penalty["symbol"].eq("D"), "rank_score"
    ].iloc[0]
    d1 = with_penalty.loc[
        with_penalty["symbol"].eq("D"), "rank_score"
    ].iloc[0]
    assert d1 < d0
    assert d1 == d0 - 0.2
    assert (
        with_penalty.set_index("symbol").loc["D", "rank_uncertainty"]
        > with_penalty.set_index("symbol").loc["A", "rank_uncertainty"]
    )
    assert with_penalty["rank_uncertainty_penalty"].eq(0.2).all()


def test_training_window_keeps_only_latest_sessions():
    from src.validation.training_window import restrict_to_lookback

    df = pd.DataFrame({
        "session_date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "value": [1, 2, 3, 4, 5],
    })
    out = restrict_to_lookback(df, 2)
    assert out["session_date"].dt.date.tolist() == [
        pd.Timestamp("2026-01-04").date(),
        pd.Timestamp("2026-01-05").date(),
    ]
    assert len(restrict_to_lookback(df, 0)) == 5


def test_production_artifact_uses_training_window_before_calibration():
    from pathlib import Path

    source = Path("scripts/build_production_artifact.py").read_text(encoding="utf-8")
    assert "training_labeled = restrict_to_lookback" in source
    assert "core, cal = split_train_cal(training_labeled)" in source
    assert "recent_sessions=min(252, training_window or 252)" in source


def test_causal_correlation_and_trend_factors():
    x = sample()
    out = add_technical_features(x)
    cols = [
        "return_autocorr_20",
        "return_volume_corr_20",
        "trend_slope_20",
        "trend_r2_20",
        "up_down_imbalance_20",
    ]
    assert set(cols).issubset(out.columns)
    group_keys=["asset_class", "symbol"] if "asset_class" in out.columns else ["symbol"]
    last = out.groupby(group_keys).tail(1)
    assert last[cols].notna().all().all()
    assert ((last["trend_r2_20"] >= 0) & (last["trend_r2_20"] <= 1)).all()


def test_retrieval_provenance_audit_rejects_inconsistent_times():
    import pandas as pd
    from src.validation.leakage import audit_retrieval_provenance

    frame = pd.DataFrame({
        "available_at": [pd.Timestamp("2026-01-02T00:00:00Z")],
        "retrieved_at": [pd.Timestamp("2026-01-02T00:00:00Z")],
    })
    assert audit_retrieval_provenance(frame)["ok"] is True
    frame.loc[0, "retrieved_at"] = pd.Timestamp("2026-01-03T00:00:00Z")
    assert audit_retrieval_provenance(frame)["ok"] is True
    frame.loc[0, "retrieved_at"] = pd.Timestamp("2026-01-01T00:00:00Z")
    assert audit_retrieval_provenance(frame)["ok"] is False


def test_recency_weights_decay_and_normalize():
    import numpy as np
    import pandas as pd
    from src.prediction.fit import training_recency_weights

    dates = pd.Series(pd.to_datetime([
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
    ]))
    weights = training_recency_weights(dates, half_life_sessions=2)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[0] < weights[-1]
    assert np.isclose(weights[1] / weights[3], 0.5, atol=1e-6)


def test_data_quality_gate_handles_retrieved_at_rows_without_unbound_session_dates(
    tmp_path, monkeypatch
):
    import json
    import pandas as pd
    import scripts.data_quality_gate as gate

    root = tmp_path / "data" / "prices"
    root.mkdir(parents=True)
    universe = tmp_path / "data" / "universe" / "latest.json"
    universe.parent.mkdir(parents=True)
    universe.write_text(
        json.dumps(
            {
                "records": [
                    {"symbol": "7203", "asset_class": "jp_stock", "tradeable": True}
                ]
            }
        )
    )
    bars = pd.DataFrame(
        [
            {
                "symbol": "7203",
                "asset_class": "jp_stock",
                "session_date": pd.Timestamp("2026-09-22").date(),
                "available_at": pd.Timestamp("2026-09-22T07:00:00Z"),
                "retrieved_at": pd.Timestamp("2026-09-22T07:30:00Z"),
                "source": "yfinance",
                "provider_symbol": "7203.T",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            }
        ]
    )
    bars.to_parquet(root / "shard.parquet", index=False)
    monkeypatch.chdir(tmp_path)
    gate.UNIVERSE = universe

    gate.main()
    result = json.loads((tmp_path / "data" / "research" / "data_quality.json").read_text())
    assert result["status"] == "PASS"
    assert result["reasons"] == []


def test_production_ranking_uncertainty_uses_asset_quantiles_when_available():
    from pathlib import Path

    source = Path("scripts/run_daily_prediction.py").read_text(encoding="utf-8")
    assert 'asset_qmodels.get(str(asset), global_qmodels)' in source
    assert 'latest["ranking_uncertainty"] = np.nan' in source


def test_future_value_poisoning_does_not_change_prior_features():
    base = sample()
    base["asset_class"] = "jp_stock"
    clean = add_technical_features(base)
    poisoned = base.copy()
    poisoned["volume"] = poisoned["volume"].astype(float)
    cutoff = 100
    poisoned.loc[poisoned.index >= cutoff, "close"] *= 100.0
    poisoned.loc[poisoned.index >= cutoff, "volume"] *= 0.01
    changed = add_technical_features(poisoned)

    # Technical features are intentionally tested in isolation here. Context
    # features depend on separate PIT datasets and are covered by the next test.
    technical_cols = [
        col for col in FEATURE_COLUMNS
        if col in clean.columns and col in changed.columns
    ]
    assert technical_cols
    pd.testing.assert_frame_equal(
        clean.loc[clean.index < cutoff, technical_cols].reset_index(drop=True),
        changed.loc[changed.index < cutoff, technical_cols].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

def test_future_market_context_poisoning_does_not_change_prior_features():
    from src.features.context import add_market_context

    prices = pd.DataFrame({
        "symbol": ["AAA", "AAA", "AAA"],
        "asset_class": ["jp_stock"] * 3,
        "session_date": pd.to_datetime(
            ["2026-09-22", "2026-09-23", "2026-09-24"]
        ).date,
        "available_at": pd.to_datetime([
            "2026-09-22T07:00:00Z",
            "2026-09-23T07:00:00Z",
            "2026-09-24T07:00:00Z",
        ]),
    })
    context = pd.DataFrame({
        "session_date": pd.to_datetime([
            "2026-09-21", "2026-09-22", "2026-09-23"
        ]).date,
        "family": ["sp500", "sp500", "sp500"],
        "available_at": pd.to_datetime([
            "2026-09-22T05:30:00Z",
            "2026-09-23T05:30:00Z",
            "2026-09-24T05:30:00Z",
        ]),
        "ret_1d": [0.01, 0.02, 0.03],
        "volatility_20": [0.10, 0.20, 0.30],
        "close": [100.0, 110.0, 120.0],
    })
    poisoned = context.copy()
    poisoned.loc[poisoned["session_date"].eq(pd.Timestamp("2026-09-23").date()), "ret_1d"] = 9.99
    clean = add_market_context(prices, context)
    changed = add_market_context(prices, poisoned)

    pd.testing.assert_series_equal(
        clean.loc[:1, "sp500_ret_1d_lag1"].reset_index(drop=True),
        changed.loc[:1, "sp500_ret_1d_lag1"].reset_index(drop=True),
        check_dtype=False,
    )


def test_frozen_holdout_uses_shared_production_ranking_and_asset_quantiles():
    from pathlib import Path

    source = Path("scripts/evaluate_frozen_holdout.py").read_text(encoding="utf-8")
    assert "from src.ranking.cross_sectional import cross_sectional_rank" in source
    assert "q_assets.get(str(asset), q_global)" in source
    assert "cross_sectional_rank(" in source



def test_recency_challenger_fit_parity_across_research_holdout_artifact():
    from pathlib import Path

    research = Path("scripts/run_daily_research.py").read_text(encoding="utf-8")
    holdout = Path("scripts/evaluate_frozen_holdout.py").read_text(encoding="utf-8")
    artifact = Path("scripts/build_production_artifact.py").read_text(encoding="utf-8")

    assert "fit_classifier(" in research
    assert "fit_classifier(" in holdout
    assert "fit_classifier(" in artifact
    assert "recency_weight_half_life_sessions" in research


def test_monitoring_workflow_does_not_silently_ignore_price_restore_failure():
    from pathlib import Path

    source = Path(".github/workflows/prediction-monitoring.yml").read_text(encoding="utf-8")
    assert "|| true" not in source
    assert "one or more price-state shards could not be restored" in source



def test_live_performance_gate_fails_closed_when_monitoring_state_is_missing(tmp_path, monkeypatch):
    import pytest
    import scripts.live_performance_gate as gate

    monkeypatch.chdir(tmp_path)
    gate.MONITOR = tmp_path / "data" / "research" / "monitor_latest.json"
    gate.OUT = tmp_path / "data" / "research" / "live_performance_gate.json"

    with pytest.raises(SystemExit, match="monitoring state is missing"):
        gate.main()

    result = __import__("json").loads(gate.OUT.read_text(encoding="utf-8"))
    assert result["status"] == "DEFERRED"
    assert result["production_action"] == "DEFERRED"


def test_independent_raw_input_audit_is_temporal_and_fail_closed():
    from src.validation.independent_audit import audit_raw_inputs

    now = pd.Timestamp("2026-09-22T10:00:00Z")
    prices = pd.DataFrame([{
        "symbol": "AAA",
        "asset_class": "jp_stock",
        "session_date": pd.Timestamp("2026-09-22").date(),
        "available_at": pd.Timestamp("2026-09-22T08:00:00Z"),
        "retrieved_at": pd.Timestamp("2026-09-22T09:00:00Z"),
        "source": "test",
        "provider_symbol": "AAA.T",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000.0,
    }])
    context = pd.DataFrame([{
        "family": "sp500",
        "session_date": pd.Timestamp("2026-09-22").date(),
        "available_at": pd.Timestamp("2026-09-22T07:00:00Z"),
    }])
    good = audit_raw_inputs(prices, context, now=now)
    assert good.ok is True

    poisoned = prices.copy()
    poisoned.loc[0, "available_at"] = pd.Timestamp("2026-09-22T10:01:00Z")
    bad = audit_raw_inputs(poisoned, context, now=now)
    assert bad.ok is False
    assert any("available_at_after_audit_time" in x for x in bad.violations)

    duplicated = pd.concat([prices, prices], ignore_index=True)
    bad_dup = audit_raw_inputs(duplicated, context, now=now)
    assert bad_dup.ok is False
    assert any("duplicates" in x for x in bad_dup.violations)

    target_leak = prices.assign(target_ret_1d=[0.1])
    bad_target = audit_raw_inputs(target_leak, context, now=now)
    assert bad_target.ok is False
    assert any("raw_target_column_present" in x for x in bad_target.violations)


def test_soft_blend_challenger_is_cloneable_and_picklable():
    import pickle
    from src.prediction.model_factories import models

    factories = models()
    name = "blend_hgb_lgbm_regularized_recent"
    assert name in factories

    estimator = factories[name]()
    assert estimator.left_weight == 0.5
    assert estimator.right_weight == 0.5
    pickle.loads(pickle.dumps(estimator))


def test_soft_blend_challenger_is_registered_and_lightgbm_artifact_checked():
    from pathlib import Path

    pipeline = Path("config/pipeline.yml").read_text(encoding="utf-8")
    artifact = Path("src/prediction/production_artifact.py").read_text(encoding="utf-8")

    assert "blend_hgb_lgbm_regularized_recent" in pipeline
    assert 'if any("lightgbm" in str(name).lower() for name in classifiers):' in artifact


def test_conservative_lightgbm_reaches_oos_selection_candidates():
    from src.research.router import CANDIDATES, ASSET_CANDIDATES

    name = "lightgbm_conservative_recent"
    for regime in ("normal", "high_vol", "trend", "event"):
        assert name in CANDIDATES[regime]
    for asset_class in ("jp_stock", "jp_etf", "jp_reit", "us_stock", "us_etf"):
        assert name in ASSET_CANDIDATES[asset_class]


def test_soft_blend_challenger_reaches_oos_selection_candidates():
    from src.research.router import CANDIDATES

    name = "blend_hgb_lgbm_regularized_recent"
    for regime in ("normal", "high_vol", "trend", "event"):
        assert name in CANDIDATES[regime]


def test_prediction_history_restore_fails_closed_on_partial_restore():
    from pathlib import Path

    script = Path("scripts/restore_prediction_history.py").read_text(
        encoding="utf-8"
    )
    assert "errors=[]" in script
    assert "errors.append({" in script
    assert (
        "DEFERRED: one or more prediction-history artifacts could not be restored"
        in script
    )


def test_regime_volatility_threshold_is_oos_train_derived():
    import numpy as np
    from src.research.regime_threshold import (
        aggregate_oos_training_thresholds,
        volatility_threshold_from_training,
    )

    train_a = volatility_threshold_from_training(
        np.array([0.01, 0.02, 0.03, 0.04])
    )
    train_b = volatility_threshold_from_training(
        np.array([0.02, 0.04, 0.06, 0.08])
    )
    assert np.isclose(train_a, 0.0325)
    assert np.isclose(
        aggregate_oos_training_thresholds([train_a, train_b]),
        0.04875,
    )

    import pathlib
    research = pathlib.Path("scripts/run_daily_research.py").read_text(encoding="utf-8")
    lock = pathlib.Path("scripts/lock_frozen_model.py").read_text(encoding="utf-8")
    gate = pathlib.Path("scripts/release_gate.py").read_text(encoding="utf-8")
    assert "oos_fold_train_median" in research
    assert "oos_fold_train_median" in lock
    assert "oos_fold_train_median" in gate

def test_market_cycle_guard_fails_closed_on_actions_api_errors():
    from pathlib import Path

    source = Path(".github/workflows/market-cycle.yml").read_text(encoding="utf-8")
    assert "actions_api_unavailable_defer" in source
    assert "actions_api_empty_response_defer" in source
    assert "Actions API unavailable; deferring cycle fail-closed" in source
    assert "Actions API returned empty response; deferring cycle fail-closed" in source
    assert "2>/dev/null || true" not in source


def test_research_state_restore_prefers_approved_production_artifacts():
    from pathlib import Path

    source = Path("scripts/restore_latest_research_state.py").read_text(encoding="utf-8")
    assert "_has_approved_production_state" in source
    assert 'payload.get("approved") is True' in source
    assert 'production_model_artifact.pkl' in source
    assert 'production_model_artifact.meta.json' in source
    assert "skipped_unapproved" in source


def test_restore_research_state_approval_predicate_is_fail_closed(tmp_path):
    import json
    from scripts.restore_latest_research_state import _has_approved_production_state

    root = tmp_path / "extract"
    research = root / "research"
    research.mkdir(parents=True)
    (research / "production_model_artifact.pkl").write_bytes(b"artifact")
    (research / "production_model_artifact.meta.json").write_text("{}", encoding="utf-8")

    (research / "release_gate.json").write_text(
        json.dumps({"approved": False}),
        encoding="utf-8",
    )
    assert _has_approved_production_state(research) is False

    (research / "release_gate.json").write_text(
        json.dumps({"approved": True}),
        encoding="utf-8",
    )
    assert _has_approved_production_state(research) is True

def test_actions_watchdog_is_hourly_and_fail_visible():
    from pathlib import Path

    source = Path(".github/workflows/actions-reliability-watchdog.yml").read_text(
        encoding="utf-8"
    )
    assert 'cron: "7 * * * *"' in source
    assert "actions: write" in source
    assert "96 hours ago" in source
    assert "bounded recovery" in source
    assert "Actions reliability watchdog: PASS" in source
    assert 'conclusion" = "failure"' in source
    assert "exit 1" in source
    assert "|| true" not in source

def test_production_invariants_include_actions_watchdog():
    from pathlib import Path

    source = Path("scripts/production_invariants.py").read_text(encoding="utf-8")
    assert "actions_reliability_watchdog" in source
    assert "actions-reliability-watchdog.yml" in source

def test_actions_watchdog_detects_stale_and_recovery_failures():
    from pathlib import Path

    source = Path(".github/workflows/actions-reliability-watchdog.yml").read_text(
        encoding="utf-8"
    )
    assert "120 minutes ago" in source
    assert 'updated_at // empty' in source
    assert "stale run" in source
    assert 'bounded-production-recovery.yml' in source
    assert "Bounded production recovery" in source


def test_capafy_inspired_anti_overfit_battery_is_deterministic():
    import numpy as np
    from src.validation.anti_overfit import (
        anti_overfit_battery,
        multiple_trial_sharpe_adjustment,
    )

    dates = pd.date_range("2022-01-03", periods=800, freq="D")
    rng = np.random.default_rng(7)
    returns = 0.0008 + rng.normal(0.0, 0.01, len(dates))

    first = anti_overfit_battery(
        dates,
        returns,
        trials=100,
        permutations=250,
        seed=42,
    )
    second = anti_overfit_battery(
        dates,
        returns,
        trials=100,
        permutations=250,
        seed=42,
    )
    assert first == second

def test_data_quality_provider_deferred_symbols_are_bounded(tmp_path, monkeypatch):
    import json
    import scripts.data_quality_gate as gate

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data"
    price_root = root / "prices"
    universe = root / "universe" / "latest.json"
    price_root.mkdir(parents=True)
    universe.parent.mkdir(parents=True)

    rows = []
    records = []
    for i in range(100):
        symbol = f"{i:04d}"
        records.append({
            "symbol": symbol,
            "asset_class": "jp_stock",
            "tradeable": True,
        })
        rows.append({
            "symbol": symbol,
            "asset_class": "jp_stock",
            "session_date": pd.Timestamp("2026-09-23").date(),
            "available_at": pd.Timestamp("2026-09-23T07:00:00Z"),
            "retrieved_at": pd.Timestamp("2026-09-23T07:30:00Z"),
            "source": "yfinance",
            "provider_symbol": f"{symbol}.T",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        })
    universe.write_text(json.dumps({"records": records}), encoding="utf-8")
    pd.DataFrame(rows[:99]).to_parquet(
        price_root / "canonical.parquet",
        index=False,
    )
    (price_root / "price_deferred_shard_0.json").write_text(
        json.dumps({
            "status": "DEFERRED",
            "retrieval_run_id": "test-run",
            "deferred": [{
                "asset_class": "jp_stock",
                "symbol": "0099",
            }],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "test-run")
    gate.UNIVERSE = universe
    gate.main()
    result = json.loads(
        (root / "research" / "data_quality.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["provider_deferred_symbols"] == 1
    assert result["provider_deferred_ratio"] < 0.05


def test_data_quality_provider_deferred_ratio_blocks_large_omission(tmp_path, monkeypatch):
    import json
    import scripts.data_quality_gate as gate

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data"
    price_root = root / "prices"
    universe = root / "universe" / "latest.json"
    price_root.mkdir(parents=True)
    universe.parent.mkdir(parents=True)
    records = [
        {"symbol": f"{i:04d}", "asset_class": "jp_stock", "tradeable": True}
        for i in range(20)
    ]
    universe.write_text(json.dumps({"records": records}), encoding="utf-8")
    rows = [{
        "symbol": f"{i:04d}",
        "asset_class": "jp_stock",
        "session_date": pd.Timestamp("2026-09-23").date(),
        "available_at": pd.Timestamp("2026-09-23T07:00:00Z"),
        "retrieved_at": pd.Timestamp("2026-09-23T07:30:00Z"),
        "source": "yfinance",
        "provider_symbol": f"{i:04d}.T",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000.0,
    } for i in range(10)]
    pd.DataFrame(rows).to_parquet(price_root / "canonical.parquet", index=False)
    (price_root / "price_deferred_shard_0.json").write_text(
        json.dumps({
            "status": "DEFERRED",
            "retrieval_run_id": "large-test",
            "deferred": [
                {"asset_class": "jp_stock", "symbol": f"{i:04d}"}
                for i in range(10,20)
            ],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "large-test")
    gate.UNIVERSE = universe
    result = gate.main()
    payload = json.loads((root / "research" / "data_quality.json").read_text(encoding="utf-8"))
    assert result == 0
    assert payload["status"] == "DEFERRED"
    assert payload["provider_deferred_ratio"] == 0.5


def test_release_gate_rejects_degraded_price_quality():
    from pathlib import Path

    source = Path("scripts/release_gate.py").read_text(encoding="utf-8")
    assert 'quality.get("status")!="PASS"' in source
    assert '"data_quality_not_pass"' in source


def test_repository_verification_prioritizes_latest_sha():
    from pathlib import Path

    source = Path(".github/workflows/repository-verification.yml").read_text(
        encoding="utf-8"
    )
    assert "cancel-in-progress: true" in source


def test_bounded_recovery_targets_current_main_for_obsolete_runs():
    from pathlib import Path

    source = Path(".github/workflows/bounded-production-recovery.yml").read_text(
        encoding="utf-8"
    )
    assert "recovery-mode=rerun-current-sha" in source
    assert "recovery-mode=dispatch-current-main" in source
    assert "git/ref/heads/main" in source
    assert "gh workflow run" in source
    assert "current-main-run-already-active=true" in source


def test_watchdog_cleans_stale_bounded_recovery_without_recovery_loop():
    from pathlib import Path

    source = Path(".github/workflows/actions-reliability-watchdog.yml").read_text(
        encoding="utf-8"
    )
    assert "stale queued recovery run" in source
    assert '[ "$workflow_name" = "Bounded production recovery" ]' in source
    assert "without dispatching a recovery-of-recovery run" in source


def test_security_route_precedes_higher_level_frozen_routes():
    from src.research.router import route_plan

    plan = route_plan(
        "jp_stock",
        "trend",
        symbol="7203",
        locked_symbol_regime={"jp_stock::7203::trend": "logistic"},
        locked_symbol={"jp_stock::7203": "hgb"},
        locked_asset_regime={"jp_stock::trend": "extra_trees"},
        locked_asset={"jp_stock": "hgb_conservative_recent"},
        locked_regime={"trend": "hgb"},
        locked_global="hgb",
    )
    assert plan.names == ("logistic",)
    assert plan.scope == "jp_stock::7203::trend"


def test_security_routing_artifacts_are_part_of_research_and_lock_contract():
    from pathlib import Path

    research = Path("scripts/run_daily_research.py").read_text(encoding="utf-8")
    lock = Path("scripts/lock_frozen_model.py").read_text(encoding="utf-8")
    holdout = Path("scripts/evaluate_frozen_holdout.py").read_text(encoding="utf-8")
    prediction = Path("scripts/run_daily_prediction.py").read_text(encoding="utf-8")
    assert "symbol_selected_models" in research
    assert "symbol_regime_selected_models" in research
    assert "symbol_selected_models" in lock
    assert "symbol_regime_selected_models" in lock
    assert "locked_symbol_regime" in holdout
    assert "locked_symbol_regime" in prediction


def test_production_artifact_validates_security_route_metadata():
    from pathlib import Path

    source = Path("src/prediction/production_artifact.py").read_text(encoding="utf-8")
    artifact = Path("scripts/build_production_artifact.py").read_text(encoding="utf-8")
    assert "symbol_selected_models" in source
    assert "symbol_regime_selected_models" in source
    assert '"symbol_selected_models": dict(frozen.get("symbol_selected_models") or {})' in artifact


def test_production_artifact_requires_frozen_security_routes_when_present():
    from pathlib import Path

    source = Path("scripts/build_production_artifact.py").read_text(encoding="utf-8")
    assert '"symbol_selected_models": dict(frozen.get("symbol_selected_models") or {})' in source
    assert '"symbol_regime_selected_models": dict(frozen.get("symbol_regime_selected_models") or {})' in source
    assert '"symbol_selected_models",' in source
    assert '"symbol_regime_selected_models",' in source


def test_security_route_requires_per_fold_oos_evidence():
    from pathlib import Path

    source = Path("scripts/run_daily_research.py").read_text(encoding="utf-8")
    assert '"n_test_min"' in source
    assert 'metric.get("n_test_min", 0.0)' in source



def test_market_cycle_pushes_cancel_obsolete_queue_entries():
    from pathlib import Path

    source = Path(".github/workflows/market-cycle.yml").read_text(encoding="utf-8")
    assert "cancel-in-progress: ${{ github.event_name == 'push' }}" in source


def test_market_context_persistence_filters_future_available_rows(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import scripts.update_market_context as updater

    monkeypatch.chdir(tmp_path)
    updater.RESULT = tmp_path / "data" / "research" / "market_context_quality.json"
    updater.OUT = tmp_path / "data" / "market_context.parquet"

    now = pd.Timestamp.now(tz="UTC")
    future = now + pd.Timedelta(hours=2)
    rows = pd.DataFrame([
        {
            "session_date": now.date(), "family": "nikkei",
            "provider_symbol": "^N225", "source": "yfinance",
            "retrieved_at": now, "retrieval_run_id": "t",
            "close": 100.0, "ret_1d": 0.01, "volatility_20": 0.02,
            "available_at": now,
        },
        {
            "session_date": now.date(), "family": "topix",
            "provider_symbol": "1306.T", "source": "yfinance",
            "retrieved_at": now, "retrieval_run_id": "t",
            "close": 100.0, "ret_1d": 0.01, "volatility_20": 0.02,
            "available_at": future,
        },
    ])

    monkeypatch.setattr(updater, "download_market_context", lambda period: rows)
    # Only the persistence filter is under test; bypass family completeness.
    updater.MAX_STALENESS_DAYS = 9999
    updater.main()
    saved = pd.read_parquet(updater.OUT)
    assert len(saved) == 1
    assert saved.iloc[0]["family"] == "nikkei"
    payload = json.loads(updater.RESULT.read_text(encoding="utf-8"))
    assert payload["status"] == "DEFERRED"


def test_situation_labels_are_deterministic():
    from src.research.router import situation_for_row

    assert situation_for_row("event", gap_pct=0.04) == "event_gap"
    assert situation_for_row("event", volume_ratio_20=3.5) == "event_volume"
    assert situation_for_row("high_vol", vix_level=31) == "high_vol_vix"
    assert situation_for_row("trend", price_vs_sma60=0.05) == "trend_up"
    assert situation_for_row("trend", price_vs_sma60=-0.05) == "trend_down"
    assert situation_for_row("normal") == "range"
    assert situation_for_row("data_stressed") == "data_stressed"
