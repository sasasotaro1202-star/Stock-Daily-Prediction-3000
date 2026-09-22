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

def test_ece_bounds():
    v=expected_calibration_error([0,1,0,1],[0.1,0.9,0.2,0.8])
    assert 0<=v<=1


def test_model_factory_matches_production_hgb_contract():
    from src.prediction.model_factories import models

    m = models()["hgb"]()
    params = m.get_params()
    assert params["histgradientboostingclassifier__max_iter"] == 300
    assert params["histgradientboostingclassifier__learning_rate"] == 0.04


def test_lightgbm_challenger_is_available_in_research_ci():
    from src.prediction.model_factories import models
    import lightgbm

    assert "lightgbm" in models()
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
