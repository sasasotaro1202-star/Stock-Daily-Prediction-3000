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
        "blend_mean_q50": {"mae": 0.0115, "rank_ic": 0.090, "rank_ic_std": 0.015, "folds": 4},
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
