from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_configured_xgboost_and_catboost_challengers_have_factories_or_optional_imports():
    source = (ROOT / "src/prediction/model_factories.py").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"xgboost"' in source
    assert '"catboost"' in source
    assert "from xgboost import XGBClassifier" in source
    assert "from catboost import CatBoostClassifier" in source
    assert "xgboost>=2.0" in pyproject
    assert "catboost>=1.2" in pyproject
    assert 'out["xgboost"] = make_xgboost' in source
    assert 'out["catboost"] = make_catboost' in source
