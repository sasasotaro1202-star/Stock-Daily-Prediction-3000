from pathlib import Path

import pytest

from src.prediction.production_artifact import load_production_artifact


def test_missing_production_artifact_fails_closed(tmp_path: Path):
    with pytest.raises(RuntimeError, match="artifact is missing"):
        load_production_artifact(tmp_path / "missing.pkl")



def test_artifact_validation_requires_exact_required_classifier_set(
    tmp_path, monkeypatch
):
    import numpy as np
    import pickle
    import sklearn
    import src.prediction.production_artifact as artifact_module
    from src.features.technical import FEATURE_COLUMNS

    monkeypatch.setattr(artifact_module, "research_fingerprint_sha256", lambda: "rfp")
    monkeypatch.setattr(artifact_module, "release_signature", lambda: "sig")

    gate_path = tmp_path / "release_gate.json"
    lock_path = tmp_path / "frozen_holdout.json"
    gate_path.write_text('{"approved": true}', encoding="utf-8")
    lock_path.write_text(
        '{"status":"FROZEN","holdout_generation":1,'
        '"research_code_fingerprint_sha256":"rfp"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(artifact_module, "RELEASE_GATE_PATH", gate_path)
    monkeypatch.setattr(artifact_module, "FROZEN_LOCK_PATH", lock_path)

    payload = {
        "metadata": {
            "artifact_version": 1,
            "code_fingerprint_sha256": "legacy-fp",
            "research_code_fingerprint_sha256": "rfp",
            "release_signature": "sig",
            "feature_columns": list(FEATURE_COLUMNS),
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "required_classifiers": ["hgb"],
            "selected_model": "hgb",
            "return_selected_estimator": "q50",
        },
        "classifiers": {
            "logistic": {"model": object(), "calibrator": object()},
            "hgb": {"model": object(), "calibrator": object()},
        },
        "return": {
            "selected": "q50",
            "global": {"mean": object(), "q50": object()},
        },
        "quantile": {"global": {}, "assets": {}},
    }
    with pytest.raises(RuntimeError, match="classifier set mismatch"):
        artifact_module.validate_artifact(payload)


def test_runtime_restores_lightgbm_for_recent_and_composite_routes():
    from pathlib import Path
    script = Path("scripts/install_production_runtime.py").read_text()
    assert 'if any("lightgbm" in str(name).lower() for name in required_models):' in script


def test_artifact_validates_lightgbm_and_composite_lightgbm_routes():
    script = Path("src/prediction/production_artifact.py").read_text()
    assert 'if any("lightgbm" in str(name).lower() for name in classifiers):' in script


def test_release_files_include_approved_gate():
    from src.prediction.production_artifact import RELEASE_FILES
    assert any(str(path) == "data/research/release_gate.json" for path in RELEASE_FILES)


def test_prediction_targets_each_security_not_only_each_asset_class():
    script = Path("scripts/run_daily_prediction.py").read_text()
    assert 'groupby(["asset_class", "symbol"], group_keys=False)' in script


def test_production_artifact_uses_research_fingerprint_for_compatibility():
    script = Path("src/prediction/production_artifact.py").read_text()
    assert "research_code_fingerprint_sha256" in script
    assert "research_fingerprint_sha256()" in script


def test_on_demand_prediction_workflow_is_present():
    workflow = Path(".github/workflows/on-demand-production-prediction.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "run_now_prediction.py" in workflow
    assert "load_production_artifact" in workflow

def test_near_production_training_enforces_pit_cutoff():
    script = Path("scripts/run_now_prediction.py").read_text()
    assert 'available = pd.to_datetime(df["available_at"], utc=True, errors="coerce")' in script
    assert "df = df.loc[available.le(asof)].copy()" in script
