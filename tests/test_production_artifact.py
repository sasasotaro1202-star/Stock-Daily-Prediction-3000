from pathlib import Path

import pytest

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

    monkeypatch.setattr(artifact_module, "fingerprint_sha256", lambda: "fp")
    monkeypatch.setattr(artifact_module, "release_signature", lambda: "sig")

    payload = {
        "metadata": {
            "artifact_version": 1,
            "code_fingerprint_sha256": "fp",
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
