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
        },
        "classifiers": {
            "logistic": {"model": object(), "calibrator": object()},
            "hgb": {"model": object(), "calibrator": object()},
        },
        "quantile": {"global": {}, "assets": {}},
    }
    with pytest.raises(RuntimeError, match="classifier set mismatch"):
        artifact_module.validate_artifact(payload)
