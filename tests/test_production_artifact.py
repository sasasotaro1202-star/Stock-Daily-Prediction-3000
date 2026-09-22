from pathlib import Path

import pytest

import pytest

from src.prediction.production_artifact import load_production_artifact


def test_missing_production_artifact_fails_closed(tmp_path: Path):
    with pytest.raises(RuntimeError, match="artifact is missing"):
        load_production_artifact(tmp_path / "missing.pkl")


def test_artifact_validation_requires_exact_required_classifier_set(tmp_path):
    import pickle
    from src.prediction.production_artifact import load_production_artifact

    payload = {
        "metadata": {
            "artifact_version": 1,
            "code_fingerprint_sha256": "bad",
            "release_signature": "bad",
            "feature_columns": [],
            "python_version": "3.12",
            "numpy_version": "0",
            "sklearn_version": "0",
            "required_classifiers": ["hgb"],
            "selected_model": "hgb",
        },
        "classifiers": {"logistic": object(), "hgb": object()},
        "quantile": {"global": {}, "assets": {}},
    }
    path = tmp_path / "artifact.pkl"
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(RuntimeError, match="classifier set mismatch"):
        load_production_artifact(path)
