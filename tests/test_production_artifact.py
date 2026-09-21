from pathlib import Path

import pytest

from src.prediction.production_artifact import load_production_artifact


def test_missing_production_artifact_fails_closed(tmp_path: Path):
    with pytest.raises(RuntimeError, match="artifact is missing"):
        load_production_artifact(tmp_path / "missing.pkl")
