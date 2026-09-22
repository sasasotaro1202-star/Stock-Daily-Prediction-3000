from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
import sys

import numpy as np
import sklearn

from src.features.technical import FEATURE_COLUMNS
from src.validation.code_fingerprint import fingerprint_sha256

ARTIFACT_PATH = Path("data/research/production_model_artifact.pkl")
RELEASE_GATE_PATH = Path("data/research/release_gate.json")
FROZEN_LOCK_PATH = Path("config/frozen_holdout.json")
RELEASE_FILES = (
    Path("data/research/latest_metrics.json"),
    Path("data/research/frozen_holdout_result.json"),
    Path("config/frozen_holdout.json"),
    Path("data/research/release_gate.json"),
)


def release_signature() -> str:
    h = hashlib.sha256()
    for path in RELEASE_FILES:
        if not path.exists():
            raise RuntimeError(f"missing release evidence: {path}")
        h.update(str(path).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def validate_artifact(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError("production model artifact is not a mapping")
    meta = payload.get("metadata")
    classifiers = payload.get("classifiers")
    quantile = payload.get("quantile")
    return_section = payload.get("return")
    if not isinstance(meta, dict) or not isinstance(classifiers, dict):
        raise RuntimeError("production model artifact metadata/classifiers are missing")
    if not isinstance(quantile, dict):
        raise RuntimeError("production model artifact quantile section is missing")
    if not isinstance(return_section, dict):
        raise RuntimeError("production return estimator section is missing")

    if meta.get("artifact_version") != 1:
        raise RuntimeError("unsupported production model artifact version")
    if meta.get("code_fingerprint_sha256") != fingerprint_sha256():
        raise RuntimeError("production model artifact code fingerprint mismatch")
    if meta.get("release_signature") != release_signature():
        raise RuntimeError("production model artifact release evidence mismatch")
    gate_path = RELEASE_GATE_PATH
    try:
        gate_payload = __import__("json").loads(
            gate_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise RuntimeError("production release gate is unreadable") from exc
    if gate_payload.get("approved") is not True:
        raise RuntimeError("production model artifact requires an approved release gate")
    lock_payload = __import__("json").loads(
        FROZEN_LOCK_PATH.read_text(encoding="utf-8")
    )
    if meta.get("holdout_generation") != lock_payload.get("holdout_generation"):
        raise RuntimeError("production holdout generation mismatch")
    if meta.get("research_code_fingerprint_sha256") != lock_payload.get(
        "research_code_fingerprint_sha256"
    ):
        raise RuntimeError("production research fingerprint mismatch")
    if meta.get("feature_columns") != list(FEATURE_COLUMNS):
        raise RuntimeError("production model artifact feature schema mismatch")
    if meta.get("python_version") != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise RuntimeError("production model artifact Python major/minor mismatch")
    if meta.get("numpy_version") != np.__version__:
        raise RuntimeError("production model artifact NumPy version mismatch")
    if meta.get("sklearn_version") != sklearn.__version__:
        raise RuntimeError("production model artifact scikit-learn version mismatch")

    required_classifiers = set(meta.get("required_classifiers") or [])
    if "hgb" not in required_classifiers:
        raise RuntimeError("production model artifact must include hgb fallback")
    if set(classifiers) != required_classifiers:
        raise RuntimeError("production model artifact classifier set mismatch")
    selected = meta.get("selected_model")
    if selected not in classifiers:
        raise RuntimeError("production model artifact selected model is missing")
    for name, entry in classifiers.items():
        if not isinstance(entry, dict) or "model" not in entry or "calibrator" not in entry:
            raise RuntimeError(f"production classifier artifact missing components: {name}")
    if "global" not in quantile or "assets" not in quantile:
        raise RuntimeError("production quantile artifact is incomplete")
    if return_section.get("selected") not in {"mean", "q50", "blend_mean_q50"}:
        raise RuntimeError("production return estimator selection is invalid")
    if set(return_section.get("global") or {}) != {"mean", "q50"}:
        raise RuntimeError("production return estimator artifacts are incomplete")
    if meta.get("return_selected_estimator") != return_section.get("selected"):
        raise RuntimeError("production return estimator metadata mismatch")
    if meta.get("calibration_method") not in {"platt", "beta", "isotonic"}:
        raise RuntimeError("production calibration method is invalid")
    runtime_versions = meta.get("runtime_dependency_versions") or {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "pyarrow",
        "yfinance",
        "PyYAML",
        "curl_cffi",
    ):
        expected = runtime_versions.get(package)
        if expected is None:
            raise RuntimeError(f"production runtime dependency version is missing: {package}")
        try:
            actual = __import__("importlib.metadata", fromlist=["version"]).version(package)
        except Exception as exc:
            raise RuntimeError(f"production runtime dependency is missing: {package}") from exc
        if actual != expected:
            raise RuntimeError(
                f"production runtime dependency version mismatch for {package}: "
                f"{actual} != {expected}"
            )

    if any("lightgbm" in str(name).lower() for name in classifiers):
        expected = runtime_versions.get("lightgbm") or meta.get("lightgbm_version")
        if expected is None:
            raise RuntimeError("production model artifact LightGBM version is missing")
        try:
            import lightgbm
        except ImportError as exc:
            raise RuntimeError("production model artifact requires LightGBM") from exc
        if meta["lightgbm_version"] != lightgbm.__version__:
            raise RuntimeError("production model artifact LightGBM version mismatch")


def load_production_artifact(path: Path = ARTIFACT_PATH) -> dict:
    if not path.exists():
        raise RuntimeError("approved production model artifact is missing")
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(f"production model artifact could not be loaded: {exc}") from exc
    validate_artifact(payload)
    return payload
