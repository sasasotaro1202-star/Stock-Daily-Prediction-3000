from __future__ import annotations
import numpy as np

def _validate_inputs(y_calibration, p_calibration, p_test):
    y = np.asarray(y_calibration, dtype=int)
    p_cal = np.asarray(p_calibration, dtype=float)
    p_test = np.asarray(p_test, dtype=float)
    if y.ndim != 1 or p_cal.ndim != 1 or p_test.ndim != 1:
        raise ValueError("inputs must be 1-D arrays")
    if len(y) != len(p_cal):
        raise ValueError("calibration labels/probabilities must align")
    if len(y) < 20:
        raise ValueError("calibration slice is too small")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("calibration labels must be binary")
    if not np.isfinite(p_cal).all() or not np.isfinite(p_test).all():
        raise ValueError("probabilities must be finite")
    return y, np.clip(p_cal, 1e-6, 1.0 - 1e-6), np.clip(p_test, 1e-6, 1.0 - 1e-6)

def conformal_prediction_sets(y_calibration, p_calibration, p_test, *, alpha=0.10):
    """Deterministic split-conformal prediction sets for binary direction."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0,1)")
    y, p_cal, p_test = _validate_inputs(y_calibration, p_calibration, p_test)
    scores = np.where(y == 1, 1.0 - p_cal, p_cal)
    pvalue_1 = (1.0 + np.sum(scores[:, None] >= (1.0 - p_test)[None, :], axis=0)) / (len(scores) + 1.0)
    pvalue_0 = (1.0 + np.sum(scores[:, None] >= p_test[None, :], axis=0)) / (len(scores) + 1.0)
    include_1 = pvalue_1 > float(alpha)
    include_0 = pvalue_0 > float(alpha)
    set_size = include_0.astype(int) + include_1.astype(int)
    predicted_class = (p_test >= 0.5).astype(int)
    predicted_class_pvalue = np.where(predicted_class == 1, pvalue_1, pvalue_0)
    return {
        "pvalue_0": pvalue_0,
        "pvalue_1": pvalue_1,
        "include_0": include_0,
        "include_1": include_1,
        "set_size": set_size,
        "predicted_class": predicted_class,
        "predicted_class_pvalue": predicted_class_pvalue,
    }

def conformal_prediction_set_metrics(
    y_calibration, p_calibration, p_test, y_test, *, alpha=0.10
) -> dict[str, float]:
    """Summarize conformal coverage, ambiguity, and case-level confidence."""
    _validate_inputs(y_calibration, p_calibration, p_test)
    y_out = np.asarray(y_test, dtype=int)
    if y_out.ndim != 1 or len(y_out) != len(p_test):
        raise ValueError("test labels must align with test probabilities")
    if not np.isin(y_out, (0, 1)).all():
        raise ValueError("test labels must be binary")
    result = conformal_prediction_sets(y_calibration, p_calibration, p_test, alpha=alpha)
    set_size = result["set_size"]
    singleton = set_size == 1
    contains = np.where(y_out == 1, result["include_1"], result["include_0"])
    singleton_accuracy = (
        float(np.mean(result["predicted_class"][singleton] == y_out[singleton]))
        if singleton.any() else float("nan")
    )
    return {
        "alpha": float(alpha),
        "n_test": float(len(y_out)),
        "set_coverage": float(np.mean(contains)),
        "mean_set_size": float(np.mean(set_size)),
        "singleton_rate": float(np.mean(singleton)),
        "singleton_accuracy": singleton_accuracy,
        "empty_rate": float(np.mean(set_size == 0)),
        "mean_predicted_class_pvalue": float(np.mean(result["predicted_class_pvalue"])),
    }

__all__ = ["conformal_prediction_sets", "conformal_prediction_set_metrics"]
