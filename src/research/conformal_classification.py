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
    # Count calibration scores >= each test threshold with a sorted vector.
    # This is algebraically identical to the broadcasted count but uses
    # O(n_cal + n_test) memory instead of O(n_cal * n_test).
    sorted_scores = np.sort(scores)
    n_scores = len(sorted_scores)
    pvalue_1 = (
        1.0
        + n_scores
        - np.searchsorted(sorted_scores, 1.0 - p_test, side="left")
    ) / (n_scores + 1.0)
    pvalue_0 = (
        1.0
        + n_scores
        - np.searchsorted(sorted_scores, p_test, side="left")
    ) / (n_scores + 1.0)
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


def group_conformal_prediction_sets(
    y_calibration,
    p_calibration,
    groups_calibration,
    p_test,
    groups_test,
    *,
    alpha=0.10,
    min_group_size=50,
):
    """Group-conditional split conformal sets with deterministic global fallback."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0,1)")
    if int(min_group_size) < 20:
        raise ValueError("min_group_size must be >= 20")
    y, p_cal, p_test = _validate_inputs(y_calibration, p_calibration, p_test)
    g_cal = np.asarray(groups_calibration, dtype=str)
    g_test = np.asarray(groups_test, dtype=str)
    if g_cal.ndim != 1 or len(g_cal) != len(y):
        raise ValueError("calibration groups must align")
    if g_test.ndim != 1 or len(g_test) != len(p_test):
        raise ValueError("test groups must align")

    scores = np.where(y == 1, 1.0 - p_cal, p_cal)
    global_sorted = np.sort(scores)
    global_n = len(global_sorted)
    pvalue_0 = np.empty(len(p_test), dtype=float)
    pvalue_1 = np.empty(len(p_test), dtype=float)
    fallback = np.zeros(len(p_test), dtype=bool)

    for group in np.unique(g_test):
        idx_test = np.flatnonzero(g_test == group)
        idx_cal = np.flatnonzero(g_cal == group)
        if len(idx_cal) < int(min_group_size):
            sorted_scores = global_sorted
            n_scores = global_n
            fallback[idx_test] = True
        else:
            sorted_scores = np.sort(scores[idx_cal])
            n_scores = len(sorted_scores)
        pvalue_1[idx_test] = (
            1.0 + n_scores
            - np.searchsorted(sorted_scores, 1.0 - p_test[idx_test], side="left")
        ) / (n_scores + 1.0)
        pvalue_0[idx_test] = (
            1.0 + n_scores
            - np.searchsorted(sorted_scores, p_test[idx_test], side="left")
        ) / (n_scores + 1.0)

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
        "fallback_to_global": fallback,
    }

def group_conformal_prediction_set_metrics(
    y_calibration,
    p_calibration,
    groups_calibration,
    p_test,
    groups_test,
    y_test,
    *,
    alpha=0.10,
    min_group_size=50,
) -> dict[str, object]:
    """Summarize group-conditional conformal coverage and fallback behavior."""
    result = group_conformal_prediction_sets(
        y_calibration,
        p_calibration,
        groups_calibration,
        p_test,
        groups_test,
        alpha=alpha,
        min_group_size=min_group_size,
    )
    y_out = np.asarray(y_test, dtype=int)
    if y_out.ndim != 1 or len(y_out) != len(p_test):
        raise ValueError("test labels must align with test probabilities")
    if not np.isin(y_out, (0, 1)).all():
        raise ValueError("test labels must be binary")
    set_size = result["set_size"]
    singleton = set_size == 1
    contains = np.where(y_out == 1, result["include_1"], result["include_0"])
    singleton_accuracy = (
        float(np.mean(result["predicted_class"][singleton] == y_out[singleton]))
        if singleton.any() else float("nan")
    )
    by_group = {}
    g_test = np.asarray(groups_test, dtype=str)
    for group in np.unique(g_test):
        idx = g_test == group
        group_set = set_size[idx]
        group_singleton = group_set == 1
        by_group[str(group)] = {
            "n_test": int(idx.sum()),
            "coverage": float(np.mean(contains[idx])),
            "mean_set_size": float(np.mean(group_set)),
            "singleton_rate": float(np.mean(group_singleton)),
            "fallback_rate": float(np.mean(result["fallback_to_global"][idx])),
        }
    return {
        "alpha": float(alpha),
        "n_test": float(len(y_out)),
        "set_coverage": float(np.mean(contains)),
        "mean_set_size": float(np.mean(set_size)),
        "singleton_rate": float(np.mean(singleton)),
        "singleton_accuracy": singleton_accuracy,
        "empty_rate": float(np.mean(set_size == 0)),
        "fallback_rate": float(np.mean(result["fallback_to_global"])),
        "mean_predicted_class_pvalue": float(np.mean(result["predicted_class_pvalue"])),
        "by_group": by_group,
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

__all__ = ["conformal_prediction_sets", "conformal_prediction_set_metrics", "group_conformal_prediction_sets", "group_conformal_prediction_set_metrics"]
