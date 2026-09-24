import numpy as np
import pytest

from src.research.selective import (
    SELECTIVE_CONFIDENCE_THRESHOLDS,
    SELECTIVE_RETAINED_WEIGHTS,
    apply_confidence_shrinkage,
    select_confidence_shrinkage_parameters,
)


def test_shrinkage_changes_only_low_confidence_values():
    p = np.array([0.50, 0.53, 0.70, 0.20])
    out = apply_confidence_shrinkage(
        p,
        base_rate=0.60,
        confidence_threshold=0.05,
        retained_weight=0.50,
    )
    assert float(out[0]) == pytest.approx(0.55)
    assert float(out[1]) == pytest.approx(0.565)
    assert float(out[2]) == pytest.approx(0.70)
    assert float(out[3]) == pytest.approx(0.20)


def test_shrinkage_weight_one_is_identity():
    p = np.array([0.1, 0.5, 0.9])
    out = apply_confidence_shrinkage(
        p,
        base_rate=0.6,
        confidence_threshold=0.2,
        retained_weight=1.0,
    )
    assert np.allclose(out, p)


def test_shrinkage_validates_bounds_and_finiteness():
    with pytest.raises(ValueError):
        apply_confidence_shrinkage(
            [0.5], base_rate=float("nan"),
            confidence_threshold=0.1, retained_weight=0.5
        )
    with pytest.raises(ValueError):
        apply_confidence_shrinkage(
            [0.5], base_rate=0.5,
            confidence_threshold=0.6, retained_weight=0.5
        )
    with pytest.raises(ValueError):
        apply_confidence_shrinkage(
            [0.5], base_rate=0.5,
            confidence_threshold=0.1, retained_weight=1.1
        )
    with pytest.raises(ValueError):
        apply_confidence_shrinkage(
            [0.5, np.nan], base_rate=0.5,
            confidence_threshold=0.1, retained_weight=0.5
        )
    with pytest.raises(ValueError):
        apply_confidence_shrinkage(
            [[0.5]], base_rate=0.5,
            confidence_threshold=0.1, retained_weight=0.5
        )


def test_selective_parameters_are_selected_from_allowed_grid():
    y = np.array([0, 0, 0, 1, 1, 1] * 8, dtype=int)
    p = np.array([0.49, 0.51, 0.48, 0.52, 0.47, 0.53] * 8, dtype=float)
    selected = select_confidence_shrinkage_parameters(
        y, p, base_rate=float(y.mean())
    )
    assert selected["confidence_threshold"] in SELECTIVE_CONFIDENCE_THRESHOLDS
    assert selected["retained_weight"] in SELECTIVE_RETAINED_WEIGHTS
    assert np.isfinite(selected["validation_logloss"])
    assert np.isfinite(selected["validation_brier"])


def test_selective_parameter_selection_is_deterministic():
    y = np.array([0, 1] * 20, dtype=int)
    p = np.array([0.49, 0.51] * 20, dtype=float)
    a = select_confidence_shrinkage_parameters(y, p, base_rate=0.5)
    b = select_confidence_shrinkage_parameters(y, p, base_rate=0.5)
    assert a == b


def test_selective_selection_rejects_tiny_or_constant_calibration():
    with pytest.raises(ValueError):
        select_confidence_shrinkage_parameters(
            [0, 1] * 9, [0.5] * 18, base_rate=0.5
        )
    with pytest.raises(ValueError):
        select_confidence_shrinkage_parameters(
            [1] * 20, [0.5] * 20, base_rate=1.0
        )
