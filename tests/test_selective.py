import numpy as np
import pytest

from src.research.selective import apply_confidence_shrinkage


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


def test_shrinkage_validates_bounds():
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
