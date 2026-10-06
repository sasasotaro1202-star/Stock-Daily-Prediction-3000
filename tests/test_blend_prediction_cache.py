from __future__ import annotations

import numpy as np
import pytest

from src.research.blend_prediction_cache import (
    CACHEABLE_BLEND_COMPONENTS,
    resolve_cached_blend_predictions,
)


def test_reuses_unweighted_blend_components_exactly():
    cache = {
        "hgb": (
            np.asarray([0.1, 0.8, 0.4]),
            np.asarray([0.2, 0.7]),
        ),
        "lightgbm_regularized": (
            np.asarray([0.5, 0.6, 0.2]),
            np.asarray([0.4, 0.3]),
        ),
    }
    expected_cal = 0.75 * cache["hgb"][0] + 0.25 * cache["lightgbm_regularized"][0]
    expected_test = 0.75 * cache["hgb"][1] + 0.25 * cache["lightgbm_regularized"][1]
    cal, test = resolve_cached_blend_predictions(
        "blend_hgb_lgbm_regularized_recent_75_25", cache
    )
    np.testing.assert_allclose(cal, expected_cal)
    np.testing.assert_allclose(test, expected_test)


def test_missing_component_falls_back_to_normal_training():
    cache = {"hgb": (np.asarray([0.1]), np.asarray([0.2]))}
    assert resolve_cached_blend_predictions(
        "blend_hgb_lgbm_regularized_recent_25_75", cache
    ) is None


def test_component_prediction_lengths_must_align():
    cache = {
        "hgb": (np.asarray([0.1, 0.2]), np.asarray([0.3])),
        "lightgbm_regularized": (np.asarray([0.4]), np.asarray([0.5])),
    }
    with pytest.raises(ValueError, match="lengths do not align"):
        resolve_cached_blend_predictions(
            "blend_hgb_lgbm_regularized_recent_25_75", cache
        )


def test_only_unweighted_component_candidates_are_cacheable():
    assert CACHEABLE_BLEND_COMPONENTS == {"hgb", "lightgbm_regularized"}
