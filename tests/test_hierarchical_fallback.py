from __future__ import annotations

import numpy as np
import pytest

from src.prediction.hierarchical_fallback import (
    blend_probabilities,
    make_symbol_logistic_model,
    SYMBOL_FALLBACK_FEATURES,
)


def test_blend_probabilities_normalizes_available_expert_weights():
    assert blend_probabilities([0.80, 0.60], [0.55, 0.25]) == pytest.approx(
        (0.80 * 0.55 + 0.60 * 0.25) / 0.80
    )


def test_blend_probabilities_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        blend_probabilities([0.8], [0.0])
    with pytest.raises(ValueError):
        blend_probabilities([0.8, np.nan], [1.0, 1.0])


def test_symbol_model_is_regularized_pipeline():
    model = make_symbol_logistic_model()
    assert len(SYMBOL_FALLBACK_FEATURES) >= 8
    names = [name for name, _ in model.steps]
    assert names == ["simpleimputer", "standardscaler", "logisticregression"]
