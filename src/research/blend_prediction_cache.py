from __future__ import annotations

from typing import Mapping

import numpy as np


# These two candidates use the exact same unweighted component fits as the
# corresponding asymmetric blend challengers. The component predictions are
# reused only when both prerequisite candidates have already been evaluated in
# the same chronological fold.
CACHEABLE_BLEND_COMPONENTS = frozenset({
    "hgb",
    "lightgbm_regularized",
})

_BLEND_SPECS = {
    "blend_hgb_lgbm_regularized_recent_75_25": (
        "hgb", "lightgbm_regularized", 0.75
    ),
    "blend_hgb_lgbm_regularized_recent_25_75": (
        "hgb", "lightgbm_regularized", 0.25
    ),
}


def resolve_cached_blend_predictions(
    model_name: str,
    component_cache: Mapping[str, tuple[np.ndarray, np.ndarray]],
):
    """Return exact raw blend predictions when both unweighted components exist."""
    spec = _BLEND_SPECS.get(str(model_name))
    if spec is None:
        return None
    left_name, right_name, left_weight = spec
    left = component_cache.get(left_name)
    right = component_cache.get(right_name)
    if left is None or right is None:
        return None
    left_cal, left_test = left
    right_cal, right_test = right
    if len(left_cal) != len(right_cal) or len(left_test) != len(right_test):
        raise ValueError("blend component prediction lengths do not align")
    return (
        float(left_weight) * np.asarray(left_cal, dtype=float)
        + (1.0 - float(left_weight)) * np.asarray(right_cal, dtype=float),
        float(left_weight) * np.asarray(left_test, dtype=float)
        + (1.0 - float(left_weight)) * np.asarray(right_test, dtype=float),
    )
