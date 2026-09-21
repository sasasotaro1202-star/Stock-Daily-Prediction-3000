from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor


def make_return_model():
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        loss="squared_error",
        random_state=42,
    )


def make_quantile_model(quantile: float):
    q=float(quantile)
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        loss="quantile",
        quantile=q,
        random_state=42,
    )
