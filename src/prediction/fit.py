from __future__ import annotations

import numpy as np
import pandas as pd


def training_recency_weights(
    session_dates: pd.Series,
    *,
    half_life_sessions: int = 252,
) -> np.ndarray:
    half_life = int(half_life_sessions)
    if half_life <= 0:
        raise ValueError("half_life_sessions must be positive")
    dates = pd.to_datetime(session_dates, errors="coerce").dt.date
    if dates.isna().any():
        raise ValueError("session_dates contain invalid values")
    unique_dates = sorted(pd.unique(dates))
    age_map = {
        value: len(unique_dates) - 1 - idx
        for idx, value in enumerate(unique_dates)
    }
    age = dates.map(age_map).to_numpy(dtype=float)
    weights = np.power(2.0, -age / float(half_life))
    weights /= float(np.mean(weights))
    return weights


def fit_regressor(
    model,
    model_name: str,
    X: pd.DataFrame,
    y,
    session_dates: pd.Series,
    *,
    half_life_sessions: int = 252,
):
    name = str(model_name)
    if name.endswith("_recent"):
        weights = training_recency_weights(
            session_dates,
            half_life_sessions=half_life_sessions,
        )
        if hasattr(model, "steps"):
            final_step = model.steps[-1][0]
            model.fit(
                X,
                y,
                **{f"{final_step}__sample_weight": weights},
            )
        else:
            model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def fit_classifier(
    model,
    model_name: str,
    X: pd.DataFrame,
    y,
    session_dates: pd.Series,
    *,
    half_life_sessions: int = 252,
):
    name = str(model_name)
    if name.endswith("_recent"):
        weights = training_recency_weights(
            session_dates,
            half_life_sessions=half_life_sessions,
        )
        if hasattr(model, "steps"):
            final_step = model.steps[-1][0]
            model.fit(
                X,
                y,
                **{f"{final_step}__sample_weight": weights},
            )
        else:
            model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y)
    return model
