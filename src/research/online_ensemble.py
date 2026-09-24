from __future__ import annotations

import numpy as np


def online_expert_average(
    predictions_by_model: dict[str, np.ndarray],
    y_true,
    session_dates,
    *,
    learning_rate: float,
):
    """Chronological online mixture with post-session expert updates.

    Predictions for session t use weights learned from sessions < t. After the
    whole session t is observed, each expert receives its log-loss update.
    """
    if not predictions_by_model:
        raise ValueError("at least one expert is required")
    eta = float(learning_rate)
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("learning_rate must be finite and non-negative")

    names = sorted(predictions_by_model)
    arrays = [np.asarray(predictions_by_model[name], dtype=float) for name in names]
    y = np.asarray(y_true, dtype=int)
    dates = np.asarray(session_dates)

    if y.ndim != 1 or dates.ndim != 1 or len(y) != len(dates):
        raise ValueError("y_true and session_dates must be aligned 1-D arrays")
    if any(arr.ndim != 1 or len(arr) != len(y) for arr in arrays):
        raise ValueError("all expert predictions must match y_true length")
    if len(y) == 0:
        raise ValueError("y_true must be non-empty")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0/1")
    if not np.isfinite(y.astype(float)).all():
        raise ValueError("y_true must be finite")
    if any(not np.isfinite(arr).all() for arr in arrays):
        raise ValueError("expert predictions must be finite")

    matrix = np.column_stack(
        [np.clip(arr, 1e-6, 1.0 - 1e-6) for arr in arrays]
    )
    unique_dates = sorted(set(dates.tolist()))
    log_weights = np.zeros(len(names), dtype=float)
    ensemble = np.empty(len(y), dtype=float)
    history = []

    for date in unique_dates:
        mask = dates == date
        current_weights = np.exp(log_weights - np.max(log_weights))
        current_weights /= current_weights.sum()
        ensemble[mask] = matrix[mask] @ current_weights

        row_losses = []
        for j in range(len(names)):
            p = matrix[mask, j]
            target = y[mask]
            loss = -(
                target * np.log(p)
                + (1 - target) * np.log(1.0 - p)
            )
            row_losses.append(float(np.mean(loss)))
        row_losses_arr = np.asarray(row_losses, dtype=float)
        log_weights -= eta * row_losses_arr

        next_weights = np.exp(log_weights - np.max(log_weights))
        next_weights /= next_weights.sum()
        history.append({
            "session_date": str(date),
            "weights_before": current_weights.tolist(),
            "weights_after": next_weights.tolist(),
            "expert_logloss": {
                name: float(row_losses_arr[j])
                for j, name in enumerate(names)
            },
        })

    final_weights = np.exp(log_weights - np.max(log_weights))
    final_weights /= final_weights.sum()
    return ensemble, final_weights, history
