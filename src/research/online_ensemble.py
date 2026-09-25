from __future__ import annotations

import numpy as np


def online_expert_average(
    predictions_by_model: dict[str, np.ndarray],
    y_true,
    session_dates,
    *,
    learning_rate: float,
    share_rate: float = 0.0,
    update_group_keys=None,
):
    """Chronological online expert mixture with optional fixed-share recovery.

    Predictions for session t use weights learned strictly before that session.
    After the whole session t is observed, each expert receives its log-loss
    update. A positive share_rate returns a small fraction of post-loss
    weight mass to the uniform prior, limiting expert lock-in after regime shifts.
    """
    if not predictions_by_model:
        raise ValueError("at least one expert is required")
    eta = float(learning_rate)
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("learning_rate must be finite and non-negative")
    share = float(share_rate)
    if not np.isfinite(share) or not 0.0 <= share <= 1.0:
        raise ValueError("share_rate must be finite and in [0, 1]")

    names = sorted(predictions_by_model)
    arrays = [np.asarray(predictions_by_model[name], dtype=float) for name in names]
    raw_y = np.asarray(y_true)
    dates = np.asarray(session_dates)
    groups = None if update_group_keys is None else np.asarray(update_group_keys)

    if raw_y.ndim != 1 or dates.ndim != 1 or len(raw_y) != len(dates):
        raise ValueError("y_true and session_dates must be aligned 1-D arrays")
    if groups is not None and (groups.ndim != 1 or len(groups) != len(raw_y)):
        raise ValueError("update_group_keys must align with y_true 1-D")
    if any(arr.ndim != 1 or len(arr) != len(raw_y) for arr in arrays):
        raise ValueError("all expert predictions must match y_true length")
    if len(raw_y) == 0:
        raise ValueError("y_true must be non-empty")
    try:
        y_float = raw_y.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must be numeric binary labels 0/1") from exc
    if not np.isfinite(y_float).all():
        raise ValueError("y_true must be finite")
    if not np.isin(y_float, (0.0, 1.0)).all():
        raise ValueError("y_true must contain only binary labels 0/1")
    y = y_float.astype(int)
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
        session_groups = groups[mask] if groups is not None else None
        unique_groups = (
            sorted(set(session_groups.tolist()))
            if session_groups is not None
            else None
        )
        for j in range(len(names)):
            p = matrix[mask, j]
            target = y[mask]
            loss = -(
                target * np.log(p)
                + (1 - target) * np.log(1.0 - p)
            )
            if unique_groups is None:
                row_losses.append(float(np.mean(loss)))
            else:
                group_losses = []
                for group_value in unique_groups:
                    group_mask = session_groups == group_value
                    if group_mask.any():
                        group_losses.append(float(np.mean(loss[group_mask])))
                row_losses.append(float(np.mean(group_losses)))
        row_losses_arr = np.asarray(row_losses, dtype=float)
        log_weights -= eta * row_losses_arr

        next_weights = np.exp(log_weights - np.max(log_weights))
        next_weights /= next_weights.sum()
        if share > 0.0:
            next_weights = (1.0 - share) * next_weights + share / len(names)
            log_weights = np.log(np.clip(next_weights, 1e-300, 1.0))
        history.append({
            "session_date": str(date),
            "share_rate": share,
            "update_grouped": groups is not None,
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
