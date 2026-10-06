from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def robust_distribution_shift_score(
    reference,
    query,
    *,
    max_features: int | None = None,
) -> float:
    """Estimate covariate shift using robust quantile differences.

    The query side may be unlabeled current-fold data. No target/outcome values
    are used. Each feature is standardized by the reference IQR with a small
    floor, then the mean absolute difference over fixed quantiles is returned.
    """
    ref = np.asarray(reference, dtype=float)
    qry = np.asarray(query, dtype=float)
    if ref.ndim != 2 or qry.ndim != 2 or ref.shape[1] != qry.shape[1]:
        raise ValueError("reference/query must be 2-D with matching columns")
    if ref.shape[0] < 20 or qry.shape[0] < 20:
        raise ValueError("reference/query require at least 20 rows")

    finite_ref = np.isfinite(ref)
    finite_qry = np.isfinite(qry)
    features = ref.shape[1]
    if max_features is not None:
        features = min(features, int(max_features))
    quantiles = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90], dtype=float)
    scores = []
    for j in range(features):
        r = ref[finite_ref[:, j], j]
        q = qry[finite_qry[:, j], j]
        if len(r) < 20 or len(q) < 20:
            continue
        iq = float(np.quantile(r, 0.75) - np.quantile(r, 0.25))
        scale = max(iq, float(np.std(r)), 1e-8)
        rq = np.quantile(r, quantiles)
        qq = np.quantile(q, quantiles)
        scores.append(float(np.mean(np.abs(rq - qq)) / scale))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def select_drift_aware_window(
    history: Mapping[int, Sequence[Mapping[str, float]]],
    current_fold: int,
    current_drift_by_window: Mapping[int, float],
    *,
    min_history_folds: int = 2,
    half_life_folds: float = 4.0,
    drift_scale: float = 0.50,
    stability_penalty: float = 0.25,
) -> tuple[int | None, dict[int, dict[str, float]]]:
    """Choose a training window from prior-fold performance conditioned on drift.

    Only history with fold < current_fold is admissible. Returns the selected
    window and diagnostics. None means insufficient prior evidence.
    """
    if current_fold < 0:
        raise ValueError("current_fold must be >= 0")
    if min_history_folds < 1:
        raise ValueError("min_history_folds must be >= 1")
    if not np.isfinite(half_life_folds) or half_life_folds <= 0:
        raise ValueError("half_life_folds must be > 0")
    if not np.isfinite(drift_scale) or drift_scale <= 0:
        raise ValueError("drift_scale must be > 0")
    if stability_penalty < 0:
        raise ValueError("stability_penalty must be >= 0")

    diagnostics: dict[int, dict[str, float]] = {}
    for raw_window, raw_drift in current_drift_by_window.items():
        window = int(raw_window)
        current_drift = float(raw_drift)
        if not np.isfinite(current_drift) or current_drift < 0:
            continue
        rows = []
        for row in history.get(window, ()):
            try:
                fold = int(row["fold"])
                logloss = float(row["logloss"])
                drift = float(row["drift"])
            except (KeyError, TypeError, ValueError):
                continue
            if fold >= current_fold:
                raise ValueError("drift-aware window history contains non-prior fold")
            if not all(np.isfinite(x) for x in (logloss, drift)):
                continue
            rows.append((fold, logloss, drift))
        if len(rows) < min_history_folds:
            continue

        folds = np.asarray([x[0] for x in rows], dtype=float)
        losses = np.asarray([x[1] for x in rows], dtype=float)
        drifts = np.asarray([x[2] for x in rows], dtype=float)
        recency = np.power(0.5, (float(current_fold) - folds) / half_life_folds)
        similarity = np.exp(-np.abs(drifts - current_drift) / drift_scale)
        weights = recency * similarity
        denom = float(weights.sum())
        if denom <= 0.0 or not np.isfinite(denom):
            continue
        mean_loss = float(np.average(losses, weights=weights))
        variance = float(np.average((losses - mean_loss) ** 2, weights=weights))
        score = mean_loss + stability_penalty * float(np.sqrt(max(variance, 0.0)))
        diagnostics[window] = {
            "score": score,
            "weighted_logloss": mean_loss,
            "weighted_std": float(np.sqrt(max(variance, 0.0))),
            "history_folds": float(len(rows)),
            "current_drift": current_drift,
        }

    if not diagnostics:
        return None, diagnostics
    selected = min(
        diagnostics,
        key=lambda w: (
            diagnostics[w]["score"],
            diagnostics[w]["weighted_logloss"],
            10**9 if w == 0 else w,
        ),
    )
    return int(selected), diagnostics
