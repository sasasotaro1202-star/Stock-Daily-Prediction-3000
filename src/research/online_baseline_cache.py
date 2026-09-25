from __future__ import annotations

from typing import Callable, Mapping

import numpy as np


def build_baseline_logloss_cache(
    online_prediction_by_fold: Mapping[int, Mapping[str, object]],
    selected_model: str,
    classification_metrics: Callable[[object, object], Mapping[str, float]],
) -> tuple[dict[int, float], dict[tuple[int, str], float]]:
    """Cache exact baseline LogLoss used by repeated research-only ablations.

    The cache is derived only from already materialized chronological OOS
    predictions. It never creates predictions, fits models, or changes labels.
    Situation entries preserve the existing minimum-size and binary-label
    eligibility conditions used by the online-ensemble evaluation.
    """
    fold_logloss: dict[int, float] = {}
    situation_logloss: dict[tuple[int, str], float] = {}

    for raw_fold_idx, bank in online_prediction_by_fold.items():
        fold_idx = int(raw_fold_idx)
        predictions = bank.get("predictions", {})
        baseline = predictions.get(selected_model) if hasattr(predictions, "get") else None
        if baseline is None:
            continue

        y = np.asarray(bank.get("y"), dtype=int)
        if y.ndim != 1 or len(y) == 0:
            continue

        fold_logloss[fold_idx] = float(classification_metrics(y, baseline)["logloss"])

        situations = bank.get("situations")
        if situations is None:
            continue
        situations_arr = np.asarray(situations, dtype=str)
        if situations_arr.ndim != 1 or len(situations_arr) != len(y):
            raise ValueError(
                f"baseline situation cache alignment mismatch in fold {fold_idx}"
            )

        for situation_name in sorted(set(situations_arr.tolist())):
            mask = situations_arr == situation_name
            if mask.sum() < 30 or len(np.unique(y[mask])) < 2:
                continue
            situation_logloss[(fold_idx, str(situation_name))] = float(
                classification_metrics(y[mask], baseline[mask])["logloss"]
            )

    return fold_logloss, situation_logloss
