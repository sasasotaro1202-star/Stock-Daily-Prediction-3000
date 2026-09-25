from __future__ import annotations

import numpy as np
import pytest

from src.research.online_baseline_cache import build_baseline_logloss_cache


def _metric(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    eps = 1e-6
    p = np.clip(p, eps, 1.0 - eps)
    return {
        "logloss": float(np.mean(
            -(y * np.log(p) + (1 - y) * np.log(1 - p))
        ))
    }


def test_baseline_cache_matches_direct_fold_and_situation_logloss():
    bank = {
        0: {
            "y": np.asarray([0, 1] * 20),
            "predictions": {
                "model": np.asarray([0.2, 0.8] * 20),
                "other": np.asarray([0.4, 0.6] * 20),
            },
            "situations": np.asarray(["normal"] * 20 + ["event"] * 20),
        }
    }
    fold_cache, situation_cache = build_baseline_logloss_cache(
        bank, "model", _metric
    )

    y = bank[0]["y"]
    p = bank[0]["predictions"]["model"]
    assert fold_cache[0] == pytest.approx(_metric(y, p)["logloss"])
    assert situation_cache[(0, "normal")] == pytest.approx(
        _metric(y[:20], p[:20])["logloss"]
    )
    assert situation_cache[(0, "event")] == pytest.approx(
        _metric(y[20:], p[20:])["logloss"]
    )


def test_baseline_cache_skips_missing_or_ineligible_situations():
    bank = {
        0: {
            "y": np.asarray([0, 1] * 15 + [0]),
            "predictions": {"model": np.asarray([0.2, 0.8] * 15 + [0.3])},
            "situations": np.asarray(["normal"] * 30 + ["tiny"]),
        }
    }
    fold_cache, situation_cache = build_baseline_logloss_cache(
        bank, "model", _metric
    )
    assert 0 in fold_cache
    assert (0, "normal") in situation_cache
    assert (0, "tiny") not in situation_cache


def test_baseline_cache_rejects_misaligned_situations():
    bank = {
        0: {
            "y": np.asarray([0, 1] * 20),
            "predictions": {"model": np.asarray([0.2, 0.8] * 20)},
            "situations": np.asarray(["normal"] * 39),
        }
    }
    with pytest.raises(ValueError, match="alignment mismatch"):
        build_baseline_logloss_cache(bank, "model", _metric)
