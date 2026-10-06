from __future__ import annotations

import math

import numpy as np


def moving_block_bootstrap_mean(
    values,
    *,
    n_bootstrap: int = 4000,
    block_length: int | None = None,
    seed: int = 20260925,
) -> tuple[float, float]:
    """Return P(mean > 0) and the 5th percentile of a moving-block bootstrap.

    Observations are assumed to be ordered chronologically. Contiguous blocks
    are sampled with replacement and concatenated until the original sample
    length is reached. The default block length follows the conservative
    cube-root heuristic and is capped to the sample size.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) < 2 or not np.isfinite(arr).all():
        return 0.0, float("-inf")

    n = len(arr)
    if block_length is None:
        block = max(1, int(math.ceil(n ** (1.0 / 3.0))))
    else:
        block = int(block_length)
    block = max(1, min(block, n))

    reps = max(100, int(n_bootstrap))
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, n - block + 1, size=(reps, int(math.ceil(n / block))))
    samples = np.empty((reps, n), dtype=float)

    for b_idx in range(starts.shape[1]):
        lo = b_idx * block
        hi = min((b_idx + 1) * block, n)
        take = hi - lo
        if take <= 0:
            break
        idx = starts[:, b_idx, None] + np.arange(take)[None, :]
        samples[:, lo:hi] = arr[idx]

    means = samples.mean(axis=1)
    return float(np.mean(means > 0.0)), float(np.quantile(means, 0.05))


__all__ = ["moving_block_bootstrap_mean"]
