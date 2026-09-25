from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def circular_block_bootstrap_mean(
    values: Iterable[float],
    *,
    n_boot: int = 4000,
    seed: int = 20260925,
    block_length: int | None = None,
) -> tuple[float, float, int]:
    """Estimate P(mean > 0) and the 5th percentile with dependent blocks.

    Circular moving blocks preserve short-range serial dependence between
    adjacent OOS folds. The block length is chosen from the number of folds
    only, so no outcome values are used to tune it.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1 or len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError("values must be a non-empty finite 1-D sequence")
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100")

    n = len(arr)
    if block_length is None:
        block_length = max(2, min(4, int(round(math.sqrt(n)))))
    block_length = int(block_length)
    if block_length < 1 or block_length > n:
        raise ValueError("block_length must be in [1, n]")

    blocks = int(math.ceil(n / block_length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, blocks))
    offsets = np.arange(block_length, dtype=int)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    indices = indices.reshape(n_boot, -1)[:, :n]
    means = arr[indices].mean(axis=1)
    return (
        float(np.mean(means > 0.0)),
        float(np.quantile(means, 0.05)),
        block_length,
    )
