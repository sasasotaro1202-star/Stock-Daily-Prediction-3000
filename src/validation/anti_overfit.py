from __future__ import annotations

from math import isfinite, log, sqrt

import numpy as np
import pandas as pd


DEFAULT_PERIODS_PER_YEAR = 252
DEFAULT_BLOCK_SIZE = 5
DEFAULT_PERMUTATIONS = 1000
DEFAULT_SEED = 42


def annualized_sharpe(
    returns: pd.Series | np.ndarray,
    *,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    std = float(np.std(values, ddof=1))
    if std <= 0.0:
        return float("nan")
    return float(np.mean(values) / std * sqrt(periods_per_year))


def sign_flip_mcpt_pvalue(
    returns: pd.Series | np.ndarray,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> float:
    """
    Two-sided block sign-flip Monte Carlo permutation test for zero mean.

    Blocks preserve short-range dependence inside each block while the null
    randomizes the sign of each block. This is a conservative diagnostic,
    not a replacement for PIT/OOS validation.
    """
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < max(20, block_size * 2):
        return float("nan")
    block_size = max(1, int(block_size))
    permutations = max(100, int(permutations))

    usable = len(values) - (len(values) % block_size)
    blocks = values[:usable].reshape(-1, block_size)
    observed = abs(float(values.mean()))

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(blocks))
        simulated = (blocks * signs[:, None]).reshape(-1)
        if abs(float(simulated.mean())) >= observed:
            hits += 1
    return float((hits + 1) / (permutations + 1))


def multiple_trial_sharpe_adjustment(
    sharpe: float,
    observations: int,
    trials: int,
) -> dict[str, float]:
    """
    Conservative selection-luck adjustment.

    This is intentionally not labelled Deflated Sharpe Ratio: it subtracts a
    normal-order-statistics selection penalty from the Sharpe standard error.
    It is a cheap diagnostic for how much an observed Sharpe may be explained
    by trying many configurations.
    """
    if not isfinite(float(sharpe)) or observations < 20 or trials < 1:
        return {
            "selection_penalty": float("nan"),
            "selection_adjusted_sharpe": float("nan"),
        }
    sr = float(sharpe)
    t = float(observations)
    trial_n = max(1, int(trials))
    se = sqrt(max(1e-12, (1.0 + 0.5 * sr * sr) / t))
    tail = max(1.0 / max(2, trial_n), 1e-12)
    # Normal quantile approximation without requiring scipy.
    # Acklam-style approximation is more machinery than needed here; use a
    # stable inverse-error-function approximation instead.
    z = sqrt(2.0) * _inverse_erf(1.0 - 2.0 * tail)
    penalty = max(0.0, z * se)
    return {
        "selection_penalty": float(penalty),
        "selection_adjusted_sharpe": float(sr - penalty),
    }


def _inverse_erf(x: float) -> float:
    # Winitzki approximation; adequate for a conservative screening metric.
    a = 0.147
    sign = 1.0 if x >= 0 else -1.0
    x = min(0.999999999, max(-0.999999999, abs(float(x))))
    ln = log(1.0 - x * x)
    first = 2.0 / (np.pi * a) + ln / 2.0
    second = ln / a
    return float(sign * sqrt(max(0.0, sqrt(first * first - second) - first)))


def per_year_robustness(
    dates: pd.Series | np.ndarray,
    returns: pd.Series | np.ndarray,
) -> dict:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "return": np.asarray(returns, dtype=float),
        }
    ).dropna()
    if frame.empty:
        return {
            "years": 0,
            "positive_year_fraction": float("nan"),
            "yearly_mean_returns": {},
        }
    frame["year"] = frame["date"].dt.year.astype(int)
    yearly = frame.groupby("year")["return"].mean().sort_index()
    positive_fraction = float((yearly > 0.0).mean())
    return {
        "years": int(len(yearly)),
        "positive_year_fraction": positive_fraction,
        "yearly_mean_returns": {
            str(year): float(value) for year, value in yearly.items()
        },
    }


def anti_overfit_battery(
    dates: pd.Series | np.ndarray,
    returns: pd.Series | np.ndarray,
    *,
    trials: int = 1,
    min_years: int = 3,
    min_positive_year_fraction: float = 0.60,
    max_mcpt_pvalue: float = 0.05,
    block_size: int = DEFAULT_BLOCK_SIZE,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> dict:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "return": np.asarray(returns, dtype=float),
        }
    ).dropna()
    if len(frame) < 20:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "rows": int(len(frame)),
            "reason": "too_few_returns",
        }

    sharpe = annualized_sharpe(
        frame["return"],
        periods_per_year=periods_per_year,
    )
    mcpt = sign_flip_mcpt_pvalue(
        frame["return"],
        block_size=block_size,
        permutations=permutations,
        seed=seed,
    )
    adjustment = multiple_trial_sharpe_adjustment(
        sharpe,
        len(frame),
        trials,
    )
    yearly = per_year_robustness(frame["date"], frame["return"])

    evidence = {
        "mcpt_pass": bool(
            isfinite(mcpt) and mcpt <= float(max_mcpt_pvalue)
        ),
        "selection_adjusted_sharpe_pass": bool(
            isfinite(adjustment["selection_adjusted_sharpe"])
            and adjustment["selection_adjusted_sharpe"] > 0.0
        ),
        "per_year_pass": bool(
            yearly["years"] >= int(min_years)
            and yearly["positive_year_fraction"]
            >= float(min_positive_year_fraction)
        ),
    }
    status = (
        "PASS"
        if all(evidence.values())
        else "INSUFFICIENT_EVIDENCE"
        if yearly["years"] < int(min_years)
        else "FAIL"
    )
    return {
        "status": status,
        "rows": int(len(frame)),
        "annualized_sharpe": float(sharpe),
        "mcpt_pvalue": float(mcpt),
        "mcpt_block_size": int(block_size),
        "mcpt_permutations": int(permutations),
        "trial_count": int(max(1, trials)),
        **adjustment,
        **yearly,
        "thresholds": {
            "min_years": int(min_years),
            "min_positive_year_fraction": float(min_positive_year_fraction),
            "max_mcpt_pvalue": float(max_mcpt_pvalue),
        },
        "evidence": evidence,
        "method": (
            "Capafy-inspired anti-overfit diagnostics: block sign-flip MCPT, "
            "multiple-trial Sharpe selection penalty, and per-year robustness. "
            "This layer does not replace PIT, chronological OOS, or frozen holdout."
        ),
    }
