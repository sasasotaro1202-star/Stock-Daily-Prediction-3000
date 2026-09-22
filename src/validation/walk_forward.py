from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Fold:
    train_end:int
    test_start:int
    test_end:int

def make_folds(
    n_samples: int,
    *,
    min_train: int = 504,
    test_size: int = 63,
    step: int = 63,
    embargo: int = 1,
    purge: int = 0,
) -> list[Fold]:
    if min_train <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("fold sizes must be positive")
    if embargo < 0 or purge < 0:
        raise ValueError("embargo and purge must be non-negative")

    folds = []
    boundary_end = min_train
    while boundary_end + embargo + test_size <= n_samples:
        # Purge training observations whose labels can overlap the first
        # test observation, then apply an additional post-boundary embargo.
        train_end = boundary_end - purge
        test_start = boundary_end + embargo
        test_end = test_start + test_size
        if train_end >= min_train:
            folds.append(Fold(train_end, test_start, test_end))
        boundary_end += step
    return folds


def make_date_folds(
    dates: list,
    *,
    min_train: int = 252,
    test_size: int = 21,
    step: int = 21,
    embargo: int = 1,
    purge: int = 0,
) -> list[Fold]:
    return make_folds(
        len(dates),
        min_train=min_train,
        test_size=test_size,
        step=step,
        embargo=embargo,
        purge=purge,
    )

def assert_chronological(frame:pd.DataFrame,time_col:str="session_date")->None:
    if time_col not in frame: raise ValueError(f"missing {time_col}")
    values=pd.to_datetime(frame[time_col],errors="coerce")
    if values.isna().any() or not values.is_monotonic_increasing:
        raise ValueError("time axis must be valid and monotonically increasing")
