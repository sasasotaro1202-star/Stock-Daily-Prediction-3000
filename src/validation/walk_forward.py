from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Fold:
    train_end:int
    test_start:int
    test_end:int

def make_folds(n_samples:int,*,min_train:int=504,test_size:int=63,step:int=63,embargo:int=1)->list[Fold]:
    folds=[]; train_end=min_train
    while train_end+embargo+test_size<=n_samples:
        test_start=train_end+embargo; test_end=test_start+test_size
        folds.append(Fold(train_end,test_start,test_end)); train_end+=step
    return folds

def make_date_folds(dates:list,*,min_train:int=252,test_size:int=21,step:int=21,embargo:int=1)->list[Fold]:
    return make_folds(len(dates),min_train=min_train,test_size=test_size,step=step,embargo=embargo)

def assert_chronological(frame:pd.DataFrame,time_col:str="session_date")->None:
    if time_col not in frame: raise ValueError(f"missing {time_col}")
    values=pd.to_datetime(frame[time_col],errors="coerce")
    if values.isna().any() or not values.is_monotonic_increasing:
        raise ValueError("time axis must be valid and monotonically increasing")
