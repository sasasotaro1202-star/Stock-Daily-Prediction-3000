from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import pandas as pd

@dataclass(frozen=True)
class LeakageAudit:
    ok: bool
    violations: tuple[str,...]

FORBIDDEN=("target_","future_","fwd_","label_","next_close","next_open","lead_","shift_-")

def audit_feature_columns(columns:Iterable[str])->LeakageAudit:
    v=[f"forbidden_feature_name:{c}" for c in columns if any(t in c.lower() for t in FORBIDDEN)]
    return LeakageAudit(not v,tuple(sorted(v)))

def audit_available_at(df:pd.DataFrame,prediction_time_col:str="prediction_time")->LeakageAudit:
    req={"available_at",prediction_time_col}
    missing=req-set(df.columns)
    if missing:return LeakageAudit(False,(f"missing:{sorted(missing)}",))
    a=pd.to_datetime(df["available_at"],utc=True,errors="coerce")
    p=pd.to_datetime(df[prediction_time_col],utc=True,errors="coerce")
    bad=a.isna()|p.isna()|(a>p)
    n=int(bad.sum())
    return LeakageAudit(n==0,(f"available_after_prediction:{n}",) if n else ())

def audit_target_separation(feature_columns:Iterable[str],target_columns:Iterable[str])->LeakageAudit:
    overlap=sorted(set(feature_columns)&set(target_columns))
    return LeakageAudit(not overlap,tuple(f"feature_target_overlap:{x}" for x in overlap))


def audit_retrieval_provenance(df):
    import pandas as pd

    required = {"available_at", "retrieved_at"}
    missing = sorted(required - set(df.columns))
    if missing:
        return {"ok": False, "violations": [f"missing_{c}" for c in missing]}
    available = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
    retrieved = pd.to_datetime(df["retrieved_at"], utc=True, errors="coerce")
    invalid = int(
        available.isna().sum()
        + retrieved.isna().sum()
    )
    impossible = int(available.gt(retrieved).fillna(False).sum())
    return {
        "ok": invalid == 0 and impossible == 0,
        "violations": [
            *([f"missing_retrieval_provenance:{invalid}"] if invalid else []),
            *([f"available_after_retrieval:{impossible}"] if impossible else []),
        ],
    }
