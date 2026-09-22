from __future__ import annotations

import pandas as pd


def cross_sectional_rank(
    df: pd.DataFrame,
    probability_col: str = "p_up_1d",
    return_col: str = "expected_return_1d",
    probability_weight: float = 0.5,
) -> pd.DataFrame:
    out = df.copy()
    if probability_col not in out or return_col not in out:
        raise ValueError("required prediction columns are missing")
    weight = min(1.0, max(0.0, float(probability_weight)))

    group_cols = ["prediction_date"]
    if "asset_class" in out.columns:
        group_cols.append("asset_class")

    out["rank_probability"] = out.groupby(group_cols)[probability_col].rank(
        method="average", ascending=False, pct=True
    )
    out["rank_expected_return"] = out.groupby(group_cols)[return_col].rank(
        method="average", ascending=False, pct=True
    )
    out["rank_probability_weight"] = weight
    out["rank_expected_return_weight"] = 1.0 - weight
    out["rank_score"] = (
        weight * out["rank_probability"]
        + (1.0 - weight) * out["rank_expected_return"]
    )
    return out.sort_values(
        group_cols + ["rank_score"],
        ascending=[True] * len(group_cols) + [False],
    )
