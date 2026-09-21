from __future__ import annotations

import pandas as pd


def cross_sectional_rank(
    df: pd.DataFrame,
    probability_col: str = "p_up_1d",
    return_col: str = "expected_return_1d",
) -> pd.DataFrame:
    out = df.copy()
    if probability_col not in out or return_col not in out:
        raise ValueError("required prediction columns are missing")

    group_cols = ["prediction_date"]
    if "asset_class" in out.columns:
        # Japan and U.S. securities do not share the same market clock.
        # Rank within the product/session family to avoid mixing asynchronous snapshots.
        group_cols.append("asset_class")

    out["rank_probability"] = out.groupby(group_cols)[probability_col].rank(
        method="average", ascending=False, pct=True
    )
    out["rank_expected_return"] = out.groupby(group_cols)[return_col].rank(
        method="average", ascending=False, pct=True
    )
    out["rank_score"] = 0.5 * out["rank_probability"] + 0.5 * out["rank_expected_return"]
    return out.sort_values(group_cols + ["rank_score"], ascending=[True] * len(group_cols) + [False])
