from __future__ import annotations

import pandas as pd

from src.ranking.cross_sectional import cross_sectional_rank


def test_cross_sectional_rank_exposes_canonical_one_based_rank():
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "asset_class": ["jp_stock", "jp_stock", "jp_stock"],
            "prediction_date": ["2026-10-08", "2026-10-08", "2026-10-08"],
            "p_up_1d": [0.8, 0.6, 0.4],
            "expected_return_1d": [0.02, 0.01, 0.00],
            "ranking_uncertainty": [0.01, 0.02, 0.03],
        }
    )
    out = cross_sectional_rank(
        frame,
        probability_weight=0.5,
        uncertainty_col="ranking_uncertainty",
    ).sort_values("rank")

    assert out["rank"].tolist() == [1.0, 2.0, 3.0]
