from __future__ import annotations

import pandas as pd

from scripts.update_prices import prune_price_batch_to_current_universe


def test_prune_removes_symbols_moved_out_of_positional_batch(tmp_path):
    path=tmp_path/"batch_000.parquet"
    df=pd.DataFrame(
        {
            "asset_class":["JP","JP","US"],
            "symbol":["A","B","C"],
            "session_date":["2026-09-23","2026-09-23","2026-09-23"],
            "close":[100.0,200.0,300.0],
        }
    )
    df.to_parquet(path,index=False)

    removed=prune_price_batch_to_current_universe(
        path,
        {("JP","A"),("US","C")},
    )

    assert removed == 1
    kept=pd.read_parquet(path)
    assert set(zip(kept["asset_class"],kept["symbol"])) == {
        ("JP","A"),
        ("US","C"),
    }


def test_prune_is_noop_when_batch_matches_current_universe(tmp_path):
    path=tmp_path/"batch_000.parquet"
    df=pd.DataFrame(
        {
            "asset_class":["JP","US"],
            "symbol":["A","C"],
            "session_date":["2026-09-23","2026-09-23"],
            "close":[100.0,300.0],
        }
    )
    df.to_parquet(path,index=False)

    removed=prune_price_batch_to_current_universe(
        path,
        {("JP","A"),("US","C")},
    )

    assert removed == 0
    assert len(pd.read_parquet(path)) == 2
