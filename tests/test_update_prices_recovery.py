from __future__ import annotations

import pandas as pd

from scripts import update_prices


def _records(n: int) -> list[dict]:
    return [{"asset_class": "us_stock", "symbol": f"S{i}"} for i in range(n)]


def _frame(records):
    return pd.DataFrame({
        "asset_class": [r["asset_class"] for r in records],
        "symbol": [r["symbol"] for r in records],
    })


def test_split_recovery_is_bounded_and_recovers_good_halves():
    calls = []

    def fake_download(records, period="5y"):
        calls.append(len(records))
        if len(records) > 10:
            raise RuntimeError("bulk failure")
        return _frame(records)

    result = update_prices.fetch_resilient(
        _records(20),
        "5y",
        downloader=fake_download,
        sleep_fn=lambda _: None,
    )

    assert len(result) == 20
    assert calls[:3] == [20, 20, 20]
    assert all(size <= 10 for size in calls[3:])


def test_split_recovery_defers_only_poisoned_leaf():
    calls = []

    def fake_download(records, period="5y"):
        calls.append([r["symbol"] for r in records])
        if any(r["symbol"] == "S0" for r in records):
            raise RuntimeError("poisoned leaf")
        return _frame(records)

    result = update_prices.fetch_resilient(
        _records(20),
        "5y",
        downloader=fake_download,
        sleep_fn=lambda _: None,
    )

    assert len(result) == 15
    assert "S0" not in set(result["symbol"])
    assert max(len(batch) for batch in calls) == 20
    assert any(len(batch) == 5 for batch in calls)
