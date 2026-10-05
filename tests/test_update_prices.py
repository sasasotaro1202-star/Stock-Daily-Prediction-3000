from __future__ import annotations

import pytest

from scripts.update_prices import validate_unique_universe_records


def test_duplicate_universe_keys_are_collapsed_when_identical():
    records = [
        {
            "asset_class": "us_stock",
            "symbol": "ABC",
            "name": "ABC Corp",
        },
        {
            "asset_class": "us_stock",
            "symbol": "ABC",
            "name": "ABC Corp",
        },
        {
            "asset_class": "jp_stock",
            "symbol": "1234",
            "name": "Example",
        },
    ]

    result = validate_unique_universe_records(records)

    assert [(r["asset_class"], r["symbol"]) for r in result] == [
        ("us_stock", "ABC"),
        ("jp_stock", "1234"),
    ]


def test_conflicting_duplicate_universe_keys_fail_closed():
    records = [
        {
            "asset_class": "us_stock",
            "symbol": "ABC",
            "name": "ABC Corp",
        },
        {
            "asset_class": "us_stock",
            "symbol": "ABC",
            "name": "ABC Holdings",
        },
    ]

    with pytest.raises(SystemExit, match="conflicting duplicate universe key"):
        validate_unique_universe_records(records)


def test_missing_universe_price_key_fails_closed():
    records = [
        {
            "asset_class": "us_stock",
            "name": "Missing ticker",
        }
    ]

    with pytest.raises(SystemExit, match="universe record missing price key"):
        validate_unique_universe_records(records)
