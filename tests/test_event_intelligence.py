from __future__ import annotations

import importlib

import pandas as pd


def _events() -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "asset_class": "us_stock",
                "available_at": "2026-01-02T10:00:00Z",
                "form": "8-K",
                "accession_number": "a1",
                "primary_document": "a1.htm",
                "source": "sec",
            },
            {
                "symbol": "AAA",
                "asset_class": "us_stock",
                "available_at": "2026-01-10T10:00:00Z",
                "form": "10-Q",
                "accession_number": "a2",
                "primary_document": "a2.htm",
                "source": "sec",
            },
            {
                "symbol": "AAA",
                "asset_class": "us_stock",
                "available_at": "2026-01-20T10:00:00Z",
                "form": "SC 13D",
                "accession_number": "a3",
                "primary_document": "a3.htm",
                "source": "sec",
            },
        ]
    )
    ledger, diag = importlib.import_module(
        "src.research.event_intelligence"
    ).build_event_ledger(raw)
    assert diag["status"] == "PASS"
    return ledger


def test_future_event_is_never_counted() -> None:
    mod = importlib.import_module("src.research.event_intelligence")
    prices = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "asset_class": "us_stock",
                "available_at": "2026-01-09T10:00:00Z",
            },
            {
                "symbol": "AAA",
                "asset_class": "us_stock",
                "available_at": "2026-01-10T10:00:00Z",
            },
        ]
    )
    out = mod.add_event_intelligence_features(prices, _events())
    assert out.loc[0, "event_count_30d"] == 1.0
    assert out.loc[0, "event_financial_report_180d"] == 0.0
    assert out.loc[1, "event_count_30d"] == 2.0
    assert out.loc[1, "event_financial_report_180d"] == 1.0


def test_same_timestamp_event_is_admitted() -> None:
    mod = importlib.import_module("src.research.event_intelligence")
    prices = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "asset_class": "us_stock",
                "available_at": "2026-01-10T10:00:00Z",
            }
        ]
    )
    out = mod.add_event_intelligence_features(prices, _events())
    assert out.loc[0, "event_count_30d"] == 2.0
    assert out.loc[0, "event_latest_is_financial_report"] == 1.0


def test_unknown_available_at_is_fail_closed() -> None:
    mod = importlib.import_module("src.research.event_intelligence")
    raw = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "asset_class": "us_stock",
                "available_at": "",
                "form": "8-K",
            }
        ]
    )
    ledger, diag = mod.build_event_ledger(raw)
    assert ledger.empty
    assert diag["status"] == "DEFERRED"
    assert diag["pit_invalid_rows"] == 1


def test_non_us_event_does_not_enter_us_features() -> None:
    mod = importlib.import_module("src.research.event_intelligence")
    raw = _events().copy()
    raw.loc[:, "asset_class"] = "non_us"
    prices = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "asset_class": "us_stock",
                "available_at": "2026-01-30T10:00:00Z",
            }
        ]
    )
    out = mod.add_event_intelligence_features(prices, raw)
    assert out.loc[0, "event_count_30d"] == 0.0
