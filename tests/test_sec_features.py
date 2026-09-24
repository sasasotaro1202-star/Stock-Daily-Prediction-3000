import pandas as pd

from src.research.sec_features import SEC_FEATURE_COLUMNS, add_sec_filing_features


def test_sec_features_respect_acceptance_time_and_windows():
    prices = pd.DataFrame(
        {
            "symbol": ["ABC"] * 3 + ["JP"],
            "asset_class": ["us_stock"] * 3 + ["jp_stock"],
            "available_at": pd.to_datetime(
                [
                    "2026-01-01T20:00:00Z",
                    "2026-01-10T20:00:00Z",
                    "2026-03-15T20:00:00Z",
                    "2026-03-15T20:00:00Z",
                ],
                utc=True,
            ),
        }
    )
    filings = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC", "ABC", "ABC"],
            "asset_class": ["us_stock"] * 4,
            "available_at": pd.to_datetime(
                [
                    "2026-01-05T15:00:00Z",
                    "2026-01-20T15:00:00Z",
                    "2026-03-10T15:00:00Z",
                    "2026-03-20T15:00:00Z",
                ],
                utc=True,
            ),
            "form": ["8-K", "10-Q", "DEF 14A", "10-K"],
        }
    )

    out = add_sec_filing_features(prices, filings)

    assert set(SEC_FEATURE_COLUMNS).issubset(out.columns)
    assert float(out.loc[0, "sec_data_available"]) == 0.0
    assert float(out.loc[1, "sec_data_available"]) == 1.0
    assert float(out.loc[1, "sec_8k_30d"]) == 1.0
    assert float(out.loc[2, "sec_filings_90d"]) == 3.0
    assert float(out.loc[2, "sec_proxy_90d"]) == 1.0
    # The filing accepted after the prediction timestamp must not leak backward.
    assert float(out.loc[2, "sec_10k_365d"]) == 0.0
    # Non-US assets remain explicitly neutral rather than inheriting another symbol.
    assert float(out.loc[3, "sec_data_available"]) == 0.0


def test_sec_feature_recency_is_monotone_until_next_filing():
    prices = pd.DataFrame(
        {
            "symbol": ["ABC"] * 2,
            "asset_class": ["us_stock"] * 2,
            "available_at": pd.to_datetime(
                ["2026-02-01T20:00:00Z", "2026-02-10T20:00:00Z"],
                utc=True,
            ),
        }
    )
    filings = pd.DataFrame(
        {
            "symbol": ["ABC"],
            "asset_class": ["us_stock"],
            "available_at": pd.to_datetime(["2026-01-15T15:00:00Z"], utc=True),
            "form": ["10-Q"],
        }
    )
    out = add_sec_filing_features(prices, filings)
    assert float(out.loc[1, "sec_days_since_filing"]) > float(
        out.loc[0, "sec_days_since_filing"]
    )


def test_sec_features_preserve_original_row_alignment_when_prices_are_unsorted():
    prices = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC", "XYZ", "ABC"],
            "asset_class": ["us_stock"] * 4,
            "available_at": pd.to_datetime(
                [
                    "2026-03-15T20:00:00Z",
                    "2026-01-10T20:00:00Z",
                    "2026-03-15T20:00:00Z",
                    "2026-01-01T20:00:00Z",
                ],
                utc=True,
            ),
        }
    )
    filings = pd.DataFrame(
        {
            "symbol": ["ABC", "XYZ"],
            "asset_class": ["us_stock", "us_stock"],
            "available_at": pd.to_datetime(
                ["2026-01-05T15:00:00Z", "2026-03-14T15:00:00Z"],
                utc=True,
            ),
            "form": ["8-K", "10-Q"],
        }
    )
    out = add_sec_filing_features(prices, filings)
    # Original row 0 is the late ABC observation and must see the filing;
    # original row 3 predates it and must remain unavailable.
    assert float(out.loc[0, "sec_data_available"]) == 1.0
    assert float(out.loc[0, "sec_8k_30d"]) == 1.0
    assert float(out.loc[1, "sec_data_available"]) == 1.0
    assert float(out.loc[2, "sec_data_available"]) == 1.0
    assert float(out.loc[3, "sec_data_available"]) == 0.0
