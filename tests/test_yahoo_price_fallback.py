from __future__ import annotations

import pandas as pd

from src.data import yahoo_price


def _bars(start: str, periods: int = 3) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="D", name="Date")
    return pd.DataFrame(
        {
            "Open": [10.0 + i for i in range(periods)],
            "High": [11.0 + i for i in range(periods)],
            "Low": [9.0 + i for i in range(periods)],
            "Close": [10.5 + i for i in range(periods)],
            "Adj Close": [10.5 + i for i in range(periods)],
            "Volume": [1000.0 + i for i in range(periods)],
        },
        index=idx,
    )


def test_download_batch_recovers_ticker_omitted_by_multiticker_download(monkeypatch):
    batch = _bars("2026-01-01")
    single = _bars("2026-01-01", periods=4)

    def fake_download(symbols, **_kwargs):
        assert symbols == ["1111.T", "2222.T"]
        return pd.concat({"1111.T": batch}, axis=1)

    class FakeTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol

        def history(self, **_kwargs):
            assert self.symbol == "2222.T"
            return single

    monkeypatch.setattr(yahoo_price.yf, "download", fake_download)
    monkeypatch.setattr(yahoo_price.yf, "Ticker", FakeTicker)

    records = [
        {"symbol": "1111", "asset_class": "jp_stock"},
        {"symbol": "2222", "asset_class": "jp_stock"},
    ]
    out = yahoo_price.download_batch(records, period="5y")

    assert set(zip(out["asset_class"], out["symbol"])) == {
        ("jp_stock", "1111"),
        ("jp_stock", "2222"),
    }
    assert (out["symbol"] == "2222").sum() == 4
