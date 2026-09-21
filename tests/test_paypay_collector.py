from __future__ import annotations

from src.data.paypay_collector import parse_rows


def test_paypay_parser_keeps_only_tradeable_rows():
    html="""
    <h2>日本株 個別銘柄</h2>
    <table>
      <tr><td>7203</td><td>トヨタ</td><td>trade_on,mini_on,cfd_on</td></tr>
    </table>
    <h2>国内ETF（上場投資信託）</h2>
    <table>
      <tr><td>1306</td><td>TOPIX ETF</td><td>trade_off,cfd_off</td></tr>
      <tr><td>1475</td><td>iShares ETF</td><td>mini_on,cfd_off</td></tr>
    </table>
    """.encode("utf-8")
    rows=parse_rows(
        html,
        "japan",
        "https://www.paypay-sec.co.jp/stock/list/",
    )
    assert {(r["symbol"],r["asset_class"]) for r in rows} == {
        ("7203","jp_stock"),
        ("1475","jp_etf"),
    }


def test_paypay_parser_detects_us_ticker_and_section():
    html="""
    <h2>米国株（アルファベット順）</h2>
    <table>
      <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>trade_on,mini_on</td></tr>
    </table>
    <h2>米国ETF（アルファベット順）</h2>
    <table>
      <tr><td>SPY</td><td>SPDR S&amp;P 500 ETF</td><td>trade_on</td></tr>
    </table>
    """.encode("utf-8")
    rows=parse_rows(
        html,
        "us",
        "https://www.paypay-sec.co.jp/us-stock/list/",
    )
    assert {(r["symbol"],r["asset_class"]) for r in rows} == {
        ("BRK.B","us_stock"),
        ("SPY","us_etf"),
    }
