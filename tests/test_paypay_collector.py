from __future__ import annotations

from src.data.paypay_collector import parse_rows


def test_paypay_parser_keeps_only_tradeable_rows():
    html=b"""
    <table>
      <tr><td>7203</td><td>トヨタ</td><td>trade_on</td></tr>
      <tr><td>1306</td><td>TOPIX ETF</td><td>trade_off</td></tr>
      <tr><td>1475</td><td>iShares ETF</td><td>mini_on</td></tr>
    </table>
    """
    rows=parse_rows(html,"japan","https://www.paypay-sec.co.jp/stock/list/")
    assert {(r["symbol"],r["asset_class"]) for r in rows} == {
        ("7203","jp_stock"),
        ("1475","jp_etf"),
    }


def test_paypay_parser_detects_us_ticker():
    html=b"""
    <table>
      <tr><td>BRK.B</td><td>Berkshire ETF</td><td>trade_on</td></tr>
      <tr><td>AAPL</td><td>Apple</td><td>mini_on</td></tr>
    </table>
    """
    rows=parse_rows(html,"us","https://www.paypay-sec.co.jp/us-stock/list/")
    symbols={r["symbol"] for r in rows}
    assert symbols == {"BRK.B","AAPL"}
