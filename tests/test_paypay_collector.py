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


def test_reader_fallback_parse_stays_tradeable_and_section_aware():
    markdown = """
# 日本株 個別銘柄
| コード | 銘柄 | 取扱いアプリ |
| 7203 | トヨタ | trade_on,mini_on,cfd_on |
# 国内ETF（上場投資信託）
| 1306 | TOPIX ETF | trade_off,cfd_off |
| 1475 | iShares ETF | trade_on,cfd_off |
# REIT（不動産投資信託）
| 8951 | 日本ビルファンド投資法人 | trade_on,mini_on,cfd_off |
"""
    from src.data.paypay_collector import parse_reader_text

    rows = parse_reader_text(
        markdown.encode("utf-8"),
        "japan",
        "https://www.paypay-sec.co.jp/stock/list/",
    )
    pairs = {(r["symbol"], r["asset_class"]) for r in rows}
    assert ("7203", "jp_stock") in pairs
    assert ("1475", "jp_etf") in pairs
    assert ("8951", "jp_reit") in pairs
    assert ("1306", "jp_etf") not in pairs


def test_browser_dump_dom_restricted_to_official_paypay(monkeypatch):
    import src.data.paypay_collector as collector

    class Result:
        returncode = 0
        stdout = b"<html>" + (b"x" * 10000) + b"</html>"
        stderr = b""

    calls = []

    monkeypatch.setattr(collector.shutil, "which", lambda name: "/bin/" + name)
    monkeypatch.setattr(
        collector.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or Result(),
    )

    body = collector._browser_dump_dom(
        "https://www.paypay-sec.co.jp/stock/list/"
    )
    assert len(body) > 10000
    assert calls
    assert "--dump-dom" in calls[0][0]
    assert calls[0][0][-1] == "https://www.paypay-sec.co.jp/stock/list/"


def test_browser_dump_dom_rejects_non_paypay_url():
    import pytest
    from src.data.paypay_collector import _browser_dump_dom

    with pytest.raises(RuntimeError, match="PayPay official hosts"):
        _browser_dump_dom("https://example.com/")


from src.data import paypay_collector


def test_short_us_name_collision_is_rejected_when_official_resource_is_missing(monkeypatch):
    monkeypatch.setattr(
        paypay_collector,
        "_official_us_symbol_resource_exists",
        lambda _symbol: False,
    )
    raw = b"""
    <div>米国株</div>
    <div>DR</div>
    <div>D.R.ホートン</div>
    <div>trade_on</div>
    """
    rows = paypay_collector.parse_visible_text(raw, "us", "https://example.test")
    assert rows == []


def test_short_us_real_symbol_is_kept_when_official_resource_exists(monkeypatch):
    monkeypatch.setattr(
        paypay_collector,
        "_official_us_symbol_resource_exists",
        lambda symbol: symbol == "IBM",
    )
    raw = b"""
    <div>米国株</div>
    <div>IBM</div>
    <div>IBM</div>
    <div>trade_on</div>
    """
    rows = paypay_collector.parse_visible_text(raw, "us", "https://example.test")
    assert [(row["symbol"], row["asset_class"]) for row in rows] == [("IBM", "us_stock")]


def test_non_ticker_us_labels_are_not_treated_as_symbols():
    raw = """
    <div>米国株</div>
    <div>NYRS</div>
    <div>ASML</div>
    <div>trade_on</div>
    """.encode()
    rows = paypay_collector.parse_visible_text(raw, "us", "https://example.test")
    assert all(row["symbol"] != "NYRS" for row in rows)
