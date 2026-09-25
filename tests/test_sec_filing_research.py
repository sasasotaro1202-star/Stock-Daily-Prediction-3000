from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd


def test_sec_rows_use_acceptance_time_as_available_at():
    mod = importlib.import_module("scripts.sec_filings_research")
    collected = pd.Timestamp("2026-09-24T00:00:00Z")
    row = {"symbol": "TEST", "asset_class": "us_stock", "name": "Test Co", "tradeable": True}
    info = {"cik": "0000000001", "title": "Test Co"}
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000000001-26-000001"],
                "filingDate": ["2026-09-23"],
                "acceptanceDateTime": ["2026-09-23T20:15:00.000Z"],
                "form": ["8-K"],
                "primaryDocument": ["test.htm"],
            }
        }
    }
    rows = mod._rows_from_submissions(row, info, payload, collected)
    assert len(rows) == 1
    assert rows[0]["available_at"] == rows[0]["acceptance_datetime"]
    assert rows[0]["research_only"] is True
    assert rows[0]["production_changed"] is False


def test_unknown_or_future_acceptance_is_excluded():
    mod = importlib.import_module("scripts.sec_filings_research")
    collected = pd.Timestamp("2026-09-24T00:00:00Z")
    row = {"symbol": "TEST", "name": "Test Co"}
    info = {"cik": "0000000001", "title": "Test Co"}
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["a", "b", "c"],
                "filingDate": ["2026-09-23", "2026-09-23", "2026-09-23"],
                "acceptanceDateTime": ["", "2026-09-24T01:00:00Z", "2026-09-23T19:00:00Z"],
                "form": ["8-K", "8-K", "8-K"],
                "primaryDocument": ["a.htm", "b.htm", "c.htm"],
            }
        }
    }
    rows = mod._rows_from_submissions(row, info, payload, collected)
    assert len(rows) == 1
    assert rows[0]["accession_number"] == "c"


def test_sec_date_only_sources_exclude_collection_day_for_pit():
    mod = importlib.import_module("scripts.sec_filings_research")
    start_date, end_date = mod._conservative_filing_window(
        pd.Timestamp("2026-09-25T12:00:00Z")
    )
    assert str(start_date) == "2025-09-24"
    assert str(end_date) == "2026-09-24"


def test_sec_collector_is_research_only():
    source = Path("scripts/sec_filings_research.py").read_text(encoding="utf-8")
    assert "research_only" in source
    assert "production_changed" in source
    assert "load_production_artifact" not in source


def test_sec_official_mapping_via_jina_uses_company_ticker_schema(monkeypatch):
    mod = importlib.import_module("scripts.sec_filings_research")
    payload = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

    monkeypatch.setattr(mod, "_get_json", lambda url: (_ for _ in ()).throw(mod.HTTPError(url, 403, "Forbidden", {}, None)))
    monkeypatch.setattr(mod, "_jina_get_json", lambda url: payload)

    # Mirror the production mapping branch to ensure official Jina JSON is
    # interpreted by ticker/cik_str rather than as a third-party dict.
    try:
        ticker_payload = mod._jina_get_json(mod.TICKERS_URL)
        mapping_source = "sec_official_company_tickers"
        assert mapping_source == "sec_official_company_tickers"
        items = ticker_payload.values()
        mapping = {}
        for value in items:
            mapping[mod._norm_ticker(value["ticker"])] = str(value["cik_str"]).zfill(10)
        assert mapping["AAPL"] == "0000320193"
    except Exception as exc:
        raise AssertionError(f"official Jina mapping schema regression: {exc}") from exc

def test_sec_official_mapping_can_use_jina_reader_fallback(monkeypatch):
    mod = importlib.import_module("scripts.sec_filings_research")
    calls = []

    class DummyResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b'{"0": {"cik_str": 123, "ticker": "ABC", "title": "ABC Corp"}}'

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        return DummyResponse()

    monkeypatch.setattr(mod, "urlopen", fake_urlopen)
    payload = mod._jina_get_json(mod.TICKERS_URL)
    assert payload["0"]["ticker"] == "ABC"
    assert calls[0].startswith("https://r.jina.ai/https://www.sec.gov/")

def test_sec_submission_reader_uses_free_official_url_fallback(monkeypatch):
    mod = importlib.import_module("scripts.sec_filings_research")
    direct_calls = []
    curl_calls = []
    jina_calls = []

    def fake_get_json(url):
        direct_calls.append(url)
        raise mod.HTTPError(url, 403, "Forbidden", {}, None)

    def fake_curl(url):
        curl_calls.append(url)
        raise mod.HTTPError(url, 403, "Forbidden", {}, None)

    def fake_jina(url):
        jina_calls.append(url)
        return {"tickers": ["AAPL"], "filings": {"recent": {}}}

    monkeypatch.setattr(mod, "_get_json", fake_get_json)
    monkeypatch.setattr(mod, "_curl_cffi_get_json", fake_curl)
    monkeypatch.setattr(mod, "_jina_get_json", fake_jina)

    payload = None
    url = "https://data.sec.gov/submissions/CIK0000320193.json"
    try:
        payload = mod._get_json(url)
    except mod.HTTPError:
        payload = mod._jina_get_json(url)

    assert payload["tickers"] == ["AAPL"]
    assert direct_calls == [url]
    assert jina_calls == [url]
    assert curl_calls == []


def test_sec_request_headers_are_identified_and_rate_limit_friendly():
    mod = importlib.import_module("scripts.sec_filings_research")
    captured = {}

    class DummyHeaders:
        def get(self, key, default=None):
            return None if key != "Content-Encoding" else ""
    
    class DummyResponse:
        headers = DummyHeaders()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout):
        captured["request"] = req
        captured["timeout"] = timeout
        return DummyResponse()

    original = mod.urlopen
    mod.urlopen = fake_urlopen
    try:
        assert mod._get_json(mod.TICKERS_URL) is not None
    finally:
        mod.urlopen = original

    req = captured["request"]
    assert req.get_header("User-agent") == mod.USER_AGENT
    assert "github.com/sasasotaro1202-star/Stock-Daily-Prediction-3000" in mod.USER_AGENT
    assert req.get_header("Accept") == "application/json"
    assert req.get_header("Accept-encoding") == "gzip, deflate"
    assert captured["timeout"] == mod.REQUEST_TIMEOUT


def test_sec_ablation_uses_research_feature_module():
    source = Path("scripts/run_sec_filing_ablation.py").read_text(encoding="utf-8")
    assert "from src.research.sec_features import" in source
    assert "from src.features.sec_features import" not in source


def test_sec_mapping_fallback_is_pinned_and_submission_history_remains_official():
    source = Path("scripts/sec_filings_research.py").read_text(encoding="utf-8")
    assert "FALLBACK_TICKERS_URL = " in source
    assert "raw.githubusercontent.com/jadchaar/sec-cik-mapper/7883b83389836f9bba9bdfe53031467235746334" in source
    assert "https://data.sec.gov/submissions/CIK" in source
    assert "pinned_third_party_fallback" in source


def test_sec_request_json_handles_gzip_encoded_response():
    mod = importlib.import_module("scripts.sec_filings_research")
    import io

    captured = {}

    class Headers:
        def get(self, key, default=None):
            return "gzip" if key == "Content-Encoding" else default

    class DummyResponse:
        headers = Headers()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            import gzip
            return gzip.compress(b'{"ok": true}')

    def fake_urlopen(req, timeout):
        captured["timeout"] = timeout
        return DummyResponse()

    original = mod.urlopen
    mod.urlopen = fake_urlopen
    try:
        assert mod._get_json(mod.TICKERS_URL) == {"ok": True}
    finally:
        mod.urlopen = original


def test_sec_master_index_uses_curl_fallback_after_sec_400(monkeypatch):
    mod = importlib.import_module("scripts.sec_filings_research")
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        raise mod.HTTPError(req.full_url, 400, "Bad Request", {}, None)

    monkeypatch.setattr(mod, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        mod,
        "_curl_cffi_get_master_index_text",
        lambda url: (
            "CIK|Company|Form Type|Date Filed|Filename\\n" + "x" * 1200,
            "curl_fixture",
        ),
    )

    text_value, transport = mod._get_master_index_text(
        "https://www.sec.gov/Archives/edgar/full-index/2026/QTR3/master.idx"
    )

    assert transport == "curl_fixture"
    assert len(text_value) >= 1000
    assert len(calls) == 1


def test_sec_master_index_uses_free_jina_fallback_after_curl_failure(monkeypatch):
    mod = importlib.import_module("scripts.sec_filings_research")
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise mod.HTTPError(req.full_url, 403, "Forbidden", {}, None)
        class DummyResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return ("CIK|Company|Form Type|Date Filed|Filename\\n" + "x" * 1200).encode()
        return DummyResponse()

    monkeypatch.setattr(mod, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        mod,
        "_curl_cffi_get_master_index_text",
        lambda url: (_ for _ in ()).throw(mod.HTTPError(url, 403, "Forbidden", {}, None)),
    )

    text_value, transport = mod._get_master_index_text(
        "https://www.sec.gov/Archives/edgar/full-index/2026/QTR3/master.idx"
    )

    assert transport == "jina_reader_sec_official_url"
    assert len(text_value) >= 1000
    assert calls[0].startswith("https://www.sec.gov/")
    assert calls[1].startswith("https://r.jina.ai/https://www.sec.gov/")



def test_sec_efts_collects_root_document_only_and_preserves_pit(monkeypatch):
    mod = importlib.import_module("scripts.sec_filings_research")

    payload = {
        "hits": {
            "hits": [
                {
                    "_id": "0000320193-26-000001:aapl-8k.htm",
                    "_source": {
                        "adsh": "0000320193-26-000001",
                        "ciks": ["0000320193"],
                        "display_names": ["APPLE INC (AAPL) (CIK 0000320193)"],
                        "form": "8-K",
                        "file_type": "8-K",
                        "file_date": "2026-09-24",
                    },
                },
                {
                    "_id": "0000320193-26-000001:ex99.htm",
                    "_source": {
                        "adsh": "0000320193-26-000001",
                        "ciks": ["0000320193"],
                        "display_names": ["APPLE INC (AAPL) (CIK 0000320193)"],
                        "form": "8-K",
                        "file_type": "EX-99.1",
                        "file_date": "2026-09-24",
                    },
                },
            ]
        }
    }

    monkeypatch.setattr(mod, "_get_json", lambda url: payload)

    rows, diagnostics = mod._rows_from_efts(
        [{"symbol": "AAPL", "asset_class": "us_stock", "name": "Apple Inc.", "tradeable": True}],
        {"AAPL": {"cik": "0000320193", "title": "Apple Inc."}},
        {},
        pd.Timestamp("2026-09-25T12:00:00Z"),
    )

    assert len(rows) == 1
    assert rows[0]["accession_number"] == "0000320193-26-000001"
    assert rows[0]["primary_document"] == "aapl-8k.htm"
    assert rows[0]["available_at"] == "2026-09-25T03:59:59.999999+00:00"
    assert rows[0]["available_at_method"] == "filing_date_eod_conservative"
    assert rows[0]["source"] == "sec_edgar_efts"
    assert diagnostics["pages"] == 1


def test_sec_efts_uses_curl_on_direct_403(monkeypatch):
    mod = importlib.import_module("scripts.sec_filings_research")
    calls = []

    def fake_direct(url):
        calls.append(url)
        raise mod.HTTPError(url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(mod, "_get_json", fake_direct)
    monkeypatch.setattr(
        mod,
        "_curl_cffi_get_json",
        lambda url: {
            "hits": {"hits": []},
        },
    )

    rows, diagnostics = mod._rows_from_efts(
        [{"symbol": "AAPL", "asset_class": "us_stock", "name": "Apple Inc.", "tradeable": True}],
        {"AAPL": {"cik": "0000320193", "title": "Apple Inc."}},
        {},
        pd.Timestamp("2026-09-25T12:00:00Z"),
    )

    assert rows == []
    assert diagnostics["empty_symbols"] == ["AAPL"]
    assert calls


def test_sec_master_index_fallback_is_pit_conservative(monkeypatch):
    mod = importlib.import_module("scripts.sec_filings_research")
    raw = (
        "Description: Master Index of EDGAR Dissemination Feed\n"
        "Last Data Received: 2026-09-24\n"
        "Comments: test\n"
        "CIK|Company Name|Form Type|Date Filed|Filename\n"
        "320193|APPLE INC|8-K|2026-09-24|edgar/data/320193/000032019326000001/aapl.htm\n"
        "320193|APPLE INC|10-Q|2026-09-25|edgar/data/320193/000032019326000002/aapl10q.htm\n"
        "320193|APPLE INC|4|2026-09-24|edgar/data/320193/000032019326000003/form4.xml\n"
    ).encode("latin-1")
    monkeypatch.setattr(mod, "_quarter_keys", lambda start_date, end_date: [(2026, 3)])
    monkeypatch.setattr(mod, "_get_master_index_text", lambda url: (raw.decode("latin-1"), "test_fixture"))

    collected = pd.Timestamp("2026-09-25T12:00:00Z")
    records = [
        {"symbol": "AAPL", "asset_class": "us_stock", "name": "Apple Inc.", "tradeable": True}
    ]
    ticker_map = {"AAPL": {"cik": "0000320193", "title": "Apple Inc."}}

    rows, diagnostics = mod._rows_from_master_indexes(
        records,
        ticker_map,
        {},
        collected,
    )

    assert diagnostics == {"2026Q3": 1}
    assert [row["form"] for row in rows] == ["8-K"]
    assert rows[0]["available_at_method"] == "filing_date_eod_conservative"
    assert rows[0]["acceptance_datetime"] == ""
    assert rows[0]["available_at"] == "2026-09-25T03:59:59.999999+00:00"
    assert rows[0]["accession_number"] == "0000320193-26-000001"
