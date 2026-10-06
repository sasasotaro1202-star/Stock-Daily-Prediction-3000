from datetime import date

import pandas as pd

from src.data.treasury_api import _fetch_year, _next_weekday_0600_jst


def test_treasury_availability_skips_weekend():
    assert (
        _next_weekday_0600_jst(date(2026, 9, 25))
        .tz_convert("Asia/Tokyo")
        .date()
        .isoformat()
        == "2026-09-28"
    )


def test_treasury_availability_is_explicit():
    jst = _next_weekday_0600_jst(date(2026, 9, 24)).tz_convert("Asia/Tokyo")
    assert (jst.hour, jst.minute) == (6, 0)


def test_treasury_xml_parser_is_pit_neutral(monkeypatch):
    xml = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
          xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata">
      <entry>
        <content>
          <m:properties>
            <d:NEW_DATE>2026-09-24T00:00:00</d:NEW_DATE>
            <d:BC_1MONTH>3.70</d:BC_1MONTH>
            <d:BC_3MONTH>3.65</d:BC_3MONTH>
            <d:BC_6MONTH>N/A</d:BC_6MONTH>
            <d:BC_1YEAR>3.45</d:BC_1YEAR>
            <d:BC_2YEAR>3.40</d:BC_2YEAR>
            <d:BC_5YEAR>3.55</d:BC_5YEAR>
            <d:BC_10YEAR>3.95</d:BC_10YEAR>
            <d:BC_30YEAR>4.60</d:BC_30YEAR>
          </m:properties>
        </content>
      </entry>
    </feed>"""
    monkeypatch.setattr(
        "src.data.treasury_api.urlopen",
        lambda request, timeout: _FakeResponse(xml.encode("utf-8")),
    )
    frame = _fetch_year(2026)
    assert len(frame) == 1
    assert float(frame.loc[0, "treasury_10y"]) == 3.95
    assert pd.isna(frame.loc[0, "treasury_6m"])


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload
