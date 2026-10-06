from datetime import date, timedelta

import pytest

from src.data.boj_api import (
    build_request_url,
    conservative_available_at,
    parse_get_data_code_payload,
)

def test_conservative_available_at_moves_friday_to_monday():
    result = conservative_available_at(date(2026, 9, 25))
    assert result.weekday() == 0
    assert result.hour == 15
    assert result.utcoffset() == timedelta(hours=9)

def test_build_request_url_uses_official_parameters():
    url = build_request_url(
        "https://www.stat-search.boj.or.jp/api/v1/getDataCode",
        db="FM01",
        codes=["STRDCLUCON", "STRDCLUCONH"],
        start_date="202601",
        end_date="202609",
    )
    assert "format=json" in url
    assert "lang=en" in url
    assert "db=FM01" in url
    assert "STRDCLUCON%2CSTRDCLUCONH" in url
    assert "startDate=202601" in url
    assert "endDate=202609" in url

def test_parse_realistic_boj_payload():
    payload = {
        "data": {"RESULTSET": [{
            "SERIES_CODE": "STRDCLUCON",
            "NAME_OF_TIME_SERIES": "Call Rate",
            "UNIT": "percent per annum",
            "FREQUENCY": "DAILY",
            "VALUES": {
                "SURVEY_DATES": [20260924, 20260925],
                "VALUES": [0.727, 0.728],
            },
        }]}
    }
    rows = parse_get_data_code_payload(payload, db="FM01")
    assert len(rows) == 2
    assert rows[0].series_code == "STRDCLUCON"
    assert rows[0].value == pytest.approx(0.727)

def test_parse_rejects_length_mismatch():
    payload = {
        "RESULTSET": [{
            "SERIES_CODE": "STRDCLUCON",
            "VALUES": {"SURVEY_DATES": [20260924], "VALUES": [0.7, 0.8]},
        }]
    }
    with pytest.raises(ValueError, match="length mismatch"):
        parse_get_data_code_payload(payload, db="FM01")
