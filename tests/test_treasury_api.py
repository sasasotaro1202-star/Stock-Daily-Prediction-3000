from datetime import date
from src.data.treasury_api import _next_weekday_0600_jst

def test_treasury_availability_skips_weekend():
    assert _next_weekday_0600_jst(date(2026,9,25)).tz_convert("Asia/Tokyo").date().isoformat()=="2026-09-28"

def test_treasury_availability_is_explicit():
    jst=_next_weekday_0600_jst(date(2026,9,24)).tz_convert("Asia/Tokyo")
    assert (jst.hour,jst.minute)==(6,0)
