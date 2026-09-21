from src.data.schema import DailyBar

def test_daily_bar_validates():
    DailyBar("TEST","2026-01-02","2026-01-02T07:00:00+09:00",100,105,99,103,1000).validate()
