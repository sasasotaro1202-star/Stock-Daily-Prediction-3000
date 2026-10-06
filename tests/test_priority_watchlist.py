from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "daily_watchlist.yml"
WORKFLOW = ROOT / ".github" / "workflows" / "daily-watchlist.yml"
GENERATOR = ROOT / "scripts" / "generate_daily_watchlist.py"
VALIDATOR = ROOT / "scripts" / "validate_daily_watchlist.py"


EXPECTED_JP = {
    "8802", "8801", "9409", "9404", "4676", "9401", "285A", "1333", "2802",
    "2897", "7267", "7201", "4689", "4385", "215A", "2379", "9843", "9432",
    "6752", "6758", "8035", "6857", "7974", "9434", "7203",
}
EXPECTED_US = {
    "DIS", "SBUX", "MCD", "NFLX", "KO", "TSLA", "NKE", "META", "AMZN",
    "GOOGL", "MSFT", "NVDA", "AAPL",
}
EXPECTED_MARKET = {"^N225", "^GSPC", "USDJPY=X"}


def test_mandatory_priority_watchlist_exactly_41():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["priority_watchlist"]["mandatory_daily_prediction"] is True
    assert cfg["priority_watchlist"]["expected_count"] == 41

    jp = {str(x["symbol"]).upper() for x in cfg["equities"]["items"]}
    us = {str(x["symbol"]).upper() for x in cfg["us_equities"]["items"]}
    market = {
        str(x["provider_symbol"]).upper()
        for x in cfg["market_instruments"]["instruments"]
    }

    assert jp == EXPECTED_JP
    assert us == EXPECTED_US
    assert market == EXPECTED_MARKET
    assert len(jp) + len(us) + len(market) == 41


def test_daily_priority_workflow_predicts_before_watchlist():
    text = WORKFLOW.read_text(encoding="utf-8")
    predict_pos = text.index("python scripts/run_now_prediction.py")
    watchlist_pos = text.index("python scripts/generate_daily_watchlist.py")
    assert predict_pos < watchlist_pos
    assert "PREDICT_ASSET_CLASSES: jp_stock,jp_etf,jp_reit,us_stock,us_etf" in text
    assert 'cron: "50 18 * * 1-5"' in text


def test_near_production_predictions_remain_explicitly_nonproduction():
    generator = GENERATOR.read_text(encoding="utf-8")
    validator = VALIDATOR.read_text(encoding="utf-8")
    assert 'USABLE_EQUITY_STATUSES = {"READY", "READY_NEAR_PRODUCTION"}' in generator
    assert 'if status not in USABLE_EQUITY_STATUSES:' in generator
    assert '"NEAR_PRODUCTION_PREDICTION_REFERENCE"' in generator
    assert '"READY_NEAR_PRODUCTION"' in validator
