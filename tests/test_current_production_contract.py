from pathlib import Path

def test_current_production_contract():
    assert "return_holdout_metrics" in Path("scripts/evaluate_frozen_holdout.py").read_text()
    assert "live_performance_gate.py" in Path(".github/workflows/market-cycle.yml").read_text()
    assert "restore_latest_monitor_state.py" in Path(".github/workflows/us-close-prediction.yml").read_text()
