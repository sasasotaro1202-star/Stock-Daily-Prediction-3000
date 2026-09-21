from pathlib import Path

def test_final_release_contract():
    assert "return_holdout_metrics" in Path("scripts/evaluate_frozen_holdout.py").read_text()
    assert "code_fingerprint_sha256" in Path("scripts/build_manifest.py").read_text()
    assert "recent_20_sessions" in Path("scripts/monitor_predictions.py").read_text()
    assert "live_performance_gate.py" in Path(".github/workflows/us-close-prediction.yml").read_text()
    assert "bounded-production-recovery" in "bounded-production-recovery.yml"
