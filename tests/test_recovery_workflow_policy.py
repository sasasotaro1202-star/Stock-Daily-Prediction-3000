from pathlib import Path

def test_recovery_is_bounded():
    text=Path(".github/workflows/bounded-production-recovery.yml").read_text()
    assert "run_attempt == 1" in text
    assert "--failed" in text
