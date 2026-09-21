from pathlib import Path

def test_core_production_invariants_present():
    text=Path("scripts/production_invariants.py").read_text()
    assert "raw_pit_price_modeling" in text
    assert "state_compatibility" in text
    assert "quantile_intervals" in text
