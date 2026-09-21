from pathlib import Path

def test_final_production_contract():
    cfg=Path("config/pipeline.yml").read_text()
    inv=Path("scripts/production_invariants.py").read_text()
    assert "selection_source: chronological_oos_only" in cfg
    assert "interval_method: conditional_quantiles_q10_q50_q90" in cfg
    assert "state_compatibility" in inv
    assert "paypay_master_catalog" in inv
    assert "regime_thresholds" in inv
