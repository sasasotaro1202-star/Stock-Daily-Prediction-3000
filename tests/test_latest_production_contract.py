from pathlib import Path


def test_production_contract_is_explicit():
    pipeline=Path("config/pipeline.yml").read_text()
    catalog=Path("config/paypay_catalog.yml").read_text()
    recovery=Path(".github/workflows/bounded-production-recovery.yml").read_text()
    monitoring=Path(".github/workflows/prediction-monitoring.yml").read_text()
    assert "selection_source: chronological_oos_only" in pipeline
    assert "interval_method: conditional_quantiles_q10_q50_q90" in pipeline
    assert "no_fixed_security_count_limit: true" in catalog
    assert "run_attempt == 1" in recovery
    assert "restore_prediction_history.py" in monitoring
