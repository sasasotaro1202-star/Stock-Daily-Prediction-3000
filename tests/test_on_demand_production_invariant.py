from pathlib import Path


def test_on_demand_workflow_uses_immutable_production_prediction_path():
    workflow = Path(".github/workflows/on-demand-production-prediction.yml").read_text(
        encoding="utf-8"
    )
    invariants = Path("scripts/production_invariants.py").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "run_now_prediction.py" in workflow
    assert "load_production_artifact" in Path(
        "scripts/run_now_prediction.py"
    ).read_text(encoding="utf-8")
    assert "run_now_prediction.py" in invariants
    assert "run_daily_prediction.py" not in workflow
    assert "Production-or-near-production prediction" in workflow
    assert "Validate production prediction output" in workflow
