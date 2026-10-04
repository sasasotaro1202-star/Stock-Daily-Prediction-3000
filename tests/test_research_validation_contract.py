from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/research-validation.yml")


def test_research_validation_requires_integrated_case_risk_artifacts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for filename in (
        "case_risk_oos.json",
        "learned_case_risk_oos.json",
    ):
        assert filename in text
        assert f'data/research/ultimate_v13/{filename}' in text

    required_block = '"case_risk_oos.json",\n              "learned_case_risk_oos.json"'
    assert required_block in text


def test_research_validation_checks_case_risk_safety_contracts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'if data.get("research_only") is not True' in text
    assert 'if data.get("production_changed") is not False' in text
    assert 'if data.get("promotion_allowed") is not False' in text
    assert 'if data.get("frozen_holdout_used") is not False' in text
    assert "Research validation workflow contract tests" in text


def test_ranking_oos_calibration_is_prequential():
    source = (ROOT / "scripts/run_daily_research.py").read_text(encoding="utf-8")
    assert "temporal_calibration_method_by_fold" in source
    assert 'temporal_calibration_rows' in source
    assert 'fold_calibration_method = temporal_calibration_method_by_fold.get(' in source
    assert 'global selected_calibration_method is not reused for ranking evidence' in source
    assert '"ranking_oos_calibration_protocol"' in source
