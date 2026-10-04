from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/research-validation.yml"


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


def test_ranking_selection_is_explicitly_blocked_until_nested_oos_evidence():
    research = (ROOT / "scripts/run_daily_research.py").read_text(encoding="utf-8")
    locker = (ROOT / "scripts/lock_frozen_model.py").read_text(encoding="utf-8")
    assert '"ranking_selection_ready_for_production": False' in research
    assert "nested prequential ranking selection is required before production freeze" in research
    assert 'if payload.get("ranking_selection_ready_for_production") is not True:' in locker



def test_nested_ranking_selection_is_prequential():
    source = (ROOT / "scripts/run_daily_research.py").read_text(encoding="utf-8")
    helper = (ROOT / "src/research/nested_ranking.py").read_text(encoding="utf-8")
    assert "nested_prequential_ranking_oos(" in source
    assert '"nested_ranking_selection_research": nested_ranking_selection_research' in source
    assert '"ranking_selection_ready_for_production": False' in source
    assert "same-OOS model/window selection" in helper or "globally selected model" in helper
    assert '"same_oos_global_model_or_window_reuse": False' in helper



def test_nested_ranking_freeze_gate_requires_structural_evidence():
    locker = (ROOT / "scripts/lock_frozen_model.py").read_text(encoding="utf-8")
    assert 'nested_ranking = payload.get("nested_ranking_selection_research")' in locker
    assert 'nested_ranking.get("status") != "EVALUATED"' in locker
    assert 'nested_ranking.get("same_oos_global_model_or_window_reuse") is not False' in locker
    assert 'nested_ranking.get("ranking_weight_selection_prequential") is not True' in locker
    assert 'nested_ranking.get("model_selection_prequential") is not True' in locker
    assert 'nested_ranking.get("return_estimator_selection_prequential") is not True' in locker
    assert 'bootstrap_probability < 0.90 or bootstrap_p05 <= 0.0' in locker


def test_nested_ranking_runs_after_production_candidate_configuration_is_available():
    source = (ROOT / "scripts/run_daily_research.py").read_text(encoding="utf-8")
    call = source.index("nested_ranking_selection_research = nested_prequential_ranking_oos(")
    assert source.index("selected_training_window =", 0, call) < call
    assert source.index("selected_rank_weight =", 0, call) < call
    assert source.index("selected_uncertainty_penalty =", 0, call) < call
    assert source.rfind("selected_return_estimator =", 0, call) < call
    assert '"classifier_training_window_sessions": selected_training_window' in source


def test_nested_ranking_freeze_gate_requires_identity_alignment():
    locker = (ROOT / "scripts/lock_frozen_model.py").read_text(encoding="utf-8")
    assert 'identity = nested_ranking.get("production_identity_alignment")' in locker
    assert 'identity.get("aligned") is not True' in locker


def test_nested_ranking_uses_full_history_window_semantics_for_prediction_bank():
    source = (ROOT / "scripts/run_daily_research.py").read_text(encoding="utf-8")
    helper = (ROOT / "src/research/nested_ranking.py").read_text(encoding="utf-8")
    call_start = source.index("nested_ranking_selection_research = nested_prequential_ranking_oos(")
    call_end = source.index("\n    )", call_start) + 6
    call = source[call_start:call_end]
    assert "prediction_generation_training_window_sessions=0" in call
    assert "prediction_generation_training_window_sessions=252" not in call
    assert "sampling safeguard, not a 252-session" in helper
