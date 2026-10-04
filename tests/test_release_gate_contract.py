from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_gate_requires_research_pit_contract_artifact():
    source = (ROOT / "scripts/release_gate.py").read_text(encoding="utf-8")
    assert 'pit_audit_path=Path("data/research/research_pit_contract_audit.json")' in source
    assert 'reasons.append("missing_research_pit_contract_audit")' in source
    assert 'reasons.append("research_pit_contract_audit_failed")' in source
    assert 'reasons.append("research_pit_prediction_ledger_not_pass")' in source


def test_release_gate_requires_nonfailed_pit_audit_and_passed_prediction_ledger():
    source = (ROOT / "scripts/release_gate.py").read_text(encoding="utf-8")
    assert 'if pit_audit.get("status") == "FAIL":' in source
    assert 'if pit_ledger.get("status") != "PASS":' in source
