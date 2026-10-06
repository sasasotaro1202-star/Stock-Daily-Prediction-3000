from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V13 = ROOT / "src/research/ultimate_v13.py"


def test_frontier_suites_are_executed_not_only_artifact_mapped():
    text = V13.read_text(encoding="utf-8")
    assert "from src.research.frontier_pattern_suite import run_frontier_pattern_suite" in text
    assert "from src.research.frontier_extreme_suite import run_extreme_pattern_suite" in text
    assert "frontier_patterns = run_frontier_pattern_suite(" in text
    assert "extreme_patterns = run_extreme_pattern_suite(" in text
    assert 'result["frontier_patterns"] = frontier_patterns' in text
    assert 'result["frontier_extreme_patterns"] = extreme_patterns' in text


def test_extreme_suite_has_explicit_breadth_gate_at_v13_boundary():
    text = V13.read_text(encoding="utf-8")
    assert "minimum_patterns=100" in text
    assert "locked_folds=2" in text
