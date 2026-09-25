from __future__ import annotations

import ast
from pathlib import Path


def test_adaptive_conformal_research_caller_uses_keyword_feedback():
    source = Path("scripts/run_daily_research.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "adaptive_conformal_prediction_sets"
    ]

    assert len(calls) == 1
    call = calls[0]
    assert len(call.args) == 4
    assert {kw.arg for kw in call.keywords} >= {
        "alpha",
        "gamma",
        "alpha_min",
        "alpha_max",
        "observed_y_test",
    }
