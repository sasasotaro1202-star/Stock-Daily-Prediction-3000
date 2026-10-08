from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_prediction_runtime_dependencies_are_core_dependencies():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_dependencies = set(payload["project"]["dependencies"])
    research_dependencies = set(payload["project"]["optional-dependencies"]["research"])

    assert any(dep.startswith("exchange-calendars") for dep in project_dependencies), (
        "exchange-calendars must be a core runtime dependency because "
        "run_now_prediction.py imports exchange_calendars"
    )
    assert not any(
        dep.startswith("exchange-calendars") for dep in research_dependencies
    ), "exchange-calendars must not be research-only"

    source = (ROOT / "scripts" / "run_now_prediction.py").read_text(
        encoding="utf-8"
    )
    assert "from exchange_calendars import get_calendar" in source


def test_prediction_runtime_keeps_q50_output():
    source = (ROOT / "scripts" / "run_now_prediction.py").read_text(encoding="utf-8")
    assert '"return_q10_1d", "q50_1d", "return_q90_1d"' in source, (
        "q50_1d is a required production prediction output and must survive "
        "the final output-column selection"
    )
