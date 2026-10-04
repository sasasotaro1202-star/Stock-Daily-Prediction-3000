from pathlib import Path

from scripts.project_source_contract import validate_source_text

ROOT = Path(__file__).resolve().parents[1]

def test_canonical_project_source_matches_contract():
    source = (ROOT / "PROJECT_SOURCE.md").read_text(encoding="utf-8")
    assert validate_source_text(source) == []
