from pathlib import Path

from scripts.project_source_contract import validate_source_text

ROOT = Path(__file__).resolve().parents[1]

def test_canonical_project_source_matches_contract():
    source = (ROOT / "PROJECT_SOURCE.md").read_text(encoding="utf-8")
    assert validate_source_text(source) == []


def test_canonical_project_source_rejects_missing_section():
    source = (ROOT / "PROJECT_SOURCE.md").read_text(encoding="utf-8")
    mutated = source.replace("60. RESEARCH FRONTIERS\n", "", 1)
    errors = validate_source_text(mutated)
    assert "section_sequence_invalid" in " ".join(errors) or "section_headings_invalid" in errors


def test_canonical_project_source_rejects_wrapped_copy():
    source = (ROOT / "PROJECT_SOURCE.md").read_text(encoding="utf-8")
    errors = validate_source_text("=== COPY START ===\n" + source + "=== COPY END ===\n")
    assert "copy_wrapper_present" in errors
