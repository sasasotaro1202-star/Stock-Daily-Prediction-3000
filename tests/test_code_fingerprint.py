from __future__ import annotations

from src.validation.code_fingerprint import evidence_file_fingerprint, evidence_fingerprint_sha256


def test_evidence_fingerprint_includes_runtime_inputs_but_excludes_docs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "x.yml").write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "research.yml").write_text("name: research\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("docs\n", encoding="utf-8")

    rows = evidence_file_fingerprint()

    assert "config/x.yml" in rows
    assert "src/x.py" in rows
    assert ".github/workflows/research.yml" in rows
    assert "pyproject.toml" in rows
    assert "README.md" not in rows
    assert len(evidence_fingerprint_sha256(rows)) == 64
