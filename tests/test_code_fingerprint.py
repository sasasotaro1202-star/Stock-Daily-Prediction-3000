from __future__ import annotations

from src.validation.code_fingerprint import evidence_file_fingerprint, evidence_fingerprint_sha256


def test_evidence_fingerprint_includes_runtime_inputs_but_excludes_docs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "x.yml").write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    workflows = tmp_path / ".github" / "workflows"
    (workflows / "research.yml").write_text("name: research\n", encoding="utf-8")
    (workflows / "heartbeat.yml").write_text("name: heartbeat\n", encoding="utf-8")
    (workflows / "automation-supervisor.yml").write_text("name: supervisor\n", encoding="utf-8")
    (workflows / "unknown-control.yml").write_text("name: unknown\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("docs\n", encoding="utf-8")

    rows = evidence_file_fingerprint()

    assert "config/x.yml" in rows
    assert "src/x.py" in rows
    assert ".github/workflows/research.yml" in rows
    assert ".github/workflows/unknown-control.yml" in rows
    assert ".github/workflows/heartbeat.yml" not in rows
    assert ".github/workflows/automation-supervisor.yml" not in rows
    assert "pyproject.toml" in rows
    assert "README.md" not in rows
    assert len(evidence_fingerprint_sha256(rows)) == 64
