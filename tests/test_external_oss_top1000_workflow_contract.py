from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/external-oss-research.yml")


def test_top1000_metadata_workflow_contract():
    text = WORKFLOW.read_text(encoding="utf-8")
    required = [
        "scripts/audit_external_oss_top1000_metadata.py",
        "config/external_prediction_system_oss_top1000_registry.json",
        "research_sources/external_prediction_system_oss_top1000_source.tsv",
        "external-oss-status",
        "promotion_allowed",
        "UNKNOWN_NOT_INFERRED",
        "UNVERIFIED_RESEARCH_ONLY",
        "external_oss_top1000_snapshot.json",
    ]
    for needle in required:
        assert needle in text, needle


def test_workflow_does_not_push_main_for_snapshot_state():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "HEAD:main" not in text
    assert "HEAD:external-oss-status" in text


def test_oss_audit_push_trigger_is_main_only():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "push:\n    branches:\n      - main" in text


def test_snapshot_publication_is_main_only():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "if: github.ref == 'refs/heads/main' && success()" in text


def test_oss_audit_concurrency_isolated_by_ref():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "external-oss-research-${{ github.repository }}-${{ github.ref }}" in text
