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


def test_verified_stock_queue_is_generated_and_published():
    text = WORKFLOW.read_text(encoding="utf-8")
    for needle in [
        "scripts/build_external_stock_research_queue.py",
        "artifacts/external_stock_research_queue.json",
        "data/research/external_stock_research_queue.json",
        "NONE_EXTERNAL_METADATA",
        "priority_is_not_predictive_evidence",
    ]:
        assert needle in text, needle


def test_static_source_inspection_is_integrated_safely():
    text = WORKFLOW.read_text(encoding="utf-8")
    for needle in [
        "scripts/inspect_external_stock_research_queue.py",
        "external_stock_source_inspection.json",
        "external_code_executed",
        "external_code_installed",
        "E2_EXTERNAL_SOURCE_INSPECTION",
    ]:
        assert needle in text, needle
