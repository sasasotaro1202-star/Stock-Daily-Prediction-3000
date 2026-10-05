from __future__ import annotations

import json

import src.research.run_provenance as provenance


def test_manifest_is_research_only_and_tracks_reproducibility(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")

    manifest = provenance.build_manifest(
        status="STARTED",
        stage="initialization",
        fold_count=7,
        fold_signature="oos:test",
        candidate_models=["logistic", "hgb"],
    )

    assert manifest["schema_version"] == 1
    assert manifest["research_only"] is True
    assert manifest["production_changed"] is False
    assert manifest["promotion_allowed"] is False
    assert manifest["runtime"]["run_id"] == "123"
    assert manifest["runtime"]["run_attempt"] == "2"
    assert manifest["runtime"]["sha"] == "abc123"
    assert manifest["folds"]["count"] == 7
    assert manifest["folds"]["signature"] == "oos:test"
    assert manifest["candidate_models"] == ["logistic", "hgb"]
    assert manifest["holdout_policy"]["frozen_holdout_used_for_selection"] is False
    assert all(
        digest is None or len(digest) == 64
        for digest in manifest["artifact_integrity"]["tracked_files"].values()
    )


def test_atomic_progress_write_preserves_run_identity(tmp_path, monkeypatch):
    progress_path = tmp_path / "oos_progress.json"
    monkeypatch.setattr(provenance, "PROGRESS_PATH", progress_path)
    monkeypatch.setenv("GITHUB_RUN_ID", "456")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")

    payload = provenance.write_progress(
        event="fold_evaluated",
        model="hgb",
        model_index=2,
        total_models=5,
        fold=3,
        total_folds=8,
        completed_folds=11,
    )

    assert payload["status"] == "RUNNING"
    assert payload["runtime"]["run_id"] == "456"
    assert payload["runtime"]["sha"] == "deadbeef"
    assert payload["progress"]["completed_folds"] == 11
    loaded = json.loads(progress_path.read_text(encoding="utf-8"))
    assert loaded == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_finalize_maps_non_success_without_claiming_performance(monkeypatch, tmp_path):
    progress_path = tmp_path / "oos_progress.json"
    manifest_path = tmp_path / "oos_run_manifest.json"
    monkeypatch.setattr(provenance, "PROGRESS_PATH", progress_path)
    monkeypatch.setattr(provenance, "MANIFEST_PATH", manifest_path)
    monkeypatch.setenv("GITHUB_RUN_ID", "789")
    monkeypatch.setenv("GITHUB_SHA", "sha789")

    provenance.write_progress(event="started")
    result = provenance.finalize_progress(outcome="cancelled")

    assert result["status"] == "CANCELLED"
    assert result["event"] == "run_finalized"
    assert result["outcome"] == "cancelled"
    assert result["research_only"] is True
    assert result["production_changed"] is False
    assert result["promotion_allowed"] is False


def test_manifest_writer_is_atomic_and_writes_json(monkeypatch, tmp_path):
    path = tmp_path / "oos_run_manifest.json"
    monkeypatch.setattr(provenance, "MANIFEST_PATH", path)

    result = provenance.write_manifest(
        status="FOLDS_READY",
        stage="fold_setup",
        fold_count=5,
        fold_signature="oos:abc",
        candidate_models=["logistic"],
    )

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == result
    assert not list(tmp_path.glob("*.tmp"))
