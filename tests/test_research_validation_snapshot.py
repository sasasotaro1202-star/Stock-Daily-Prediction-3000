from __future__ import annotations

import hashlib
import json

import pytest

from scripts.persist_research_validation_snapshot import main


def _write_metrics(tmp_path, payload):
    data_dir = tmp_path / "data" / "research"
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "latest_metrics.json"
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    source.write_bytes(raw)
    return source


def _payload(**overrides):
    payload = {
        "status": "OOS_COMPLETE",
        "selected_model": "hgb",
        "results": {
            "hgb": {
                "metrics": {
                    "logloss": 0.61,
                    "brier": 0.21,
                    "ece": 0.04,
                    "accuracy": 0.68,
                    "rank_ic": 0.09,
                    "folds": 8,
                    "oos_fold_count": 8,
                    "oos_fold_signature": "oos:abcdef1234567890",
                }
            },
            "logistic": {
                "metrics": {
                    "logloss": 0.64,
                    "brier": 0.22,
                    "ece": 0.05,
                    "accuracy": 0.65,
                }
            },
        },
        "return_oos": {
            "status": "OOS_COMPLETE",
            "metrics": {
                "mae": 0.012,
                "rmse": 0.019,
                "sign_accuracy": 0.57,
                "rank_ic": 0.03,
                "range_80_coverage": 0.81,
            },
        },
        "global_selection_evidence": {"status": "SUPPORTED"},
        "calibration_method": "beta",
    }
    payload.update(overrides)
    return payload


def test_research_snapshot_persists_core_scores_and_integrity_metadata(
    tmp_path, monkeypatch
) -> None:
    source = _write_metrics(tmp_path, _payload())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "research-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "987")
    main()

    out = json.loads(
        (tmp_path / "artifacts" / "research_validation_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert out["selected_model"] == "hgb"
    assert out["workflow_sha"] == "research-sha"
    assert out["research_workflow_sha"] == "research-sha"
    assert out["status_branch_main_sha"] is None
    assert out["evidence_freshness"] == "UNKNOWN"
    assert out["selected_model_score"]["logloss"] == 0.61
    assert out["selected_model_score"]["brier"] == 0.21
    assert out["selected_model_score"]["ece"] == 0.04
    assert out["selected_model_score"]["accuracy"] == 0.68
    assert out["selected_model_score"]["oos_fold_signature"] == "oos:abcdef1234567890"
    assert out["candidate_scores"]["logistic"]["logloss"] == 0.64
    assert out["return_oos_score"]["mae"] == 0.012
    assert out["return_oos_score"]["range_80_coverage"] == 0.81
    assert out["global_selection_evidence"]["status"] == "SUPPORTED"
    assert (
        out["evidence_scope"]["source_sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert out["evidence_scope"]["frozen_holdout_excluded_from_selection"] is True


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"status": "INCOMPLETE"}, "OOS_COMPLETE"),
        ({"selected_model": "missing"}, "absent from research results"),
        ({"results": {"hgb": {"metrics": {}}}}, "LogLoss"),
    ],
)
def test_research_snapshot_fails_closed_on_missing_or_incomplete_evidence(
    tmp_path, monkeypatch, overrides, message
) -> None:
    _write_metrics(tmp_path, _payload(**overrides))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match=message):
        main()

    assert not (tmp_path / "artifacts" / "research_validation_latest.json").exists()


def test_research_snapshot_marks_stale_execution_sha(monkeypatch, tmp_path) -> None:
    source = _write_metrics(tmp_path, _payload())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "research-old")
    monkeypatch.setattr(
        "scripts.persist_research_validation_snapshot._git_head_sha",
        lambda: "main-new",
    )
    main()
    out = json.loads(
        (tmp_path / "artifacts" / "research_validation_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert out["research_workflow_sha"] == "research-old"
    assert out["status_branch_main_sha"] == "main-new"
    assert out["evidence_freshness"] == "STALE"
    assert out["evidence_freshness_basis"] == "execution_sha"
    assert out["evidence_scope"]["freshness"] == "STALE"
    assert out["evidence_scope"]["freshness_basis"] == "execution_sha"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == out["evidence_scope"]["source_sha256"]


def test_research_snapshot_keeps_evidence_fresh_across_control_plane_sha_change(monkeypatch, tmp_path):
    _write_metrics(tmp_path, _payload(evidence_code_fingerprint_sha256="evidence-fp"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "research-old")
    monkeypatch.setattr("scripts.persist_research_validation_snapshot._git_head_sha", lambda: "main-new")
    monkeypatch.setattr("scripts.persist_research_validation_snapshot.evidence_fingerprint_sha256", lambda: "evidence-fp")
    main()
    out = json.loads((tmp_path / "artifacts" / "research_validation_latest.json").read_text(encoding="utf-8"))
    assert out["evidence_freshness"] == "FRESH"
    assert out["evidence_freshness_basis"] == "evidence_code_fingerprint"


def test_research_snapshot_marks_evidence_fingerprint_mismatch_stale(monkeypatch, tmp_path):
    _write_metrics(tmp_path, _payload(evidence_code_fingerprint_sha256="research-fp"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "same-sha")
    monkeypatch.setattr("scripts.persist_research_validation_snapshot._git_head_sha", lambda: "same-sha")
    monkeypatch.setattr("scripts.persist_research_validation_snapshot.evidence_fingerprint_sha256", lambda: "main-fp")
    main()
    out = json.loads((tmp_path / "artifacts" / "research_validation_latest.json").read_text(encoding="utf-8"))
    assert out["evidence_freshness"] == "STALE"
    assert out["evidence_freshness_basis"] == "evidence_code_fingerprint"
