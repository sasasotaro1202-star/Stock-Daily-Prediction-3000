from __future__ import annotations

from src.research.pit_contract import (
    audit_pit_row,
    audit_pit_timestamps,
    audit_pit_rows,
    audit_source_config,
    canonical_case_key,
)


def _row(**overrides):
    row = {
        "asset_class": "JP_STOCK",
        "symbol": "1301",
        "session_date": "2026-10-02",
        "prediction_time": "2026-10-02T09:00:00+09:00",
        "prediction_cutoff": "2026-10-02T09:00:00+09:00",
        "available_at": "2026-10-02T08:59:00+09:00",
        "published_at": "2026-10-02T08:58:00+09:00",
        "retrieved_at": "2026-10-02T08:59:10+09:00",
        "pit_status": "PASS",
    }
    row.update(overrides)
    return row


def test_pit_row_requires_available_before_cutoff_and_pass_status():
    assert audit_pit_row(_row())["ok"] is True
    late = audit_pit_row(
        _row(available_at="2026-10-02T09:01:00+09:00")
    )
    assert late["ok"] is False
    assert "available_at_after_prediction_cutoff" in late["violations"]
    unknown = audit_pit_row(_row(pit_status="UNVERIFIED"))
    assert unknown["ok"] is False
    assert "pit_status_not_pass" in unknown["violations"]


def test_pit_row_rejects_naive_and_reversed_lineage():
    naive = audit_pit_row(
        _row(prediction_time="2026-10-02T09:00:00")
    )
    assert naive["ok"] is False
    assert "invalid_prediction_time" in naive["violations"]

    reversed_times = audit_pit_row(
        _row(
            published_at="2026-10-02T08:59:30+09:00",
            available_at="2026-10-02T08:59:00+09:00",
            retrieved_at="2026-10-02T08:58:00+09:00",
        )
    )
    assert reversed_times["ok"] is False
    assert "retrieved_at_before_available_at" in reversed_times["violations"]


def test_canonical_case_key_is_stable_for_prediction_ledger():
    assert canonical_case_key(_row()) == (
        "JP_STOCK",
        "1301",
        "2026-10-02",
        "2026-10-02T09:00:00+09:00",
    )


def test_pit_rows_detect_duplicate_canonical_cases():
    result = audit_pit_rows([_row(), _row()])
    assert result["status"] == "FAIL"
    assert any(
        "duplicate_canonical_case_key" in row.get("violations", [])
        for row in result["violations"]
    )


def test_source_config_fails_closed_on_unverified_claims():
    config = {
        "policy": {"research_only": True},
        "good_source": {
            "enabled": True,
            "pit_policy": {
                "research_use_only_until_pit_validation": True,
            },
        },
        "bad_source": {
            "enabled": True,
            "pit_policy": {
                "research_use_only_until_pit_validation": True,
                "available_at_verified": True,
            },
        },
    }
    result = audit_source_config(config)
    assert result["status"] == "FAIL"
    assert any(
        row["reason"] == "availability_claim_has_no_verification_evidence"
        for row in result["violations"]
    )


def test_source_config_preserves_research_only_unverified_state():
    config = {
        "policy": {"research_only": True},
        "frontier": {
            "enabled": True,
            "pit_policy": {
                "research_use_only_until_pit_validation": True,
                "available_at_verified": False,
            },
        },
    }
    result = audit_source_config(config)
    assert result["status"] == "PASS"
    assert result["unverified_sources"] == 1
    assert result["pit_ready_sources"] == 0


def test_prediction_cutoff_is_the_pit_boundary_when_prediction_time_is_later():
    row = _row(
        prediction_cutoff="2026-10-02T08:55:00+09:00",
        prediction_time="2026-10-02T09:00:00+09:00",
        available_at="2026-10-02T08:59:00+09:00",
    )
    result = audit_pit_row(row)
    assert result["ok"] is False
    assert "available_at_after_prediction_cutoff" in result["violations"]


def test_missing_case_identity_is_blocking():
    result = audit_pit_rows([_row(symbol="")])
    assert result["status"] == "FAIL"
    assert any(
        "missing_canonical_case_identity" in row.get("violations", [])
        for row in result["violations"]
    )


def test_present_but_malformed_secondary_prediction_timestamp_is_blocking():
    result = audit_pit_row(
        _row(prediction_cutoff="2026-10-02T08:55:00+09:00", prediction_time="bad")
    )
    assert result["ok"] is False
    assert "invalid_prediction_time" in result["violations"]

    result = audit_pit_row(
        _row(prediction_cutoff="bad", prediction_time="2026-10-02T09:00:00+09:00")
    )
    assert result["ok"] is False
    assert "invalid_prediction_cutoff" in result["violations"]


def test_shared_pit_timestamp_contract_catches_publication_and_retrieval_ordering():
    published_late = audit_pit_timestamps(
        {
            "prediction_cutoff": "2026-10-02T09:00:00+09:00",
            "prediction_time": "2026-10-02T09:00:00+09:00",
            "available_at": "2026-10-02T08:59:00+09:00",
            "published_at": "2026-10-02T09:01:00+09:00",
            "retrieved_at": "2026-10-02T09:02:00+09:00",
        }
    )
    assert published_late["ok"] is False
    assert "published_at_after_available_at" in published_late["violations"]

    retrieved_early = audit_pit_timestamps(
        {
            "prediction_cutoff": "2026-10-02T09:00:00+09:00",
            "prediction_time": "2026-10-02T09:00:00+09:00",
            "available_at": "2026-10-02T08:59:00+09:00",
            "retrieved_at": "2026-10-02T08:58:00+09:00",
        }
    )
    assert retrieved_early["ok"] is False
    assert "retrieved_at_before_available_at" in retrieved_early["violations"]


def test_retrieval_after_prediction_cutoff_is_blocking():
    result = audit_pit_timestamps(
        {
            "prediction_cutoff": "2026-10-02T09:00:00+09:00",
            "prediction_time": "2026-10-02T09:00:00+09:00",
            "available_at": "2026-10-02T08:59:00+09:00",
            "retrieved_at": "2026-10-02T09:00:01+09:00",
        }
    )
    assert result["ok"] is False
    assert "retrieved_at_after_prediction_cutoff" in result["violations"]


def test_missing_explicit_prediction_cutoff_is_blocking():
    row = _row()
    row.pop("prediction_cutoff")
    result = audit_pit_row(row)
    assert result["ok"] is False
    assert "missing_prediction_cutoff" in result["violations"]
