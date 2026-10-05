from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _read(name: str) -> str:
    path = WORKFLOWS / name
    if not path.exists():
        raise SystemExit(f"FAIL: required workflow missing: {name}")
    return path.read_text(encoding="utf-8")


def _assert_once(source: str, needle: str, label: str) -> None:
    count = source.count(needle)
    if count != 1:
        raise SystemExit(
            f"FAIL: automation invariant {label}: expected 1 occurrence, got {count}"
        )


def _assert_absent(source: str, needle: str, label: str) -> None:
    if needle in source:
        raise SystemExit(f"FAIL: automation invariant {label}: forbidden text present")


def _assert_contains(source: str, needle: str, label: str) -> None:
    if needle not in source:
        raise SystemExit(f"FAIL: automation invariant {label}: required text missing")


def _assert_feature_pit_guard() -> None:
    from src.research.learned_case_risk_oos import _pit_ready

    base = {
        "pit_status": "PASS",
        "prediction_time": "2026-01-02T00:00:00+00:00",
        "available_at": "2026-01-01T23:00:00+00:00",
        "case_predictability": 0.8,
        "case_ood": 0.1,
        "case_failure_risk": 0.2,
        "case_disagreement": 0.1,
        "prediction": 0.8,
        "failed": 0,
    }
    future_snapshot = {
        **base,
        "feature_snapshot_cutoff": "2026-01-02T00:01:00+00:00",
    }
    unknown_lineage = {**base, "feature_pit_status": "UNKNOWN"}
    if _pit_ready(future_snapshot):
        raise SystemExit("FAIL: feature snapshot after prediction cutoff was accepted")
    if _pit_ready(unknown_lineage):
        raise SystemExit("FAIL: unknown feature PIT lineage was accepted")


def main() -> int:
    _assert_feature_pit_guard()
    market = _read("market-cycle.yml")
    monitoring = _read("prediction-monitoring.yml")
    watchdog = _read("actions-reliability-watchdog.yml")
    heartbeat = _read("heartbeat.yml")
    recovery = _read("bounded-production-recovery.yml")
    price_restore = str((WORKFLOWS.parent.parent / "scripts" / "restore_latest_price_state.py").read_text(encoding="utf-8"))
    source_discovery = _read("free-data-source-discovery.yml")
    autopilot = _read("research-autopilot.yml")
    source_discovery_script = str(
        (ROOT / "scripts" / "discover_free_data_sources.py").read_text(encoding="utf-8")
    )

    project_source = (ROOT / "PROJECT_SOURCE.md").read_text(encoding="utf-8")
    project_source_contract = str(
        (ROOT / "scripts" / "project_source_contract.py").read_text(encoding="utf-8")
    )
    repository_verification = _read("repository-verification.yml")

    _assert_once(
        repository_verification,
        "Validate canonical Project Source contract",
        "repository_verification_runs_project_source_contract",
    )
    _assert_once(
        repository_verification,
        "project-source-contract-${{ github.run_id }}",
        "repository_verification_archives_project_source_contract",
    )
    _assert_once(
        project_source_contract,
        "CANONICAL_SECTION_HEADINGS = (",
        "project_source_contract_locks_exact_headings",
    )
    source_section_count = len(re.findall(r"(?m)^(\d+)\.\s+[A-Z][A-Z0-9 /&._-]*$", project_source))
    if source_section_count != 99:
        raise SystemExit(
            f"FAIL: canonical Project Source section count expected 99, got {source_section_count}"
        )

    # The predictive-state architecture is intentionally research-only.
    predictive_state = (ROOT / "src" / "research" / "predictive_state.py").read_text(encoding="utf-8")
    _assert_contains(
        predictive_state,
        "chronological_transition_forecast",
        "predictive_state_transition_contract",
    )
    _assert_contains(
        predictive_state,
        "Research-only soft regime posterior",
        "predictive_state_research_only_documentation",
    )

    # One canonical schedule per expensive/critical daily workflow. Missed
    # schedules are recovered by the watchdog rather than by duplicate cron
    # entries, which avoids needless queued/cancelled executions.
    _assert_once(
        market,
        '    - cron: "37 18 * * 1-5"',
        "market_cycle_canonical_schedule",
    )
    _assert_absent(
        market,
        '    - cron: "17 18 * * 1-5"',
        "market_cycle_duplicate_schedule",
    )

    _assert_once(
        monitoring,
        '    - cron: "27 9 * * 1-5"',
        "prediction_monitoring_canonical_schedule",
    )
    _assert_absent(
        monitoring,
        '    - cron: "17 9 * * 1-5"',
        "prediction_monitoring_duplicate_schedule",
    )

    _assert_once(
        watchdog,
        '    - cron: "*/5 * * * *"',
        "watchdog_5m_schedule",
    )
    _assert_once(
        watchdog,
        "cancel-in-progress: true",
        "watchdog_latest_run_wins",
    )
    for due in ("07:17", "09:27", "18:37"):
        _assert_once(watchdog, due, f"watchdog_recovery_window_{due.replace(':', '_')}")

    _assert_once(
        heartbeat,
        '    - cron: "17 */6 * * *"',
        "heartbeat_6h_schedule",
    )

    _assert_once(
        source_discovery,
        '    - cron: "17 5,17 * * *"',
        "free_source_discovery_twice_daily",
    )
    _assert_once(
        source_discovery,
        "cancel-in-progress: true",
        "source_discovery_latest_run_wins",
    )
    _assert_once(
        source_discovery_script,
        '"free_only": True',
        "source_discovery_free_only",
    )
    _assert_once(
        source_discovery_script,
        '"research_only": True',
        "source_discovery_research_only",
    )
    _assert_once(
        source_discovery_script,
        '"production_changed": False',
        "source_discovery_no_production_mutation",
    )
    _assert_once(
        source_discovery_script,
        "discovery_does_not_adopt",
        "source_discovery_never_auto_adopts",
    )
    _assert_once(
        source_discovery_script,
        "retrieval_is_not_historical_pit",
        "source_discovery_separates_retrieval_from_pit",
    )
    _assert_once(
        source_discovery_script,
        'if candidate.get("blocked"):',
        "source_discovery_blocks_uncertain_commercial_providers",
    )


    _assert_once(
        watchdog,
        "heartbeat_cutoff=\"$(date -u -d '13 hours ago' +%s)\"",
        "heartbeat_stale_threshold_matches_6h_cadence",
    )

    _assert_once(
        market,
        'if [ "$run_id" -gt "$GITHUB_RUN_ID" ]; then',
        "market_cycle_ignores_future_runs",
    )

    _assert_once(
        recovery,
        "github.event.workflow_run.conclusion == 'failure'",
        "failure_only_bounded_recovery",
    )
    _assert_absent(
        recovery,
        "github.event.workflow_run.conclusion == 'cancelled'",
        "cancellation_triggered_recovery",
    )

    # Production freezes must consume explicit, conservative OOS selection
    # evidence rather than treating the top raw score as sufficient.
    research = str(
        (ROOT / "scripts" / "run_daily_research.py").read_text(encoding="utf-8")
    )
    release_gate = str(
        (ROOT / "scripts" / "release_gate.py").read_text(encoding="utf-8")
    )
    locker = str(
        (ROOT / "scripts" / "lock_frozen_model.py").read_text(encoding="utf-8")
    )
    pipeline = str(
        (ROOT / "config" / "pipeline.yml").read_text(encoding="utf-8")
    )
    _assert_once(
        research,
        "paired_logloss_selection_evidence(",
        "oos_selection_evidence_used",
    )
    _assert_once(
        research,
        "candidate_models = make_models()",
        "oos_model_candidates_frozen_per_run",
    )
    _assert_once(
        research,
        '"model_candidate_manifest":',
        "oos_model_candidate_manifest_recorded",
    )
    _assert_once(
        research,
        'row["fold"] = float(fold_idx)',
        "oos_fold_identity_recorded",
    )
    _assert_once(
        locker,
        "global OOS model selection lacks statistically supported",
        "freeze_requires_selection_evidence",
    )
    _assert_once(
        release_gate,
        'pit_audit_path=Path("data/research/research_pit_contract_audit.json")',
        "release_gate_requires_pit_audit",
    )
    _assert_once(
        release_gate,
        'if pit_audit.get("status") not in {"PASS", "RESEARCH_ONLY_UNVERIFIED"}:',
        "release_gate_rejects_failed_pit_audit",
    )
    _assert_once(