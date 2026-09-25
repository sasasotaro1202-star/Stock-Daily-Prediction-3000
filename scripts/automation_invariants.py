from __future__ import annotations

from pathlib import Path


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


def main() -> int:
    market = _read("market-cycle.yml")
    monitoring = _read("prediction-monitoring.yml")
    watchdog = _read("actions-reliability-watchdog.yml")
    heartbeat = _read("heartbeat.yml")
    recovery = _read("bounded-production-recovery.yml")
    price_restore = str((WORKFLOWS.parent.parent / "scripts" / "restore_latest_price_state.py").read_text(encoding="utf-8"))

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
    for due in ("07:17", "09:27", "18:37"):
        _assert_once(watchdog, due, f"watchdog_recovery_window_{due.replace(':', '_')}")

    _assert_once(
        heartbeat,
        '    - cron: "17 */6 * * *"',
        "heartbeat_6h_schedule",
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
        'row["fold"] = float(fold_idx)',
        "oos_fold_identity_recorded",
    )
    _assert_once(
        locker,
        "global OOS model selection lacks statistically supported",
        "freeze_requires_selection_evidence",
    )
    _assert_once(
        pipeline,
        "selection_evidence_min_relative_improvement: 0.03",
        "selection_evidence_min_three_percent_gain",
    )
    _assert_once(
        pipeline,
        "selection_evidence_min_common_oos_folds: 5",
        "selection_evidence_min_five_common_folds",
    )
    _assert_once(
        pipeline,
        "selection_evidence_alpha: 0.05",
        "selection_evidence_alpha_floor",
    )

    _assert_once(
        research,
        "select_temporal_calibration_method(",
        "temporal_calibration_router_used",
    )
    _assert_once(
        pipeline,
        "temporal_router_research_only: true",
        "temporal_calibration_is_research_only",
    )
    _assert_once(
        pipeline,
        "temporal_min_history_folds: 2",
        "temporal_calibration_min_history",
    )

    _assert_once(
        research,
        "select_drift_aware_window(",
        "drift_aware_window_router_used",
    )
    _assert_once(
        pipeline,
        "drift_aware_window:",
        "drift_aware_window_config",
    )
    _assert_once(
        pipeline,
        "drift_scale: 0.50",
        "drift_aware_window_scale",
    )
    _assert_once(
        pipeline,
        "sequential_selection_research:",
        "sequential_selection_research_config",
    )
    _assert_once(
        research,
        "chronological_policy_oos(",
        "sequential_selection_research_call",
    )

    _assert_once(price_restore, "falling back to bounded fresh price fetch", "price_restore_auth_fallback")

    validation_workflow = _read("research-validation.yml")
    quality_gate = str(
        (ROOT / "scripts" / "data_quality_gate.py").read_text(encoding="utf-8")
    )
    _assert_once(
        quality_gate,
        '"provider_deferred_ratio_over_5pct",',
        "excessive_provider_deferral_fails_closed",
    )
    _assert_once(
        validation_workflow,
        "cancel-in-progress: false",
        "research_validation_preserves_active_oos_runs",
    )
    _assert_once(
        watchdog,
        'inspect_research_validation()',
        "watchdog_monitors_research_validation",
    )

    _assert_once(
        watchdog,
        '  push:\n    paths:\n      - ".github/research_validation.trigger"',
        "watchdog_wakes_on_research_trigger",
    )
    _assert_once(
        watchdog,
        "current_sha_queued=true",
        "watchdog_suppresses_duplicate_research_dispatch",
    )
    _assert_once(
        watchdog,
        "current-SHA queued run already exists after stale queue cleanup; suppressing duplicate dispatch",
        "watchdog_reports_duplicate_dispatch_suppression",
    )
    _assert_once(
        watchdog,
        'research_stale_epoch="$((now_epoch - 150 * 60))"',
        "watchdog_research_stale_timeout",
    )
    _assert_once(
        watchdog,
        'active_run_started_epoch="$(date -d "$run_started" +%s 2>/dev/null || echo 0)"',
        "watchdog_research_hard_age_source",
    )
    _assert_once(
        watchdog,
        'active_run_started_epoch" -gt 0',
        "watchdog_research_hard_age_guard",
    )
    _assert_once(
        watchdog,
        'if [[ "$head_sha" != "$current_sha" ]]; then',
        "watchdog_research_queue_compares_current_sha",
    )
    _assert_once(
        validation_workflow,
        'pip install -e ".[dev,research]"',
        "research_validation_installs_test_dependencies",
    )
    status_workflow = _read("research-validation-status.yml")
    _assert_absent(
        validation_workflow,
        "research-status:",
        "research_validation_no_inline_status_job",
    )
    _assert_once(
        status_workflow,
        'workflows: ["Research validation"]',
        "research_validation_status_workflow_trigger",
    )
    _assert_once(
        status_workflow,
        "types: [in_progress, completed]",
        "research_validation_status_workflow_tracks_active_and_completed",
    )
    _assert_once(
        status_workflow,
        "group: research-validation-status",
        "research_validation_status_serialized_commits",
    )
    _assert_once(
        status_workflow,
        "RESEARCH_WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id }}",
        "research_validation_status_targets_research_run",
    )
    _assert_once(
        status_workflow,
        "RESEARCH_WORKFLOW_SHA: ${{ github.event.workflow_run.head_sha }}",
        "research_validation_status_targets_research_sha",
    )
    _assert_once(
        status_workflow,
        "python scripts/persist_research_validation_status.py",
        "research_validation_status_script",
    )
    _assert_once(
        validation_workflow,
        "if: always()\n        uses: actions/upload-artifact@v7",
        "research_validation_uploads_partial_evidence",
    )
    _assert_once(
        validation_workflow,
        "if-no-files-found: warn",
        "research_validation_partial_evidence_warning",
    )
    _assert_once(
        validation_workflow,
        '      - ".github/research_validation.trigger"',
        "research_validation_explicit_trigger",
    )
    _assert_absent(
        validation_workflow,
        "build_production_artifact.py",
        "research_validation_no_production_artifact_build",
    )
    _assert_absent(
        validation_workflow,
        "lock_frozen_model.py",
        "research_validation_no_model_lock",
    )

    # Guard against silently masking automation failures in the critical lane.
    for name, source in {
        "market-cycle": market,
        "prediction-monitoring": monitoring,
        "watchdog": watchdog,
        "recovery": recovery,
    }.items():
        _assert_absent(source, "|| true", f"{name}_no_failure_mask")

    print("automation-invariants: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
