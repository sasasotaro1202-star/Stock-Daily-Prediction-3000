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
    source_section_count = len(re.findall(r"(?m)^(\\d+)\\.\\s+[A-Z][A-Z0-9 /&._-]*$", project_source))
    if source_section_count != 98:
        raise SystemExit(
            f"FAIL: canonical Project Source section count expected 98, got {source_section_count}"
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
        release_gate,
        'reasons.append("research_pit_prediction_ledger_not_pass")',
        "release_gate_requires_pit_ledger_pass",
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

    market_cycle = _read("market-cycle.yml")
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
        market_cycle,
        'env:\n          REQUIRE_PIT_LEDGER: "1"',
        "market_cycle_enforces_pit_ledger_audit",
    )
    _assert_once(
        market_cycle,
        "run: python scripts/audit_research_pit_contract.py",
        "market_cycle_runs_pit_contract_audit_before_release_gate",
    )
    _assert_once(
        research,
        '"ranking_selection_ready_for_production": False',
        "ranking_selection_is_not_production_ready_without_nested_oos",
    )
    _assert_once(
        research,
        "nested_prequential_ranking_oos(",
        "nested_prequential_ranking_selector_used",
    )
    nested_ranking = str(
        (ROOT / "src" / "research" / "nested_ranking.py").read_text(encoding="utf-8")
    )
    _assert_once(
        nested_ranking,
        '"same_oos_global_model_or_window_reuse": False',
        "nested_ranking_blocks_same_oos_global_reuse",
    )
    _assert_once(
        nested_ranking,
        '"ranking_weight_selection_prequential": True',
        "nested_ranking_weight_selection_is_prequential",
    )
    _assert_once(
        nested_ranking,
        '"model_selection_prequential": True',
        "nested_ranking_model_selection_is_prequential",
    )
    _assert_once(
        nested_ranking,
        '"return_estimator_selection_prequential": True',
        "nested_ranking_return_selection_is_prequential",
    )
    _assert_once(
        nested_ranking,
        '"training_window_selection_prequential": training_window_selection_prequential',
        "nested_ranking_training_window_selection_is_prequential",
    )
    _assert_once(
        nested_ranking,
        '"window_selection_requires_contiguous_prior_folds": True',
        "nested_ranking_window_selection_requires_contiguous_evidence",
    )
    _assert_once(
        nested_ranking,
        "moving_block_bootstrap_mean(",
        "nested_ranking_bootstrap_evidence",
    )
    _assert_once(
        nested_ranking,
        '"bootstrap_method": "moving_block"',
        "nested_ranking_bootstrap_method_recorded",
    )
    learned_case_risk = str(
        (ROOT / "src" / "research" / "learned_case_risk_oos.py").read_text(encoding="utf-8")
    )
    _assert_once(
        learned_case_risk,
        "from src.research.statistics import moving_block_bootstrap_mean",
        "learned_case_risk_bootstrap_dependency",
    )
    _assert_once(
        learned_case_risk,
        "def _cluster_bootstrap_delta(",
        "learned_case_risk_session_cluster_bootstrap",
    )
    _assert_once(
        learned_case_risk,
        '"locked_logloss_bootstrap": locked_logloss_bootstrap',
        "learned_case_risk_bootstrap_artifact",
    )
    validation_learned_case = _read(
        "research-validation.yml",
    )
    _assert_once(
        validation_learned_case,
        "pytest -q tests/test_learned_case_risk_oos.py",
        "research_validation_runs_learned_case_risk_tests",
    )
    _assert_once(
        locker,
        'if payload.get("ranking_selection_ready_for_production") is not True:',
        "freeze_requires_nested_ranking_evidence",
    )
    _assert_once(
        locker,
        'if nested_ranking.get("same_oos_global_model_or_window_reuse") is not False:',
        "freeze_rejects_same_oos_ranking_reuse",
    )
    _assert_once(
        locker,
        'identity = nested_ranking.get("production_identity_alignment")',
        "freeze_reads_nested_ranking_identity_alignment",
    )
    _assert_once(
        locker,
        'identity.get("aligned") is not True',
        "freeze_requires_nested_ranking_identity_alignment",
    )
    _assert_once(
        locker,
        '"ranking_weight_selection_prequential",',
        "freeze_requires_prequential_ranking_weights",
    )
    _assert_once(
        locker,
        '"model_selection_prequential",',
        "freeze_requires_prequential_ranking_model_selection",
    )
    _assert_once(
        locker,
        '"return_estimator_selection_prequential",',
        "freeze_requires_prequential_ranking_return_selection",
    )
    _assert_once(
        locker,
        'bootstrap_probability < 0.90 or bootstrap_p05 <= 0.0',
        "freeze_requires_positive_bootstrap_ranking_evidence",
    )
    _assert_once(
        watchdog,
        'inspect_workflow "repository-verification.yml" "Repository verification" true true',
        "watchdog_recovers_stale_verification_queue",
    )
    _assert_once(
        watchdog,
        "rerunning the same current-main execution once",
        "watchdog_bounded_control_plane_failure_retry",
    )
    _assert_once(
        watchdog,
        'inspect_workflow "research-autopilot.yml" "Research autopilot" true',
        "watchdog_recovers_stale_autopilot_queue",
    )
    _assert_once(
        watchdog,
        'inspect_workflow "24h-research-marathon.yml" "24H Research Marathon" true',
        "watchdog_recovers_stale_marathon_queue",
    )
    _assert_once(
        str((WORKFLOWS / "24h-research-marathon-watchdog.yml").read_text(encoding="utf-8")),
        'HOLD marathon; verification_passed=',
        "marathon_watchdog_research_priority_hold",
    )


    _assert_once(
        watchdog,
        'inspect_research_validation()',
        "watchdog_monitors_research_validation",
    )
    _assert_once(
        watchdog,
        'inspect_workflow "automation-failure-learning.yml" "Automation failure learning" true true',
        "watchdog_recovers_stale_failure_learning",
    )
    _assert_once(
        watchdog,
        'inspect_workflow "experience-review.yml" "Experience review" true true',
        "watchdog_recovers_stale_experience_review",
    )
    _assert_once(
        watchdog,
        '".github/workflows/automation-failure-learning.yml"',
        "watchdog_wakes_on_failure_learning_changes",
    )

    _assert_once(
        autopilot,
        '    - cron: "17 12 * * *"',
        "research_autopilot_daily_schedule",
    )
    _assert_absent(
        autopilot,
        "  push:\n    branches: [main]",
        "research_autopilot_no_direct_push_trigger",
    )

    _assert_once(
        autopilot,
        'workflows: ["Repository verification"]',
        "research_autopilot_verification_event",
    )
    _assert_once(
        autopilot,
        "types: [completed]",
        "research_autopilot_verification_completed_event",
    )
    _assert_once(
        autopilot,
        "actions: write",
        "research_autopilot_can_recover_queues_and_dispatch",
    )
    _assert_once(
        autopilot,
        'gh workflow run research-validation.yml --repo "$GITHUB_REPOSITORY" --ref main',
        "research_autopilot_dispatches_current_main",
    )
    _assert_once(
        autopilot,
        'if [ "$verification_state" != "PASS" ]; then',
        "research_autopilot_requires_current_main_verification",
    )
    _assert_once(
        autopilot,
        'if [ "$active_current" -gt 0 ]; then',
        "research_autopilot_suppresses_duplicate_current_runs",
    )
    _assert_once(
        autopilot,
        "24 * 60 * 60",
        "research_autopilot_24h_success_window",
    )
    _assert_once(
        autopilot,
        "Stale queued Research runs cancelled",
        "research_autopilot_reports_stale_queue_recovery",
    )
    _assert_once(
        research,
        "workflow_dispatch:",
        "research_validation_manual_dispatch_retained",
    )
    _assert_absent(
        research,
        "  push:\n    branches: [main]",
        "research_validation_no_direct_push_dispatch",
    )

    _assert_once(
        watchdog,
        '  push:\n    paths:\n      - ".github/research_validation.trigger"',
        "watchdog_wakes_on_research_trigger",
    )
    _assert_absent(
        watchdog,
        'workflows: ["Research validation"]',
        "watchdog_not_triggered_by_research_workflow_run",
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
        'research_stale_epoch="$((now_epoch - 350 * 60))"',
        "watchdog_research_stale_timeout",
    )
    _assert_once(
        watchdog,
        'active_run_started_epoch="$(date -d "$run_started" +%s 2>/dev/null || echo 0)"',
        "watchdog_research_hard_age_source",
    )
    _assert_once(
        watchdog,
        "Only the newest watchdog should execute",
        "watchdog_latest_run_wins_comment",
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
    _assert_absent(
        watchdog,
        'active_head_sha" != "$current_sha"',
        "watchdog_research_does_not_cancel_only_for_superseded_sha",
    )
    _assert_once(
        watchdog,
        "Preserve an active chronological OOS across control-plane-only main changes.",
        "watchdog_research_preserves_active_oos",
    )
    _assert_once(
        watchdog,
        "Evidence-affecting changes invalidate the run and must release the concurrency",
        "watchdog_research_invalidates_evidence_changes",
    )
    _assert_absent(
        watchdog,
        'WATCHDOG_RESEARCH workflow=${workflow_name} active_run=$active_run_id head_sha=$active_head_sha current_sha=$current_sha"\n                return 0\n              fi\n            fi\n\n            local recovered_queue=false',
        "watchdog_research_no_duplicate_active_tail",
    )
    _assert_once(
        validation_workflow,
        'pip install -e ".[dev,research]"',
        "research_validation_installs_test_dependencies",
    )
    _assert_once(
        validation_workflow,
        '- "src/prediction/model_factories.py"',
        "research_validation_triggers_on_model_factory_changes",
    )
    _assert_once(
        validation_workflow,
        "pytest -q tests/test_model_factories_contract.py",
        "research_validation_runs_model_factory_contract",
    )
    status_workflow = _read("research-validation-status.yml")
    experience_workflow = _read("experience-review.yml")
    failure_workflow = _read("automation-failure-learning.yml")
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
        "types: [completed, in_progress, requested]",
        "research_validation_status_workflow_tracks_requested_and_completed",
    )
    _assert_once(
        status_workflow,
        "group: research-status-writer",
        "research_status_writer_shared_concurrency",
    )
    _assert_once(
        experience_workflow,
        'workflows: ["Prediction monitoring"]',
        "experience_review_runs_after_prediction_monitoring",
    )
    _assert_once(
        experience_workflow,
        "github.event.workflow_run.conclusion == 'success'",
        "experience_review_requires_successful_monitoring",
    )
    _assert_once(
        failure_workflow,
        "  workflow_dispatch:",
        "failure_learning_manual_recovery_trigger",
    )
    _assert_once(
        failure_workflow,
        'workflows:
      - "Research validation"',
        "failure_learning_research_validation_trigger",
    )
    _assert_once(
        failure_workflow,
        "research-status-writer",
        "failure_learning_shared_single_writer",
    )
    _assert_once(
        failure_workflow,
        "git push origin HEAD:research-status",
        "failure_learning_never_mutates_main",
    )
    _assert_absent(
        failure_workflow,
        "git push origin HEAD:main",
        "failure_learning_does_not_mutate_main",
    )
    _assert_once(
        status_workflow,
        "git push origin HEAD:research-status",
        "research_status_never_pushes_main",
    )
    _assert_absent(
        status_workflow,
        "git push origin HEAD:main",
        "research_status_does_not_mutate_main",
    )
    _assert_absent(
        validation_workflow,
        "git push origin HEAD:main",
        "research_snapshot_does_not_mutate_main",
    )
    _assert_absent(
        validation_workflow,
        "git push origin HEAD:research-status",
        "research_validation_no_direct_status_branch_push",
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