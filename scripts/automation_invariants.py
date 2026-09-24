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
        '    - cron: "7,22,37,52 * * * *"',
        "watchdog_15m_schedule",
    )
    for due in ("07:17", "09:27", "18:37"):
        _assert_once(watchdog, due, f"watchdog_recovery_window_{due.replace(':', '_')}")

    _assert_once(
        heartbeat,
        '    - cron: "17 */6 * * *"',
        "heartbeat_6h_schedule",
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
