from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

MONITOR = Path("data/research/monitor_latest.json")
SNAPSHOT = Path("data/research/performance_snapshot.json")
CHANGE = Path("data/research/performance_change.json")
RELEASE_GATE = Path("data/research/release_gate.json")
HOLDOUT = Path("data/research/frozen_holdout_result.json")
FROZEN_MODEL = Path("config/frozen_holdout.json")

SCOPES = ("overall", "recent_20_sessions")
LOWER_IS_BETTER = {"logloss", "brier", "ece", "return_mae", "return_rmse"}
METRICS = (
    "logloss",
    "brier",
    "ece",
    "accuracy",
    "roc_auc",
    "return_mae",
    "return_rmse",
)


def _finite_number(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _scope_snapshot(payload: dict[str, Any], scope: str) -> dict[str, Any]:
    row = payload.get(scope)
    if not isinstance(row, dict):
        return {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    out: dict[str, Any] = {"rows": row.get("rows")}
    for key in METRICS:
        source = metrics.get(key)
        if source is None and key in {"return_mae", "return_rmse"}:
            source = row.get(key)
        if _finite_number(source):
            out[key] = float(source)
    if isinstance(row.get("beats_baseline"), bool):
        out["beats_baseline"] = row["beats_baseline"]
    if not any(key in out for key in METRICS):
        return {}
    return out


def build_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": str(payload.get("status", "UNKNOWN")),
        "evaluated": int(payload.get("evaluated", 0) or 0),
        "latest_outcome_date": payload.get("latest_outcome_date"),
        "evaluation_period": {
            "start": payload.get("evaluation_start_date"),
            "end": payload.get("latest_outcome_date"),
        },
        "sample_size": {
            "evaluated": int(payload.get("evaluated", 0) or 0),
            "overall": int(((payload.get("overall") or {}).get("rows", 0) or 0)),
            "recent_20_sessions": int(
                ((payload.get("recent_20_sessions") or {}).get("rows", 0) or 0)
            ),
        },
        "context": _report_context(payload),
        "scopes": {
            scope: snapshot
            for scope in SCOPES
            if (snapshot := _scope_snapshot(payload, scope))
        },
    }


def _report_context(payload: dict[str, Any]) -> dict[str, Any]:
    context = {
        "champion": payload.get("champion")
        or payload.get("current_champion")
        or payload.get("production_model_id"),
        "prior_champion": payload.get("prior_champion"),
        "candidate": payload.get("candidate")
        or payload.get("challenger")
        or payload.get("candidate_model"),
        "decision": payload.get("decision"),
        "production_status": payload.get("production_status"),
        "pit_status": payload.get("pit_status"),
        "latest_holdout": payload.get("latest_holdout"),
        "robustness": payload.get("robustness"),
        "benchmark": payload.get("benchmark"),
        "cost": payload.get("cost"),
    }

    # Optional artifacts are evidence only. Never infer a production/model state
    # from the mere existence of a file.
    release = _load_json(RELEASE_GATE)
    holdout = _load_json(HOLDOUT)
    frozen = _load_json(FROZEN_MODEL)

    if context["decision"] is None and release:
        if isinstance(release.get("decision"), str):
            context["decision"] = release["decision"]
        elif isinstance(release.get("approved"), bool):
            context["decision"] = "APPROVED" if release["approved"] else "NOT_APPROVED"

    if context["production_status"] is None and isinstance(frozen.get("status"), str):
        context["production_status"] = frozen["status"]

    if context["latest_holdout"] is None and holdout:
        context["latest_holdout"] = {
            "status": holdout.get("status", "UNVERIFIABLE"),
            "evidence": holdout.get("metrics")
            or holdout.get("overall")
            or holdout.get("evaluation"),
        }

    def state(value: Any, default: str) -> str:
        if isinstance(value, str) and value:
            return value
        return default

    holdout_status = (
        context["latest_holdout"].get("status")
        if isinstance(context["latest_holdout"], dict)
        else context["latest_holdout"]
    )
    robustness_status = (
        context["robustness"].get("status")
        if isinstance(context["robustness"], dict)
        else context["robustness"]
    )

    for key in ("champion", "prior_champion", "candidate"):
        if context[key] is None:
            context[key] = "UNVERIFIABLE"

    context["decision"] = state(context["decision"], "UNKNOWN")
    context["production_status"] = state(context["production_status"], "UNVERIFIABLE")
    context["pit_status"] = state(context["pit_status"], "UNVERIFIABLE")
    context["latest_holdout_status"] = state(holdout_status, "UNVERIFIABLE")
    context["robustness_status"] = state(robustness_status, "UNVERIFIABLE")
    context["evidence_sources"] = {
        "release_gate": "AVAILABLE" if release else "UNVERIFIABLE",
        "frozen_holdout": "AVAILABLE" if holdout else "UNVERIFIABLE",
        "frozen_model": "AVAILABLE" if frozen else "UNVERIFIABLE",
    }
    return context


def compare_snapshots(previous: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    previous_scopes = previous.get("scopes") if isinstance(previous.get("scopes"), dict) else {}
    current_scopes = current.get("scopes") if isinstance(current.get("scopes"), dict) else {}
    for scope in SCOPES:
        old = previous_scopes.get(scope) if isinstance(previous_scopes.get(scope), dict) else {}
        new = current_scopes.get(scope) if isinstance(current_scopes.get(scope), dict) else {}
        for metric in METRICS:
            ov = old.get(metric)
            nv = new.get(metric)
            if not (_finite_number(ov) and _finite_number(nv)):
                continue
            if float(ov) == float(nv):
                continue
            delta = float(nv) - float(ov)
            improved = delta < 0 if metric in LOWER_IS_BETTER else delta > 0
            relative_delta = delta / abs(float(ov)) if float(ov) != 0.0 else None
            row_sample = new.get("rows") or old.get("rows")
            changes.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "previous": float(ov),
                    "current": float(nv),
                    "delta": round(delta, 10),
                    "relative_delta": (
                        round(relative_delta, 10)
                        if relative_delta is not None
                        else None
                    ),
                    "direction": "improved" if improved else "worsened",
                    "sample_size": (
                        int(row_sample)
                        if _finite_number(row_sample)
                        else None
                    ),
                }
            )
    return changes


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or not path.stat().st_size:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _append_summary(
    snapshot: dict[str, Any],
    changes: list[dict[str, Any]],
    reason: str | None = None,
) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = ["## Stock prediction performance"]
    if reason:
        lines.append(f"- State: `{reason}`")
    lines.append(f"- Monitor status: `{snapshot.get('status', 'UNKNOWN')}`")
    lines.append(f"- Evaluated outcomes: `{snapshot.get('evaluated', 0)}`")
    period = snapshot.get("evaluation_period") or {}
    lines.append(
        f"- Evaluation period: `{period.get('start', 'UNKNOWN')}` → "
        f"`{period.get('end', 'UNKNOWN')}`"
    )
    context = snapshot.get("context") or {}
    lines.append(f"- Champion: `{context.get('champion', 'UNVERIFIABLE')}`")
    lines.append(f"- Prior champion: `{context.get('prior_champion', 'UNVERIFIABLE')}`")
    lines.append(f"- Candidate: `{context.get('candidate', 'UNVERIFIABLE')}`")
    lines.append(f"- Decision: `{context.get('decision', 'UNKNOWN')}`")
    lines.append(
        f"- Production status: `{context.get('production_status', 'UNVERIFIABLE')}`"
    )
    lines.append(f"- Latest holdout: `{context.get('latest_holdout_status', 'UNVERIFIABLE')}`")
    lines.append(f"- Robustness: `{context.get('robustness_status', 'UNVERIFIABLE')}`")
    lines.append(f"- PIT: `{context.get('pit_status', 'UNVERIFIABLE')}`")
    evidence = context.get("evidence_sources") or {}
    lines.append(
        "- Evidence files: "
        f"release_gate={evidence.get('release_gate', 'UNVERIFIABLE')}, "
        f"frozen_holdout={evidence.get('frozen_holdout', 'UNVERIFIABLE')}, "
        f"frozen_model={evidence.get('frozen_model', 'UNVERIFIABLE')}"
    )
    for scope in SCOPES:
        row = (snapshot.get("scopes") or {}).get(scope)
        if not isinstance(row, dict):
            continue
        values = []
        for metric in METRICS:
            if metric in row:
                values.append(f"{metric}={row[metric]}")
        if values:
            lines.append(f"- **{scope}**: " + ", ".join(values))
    if changes:
        lines.append(f"- **Metric changes vs previous snapshot: {len(changes)}**")
        for change in changes:
            lines.append(
                f"  - `{change['scope']}` `{change['metric']}`: "
                f"{change['previous']} → {change['current']} "
                f"(Δ {change['delta']:+g}, {change.get('relative_delta', 0):+.4%} relative, "
                f"{change['direction']}, n={change.get('sample_size', 'n/a')})"
            )
    else:
        lines.append("- **No numeric metric change vs previous snapshot detected.**")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    if not MONITOR.is_file():
        context = _report_context({})
        change = {
            "schema_version": 1,
            "changed": False,
            "reason": "monitor_missing",
            "changes": [],
            "champion": context["champion"],
            "prior_champion": context["prior_champion"],
            "candidate": context["candidate"],
            "decision": context["decision"],
            "production_status": context["production_status"],
            "latest_holdout_status": context["latest_holdout_status"],
            "robustness_status": context["robustness_status"],
            "pit_status": context["pit_status"],
            "sample_size": {"evaluated": 0},
            "evaluation_period": {"start": None, "end": None},
            "report_context": context,
        }
        CHANGE.parent.mkdir(parents=True, exist_ok=True)
        CHANGE.write_text(json.dumps(change, indent=2, sort_keys=True), encoding="utf-8")
        _append_summary({}, [], "monitor_missing")
        return 0

    payload = _load_json(MONITOR)
    current = build_snapshot(payload)
    previous = _load_json(SNAPSHOT)
    changes = compare_snapshots(previous, current) if previous else []
    has_metrics = bool(current.get("scopes"))

    context = current.get("context") or {}
    change = {
        "schema_version": 1,
        "changed": bool(changes),
        "reason": "metric_change" if changes else "no_change",
        "changes": changes,
        "champion": context.get("champion"),
        "prior_champion": context.get("prior_champion"),
        "candidate": context.get("candidate"),
        "decision": context.get("decision"),
        "production_status": context.get("production_status"),
        "latest_holdout_status": context.get("latest_holdout_status"),
        "robustness_status": context.get("robustness_status"),
        "pit_status": context.get("pit_status"),
        "sample_size": current.get("sample_size"),
        "evaluation_period": current.get("evaluation_period"),
        "report_context": context,
    }
    CHANGE.parent.mkdir(parents=True, exist_ok=True)
    CHANGE.write_text(json.dumps(change, indent=2, sort_keys=True), encoding="utf-8")

    if has_metrics:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(
            json.dumps(current, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    _append_summary(current, changes, None if has_metrics else str(current.get("status", "UNKNOWN")))
    print(json.dumps(change, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
