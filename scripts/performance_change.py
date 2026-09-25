from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

MONITOR = Path("data/research/monitor_latest.json")
SNAPSHOT = Path("data/research/performance_snapshot.json")
CHANGE = Path("data/research/performance_change.json")

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
        "scopes": {
            scope: snapshot
            for scope in SCOPES
            if (snapshot := _scope_snapshot(payload, scope))
        },
    }


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
            changes.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "previous": float(ov),
                    "current": float(nv),
                    "delta": round(delta, 10),
                    "direction": "improved" if improved else "worsened",
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
                f"(Δ {change['delta']:+g}, {change['direction']})"
            )
    else:
        lines.append("- **No numeric metric change vs previous snapshot detected.**")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    if not MONITOR.is_file():
        change = {"schema_version": 1, "changed": False, "reason": "monitor_missing", "changes": []}
        CHANGE.parent.mkdir(parents=True, exist_ok=True)
        CHANGE.write_text(json.dumps(change, indent=2, sort_keys=True), encoding="utf-8")
        _append_summary({}, [], "monitor_missing")
        return 0

    payload = _load_json(MONITOR)
    current = build_snapshot(payload)
    previous = _load_json(SNAPSHOT)
    changes = compare_snapshots(previous, current) if previous else []
    has_metrics = bool(current.get("scopes"))

    change = {
        "schema_version": 1,
        "changed": bool(changes),
        "reason": "metric_change" if changes else "no_change",
        "changes": changes,
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
