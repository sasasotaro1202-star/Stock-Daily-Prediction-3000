from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MEMORY_VERSION = 1
MAX_HARD_CASES = 500
MAX_ANOMALIES = 50

_BUCKETS = (
    "p<=0.25",
    "0.25<p<0.50",
    "0.50<=p<0.75",
    "p>=0.75",
)

_UNCERTAINTY_BUCKETS = (
    "unknown",
    "disagreement<0.01",
    "0.01<=disagreement<0.03",
    "0.03<=disagreement<0.06",
    "disagreement>=0.06",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _blank_group() -> dict[str, float]:
    return {
        "n": 0.0,
        "correct": 0.0,
        "logloss_sum": 0.0,
        "brier_sum": 0.0,
        "abs_return_error_sum": 0.0,
        "squared_return_error_sum": 0.0,
    }


def _safe_name(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    return str(value)


def _direction_bucket(p: float) -> str:
    if p <= 0.25:
        return "p<=0.25"
    if p < 0.50:
        return "0.25<p<0.50"
    if p < 0.75:
        return "0.50<=p<0.75"
    return "p>=0.75"


def _uncertainty_bucket(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "unknown"
    if value < 0.01:
        return "disagreement<0.01"
    if value < 0.03:
        return "0.01<=disagreement<0.03"
    if value < 0.06:
        return "0.03<=disagreement<0.06"
    return "disagreement>=0.06"


def _group_update(group: dict[str, Any], *, p: float, y: int, ret_error: float) -> None:
    eps = 1e-6
    pp = float(np.clip(p, eps, 1.0 - eps))
    logloss = -(y * np.log(pp) + (1 - y) * np.log(1 - pp))
    brier = (pp - y) ** 2
    group["n"] = float(group.get("n", 0.0) + 1.0)
    group["correct"] = float(group.get("correct", 0.0) + float((pp >= 0.5) == bool(y)))
    group["logloss_sum"] = float(group.get("logloss_sum", 0.0) + logloss)
    group["brier_sum"] = float(group.get("brier_sum", 0.0) + brier)
    group["abs_return_error_sum"] = float(
        group.get("abs_return_error_sum", 0.0) + abs(ret_error)
    )
    group["squared_return_error_sum"] = float(
        group.get("squared_return_error_sum", 0.0) + ret_error * ret_error
    )


def _new_memory() -> dict[str, Any]:
    return {
        "version": MEMORY_VERSION,
        "updated_at": _now(),
        "total_resolved": 0,
        "total_files_processed": 0,
        "by_asset_class": {},
        "by_model_id": {},
        "by_regime": {},
        "by_market_situation": {},
        "by_direction_confidence": {bucket: _blank_group() for bucket in _BUCKETS},
        "by_model_disagreement": {
            bucket: _blank_group() for bucket in _UNCERTAINTY_BUCKETS
        },
        "hard_cases": [],
        "processed_prediction_files": {},
        "anomalies": [],
        "research_priority": [],
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _new_memory()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"experience memory is unreadable: {exc}") from exc
    if payload.get("version") != MEMORY_VERSION:
        raise RuntimeError("experience memory version mismatch")
    return payload


def _append_anomaly(memory: dict[str, Any], message: str) -> None:
    rows = list(memory.get("anomalies", []))
    rows.append({"at": _now(), "message": message})
    memory["anomalies"] = rows[-MAX_ANOMALIES:]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _group_map(memory: dict[str, Any], key: str, value: Any) -> dict[str, float]:
    groups = memory.setdefault(key, {})
    name = _safe_name(value)
    group = groups.setdefault(name, _blank_group())
    return group


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    n = max(float(group.get("n", 0.0)), 1.0)
    out = dict(group)
    out["accuracy"] = float(group.get("correct", 0.0) / n)
    out["logloss"] = float(group.get("logloss_sum", 0.0) / n)
    out["brier"] = float(group.get("brier_sum", 0.0) / n)
    out["return_mae"] = float(group.get("abs_return_error_sum", 0.0) / n)
    out["return_rmse"] = float(
        max(group.get("squared_return_error_sum", 0.0), 0.0) ** 0.5 / (n**0.5)
    )
    return out


def _rebuild_priority(memory: dict[str, Any]) -> None:
    total = max(int(memory.get("total_resolved", 0)), 1)
    overall_ll = (
        sum(
            float(v.get("logloss_sum", 0.0))
            for v in memory.get("by_regime", {}).values()
        )
        / total
        if memory.get("by_regime")
        else 0.0
    )
    priorities: list[dict[str, Any]] = []
    for dimension in ("by_asset_class", "by_model_id", "by_regime", "by_market_situation"):
        for name, group in memory.get(dimension, {}).items():
            n = int(group.get("n", 0))
            if n < 50:
                continue
            ll = float(group.get("logloss_sum", 0.0)) / max(n, 1)
            impact = (ll - overall_ll) * n
            priorities.append(
                {
                    "dimension": dimension,
                    "segment": name,
                    "n": n,
                    "logloss": ll,
                    "impact_vs_global_logloss": float(impact),
                }
            )
    priorities.sort(key=lambda row: (row["impact_vs_global_logloss"], row["n"]), reverse=True)
    memory["research_priority"] = priorities[:20]


def update_experience_memory(
    predictions: pd.DataFrame,
    prediction_dir: Path,
    memory_path: Path,
    file_complete: dict[str, bool] | None = None,
) -> dict[str, Any]:
    memory = _load(memory_path)
    if predictions.empty or "_prediction_file" not in predictions.columns:
        return memory

    processed = dict(memory.get("processed_prediction_files", {}))
    new_frames: list[pd.DataFrame] = []
    completeness = file_complete or {}

    for filename in sorted(predictions["_prediction_file"].dropna().unique()):
        source = prediction_dir / str(filename)
        if not source.exists():
            _append_anomaly(memory, f"missing prediction file: {filename}")
            continue
        fingerprint = _file_sha256(source)
        old = processed.get(str(filename))
        if old == fingerprint:
            continue
        if old is not None and old != fingerprint:
            _append_anomaly(memory, f"prediction file changed after processing; skipped: {filename}")
            continue
        if not bool(completeness.get(str(filename), file_complete is None)):
            _append_anomaly(memory, f"prediction file outcome incomplete; deferred: {filename}")
            continue
        frame = predictions[predictions["_prediction_file"] == filename].copy()
        if frame.empty:
            processed[str(filename)] = fingerprint
            continue
        new_frames.append(frame)
        processed[str(filename)] = fingerprint

    if not new_frames:
        memory["updated_at"] = _now()
        memory["processed_prediction_files"] = processed
        memory["total_files_processed"] = len(processed)
        _rebuild_priority(memory)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(json.dumps(memory, indent=2, default=str), encoding="utf-8")
        return memory

    new_data = pd.concat(new_frames, ignore_index=True)
    new_data = new_data.dropna(
        subset=["forward_return_1d", "p_up_1d", "expected_return_1d"]
    )

    for row in new_data.itertuples(index=False):
        p = float(row.p_up_1d)
        y = int(float(row.forward_return_1d) > 0.0)
        ret_error = float(row.expected_return_1d) - float(row.forward_return_1d)
        memory["total_resolved"] = int(memory.get("total_resolved", 0)) + 1

        for key, value in (
            ("by_asset_class", getattr(row, "asset_class", None)),
            ("by_model_id", getattr(row, "model_id", None)),
            ("by_regime", getattr(row, "regime", None)),
            ("by_market_situation", getattr(row, "market_situation", None)),
        ):
            _group_update(_group_map(memory, key, value), p=p, y=y, ret_error=ret_error)

        _group_update(
            memory["by_direction_confidence"][_direction_bucket(p)],
            p=p,
            y=y,
            ret_error=ret_error,
        )
        disagreement = getattr(row, "model_disagreement", np.nan)
        _group_update(
            memory["by_model_disagreement"][
                _uncertainty_bucket(float(disagreement) if pd.notna(disagreement) else None)
            ],
            p=p,
            y=y,
            ret_error=ret_error,
        )

    memory["processed_prediction_files"] = processed
    memory["total_files_processed"] = len(processed)

    # Keep a bounded memory of the most informative failures. These are
    # diagnostics/research inputs only; they never auto-promote a model.
    hard_rows: list[dict[str, Any]] = list(memory.get("hard_cases", []))
    for row in new_data.itertuples(index=False):
        p = float(row.p_up_1d)
        y = int(float(row.forward_return_1d) > 0.0)
        ret_error = float(row.expected_return_1d) - float(row.forward_return_1d)
        direction_wrong = int((p >= 0.5) != bool(y))
        surprise = abs(p - y)
        hardness = abs(ret_error) + 0.02 * surprise + 0.02 * direction_wrong
        hard_rows.append(
            {
                "hardness": float(hardness),
                "symbol": _safe_name(getattr(row, "symbol", None)),
                "asset_class": _safe_name(getattr(row, "asset_class", None)),
                "session_date": _safe_name(getattr(row, "session_date", None)),
                "prediction_time": _safe_name(getattr(row, "prediction_time", None)),
                "model_id": _safe_name(getattr(row, "model_id", None)),
                "regime": _safe_name(getattr(row, "regime", None)),
                "market_situation": _safe_name(getattr(row, "market_situation", None)),
                "p_up_1d": p,
                "actual_up": y,
                "expected_return_1d": float(row.expected_return_1d),
                "realized_return_1d": float(row.forward_return_1d),
                "return_error": ret_error,
                "model_disagreement": (
                    float(row.model_disagreement)
                    if pd.notna(getattr(row, "model_disagreement", np.nan))
                    else None
                ),
                "model_version": _safe_name(getattr(row, "model_version", None)),
            }
        )

    hard_rows.sort(key=lambda x: float(x.get("hardness", 0.0)), reverse=True)
    memory["hard_cases"] = hard_rows[:MAX_HARD_CASES]
    memory["updated_at"] = _now()
    _rebuild_priority(memory)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(json.dumps(memory, indent=2, default=str), encoding="utf-8")
    return memory


def compact_view(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": memory.get("version"),
        "updated_at": memory.get("updated_at"),
        "total_resolved": memory.get("total_resolved", 0),
        "total_files_processed": memory.get("total_files_processed", 0),
        "research_priority": memory.get("research_priority", []),
        "top_hard_cases": memory.get("hard_cases", [])[:20],
        "anomalies": memory.get("anomalies", [])[-10:],
        "by_model_id": {
            k: _finalize_group(v) for k, v in memory.get("by_model_id", {}).items()
        },
        "by_regime": {
            k: _finalize_group(v) for k, v in memory.get("by_regime", {}).items()
        },
        "by_market_situation": {
            k: _finalize_group(v)
            for k, v in memory.get("by_market_situation", {}).items()
        },
    }
