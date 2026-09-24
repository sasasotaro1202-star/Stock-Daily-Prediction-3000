from __future__ import annotations

import itertools
import json
from math import comb
from pathlib import Path

import pandas as pd

PRICE = Path("data/prices/canonical.parquet")
OUT = Path("data/research/cpcv_validation_audit.json")


def _build_groups(n_dates: int, n_groups: int) -> list[list[int]]:
    if n_dates < n_groups:
        raise ValueError("n_dates must be >= n_groups")
    return [
        list(range((g * n_dates) // n_groups, ((g + 1) * n_dates) // n_groups))
        for g in range(n_groups)
    ]


def _forbidden_positions(
    test_positions: list[int] | tuple[int, ...] | set[int],
    n_dates: int,
    *,
    purge: int,
    embargo: int,
) -> set[int]:
    forbidden: set[int] = set()
    for pos in test_positions:
        for delta in range(-max(0, purge), max(0, embargo) + 1):
            candidate = int(pos) + delta
            if 0 <= candidate < n_dates:
                forbidden.add(candidate)
    return forbidden


def validate_split(
    train_positions: list[int] | set[int],
    test_positions: list[int] | set[int],
    n_dates: int,
    *,
    purge: int,
    embargo: int,
) -> list[str]:
    train = set(int(x) for x in train_positions)
    test = set(int(x) for x in test_positions)
    violations: list[str] = []
    if train & test:
        violations.append("train_test_overlap")
    expected_forbidden = _forbidden_positions(
        test,
        n_dates,
        purge=purge,
        embargo=embargo,
    )
    leaked = train & expected_forbidden
    if leaked:
        violations.append(
            f"purge_or_embargo_violation:{sorted(leaked)[:10]}"
        )
    # For the repository's 1-session forward label, intervals are
    # [position, position + 1]. Any train start within one session before,
    # on, or one session after a test start is unsafe.
    for train_pos in train:
        for test_pos in test:
            if train_pos in {
                test_pos - 1,
                test_pos,
                test_pos + 1,
            }:
                violations.append(
                    f"label_interval_overlap:{train_pos}:{test_pos}"
                )
                return violations
    return violations


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not PRICE.exists():
        raise SystemExit("DEFERRED: canonical price dataset is absent")

    frame = pd.read_parquet(PRICE)
    if "session_date" not in frame.columns:
        raise SystemExit("FAIL: canonical price dataset has no session_date")

    dates = sorted(
        pd.to_datetime(frame["session_date"], errors="coerce")
        .dt.date.dropna().unique()
    )
    if len(dates) < 30:
        raise SystemExit(f"DEFERRED: insufficient sessions for CPCV audit ({len(dates)})")

    # Six groups / two held-out groups gives multiple combinatorial test
    # paths while remaining lightweight enough for every market-cycle run.
    n_groups = 6 if len(dates) >= 60 else max(3, len(dates) // 10)
    n_test_groups = 2 if n_groups >= 4 else 1
    purge = 1
    embargo = 1
    groups = _build_groups(len(dates), n_groups)

    fold_records = []
    violations: list[dict[str, object]] = []
    for combo in itertools.combinations(range(n_groups), n_test_groups):
        test_positions = sorted(
            pos for group_id in combo for pos in groups[group_id]
        )
        test_set = set(test_positions)
        train_positions = [
            pos
            for pos in range(len(dates))
            if pos not in test_set
            and pos not in _forbidden_positions(
                test_positions,
                len(dates),
                purge=purge,
                embargo=embargo,
            )
        ]
        split_violations = validate_split(
            train_positions,
            test_positions,
            len(dates),
            purge=purge,
            embargo=embargo,
        )
        record = {
            "test_groups": list(combo),
            "n_train_dates": len(train_positions),
            "n_test_dates": len(test_positions),
            "violations": split_violations,
        }
        fold_records.append(record)
        if split_violations:
            violations.append(record)

    expected_paths = comb(n_groups - 1, n_test_groups - 1)
    payload = {
        "status": "PASS" if not violations else "FAIL",
        "research_only": True,
        "production_changed": False,
        "production_promotion": False,
        "method": "combinatorial_purged_cross_validation_boundary_audit",
        "n_sessions": len(dates),
        "n_groups": n_groups,
        "n_test_groups": n_test_groups,
        "purge_sessions": purge,
        "embargo_sessions": embargo,
        "number_of_combinatorial_splits": len(fold_records),
        "expected_canonical_backtest_paths": expected_paths,
        "verified_splits": len(fold_records) - len(violations),
        "violations": violations,
        "note": (
            "This is a split-boundary leakage audit, not a replacement for "
            "the repository's chronological WFO or frozen holdout. It does "
            "not prove feature-level causality."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if violations:
        raise SystemExit("FAIL: CPCV boundary audit found leakage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
