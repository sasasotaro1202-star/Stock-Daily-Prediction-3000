from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SOURCE = Path("research_sources/external_prediction_system_oss_top1000_source.tsv")
SCRIPT = Path("scripts/build_external_prediction_system_oss_top1000_registry.py")


def test_source_has_exactly_1000_ranked_rows():
    rows = [
        line for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line and line[0].isdigit()
    ]
    ranks = [int(line.split("\t", 1)[0]) for line in rows]
    assert len(rows) == 1000
    assert len(set(ranks)) == 1000
    assert sorted(ranks) == list(range(1, 1001))


def test_registry_builder_is_fail_closed_and_research_only():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data = json.loads(
        Path("config/external_prediction_system_oss_top1000_registry.json")
        .read_text(encoding="utf-8")
    )
    assert data["source_rank_integrity"]["complete"] is True
    assert data["source_rank_integrity"]["actual_count"] == 1000
    assert data["safety_contract"]["research_only"] is True
    assert data["safety_contract"]["production_changed"] is False
    assert data["safety_contract"]["promotion_allowed"] is False
    assert data["safety_contract"]["frozen_holdout_used"] is False
    assert len(data["rows"]) == 1000
