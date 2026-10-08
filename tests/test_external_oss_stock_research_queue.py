from __future__ import annotations

import json
from pathlib import Path

RESOLUTION = Path("scripts/resolve_external_oss_top1000_not_found.py")
QUEUE = Path("scripts/build_external_oss_stock_research_queue.py")


def test_identity_resolution_is_research_only():
    text = RESOLUTION.read_text(encoding="utf-8")
    assert "auto_rewrite_allowed" in text
    assert "promotion_allowed" in text
    assert "source_changed" not in text


def test_stock_queue_is_research_only():
    text = QUEUE.read_text(encoding="utf-8")
    assert "triage_is_not_performance_evidence" in text
    assert "promotion_allowed" in text
    assert "FROZEN_HOLDOUT" in text
