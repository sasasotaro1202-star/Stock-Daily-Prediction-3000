from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.run_daily_research import _refresh_universe_with_bounded_retry




def test_adaptive_universe_refresh_skips_recent_verified_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    universe_dir = tmp_path / "data" / "universe"
    universe_dir.mkdir(parents=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    (universe_dir / "latest.json").write_text(
        json.dumps(
            {
                "retrieved_at": retrieved_at,
                "source_hashes": {"jp": "hash-jp", "us": "hash-us"},
                "record_count": 100,
                "records": [
                    {
                        "asset_class": "us_stock",
                        "symbol": f"T{i}",
                        "name": f"Test {i}",
                        "tradeable": True,
                        "source_url": "https://www.paypay-sec.co.jp/test",
                    }
                    for i in range(100)
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    calls: list[list[str]] = []

    def fake_run(args, check=True, timeout=None, **kwargs):
        calls.append(list(args))
        assert args == ["python", "scripts/universe_quality_gate.py"]
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("scripts.run_daily_research.subprocess.run", fake_run)

    result = _refresh_universe_with_bounded_retry(max_age_seconds=6 * 60 * 60)

    assert result["status"] == "SKIPPED_RECENT_EXISTING"
    assert result["refresh_available"] is True
    assert result["fallback_used"] is False
    assert result["reason"] == "existing_universe_within_bounded_refresh_window"
    assert result["record_count"] == 100
    assert calls == [["python", "scripts/universe_quality_gate.py"]]


def test_adaptive_universe_refresh_retains_verified_snapshot_after_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    universe_dir = tmp_path / "data" / "universe"
    universe_dir.mkdir(parents=True)
    (universe_dir / "latest.json").write_text(
        json.dumps(
            {
                "retrieved_at": "2026-10-08T00:00:00+00:00",
                "source_hashes": {"jp": "hash-jp", "us": "hash-us"},
                "record_count": 100,
                "records": [
                    {
                        "asset_class": "us_stock",
                        "symbol": f"T{i}",
                        "name": f"Test {i}",
                        "tradeable": True,
                        "source_url": "https://www.paypay-sec.co.jp/test",
                    }
                    for i in range(100)
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    calls: list[tuple[list[str], int]] = []

    def fake_run(args, check=True, timeout=None, **kwargs):
        calls.append((list(args), int(timeout)))
        if args[-1] == "scripts/refresh_universe.py":
            raise subprocess.TimeoutExpired(args, timeout)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("scripts.run_daily_research.subprocess.run", fake_run)

    result = _refresh_universe_with_bounded_retry()

    assert result["status"] == "DEFERRED_RETAINED_EXISTING"
    assert result["refresh_available"] is False
    assert result["fallback_used"] is True
    assert result["record_count"] == 100
    assert result["reason"] == "live_refresh_timeout"
    assert [timeout for _, timeout in calls[:2]] == [120, 240]
    assert calls[2][0] == ["python", "scripts/universe_quality_gate.py"]


def test_adaptive_universe_refresh_still_fails_closed_without_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    def fake_run(args, check=True, timeout=None, **kwargs):
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr("scripts.run_daily_research.subprocess.run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired):
        _refresh_universe_with_bounded_retry()


def test_adaptive_universe_refresh_rejects_stale_retained_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    universe_dir = tmp_path / "data" / "universe"
    universe_dir.mkdir(parents=True)
    (universe_dir / "latest.json").write_text(
        json.dumps(
            {
                "retrieved_at": "2020-01-01T00:00:00+00:00",
                "source_hashes": {"jp": "hash-jp", "us": "hash-us"},
                "record_count": 100,
                "records": [
                    {
                        "asset_class": "us_stock",
                        "symbol": f"T{i}",
                        "name": f"Test {i}",
                        "tradeable": True,
                        "source_url": "https://www.paypay-sec.co.jp/test",
                    }
                    for i in range(100)
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    def fake_run(args, check=True, timeout=None, **kwargs):
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr("scripts.run_daily_research.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="outside bounded 7-day window"):
        _refresh_universe_with_bounded_retry()
