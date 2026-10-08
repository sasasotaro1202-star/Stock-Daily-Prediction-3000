from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.refresh_universe_with_bounded_fallback import main


def _write_snapshot(path: Path, retrieved_at: str, *, source_hashes: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "retrieved_at": retrieved_at,
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
    if source_hashes:
        payload["source_hashes"] = {"jp": "hash-jp", "us": "hash-us"}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_timeout_retains_recent_verified_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    latest = tmp_path / "data" / "universe" / "latest.json"
    _write_snapshot(latest, (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat())
    monkeypatch.chdir(tmp_path)

    calls: list[tuple[list[str], int | None]] = []

    def fake_run(args, check=True, timeout=None, **kwargs):
        calls.append((list(args), timeout))
        if args[-1] == "scripts/refresh_universe.py":
            raise subprocess.TimeoutExpired(args, timeout)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("scripts.refresh_universe_with_bounded_fallback.subprocess.run", fake_run)

    main()

    assert calls[0] == (["python", "scripts/refresh_universe.py"], 240)
    assert calls[1][0] == ["python", "scripts/universe_quality_gate.py"]
    assert "UNIVERSE_REFRESH status=DEFERRED_RETAINED_EXISTING" not in ""


def test_timeout_fails_closed_without_valid_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    def fake_run(args, check=True, timeout=None, **kwargs):
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr("scripts.refresh_universe_with_bounded_fallback.subprocess.run", fake_run)

    with pytest.raises(SystemExit, match="retained universe snapshot missing"):
        main()


def test_timeout_fails_closed_on_stale_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    latest = tmp_path / "data" / "universe" / "latest.json"
    _write_snapshot(latest, "2020-01-01T00:00:00+00:00")
    monkeypatch.chdir(tmp_path)

    def fake_run(args, check=True, timeout=None, **kwargs):
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr("scripts.refresh_universe_with_bounded_fallback.subprocess.run", fake_run)

    with pytest.raises(SystemExit, match="outside bounded 7-day window"):
        main()


def test_timeout_fails_closed_without_provenance_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    latest = tmp_path / "data" / "universe" / "latest.json"
    _write_snapshot(
        latest,
        (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
        source_hashes=False,
    )
    monkeypatch.chdir(tmp_path)

    def fake_run(args, check=True, timeout=None, **kwargs):
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr("scripts.refresh_universe_with_bounded_fallback.subprocess.run", fake_run)

    with pytest.raises(SystemExit, match="lacks source_hashes"):
        main()
