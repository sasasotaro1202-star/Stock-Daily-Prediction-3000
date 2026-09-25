from datetime import datetime, timezone
import json

import pytest

from scripts.restore_latest_universe_state import (
    _recent_market_cycle_universe_artifacts,
    _validate_snapshot,
)


def _write_snapshot(tmp_path, retrieved_at, count=100):
    path = tmp_path / "latest.json"
    payload = {
        "retrieved_at": retrieved_at.isoformat(),
        "records": [
            {
                "symbol": str(i),
                "asset_class": "us_stock",
                "name": f"Name {i}",
                "tradeable": True,
            }
            for i in range(count)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_recent_universe_snapshot_is_accepted(tmp_path):
    now = datetime.now(timezone.utc)
    path = _write_snapshot(tmp_path, now)
    _validate_snapshot(path, now.isoformat().replace("+00:00", "Z"))


def test_old_universe_snapshot_is_rejected(tmp_path):
    old = datetime.now(timezone.utc).replace() - __import__("datetime").timedelta(days=8)
    path = _write_snapshot(tmp_path, old)
    with pytest.raises(ValueError, match="outside bounded 7-day window"):
        _validate_snapshot(path, old.isoformat().replace("+00:00", "Z"))


def test_small_universe_snapshot_is_rejected(tmp_path):
    now = datetime.now(timezone.utc)
    path = _write_snapshot(tmp_path, now, count=99)
    with pytest.raises(ValueError, match="fewer than 100"):
        _validate_snapshot(path, now.isoformat().replace("+00:00", "Z"))


def test_recent_market_cycle_artifact_lookup_includes_cancelled_runs(monkeypatch):
    now = datetime.now(timezone.utc)
    run_created = now.isoformat().replace("+00:00", "Z")
    calls = []

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, **kwargs):
        calls.append(url)
        if "/actions/workflows/market-cycle.yml/runs" in url:
            return Response({
                "workflow_runs": [{
                    "id": 36149391445,
                    "status": "completed",
                    "conclusion": "cancelled",
                    "created_at": run_created,
                }]
            })
        if "/actions/runs/36149391445/artifacts" in url:
            return Response({
                "artifacts": [{
                    "id": 10870717801,
                    "name": "universe-36149391445",
                    "expired": False,
                    "created_at": run_created,
                }]
            })
        raise AssertionError(f"unexpected URL: {url}")

    import scripts.restore_latest_universe_state as restore

    monkeypatch.setattr(restore.requests, "get", fake_get)
    artifacts = _recent_market_cycle_universe_artifacts("owner/repo", "token")

    assert [a["id"] for a in artifacts] == [10870717801]
    assert any("actions/runs/36149391445/artifacts" in url for url in calls)
