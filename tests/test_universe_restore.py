from datetime import datetime, timezone
import json

import pytest

from scripts.restore_latest_universe_state import _validate_snapshot


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
