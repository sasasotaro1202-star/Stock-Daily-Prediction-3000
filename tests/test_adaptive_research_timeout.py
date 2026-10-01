from __future__ import annotations

import subprocess

from scripts import run_daily_research


def test_universe_refresh_retries_only_timeout(monkeypatch):
    calls = []

    def fake_run(args, *, check, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(args, timeout)
        return None

    monkeypatch.setattr(run_daily_research.subprocess, "run", fake_run)
    run_daily_research._refresh_universe_with_bounded_retry()

    assert calls == [120, 240]


def test_universe_refresh_does_not_retry_non_timeout_failure(monkeypatch):
    calls = []

    def fake_run(args, *, check, timeout):
        calls.append(timeout)
        raise subprocess.CalledProcessError(2, args)

    monkeypatch.setattr(run_daily_research.subprocess, "run", fake_run)

    try:
        run_daily_research._refresh_universe_with_bounded_retry()
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("non-timeout failure must propagate")

    assert calls == [120]
