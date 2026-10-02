from __future__ import annotations

import inspect
from pathlib import Path

from scripts.restore_latest_price_state import (
    GITHUB_API_RETRY_ATTEMPTS as PRICE_GITHUB_API_RETRY_ATTEMPTS,
    GITHUB_API_TIMEOUT_SECONDS as PRICE_GITHUB_API_TIMEOUT_SECONDS,
)
from scripts.restore_latest_universe_state import (
    GITHUB_API_RETRY_ATTEMPTS as UNIVERSE_GITHUB_API_RETRY_ATTEMPTS,
    GITHUB_API_TIMEOUT_SECONDS as UNIVERSE_GITHUB_API_TIMEOUT_SECONDS,
)
from src.data.github_artifact import download_workflow_artifact
from src.data.market_context import (
    YF_CONTEXT_RETRY_ATTEMPTS,
    YF_CONTEXT_TIMEOUT_SECONDS,
)
from src.data.paypay_collector import (
    PAYPAY_BROWSER_TIMEOUT_SECONDS,
    PAYPAY_HTTP_TIMEOUT_SECONDS,
    PAYPAY_JINA_TIMEOUT_SECONDS,
)
from src.data.yahoo_price import (
    YF_RETRY_BACKOFF_SECONDS,
    YF_SINGLE_RETRY_ATTEMPTS,
    YF_TIMEOUT_SECONDS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_external_request_timeouts_are_longer_and_bounded():
    assert YF_TIMEOUT_SECONDS >= 60
    assert YF_SINGLE_RETRY_ATTEMPTS >= 2
    assert 1 <= YF_RETRY_BACKOFF_SECONDS <= 15

    assert YF_CONTEXT_TIMEOUT_SECONDS >= 60
    assert YF_CONTEXT_RETRY_ATTEMPTS >= 2

    assert PRICE_GITHUB_API_TIMEOUT_SECONDS >= 60
    assert PRICE_GITHUB_API_RETRY_ATTEMPTS >= 2
    assert UNIVERSE_GITHUB_API_TIMEOUT_SECONDS >= 60
    assert UNIVERSE_GITHUB_API_RETRY_ATTEMPTS >= 2

    assert PAYPAY_HTTP_TIMEOUT_SECONDS >= 60
    assert PAYPAY_JINA_TIMEOUT_SECONDS >= 60
    assert PAYPAY_BROWSER_TIMEOUT_SECONDS >= 60

    artifact_timeout = inspect.signature(
        download_workflow_artifact
    ).parameters["timeout_seconds"].default
    assert artifact_timeout >= 180


def test_workflow_time_budgets_match_extended_price_requests():
    market = (ROOT / ".github/workflows/market-cycle.yml").read_text(encoding="utf-8")
    research = (ROOT / ".github/workflows/research-validation.yml").read_text(encoding="utf-8")
    us_close = (ROOT / ".github/workflows/us-close-prediction.yml").read_text(encoding="utf-8")

    assert (
        "  price-shards:\n"
        "    if: needs.prepare-cycle.outputs.run_required == 'true'\n"
        "    needs: [prepare-cycle, refresh-universe]\n"
        "    runs-on: ubuntu-latest\n"
        "    timeout-minutes: 60"
    ) in market
    assert (
        "  research-and-gate:\n"
        "    if: needs.prepare-cycle.outputs.run_required == 'true'\n"
        "    needs: [prepare-cycle, price-shards]\n"
        "    runs-on: ubuntu-latest\n"
        "    timeout-minutes: 60"
    ) in market
    assert (
        "  price-shards:\n"
        "    needs: restore-universe\n"
        "    runs-on: ubuntu-latest\n"
        "    timeout-minutes: 60"
    ) in research
    assert (
        "  price-shards:\n"
        "    needs: restore-universe\n"
        "    runs-on: ubuntu-latest\n"
        "    timeout-minutes: 60"
    ) in us_close
    assert (
        "  predict:\n"
        "    needs: price-shards\n"
        "    runs-on: ubuntu-latest\n"
        "    timeout-minutes: 45"
    ) in us_close
