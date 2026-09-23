from pathlib import Path


def test_market_cycle_main_sha_lookup_failure_fails_closed():
    source = Path(".github/workflows/market-cycle.yml").read_text(encoding="utf-8")
    assert 'if ! branch_sha="$(gh api "repos/${GITHUB_REPOSITORY}/git/ref/heads/main" --jq ".object.sha")"; then' in source
    assert "deferring cycle fail-closed" in source
    assert 'run_required=false' in source
    assert 'reason="current_main_sha_unavailable"' in source


def test_market_cycle_superseded_sha_defers_before_expensive_jobs():
    source = Path(".github/workflows/market-cycle.yml").read_text(encoding="utf-8")
    assert 'reason="superseded_workflow_sha"' in source
    assert 'if [ "$run_required" = false ]; then' in source
    assert "refresh-universe:" in source
    assert "price-shards:" in source
    assert "research-and-gate:" in source

def test_market_cycle_actions_api_failure_fails_closed():
    source = Path(".github/workflows/market-cycle.yml").read_text(encoding="utf-8")
    assert 'if ! runs="$(gh api "repos/${GITHUB_REPOSITORY}/actions/workflows/market-cycle.yml/runs?per_page=20" 2>/dev/null)"; then' in source
    assert 'guard_reason=actions_api_unavailable_defer' in source
    assert "deferring cycle fail-closed" in source


def test_market_cycle_empty_actions_response_fails_closed():
    source = Path(".github/workflows/market-cycle.yml").read_text(encoding="utf-8")
    assert 'guard_reason=actions_api_empty_response_defer' in source
    assert "Actions API returned empty response" in source