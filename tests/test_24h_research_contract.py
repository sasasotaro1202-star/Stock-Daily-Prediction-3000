from pathlib import Path

# Recovery invariant: keep this contract dependency-light so CI can validate it early.

ROOT = Path(__file__).resolve().parents[1]
MARATHON = ROOT / ".github" / "workflows" / "24h-research-marathon.yml"
WATCHDOG = ROOT / ".github" / "workflows" / "24h-research-marathon-watchdog.yml"


def test_marathon_is_schedule_or_manual_only():
    text = MARATHON.read_text(encoding="utf-8")
    assert "\n  push:" not in text
    assert "workflow_dispatch:" in text
    assert '    - cron: "5 0 * * *"' in text


def test_marathon_price_shard_pid_is_real_background_pid():
    text = MARATHON.read_text(encoding="utf-8")
    assert '$$!' not in text
    assert text.count('pids+=( "$!" )') == 3


def test_marathon_final_checkpoint_is_fail_closed():
    text = MARATHON.read_text(encoding="utf-8")
    assert "Write immutable final checkpoint" in text
    assert text.count('if: always()') >= 3
    assert '"research_only":True' in text
    assert '"production_changed":False' in text
    assert 'if status != "COMPLETED":' in text
    assert 'if-no-files-found: error' in text
    for lane in ("core_oos", "finra", "treasury", "audits"):
        assert f'"{lane}"' in text
    for checkpoint in ("core_oos.json", "finra.json", "treasury.json", "audits.json"):
        assert f"marathon-checkpoints/{checkpoint}" in text
    assert text.count("if: success()") >= 4


def test_watchdog_has_bounded_recovery_and_failure_cooldown():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert "MAIN_SHA=" in text
    assert "SELF_RUN_ID=" in text
    assert "age >= 30" in text
    assert "timeout-minutes: 360" in MARATHON.read_text(encoding="utf-8")
    assert "age >= 350" in text
    assert "age >= 1450" not in text
    assert "len(recent_failures) < 3" in text
    assert "three_recent_failures_cooldown" in text
    assert "bounded dispatch retries exhausted" in text


def test_watchdog_does_not_cancel_itself_on_next_scheduled_tick():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert "group: research-marathon-watchdog" in text
    assert "cancel-in-progress: false" in text


def test_marathon_price_guard_is_statement_based():
    text = MARATHON.read_text(encoding="utf-8")
    assert 'raise SystemExit("FAIL: provider deferred ratio over 5%") if ratio>0.05 else None' not in text
    assert text.count("if ratio > 0.05:") == 3
    assert text.count("FAIL: universe record count is zero") == 3


def test_metric_aggregation_ignores_non_numeric_metadata():
    from src.research.metrics import aggregate_metric_rows

    rows = [
        {"logloss": 1.0, "brier": 0.60, "method": "platt", "fold": 1.0},
        {"logloss": 0.8, "brier": 0.50, "method": "isotonic", "fold": 2.0},
    ]

    aggregated = aggregate_metric_rows(rows)

    assert aggregated == {
        "brier": 0.55,
        "fold": 1.5,
        "logloss": 0.9,
    }

def test_research_has_bounded_adaptive_data_acquisition_loop():
    text = (ROOT / "scripts" / "run_daily_research.py").read_text(encoding="utf-8")
    pipeline = (ROOT / "config" / "pipeline.yml").read_text(encoding="utf-8")
    prices = (ROOT / "scripts" / "update_prices.py").read_text(encoding="utf-8")
    assert "def _ensure_adaptive_research_data()" in text
    assert "_run_acquisition_once(cfg, iteration)" in text
    assert "max_acquisition_iterations: 3" in pipeline
    assert '"selected_at_each_iteration": True' in text
    assert '"refresh_official_universe_every_iteration"' in text
    assert 'PRICE_MIN_HISTORY_SESSIONS' in prices
    assert 'select_history_warmup_targets' in prices
    assert 'history_below_oos_minimum_after_warmup' in prices
    assert 'snapshot["reasons"].append("shallow_symbol_history")' not in text

def test_adaptive_data_loop_separates_blocking_and_acquisition_debt():
    text = (ROOT / "scripts" / "run_daily_research.py").read_text(encoding="utf-8")
    assert '"blocking_reasons": []' in text
    assert '"acquisition_reasons": []' in text
    assert '"READY_WITH_PENDING_ACQUISITION"' in text
    assert '"model_candidates_reselected_after_data_loop": True' in text


def test_source_discovery_rechecks_after_research_completion():
    text = (ROOT / ".github" / "workflows" / "free-data-source-discovery.yml").read_text(encoding="utf-8")
    assert "workflow_run:" in text
    assert '"Research validation"' in text
    assert '"24H Research Marathon"' in text
    assert "types: [completed]" in text


def test_marathon_uses_diverse_model_budget_without_removing_primary_candidates():
    text = MARATHON.read_text(encoding="utf-8")
    expected = (
        "logistic,extra_trees,hgb,lightgbm,"
        "hgb_conservative_recent,blend_hgb_lgbm_regularized_recent"
    )
    assert 'RESEARCH_MODEL_ALLOWLIST: "' + expected + '"' in text


def test_research_model_allowlist_is_opt_in_and_keeps_all_primary_candidates(monkeypatch):
    from scripts.run_daily_research import make_models
    monkeypatch.delenv("RESEARCH_MODEL_ALLOWLIST", raising=False)
    full = list(make_models())
    assert {"logistic", "extra_trees", "hgb"}.issubset(set(full))

