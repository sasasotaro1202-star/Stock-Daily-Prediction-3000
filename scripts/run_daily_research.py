    if not snapshot["market_context"]:
        snapshot["reasons"].append("market_context_missing")
    snapshot["status"] = "READY" if not snapshot["reasons"] else "SEARCHING"
    return snapshot

def _run_acquisition_once(cfg: dict, iteration: int) -> None:
    print(
        "ADAPTIVE_DATA_ACQUIRE "
        f"iteration={iteration} min_history_sessions={cfg['min_history_sessions']}",
        flush=True,
    )

    # Re-discover free research sources first. Discovery is evidence collection
    # only; no discovered source is automatically admitted to production.
    subprocess.run(
        ["python", "scripts/discover_free_data_sources.py"],
        check=True,
        timeout=180,
    )

    # Re-discover the current official universe every iteration. This is
    # research-only state in the runner and is never promoted directly.
    subprocess.run(
        ["python", "scripts/refresh_universe.py"],
        check=True,
        timeout=120,
    )
    subprocess.run(
        ["python", "scripts/universe_quality_gate.py"],
        check=True,
        timeout=120,
    )

    env_base = os.environ.copy()
    env_base["PRICE_SHARD_COUNT"] = str(cfg["price_shards"])
    env_base["PRICE_MAX_WORKERS"] = str(cfg["price_max_workers"])
    env_base["PRICE_MIN_HISTORY_SESSIONS"] = str(cfg["min_history_sessions"])

    def run_shard(shard: int):
        env = dict(env_base)
        env["PRICE_SHARD_INDEX"] = str(shard)
        return subprocess.run(
            ["python", "scripts/update_prices.py"],
            check=True,