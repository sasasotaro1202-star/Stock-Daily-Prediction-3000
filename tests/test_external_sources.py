from src.data.external_sources import EXTERNAL_SOURCES, research_sources_for, source_ids


def test_catalog_is_nonempty_and_unique():
    ids = source_ids()
    assert ids
    assert len(ids) == len(set(ids))


def test_enabled_sources_have_explicit_cost_and_pit():
    for source in EXTERNAL_SOURCES:
        assert source.cost
        assert source.pit_mode
        if source.enabled_research:
            assert source.cost.startswith("free")


def test_global_sources_cover_multiple_asset_families():
    broad = [
        s for s in EXTERNAL_SOURCES
        if len(s.asset_classes) >= 4 and s.enabled_research
    ]
    assert len(broad) >= 8


def test_us_and_jp_research_paths_are_discoverable():
    assert {"gdelt_doc", "fred_alfred", "us_treasury_curve"} <= set(
        source.id for source in research_sources_for("us_stock")
    )
    assert {"gdelt_doc", "boj_timeseries", "edinet"} <= set(
        source.id for source in research_sources_for("jp_stock")
    )


def test_no_paid_only_source_is_enabled():
    paid_only = {"paid", "enterprise", "paid_only"}
    assert not any(
        s.enabled_research and s.cost in paid_only
        for s in EXTERNAL_SOURCES
    )
