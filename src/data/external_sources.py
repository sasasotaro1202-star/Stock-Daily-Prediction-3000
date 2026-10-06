from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class ExternalSource:
    id: str
    family: str
    enabled_research: bool
    auth: str
    cost: str
    coverage: str
    asset_classes: FrozenSet[str]
    temporal_resolution: str
    pit_mode: str
    historical_window: str
    expected_features: tuple[str, ...]


def _f(*values: str) -> FrozenSet[str]:
    return frozenset(values)


EXTERNAL_SOURCES: tuple[ExternalSource, ...] = (
    ExternalSource(
        "gdelt_doc", "news_events", True, "none", "free", "global",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "15m+", "publication_time", "rolling_3_months",
        ("mention_count", "source_diversity", "burst", "entity_attention",
         "sentiment_proxy", "event_intensity"),
    ),
    ExternalSource(
        "finra_short_sale", "equity_microstructure", True, "none",
        "free_noncommercial", "us_equities", _f("us_stock"), "daily",
        "same_day_post_publication", "2018-08-01+",
        ("short_volume_ratio", "short_volume_zscore", "venue_disagreement",
         "short_volume_burst"),
    ),
    ExternalSource(
        "fred_alfred", "macro", True, "free_registration", "free",
        "us_macro",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "daily", "realtime_vintage", "series_dependent",
        ("level", "change", "surprise", "revision", "macro_regime"),
    ),
    ExternalSource(
        "us_treasury_curve", "rates", True, "none", "free", "us_rates",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "daily", "publication_time", "long_history",
        ("2s10s", "3m10y", "level", "slope", "curvature", "daily_change"),
    ),
    ExternalSource(
        "cftc_cot", "positioning", True, "none", "free", "futures_macro",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "weekly", "release_time_required", "1986+",
        ("net_speculative_position", "commercial_position", "positioning_change"),
    ),
    ExternalSource(
        "boj_timeseries", "macro", True, "none", "free", "japan_macro",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "daily_or_lower", "publication_time", "series_dependent",
        ("policy_rate", "tankan", "money", "prices", "credit", "yield_curve"),
    ),
    ExternalSource(
        "bls_public_data", "macro", True, "none", "free", "us_macro",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "release_event", "release_schedule", "series_dependent",
        ("cpi", "ppi", "payrolls", "unemployment", "jolts", "wage_growth"),
    ),
    ExternalSource(
        "bea_release_data", "macro", True, "none", "free", "us_macro",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "release_event", "release_schedule", "series_dependent",
        ("gdp", "income", "consumption", "trade", "corporate_profits"),
    ),
    ExternalSource(
        "ecb_data", "macro", True, "none", "free", "euro_macro",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "daily_or_lower", "historical_versions_available", "series_dependent",
        ("ecb_policy", "euro_rates", "credit", "monetary_conditions"),
    ),
    ExternalSource(
        "bis_stats", "global_finance", True, "none", "free", "global_finance",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "monthly_or_lower", "publication_time", "series_dependent",
        ("credit_growth", "debt", "cross_border_flows", "banking_conditions"),
    ),
    ExternalSource(
        "edinet", "corporate_disclosure", True, "free_registration", "free",
        "japan_corporates", _f("jp_stock", "jp_reit"), "filing_event",
        "document_available_time", "multi_year",
        ("revenue", "profit", "guidance", "ownership", "capital_actions",
         "narrative_events"),
    ),
    ExternalSource(
        "estat", "macro", True, "free_registration", "free",
        "japan_macro", _f("jp_stock", "jp_reit", "us_stock", "us_etf"),
        "release_event", "release_time_required", "series_dependent",
        ("cpi", "labor", "trade", "production", "household_activity"),
    ),
    ExternalSource(
        "cme_cboe_context", "volatility_and_derivatives", True, "none",
        "free", "us_derivatives",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "daily", "official_publication_time", "series_dependent",
        ("volatility_term_structure", "volatility_change", "index_volume",
         "market_breadth"),
    ),
    ExternalSource(
        "usgs_events", "exogenous_shock", True, "none", "free",
        "global_events",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "minutes", "event_time_plus_publication", "feed_dependent",
        ("earthquake_intensity", "proximity", "affected_region", "event_burst"),
    ),
    ExternalSource(
        "noaa_ncei", "weather_and_climate", True, "free_registration",
        "free", "global_weather",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "hourly_or_daily", "observation_publication_required", "station_dependent",
        ("temperature_anomaly", "precipitation", "storm", "degree_days",
         "climate_shock"),
    ),
    ExternalSource(
        "openfigi", "reference_data", True, "optional", "free",
        "global_securities",
        _f("jp_stock", "us_stock", "jp_etf", "us_etf", "jp_reit"),
        "static_reference", "reference_snapshot", "current_reference",
        ("security_id", "exchange", "share_class", "security_type",
         "identifier_crosswalk"),
    ),
)


def research_sources_for(asset_class: str) -> tuple[ExternalSource, ...]:
    return tuple(
        source for source in EXTERNAL_SOURCES
        if source.enabled_research and asset_class in source.asset_classes
    )


def source_ids() -> tuple[str, ...]:
    return tuple(source.id for source in EXTERNAL_SOURCES)
