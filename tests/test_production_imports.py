from __future__ import annotations


def test_production_modules_import():
    import scripts.run_daily_research
    import scripts.run_daily_prediction
    import scripts.evaluate_frozen_holdout
    import scripts.release_gate
    import scripts.data_quality_gate
    import src.data.market_context
    import src.data.yahoo_price
    import src.features.context
    import src.features.technical
    import src.prediction.regression
    import src.prediction.targets
    import src.research.router
