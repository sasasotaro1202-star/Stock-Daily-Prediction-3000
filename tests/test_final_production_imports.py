from __future__ import annotations


def test_final_production_modules_import():
    import scripts.data_quality_gate
    import scripts.evaluate_frozen_holdout
    import scripts.monitor_predictions
    import scripts.production_invariants
    import scripts.release_gate
    import scripts.run_daily_prediction
    import scripts.run_daily_research
    import scripts.run_backtest
    import scripts.update_market_context
    import scripts.update_prices
    import src.data.market_context
    import src.data.paypay_collector
    import src.data.yahoo_price
    import src.features.context
    import src.features.technical
    import src.prediction.regression
    import src.prediction.targets
    import src.research.router
    import src.validation.training_sample
