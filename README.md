# PayPay Securities Daily Prediction

PayPay証券で現時点で取引可能な**日次市場価格を持つ対象を原則すべて**対象にする、無料優先の日次予測・研究基盤です。

## Universe

The authoritative universe is refreshed from PayPay証券's official trading-list pages:
- Japan: individual stocks, domestic ETFs, and REITs
- U.S.: individual stocks and ETFs

The current target is **dynamic**, not a fixed 3,000-name cap. Newly added PayPay listings are included automatically after the universe refresh; removed/delisted names are retained in historical data for survivorship-bias control.

Mutual funds, CFDs, leveraged CFDs, and iDeCo are excluded from this stock/market-price pipeline because their execution/NAV timing is materially different from exchange-traded daily securities. They can be handled by a separate NAV prediction pipeline without mixing targets.

## Production principles

- PIT / no-look-ahead: available_at <= prediction_time
- chronological Walk-Forward OOS
- frozen holdout evaluated only after configuration freeze
- independent leakage/model-selection audit
- fail-closed data quality
- incremental/cached processing
- provenance and universe snapshots
- daily GitHub Actions automation
- free-first data sources; paid services are never required for the core pipeline

Research system only; not investment advice or a profit guarantee.
