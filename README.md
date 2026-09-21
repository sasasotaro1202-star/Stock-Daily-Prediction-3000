# PayPay Securities Daily Prediction

PayPay証券で現時点で取引可能な日次市場価格対象を原則すべて対象にする、無料優先の日次株価予測・研究基盤です。

## Universe

Current universe is dynamic rather than capped at 3,000 names. The authoritative source is PayPay証券's official trading-list pages for Japan and the U.S. Japan covers individual stocks, ETFs and REITs; the U.S. list covers individual stocks and ETFs. Newly listed/removed names are reflected by the next universe snapshot while historical snapshots are preserved for survivorship-bias control.

Investment trusts, CFDs, leveraged CFDs and iDeCo are kept outside this exchange-traded price pipeline because their pricing/execution clocks differ.

## Prediction

The system produces, when the production gate is approved:
- next-session up probability
- expected next-session return
- expected next close
- model-implied low/high range
- cross-sectional ranking

The research layer evaluates Logistic Regression, ExtraTrees and HistGradientBoosting using chronological OOS folds, fold-local Platt calibration, and OOS-only regime routing. Optional challengers include LightGBM, XGBoost, CatBoost and PatchTST.

## Safety

PIT/no-look-ahead, causal features, independent leakage audit, data-quality fail-closed behavior, frozen holdout, reproducibility manifest and independent release gate are required. Missing critical data produces DEFERRED instead of fabricated values.

## Automation

GitHub Actions runs a six-hour health heartbeat and a weekday market cycle at 18:17 Asia/Tokyo. The market cycle refreshes the PayPay universe, updates price data in four parallel shards, runs data-quality checks, OOS research, calibration, release gate and—only when approved—production prediction. Artifacts are retained for seven days.

Research system only; not investment advice or a profit guarantee.
