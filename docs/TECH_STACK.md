# Technical stack and reference policy

## Core implementation
- Qlib is an architecture reference for an end-to-end quantitative research workflow.
- LightGBM, XGBoost and CatBoost are challenger families, not mandatory dependencies.
- LogisticRegression, ExtraTrees and HistGradientBoosting are the free CPU baseline.
- Technical indicators are implemented locally first to control dependencies and preserve causal semantics.
- Vectorized/backtesting projects are references; production validation must be independently reproducible.
- PatchTST and similar transformer models are deferred challengers after the classical OOS baseline is stable.

## Selection rule
Complexity, backtest score, or in-sample accuracy alone never promotes a model. Promotion requires chronological OOS, independent leakage audit, calibration, frozen holdout, reproducibility and release-gate approval.

## Data rule
PayPay Securities official trading lists define the current universe. Every observation that can influence a prediction must carry provenance and available_at; data unavailable at prediction time is excluded.

## Free-first rule
The core pipeline must run without paid data subscriptions. Optional keyed providers can be added later without silently replacing the free baseline.

## Automation
Work is split into bounded, rerunnable GitHub Actions stages. Public-repository standard runners are currently free and unlimited, while scheduled events can be delayed, so the pipeline uses non-hour-aligned schedules and idempotent stages.
