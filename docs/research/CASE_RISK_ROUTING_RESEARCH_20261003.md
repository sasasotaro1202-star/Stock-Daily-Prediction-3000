# Case-Risk Routing Research — 2026-10-03

## Purpose

Research-only design for Issue #160. No production mutation, no automatic promotion, and no frozen-holdout tuning.

## External evidence

FAME proposes forecastability-aware sparse expert routing: build a multidimensional forecastability fingerprint, learn expert suitability from prior validation performance, and activate a budgeted subset of experts.

Source: https://arxiv.org/abs/2606.08896

Expert-loss integration proposes incorporating expert-specific historical loss into mixture-of-experts training and shows a computationally lighter partial-online update strategy.

Source: https://arxiv.org/abs/2605.10330

Regime-dependent volatility/return research reports that return predictability can be weak and state-dependent, while implementation realism matters when translating forecasts into economic strategies.

Source: https://arxiv.org/abs/2606.09478

These results are not production evidence for this repository. They motivate local experiments only.

## Local adaptation

Target: next-business-day stock direction.

Candidate prediction-time fingerprint:
- prior-only calibrated predictability
- row-level OOD
- model disagreement
- prior-only case-failure risk
- prediction confidence / entropy
- market-regime descriptors
- liquidity / volatility state
- recentness and data-quality state

Candidate routing target:
- prior-fold expert loss for each baseline/challenger
- predicted future expert loss
- cost-aware sparse expert selection

## PIT contract

For every case:
- prediction_time is the cutoff
- available_at <= prediction_time
- published_at, when present, <= prediction_time
- retrieved_at, when present, satisfies available_at <= retrieved_at; retrieval time may be after the prediction cutoff because availability_at is the causal PIT boundary
- retrieved_at must not precede published_at when both exist
- invalid/missing provenance is fail-closed
- current-fold outcomes cannot train the router or set its threshold
- locked/frozen holdout outcomes cannot tune any component

## Evaluation

Chronological WFO only.

Compare:
1. fixed arithmetic case-risk baseline
2. learned case-risk model
3. existing dynamic routing
4. learned expert-loss routing
5. optional sparse/top-k routing

Primary:
- LogLoss

Secondary:
- Brier
- Accuracy
- ECE
- high-risk coverage
- error-capture rate
- high-vs-low failure lift
- worst-fold performance
- fold stability
- compute cost

Economic outputs must be evaluated separately with transaction-cost/slippage assumptions where applicable.

## Adoption gate

Research result remains HOLD unless all configured release evidence is available:
- sufficient common chronological OOS folds
- required relative OOS improvement
- no material deterioration across evaluation periods
- latest unused holdout not worse
- calibration safeguards pass
- PIT/leakage/meta-leakage violations = 0
- robustness and ablation evidence pass
- reproducible artifacts
- no production mutation during research

