# PROJECT_INSTRUCTIONS — Stock-Daily-Prediction-3000

## Mission
JP stock/ETF/REIT and US stock/ETF next-business-day prediction from a time-dependent PayPay Securities universe, with direction, expected return, expected close, q10/q50/q90, rank and uncertainty.
Optimize Future Generalization, case-level correctness, probabilistic quality, calibration, uncertainty, predictability awareness, robustness, PIT integrity, information value, selective prediction and operational reliability. Historical fit alone is not success.

## Every run
Re-check the latest GitHub HEAD/default branch, code/config, tests, workflows, Actions, artifacts, registries, production/champion/challenger state, current research/OOS/holdout evidence, failures and backlog. Prefer REUSE → REPAIR → INTEGRATE → TEST → VERIFY. Never rewrite past evidence to make the current state look consistent.

## PIT / time
Separate event/market time, prediction cutoff, source availability, publication, retrieval, effective time and revision time. Only use information demonstrably available by cutoff. Unknown/unverifiable availability is fail-closed for production-quality OOS.

## Data expansion
Increase data breadth by adding independent information axes (market microstructure, filings/timestamps, rates, macro, flows, events and fundamentals) rather than maximizing feature count. New sources remain research-only until license/access/cost, PIT availability, schema/revision, survivorship/universe scope, missingness, and incremental OOS value are validated. Prefer raw-data preservation and source lineage so later feature ablations can distinguish information value from feature-volume effects.

## Evaluation
Use chronological walk-forward OOS/WFO. Random splits are prohibited for temporal prediction. Separate candidate selection from final OOS. Frozen holdout is final evidence only and may not be tuned.

## Data integrity
Missing ≠ zero. Preserve explicit unavailable/unknown/delayed/not-yet-public/source-failed/malformed/not-applicable states. Preserve source lineage, snapshots, schema, revision behavior and identity history.

## Models / routing
Maintain simple baselines. Add challengers only for demonstrated incremental OOS value plus robustness. Specialist routing requires sufficient sample/folds/class coverage where relevant, calibration evidence and recent stability; otherwise fallback to broader validated scope.

## Calibration / uncertainty
Calibrate chronologically. Confidence ≠ predictability. Track disagreement, OOD, data/source uncertainty, regime ambiguity and event/volatility/liquidity uncertainty when relevant. Permit FALLBACK/ABSTAIN/DEFER/WAIT/ACQUIRE_MORE/RECOMPUTE instead of forced prediction.

## Research / adoption
External methods must pass discovery, source verification, PIT/cost checks, local reproduction, chronological OOS, robustness and frozen holdout before adoption. A candidate is never promoted from a single fold or external claim.

## Adoption / experience / cross-project transfer
Compare candidates against the incumbent on the same chronological OOS observations. Reference gates are: primary relative OOS LogLoss improvement >=3%, auxiliary improvement >=1%, no worsening in >=70% of evaluation periods, no worsening on the newest holdout, no material calibration degradation, and zero PIT violations. Also consider sample size, variance, confidence intervals, cross-sectional/serial dependence, turnover, transaction costs and operational risk; thresholds alone never auto-promote a candidate.

Matured predictions are stored at canonical instrument-date-cutoff granularity. Do not inflate experience weight with duplicate snapshots or revisions. Reconcile prediction -> actual -> metric -> error -> state, and keep current prediction artifacts separate from historical experience.

Transfer from other projects is mechanism-level only: DISCOVER -> ABSTRACT_MECHANISM -> COMPATIBILITY -> ADAPT -> LOCAL_PIT -> LOCAL_OOS -> LOCAL_HOLDOUT -> SHADOW -> PROMOTE. Cross-domain success is never copied directly into stock production.

## Reliability / cost / security
Use checkpoint/resume, idempotency, bounded retry/backoff, watchdog/heartbeat, stale-run detection, deterministic writes, artifact preservation, concurrency control, recovery and rollback. Never hide failure. Prefer verified free/OSS/local/cache; unknown-cost or billing-risk services are not automatic dependencies. Never expose credentials.

## Completion
Green CI, generated artifacts or a completed workflow do not prove performance verification or production completion. Evidence must cover tests, PIT/leakage/meta-leakage, relevant universe/scope audits, chronological OOS/WFO, calibration, ablation, robustness, frozen holdout, artifact integrity, reproducibility, recovery, release gate, monitoring and rollback.

## Status
Use IMPLEMENTED / EXECUTED / VERIFIED / PERFORMANCE_VERIFIED / PROMOTION_CANDIDATE / ADOPTED / PRODUCTION / STABLE / HOLD / REJECTED / FAILED / BLOCKED / DEFERRED / ROLLED_BACK / UNKNOWN / UNVERIFIABLE / SUPERSEDED / RETIRED distinctly.

## Loop
MONITOR → DETECT → TRIAGE → RESEARCH → IMPLEMENT → TEST → PIT → OOS/WFO → CALIBRATION → ROBUSTNESS → HOLDOUT → ADOPT/HOLD/REJECT → RELEASE → PRODUCTION → RECONCILE → FAILURE ANALYSIS → MEMORY → NEXT RESEARCH.
