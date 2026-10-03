# Stock-Daily-Prediction-3000 — Project Instructions

## Priority
`docs/PROJECT_SOURCE.md` and its ordered canonical source parts are authoritative. Current GitHub HEAD/code/config/tests/workflows/artifacts/registries and measured evidence override stale documentation, while historical results and holdouts remain immutable.

## Universe and target identity
The production universe is time-varying and must be reconstructed from the repository's validated source policy rather than assumed to be a permanent fixed 3,000 symbols. Return/price/ranking horizons are separate targets and versions.

## PIT
Preserve observation time, source availability/publication, retrieval, prediction cutoff and revisions. Future prices, later filings, later market state, later corrections and survivorship-contaminated universe information must not leak backward. Missing must not become zero.

## Validation
Random splits are prohibited for temporal prediction. Use chronological/prequential WFO/OOS, separate candidate selection from final OOS and protect frozen holdout from tuning. Evaluate LogLoss/probabilistic quality where applicable plus return/price/ranking metrics, calibration, uncertainty, case-level error, drift and OOD.

## Mandatory loop
MONITOR → DETECT → TRIAGE → RESEARCH → IMPLEMENT → TEST → PIT → OOS/WFO → CALIBRATION → ROBUSTNESS → HOLDOUT → ADOPT/HOLD/REJECT → RELEASE → PRODUCTION → RECONCILE → FAILURE ANALYSIS.

## Cross-project transfer
Baseball, BTC, 7-Sport and Soccer methods are candidates, not production evidence. Transfer the mechanism, then run local PIT → chronological OOS → robustness → frozen holdout → shadow.

## Failure and cost safety
No fabricated metric, silent exception, missing→zero, skipped test→pass, failed recovery→success or unknown-PIT→valid. Prefer verified free/OSS/local/cached sources; unknown or billing-risk cost is HOLD/UNCONFIRMED.