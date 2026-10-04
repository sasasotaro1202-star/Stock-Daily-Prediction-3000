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
## Evidence freshness
Chronological OOS evidence is bound to the exact research execution SHA and the main SHA observed when that evidence was persisted. If those SHAs differ, evidence is STALE and must not be treated as current performance or release evidence. Unknown provenance fails closed. Historical evidence remains immutable.

## Prequential frontier selection
For research-only frontier matrices, the final winner is the candidate selected by the last development decision made before that fold is scored. The decision-fold outcome cannot determine its own selection. Aggregate rankings of selected-candidate outcomes are diagnostics only; they cannot define the winner.

## Risk/OOD missingness
Risk-derived OOD must distinguish missing evidence from neutral observed OOD. When risk history/current values are unavailable, the underlying OOD score stays finite-neutral; any missingness-aware shrink is an explicit research candidate and must not silently convert missingness into observed risk.

## GitHub-side research autopilot
The Research validation workflow is not directly push-triggered. `research-autopilot.yml` owns autonomous dispatch: it runs daily and after current-main Repository verification completes, requires verification PASS before expensive OOS, suppresses duplicate current-main runs, and cancels only stale queued Research entries on superseded SHAs. Active research runs remain under the dedicated evidence-fingerprint/stale-run watchdog policy.

## Reliability state
Requested, queued, pending, waiting and in-progress research runs are transient active states. Long-running chronological OOS is preserved across ordinary main-branch commits; stale recovery must be bounded and must not create self-recovery loops. Terminal workflow_run events must not overwrite a newer active Research run; compare verifiable Research run numbers when terminal and active contexts race. Failed, cancelled, skipped and stale evidence remains non-success until independently rerun and verified.

## Evidence identity and freshness
Research evidence records the exact execution SHA plus an evidence-affecting fingerprint. Evidence is FRESH when the execution fingerprint matches the current main fingerprint, even if commit SHAs differ because of non-evidence-affecting control-plane changes. Any runtime/config/workflow/dependency fingerprint mismatch requires fresh validation/OOS. Unknown fingerprint provenance is fail-closed.
