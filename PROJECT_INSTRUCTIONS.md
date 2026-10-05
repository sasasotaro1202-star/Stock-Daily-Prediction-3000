# PROJECT_INSTRUCTIONS — Stock-Daily-Prediction-3000

## Mission
JP stock/ETF/REIT and US stock/ETF next-business-day prediction from a time-dependent PayPay Securities universe, with direction, expected return, expected close, q10/q50/q90, rank and uncertainty.
Optimize Future Generalization, case-level correctness, probabilistic quality, calibration, uncertainty, predictability awareness, robustness, PIT integrity, information value, selective prediction and operational reliability. Historical fit alone is not success.

## Canonical Project Source

`PROJECT_SOURCE.md` is the expanded technical/data/research/validation/operations contract supplied for this repository. It is the canonical source companion to this file and is audited automatically by Repository verification before Research autopilot proceeds. Missing, malformed, incomplete, or contradictory source-contract evidence is fail-closed; historical evidence is never rewritten.
## Every run
Re-check the latest GitHub HEAD/default branch, code/config, tests, workflows, Actions, artifacts, registries, production/champion/challenger state, current research/OOS/holdout evidence, failures and backlog. Prefer REUSE → REPAIR → INTEGRATE → TEST → VERIFY. Never rewrite past evidence to make the current state look consistent.

## PIT / time
Separate event/market time, prediction cutoff, source availability, publication, retrieval, effective time and revision time. Only use information demonstrably available by cutoff. Unknown/unverifiable availability is fail-closed for production-quality OOS.

## Data expansion
Increase data breadth by adding independent information axes (market microstructure, filings/timestamps, rates, macro, flows, events and fundamentals) rather than maximizing feature count. New sources remain research-only until license/access/cost, PIT availability, schema/revision, survivorship/universe scope, missingness, and incremental OOS value are validated. Prefer raw-data preservation and source lineage so later feature ablations can distinguish information value from feature-volume effects.

## Evaluation
Use chronological walk-forward OOS/WFO. Random splits are prohibited for temporal prediction. Separate candidate selection from final OOS. Frozen holdout is final evidence only and may not be tuned.

## Data integrity
Missing ≠ zero. Preserve explicit unavailable/unknown/delayed/not-yet-public/source-failed/malformed/not-applicable states. Preserve source lineage, snapshots, schema, revision behavior and identity history. Risk-derived OOD features must also fail closed: when historical or current risk evidence is entirely unavailable, emit a finite neutral score rather than NaN, and never allow missingness to masquerade as observed risk.

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

## Evidence freshness

Chronological OOS evidence is bound to the exact Research validation execution SHA and an evidence-affecting fingerprint. The status/snapshot branch records both the execution SHA and the current main SHA observed when the evidence was persisted. Freshness is determined first from the evidence-affecting fingerprint: matching fingerprints are `FRESH` even when commit SHAs differ because of non-evidence-affecting control-plane changes; a fingerprint mismatch is `STALE` and requires fresh OOS/WFO. The evidence fingerprint uses an explicit allowlist of control-plane-only workflows (`heartbeat`, `automation-supervisor`, `actions-reliability-watchdog`, `research-autopilot`, `long-research-recovery`, `automation-failure-learning`, `research-validation-status`, `24h-research-marathon-watchdog`, `bounded-production-recovery`). Any unlisted workflow change is evidence-affecting and invalidates active OOS. This allowlist is mirrored by the Watchdog invalidation guard. The allowlist itself is versioned code and changes to the fingerprint implementation require fresh evidence. If the fingerprint is absent or unverifiable, exact SHA equality is only a legacy fallback; otherwise freshness is `UNKNOWN` and release decisions fail closed. Raw historical evidence remains immutable.

## Reliability / cost / security
Use checkpoint/resume, idempotency, bounded retry/backoff, watchdog/heartbeat, stale-run detection, deterministic writes, artifact preservation, concurrency control, recovery and rollback. The GitHub-side Research autopilot dispatches expensive chronological OOS only after a successful current-main Repository verification, runs daily, suppresses duplicate current-main research runs, and cleans stale queued runs on superseded SHAs. The Actions reliability watchdog also monitors the verification, autopilot, 24H Research, Automation failure-learning and Experience review queues and performs bounded stale-queue recovery so obsolete or stalled queue entries do not become permanent manual blockers or consume resources indefinitely. For current-main Repository verification and Research autopilot only, a first-attempt workflow failure may be retried once; a second failure remains FAILED and is not hidden. Heartbeat status must treat requested/queued/pending/waiting/in_progress as transient active states and tolerate a short job-creation race without converting it into false failure. A terminal workflow_run event must not overwrite a newer active Research run; when terminal and active contexts race, compare Research run_number and persist the newest verifiable context. The Research status writer also reconciles missed terminal events against the workflow-scoped Research validation run list, reuses the recovered run context for evidence retrieval, and never treats a control-plane timestamp-only revisit as new evidence. A tertiary Automation heartbeat runs a lightweight 30-minute control-plane check and also reacts to Automation supervisor / Actions reliability watchdog completion events. It may dispatch only a bounded current-main supervisor/watchdog recovery when no active run and no recent current-main controller run exist; it does not modify production or research evidence. The original deep repository health suite remains on its 6-hour schedule. Never hide failure. A GitHub-side Automation failure-learning workflow persists immutable failure records to the research-status branch and automatically converts recent failures into a deterministic research/repair backlog. The backlog is diagnostic only: it never mutates production, promotes a candidate, or treats a recorded failure as proof that the current runtime remains failed. The workflow is also manually dispatchable and scheduled every 15 minutes as a reconciliation safety net for missed workflow_run deliveries. Prefer verified free/OSS/local/cache; unknown-cost or billing-risk services are not automatic dependencies. Never expose credentials.

## Completion
Green CI, generated artifacts or a completed workflow do not prove performance verification or production completion. Evidence must cover tests, PIT/leakage/meta-leakage, relevant universe/scope audits, chronological OOS/WFO, calibration, ablation, robustness, frozen holdout, artifact integrity, reproducibility, recovery, release gate, monitoring and rollback.

## Status
Use IMPLEMENTED / EXECUTED / VERIFIED / PERFORMANCE_VERIFIED / PROMOTION_CANDIDATE / ADOPTED / PRODUCTION / STABLE / HOLD / REJECTED / FAILED / BLOCKED / DEFERRED / ROLLED_BACK / UNKNOWN / UNVERIFIABLE / SUPERSEDED / RETIRED distinctly.

## Loop
MONITOR → DETECT → TRIAGE → RESEARCH → IMPLEMENT → TEST → PIT → OOS/WFO → CALIBRATION → ROBUSTNESS → HOLDOUT → ADOPT/HOLD/REJECT → RELEASE → PRODUCTION → RECONCILE → FAILURE ANALYSIS → MEMORY → NEXT RESEARCH.

## Cross-project validated operating contracts

The following patterns are adopted as reusable controls after comparison with the current main branches of BTC-Prediction-Research, 7-Sport-Prediction-Research, Soccer-Prediction-Research and Baseball-Prediction-System. Transfer mechanisms, not performance claims.

- PIT maturity firewall: prior outcomes used to train any meta-model, confidence-risk model, predictability model, routing layer or calibration layer must be demonstrably mature by the target prediction cutoff. A row whose outcome became known after its own prediction cutoff is not eligible merely because it is earlier in dataframe order. Missing or contradictory maturity timestamps fail closed.
- Partial-failure semantics: a selected universe member, source batch, enrichment batch or required acquisition error must be surfaced as DEGRADED/FAILED rather than converted to OK because other rows succeeded. Required failure must propagate to the workflow exit and downstream evidence gate.
- Long-running OOS continuity: chronological OOS workflows must not be cancelled by ordinary main-branch commits. They may remain valid across explicitly allowlisted control-plane-only changes only when the changed files cannot affect runtime data, features, targets, PIT, scoring, routing, calibration, selection, adoption or candidate identity. Evidence-affecting changes invalidate the run and require fresh OOS.
- Evidence freshness: any change capable of altering OOS selection, calibration, scoring, routing, adoption gates, PIT semantics, candidate identity or runtime feature schema must trigger the relevant validation/OOS workflow before prior evidence is considered current.
- Feature runtime contract: production features must have a canonical manifest plus machine-readable policy, deterministic assembly order and a runtime schema/hash that can be reconciled with the frozen candidate. A feature name in source code is not evidence of production use.
- Analysis/evidence identity: analysis artifacts must bind to the exact code/config SHA that performed the analysis. Later bot commits that publish or summarize evidence are separate commits and must not silently replace the analysis identity.
- Concurrency and durable state: critical checkpoint/state writes use single-writer or compare-and-reconcile semantics. A stale collector/OOS process must not overwrite a newer remote snapshot.
- Research handoff: expensive OOS may run only after explicit TESTS_PASSED and AUDIT_PASSED prerequisites; absent/invalid handoff is BLOCKED rather than an assumed pass.
- Cross-project evidence is mechanism-level only. A successful method in another repository never enters Stock Production without local compatibility checks, local PIT audit, chronological OOS/WFO, calibration, robustness, frozen holdout and release-gate evidence.

## Current priority hierarchy

PIT integrity > apparent backtest gain.
Future generalization > historical fit.
Case-level correctness > aggregate-only optimization.
Calibration and predictability awareness > raw confidence.
Independent evidence > source count.
Robustness > single-period improvement.
Safe degradation > forced prediction.
Reproducibility and state consistency > convenient output.
Failure learning > repeated failure.

## Production-readiness invariant

A candidate is not production-eligible unless the complete bundle is internally consistent across model, training window, target, features/schema, calibration, routing, uncertainty policy, universe/scope, source/PIT policy, frozen holdout, reproducibility manifest, release gate, monitoring and rollback target.



## Temporal state research layer

時間方向の情報は、日次OHLCVで確認できる範囲から段階的に拡張する。現在は `temporal_state_research` をResearch-only Challengerとして運用し、過去数セッションのリターン、経路リターン、方向持続性、リターン加速度、ボラティリティ遷移、出来高圧力、日中レンジ変化、終値位置変化を因果的に生成する。

この層は `available_at` の存在を必須とし、retrieved_atを過去時点の可用性証拠として代用しない。将来行を参照しないことを構造テストで監査し、current feature setとの同一chronological OOS比較を行う。Frozen holdoutは使用せず、production_changed=false、promotion_allowed=falseを維持する。
候補特徴構成を比較するときは、可能な範囲でprequentialに選択する。fold t の構成選択は t より前に完了したOOS結果だけで行い、初回はincumbentをアンカーとする。fold t 自身の結果をそのfoldの選択へ使用しない。候補選択の集計値は診断情報であり、単一実行だけでproductionへ昇格させない。

将来intradayデータを利用できる場合も、1つの巨大モデルへ直接統合せず、複数時間粒度のstate encoder、state transition、dynamic expert routing、predictability/uncertaintyをResearch-firstで段階検証する。時間粒度の追加自体を成果とせず、LogLoss、Brier、ECE、Accuracy、case-level stability、robustness、PITを含むincremental OOS valueで判断する。
## Frontier research ecology

When broad research is requested, search beyond ordinary hyperparameter tuning. Explore materially different mechanisms across probability geometry, robust aggregation, prior/recency expert weighting, diversity-aware weighting, calibration order, case-level shrinkage, selective prediction, ranking transforms, and regime/uncertainty interactions.

Every candidate must use the same chronological OOS case set where feasible. Learned or tuned quantities must use strictly prior evidence. Frozen holdout remains untouched until candidate specification is locked. External methods are discovery inputs only; external performance is never local evidence. For prequential frontier selection, the final research winner must be the candidate selected by the last development decision made before scoring that fold; the decision-fold outcome itself must never determine that selection. Any aggregate ranking of selected candidates is diagnostic only.
