* delisted instrument gap
* source gap
* PIT metadata gap
* filing timestamp gap
* event gap
* market-calendar gap
* feature gap
* regime gap
* OOS gap

に分解する。

model complexityだけでDebtを隠さない。

⸻

95. UNKNOWN FRONTIER

優先研究対象:

* high disagreement
* high-confidence wrong cases
* low predictability
* regime transition
* new listings
* delistings
* event shocks
* source conflict
* sparse histories
* OOD
* extreme volatility
* unexplained ranking failure
* unexplained calibration drift

UNKNOWNを既知カテゴリへ強制分類しない。

⸻

96. COMPLETION DEFINITION

「24時間Workflowが動いた」
「Actionsがgreen」
「prediction fileが生成された」
ではcompleteとしない。

completion evidence:

* tests
* PIT audit
* leakage/meta-leakage audit
* survivorship/universe audit
* market-calendar audit
* chronological WFO/OOS
* calibration
* ablation
* robustness
* frozen holdout
* artifact integrity
* reproducibility
* recovery
* release gate
* monitoring
* rollback
* report

未実施項目は未実施と明示する。

⸻

97. FINAL OPERATING LOOP

MONITOR
→ DETECT
→ TRIAGE
→ RESEARCH
→ IMPLEMENT
→ TEST
→ PIT
→ OOS/WFO
→ CALIBRATION
→ ROBUSTNESS
→ HOLDOUT
→ ADOPT/HOLD/REJECT
→ RELEASE
→ PRODUCTION
→ RECONCILE
→ FAILURE ANALYSIS
→ MEMORY
→ NEXT RESEARCH

を継続する。

⸻

98. ULTIMATE PRINCIPLE

Stock-Daily-Prediction-3000の最適化対象は単なるdirection accuracyではない。

Future Generalization
×
Case-Level Correctness
×
Calibration
×
Expected Return Quality
×
Price Forecast Quality
×
Cross-Sectional Ranking
×
Predictability Awareness
×
Uncertainty Quality
×
Robustness
×
PIT Integrity
×
Survivorship Integrity
×
Information Efficiency
×
Operational Reliability
×
Recovery
×
Reproducibility

を総合的に最大化する。

特に、

PIT > Apparent Backtest Gain
Survivorship Integrity > Convenient Universe
Future Generalization > Historical Fit
Case-Level Analysis > Aggregate Average
Calibration > Raw Confidence
Robustness > Single-Fold Gain
Information Value > Source Count
Evidence > Assumption
Safe Degradation > Forced Prediction
Reproducibility > Convenient Output
Failure Learning > Repeated Failure

を基本原則とする。

最終的に構築するのは「3000銘柄を予測する巨大モデル」ではない。

どの市場・銘柄を対象にするか、
その銘柄がその時点で本当にeligibleだったか、
何がそのcutoffまでに利用可能だったか、
どの予測targetを使うか、
どのmodel/routingが適切か、
どの程度の不確実性があるか、
追加情報を取得する価値があるか、
いつ予測を控えるべきか、
なぜ外れたか、
その失敗を次の改善へどう変換するか

まで制御できるAdaptive Stock Prediction Intelligenceを構築することを最終目標とする。

=== COPY END ===

⸻

99. NESTED PREQUENTIAL RANKING SELECTION — 2026-10-04

Cross-sectional ranking selection is divided into two evidence lanes.

Legacy same-run ranking diagnostics may still be emitted for comparison, but they are not production-selection evidence because they can condition on globally selected model/training-window state from the same OOS run.

The canonical research challenger is:
nested_prequential_ranking_selection

For outer fold t:

1. classifier model selection uses only classifier LogLoss from folds < t
2. return-estimator selection uses only prior-fold Rank IC
3. ranking probability weight and uncertainty penalty use only ranking outcomes from prior outer folds
4. current fold is scored only after those selections are fixed
5. current outcomes enter future selection history only after scoring

The nested ranking selector does not accept the globally selected model or globally selected training-window value. This prevents silent same-OOS reuse of global model/window evidence.

The current candidate remains:

research_only = true
production_changed = false
promotion_allowed = false
same_oos_global_model_or_window_reuse = false

ranking_selection_ready_for_production remains false until the selected nested evidence is identity-bound to the actual frozen production classifier training window and return-estimator configuration and then survives frozen-holdout evaluation.

Nested ranking evidence is therefore an incremental research-control improvement, not a production promotion claim.


100. NESTED RANKING IDENTITY BINDING — 2026-10-04

The nested ranking evidence is computed only after the current OOS run has materialized the production candidate configuration. The research result records a fail-closed identity audit covering:

* selected classifier model versus final prequential model
* selected return estimator versus final prequential estimator
* frozen classifier training-window sessions versus the training window used to generate the nested ranking prediction bank
* production ranking probability weight versus final prequential ranking weight
* production ranking uncertainty penalty versus final prequential uncertainty penalty

The nested selector itself remains prior-fold-only. Production configuration values are used only for post-scoring identity reconciliation and never for selecting the current fold.

Identity alignment is not a promotion signal by itself. Even when aligned, nested ranking remains research-only until bootstrap evidence, chronological OOS/robustness, calibration, PIT, and the frozen-holdout gate are all satisfied.

A production freeze must fail closed when nested ranking evidence is missing, unevaluated, structurally non-prequential, bootstrap-insufficient, or not identity-aligned.

101. LEARNED CASE-RISK SESSION-CLUSTER BOOTSTRAP — 2026-10-04

The learned case-risk locked-suffix comparison now records paired LogLoss improvement with dependence-aware moving-block bootstrap evidence.

* each unique prediction_cutoff is treated as an ordered dependence cluster
* cluster delta = fixed case-risk LogLoss - learned case-risk LogLoss, so positive means learned improvement
* fewer than 5 clusters => INSUFFICIENT_CLUSTERS; bootstrap promotion evidence is unavailable
* 5 or more clusters => moving-block bootstrap probability and p05 are recorded
* locked outcomes are never used to fit the learned model or select its threshold
* frozen holdout remains outside this research artifact

This bootstrap is evidence for stability and statistical uncertainty only. It does not by itself authorize production routing or replace chronological OOS, calibration, PIT, robustness, and frozen-holdout gates.


## 102. Nested ranking: model × training-window prequential identity

Nested ranking research now consumes a window-conditioned classifier prediction bank in addition to the existing fold OOS bank. For every outer fold, the selector jointly chooses the classifier model and classifier training-window length using only LogLoss evidence from earlier outer folds. When sufficient prior evidence does not exist, the selector uses a warmup default of full eligible history (window 0) and records that status explicitly.

For the current outer fold, probabilities are produced from the selected model/window pair before any current outcome is appended to selection history. After scoring, the current fold outcome is appended to every valid model/window pair in the window-conditioned bank. This preserves strict prequential ordering while making the eventual production training-window identity auditable.

The evidence exposes selected training window by fold, final prequential training window, per-fold selection status, and a boolean named training_window_selection_prequential. Production identity alignment now compares the frozen classifier training window directly against the final prequential window rather than against a fixed prediction-generation window of 0. The existing production gate remains fail-closed; implementation alone does not authorize promotion.


## 103. Nested window evidence requires contiguous prior folds

The nested model × training-window selector now requires the immediately preceding min_history_folds outer folds to be present for a candidate pair. Sparse or gapped model-window history is excluded rather than allowing a candidate to qualify from a non-contiguous subset. The artifact also preserves per-window/model fold coverage so missingness is inspectable instead of being silently treated as equivalent evidence.

⸻

104. CROSS-PROJECT VALIDATED PATTERN TRANSFER — 2026-10-04

他projectはperformance benchmarkではなく、failure prevention / evidence integrity / information efficiencyのmechanism sourceとして利用する。

参照対象:
* BTC-Prediction-Research
* 7-Sport-Prediction-Research
* Soccer-Prediction-Research
* Baseball-Prediction-System

Transfer pipeline:

DISCOVER
→ SOURCE_COMPARE
→ ABSTRACT_MECHANISM
→ COMPATIBILITY_CHECK
→ LOCAL_IMPLEMENTATION
→ LOCAL_UNIT/INTEGRATION_TEST
→ LOCAL_PIT
→ LOCAL_OOS/WFO
→ ROBUSTNESS
→ FROZEN_HOLDOUT
→ RELEASE_GATE
→ ADOPT / HOLD / REJECT

他projectでのmetric gain、production status、green CI、paper claimはStockのevidenceとして直接使用しない。

⸻

105. PIT MATURITY / PRIOR-OUTCOME FIREWALL — 2026-10-04

OOS predictionを対象とするlearned case-risk、meta-label、predictability、failure prediction、routing、calibration等の学習では、過去predictionの行順だけではprior evidenceとみなさない。

各training rowについて可能な限り、

prediction_cutoff
< outcome/event time
< experience_available_at

の成熟順序を明示し、target evaluation cutoffまでにoutcomeがmatureしていることを要求する。

特に、label i が transition i→i+1 の outcome に依存する場合、prediction transition i の時点で label i は未成熟として扱う。

minimum条件を満たさないrowは学習から除外し、その除外数を evidenceへ保存する。

missing / malformed / contradictory maturity timestamp:
→ PIT FAILURE / BLOCKED

「後から正解が判明している」という知識を過去prediction時点へ遡及して学習しない。

retrieved_atはmaturity evidenceではない。available_at、publication timing、outcome maturityを独立管理する。

⸻

106. PARTIAL FAILURE / DEGRADED STATE — 2026-10-04

selected future rows、universe members、source batches、enrichment batches、price shards等のrequired acquisitionで一部failureが発生した場合、成功したrowが存在してもbatch全体をOKとは扱わない。

required failureは、

OK
→ DEGRADED
→ FAILED / BLOCKED

の適切なstateへ伝播させる。

workflowはrequired failureをexit statusへ反映し、downstream OOS / release gateへ誤ってPASSを渡さない。

ただしoptional sourceのfailureを必ず全system failureへ昇格させるのではなく、scope / requiredness / fallback policyを明示する。

「一部成功したから完全成功」は禁止する。

partial outputsは、
* selected_count
* processed_count
* success_count
* failure_count
* skipped_count
* reason taxonomy
* usable_scope
を保持する。

⸻

107. LONG-RUNNING OOS / CONTROL-PLANE CONCURRENCY — 2026-10-04

長時間chronological OOSは、main commitだけを理由にcancelしてはならない。

Research validationのconcurrencyは、
cancel-in-progress = false
を基本とする。

ただし以下を明確に分類する:

A. NON-EVIDENCE-AFFECTING CONTROL CHANGE
* operational status
* documentation
* tests
* watchdog/control-plane
* append-only experience outputs

B. EVIDENCE-AFFECTING CHANGE
* model
* feature
* target
* dataset
* source
* PIT
* calendar
* scoring
* routing
* calibration
* selection
* config
* candidate identity

Bがruntime evidenceへ影響する場合、実行中runをvalid current evidenceとして継続利用せず、fresh OOSを要求する。

Aについてもallowlistを明示し、曖昧な変更を自動的にsafe扱いしない。

長時間collector / checkpoint writerではstable concurrency groupを使い、同一stateへの複数writer競合を避ける。

checkpoint publish前後にremote stateとlocal stateを再比較し、古いsnapshotが新しいsnapshotを上書きしないようにする。

⸻

108. EVIDENCE FRESHNESS / TRIGGER CONTRACT — 2026-10-04

次の変更はrelevant validation / OOSをfreshに実行しない限り、過去evidenceをcurrent evidenceとして扱わない:

* model implementation
* feature implementation / feature list
* feature schema
* target semantics
* data acquisition
* source parser
* PIT contract
* calibration
* routing
* OOS scoring
* release gate
* candidate generation
* adoption logic

trigger coverageそのものをtests / workflow contractで検証する。

「旧runがgreen」
「artifactが残っている」
「同じfile名のresultが存在する」
だけではcurrent evidenceと認めない。

analysis snapshotとevidence snapshotのSHAを分離して保持する。

⸻

109. FEATURE MANIFEST / RUNTIME SCHEMA GOVERNANCE — 2026-10-04

Production featureはsource code上の存在だけではactiveとみなさない。

canonical feature manifest、
machine-readable feature policy、
deterministic feature assembly、
runtime feature count、
runtime feature schema hash、
manifest version
を管理し、frozen production candidateと照合する。

feature statusを少なくとも、

ACTIVE
CONDITIONAL
OBSERVATION_ONLY
RESEARCH_CANDIDATE
DISABLED

へ分類する。

conditional featureはPIT-safe availabilityとconfig activationが両方確認できない限りproduction probabilityへ影響させない。

feature assembly orderはdeterministicでなければならない。

feature changeがprediction valueへ影響し得る場合は、
TEST
→ PIT
→ chronological OOS/WFO
→ calibration
→ ablation
→ robustness
→ frozen holdout
→ release gate
を要求する。

⸻

110. ANALYSIS SHA / PUBLISH COMMIT SEPARATION — 2026-10-04

研究artifactは、可能な限り、

analysis_git_sha
analysis_code_fingerprint
config_fingerprint
dataset/universe/source hashes
generation_run_id

を保存する。

後続bot commitによるsummary、manifest、status更新はpublication commitであり、analysisを実行したSHAの代替ではない。

release evidenceでは、
analysis identity
と
publication identity
を混同しない。

analysis SHAとworkflow GITHUB_SHAが一致すべきresearch laneでは、その一致を明示的に検査する。必要な場合はmismatchをFAIL-CLOSEDとする。

⸻

111. RESEARCH HANDOFF FIREWALL — 2026-10-04

高コストなchronological OOS / WFOを開始する前に、少なくとも、

TESTS_PASSED = true
AUDIT_PASSED = true

の明示的handoff状態を要求する。

missing
invalid
stale
contradictory
handoff
は、

BLOCKED

とする。

dependency timeoutなどのtransient failureと、schema / code / contract failureなどのdeterministic failureを分離する。

bounded retryはtransient classに限定し、implementation failureをretryで成功扱いしない。

⸻

112. SOURCE / FEATURE / DATA INDEPENDENCE — 2026-10-04

source countをinformation diversityと同一視しない。

同一upstreamの、

mirror
wrapper
republisher
derived archive
duplicate endpoint
secondary scraper

は原則として独立source数へ加算しない。

source graphとindependence classを保存し、ensemble weighting / evidence aggregationへ反映する。

feature countを増やすより、独立information axisを増やすことを優先する。

⸻

113. EXPERIENCE / FAILURE EVIDENCE INTEGRITY — 2026-10-04

experienceは「結果がある行を増やす」ことを目的としない。

canonical identity:
instrument
+
prediction date
+
prediction cutoff
+
target
+
prediction state

等で重複を排除し、same-event revisionや同一predictionの再取得でexperience weightを水増ししない。

failureは少なくとも、

data
timestamp
PIT
universe
source
feature
model
calibration
routing
regime
OOD
timing
automation
recovery

へ分類し、

Failure
→ Root Cause
→ Hypothesis
→ Experiment
→ OOS
→ Robustness
→ Decision
→ Memory

のchainを保存する。

negative evidenceも将来のresearch priorityと再開条件を持つknowledgeとして保存する。

⸻

114. STOCK-SPECIFIC ADAPTIVE INFORMATION LOOP — 2026-10-04

Stockでは「より多く取得する」を常に正解としない。

candidate information source / refresh / recomputeについて、

Expected Information Value
× Probability of usable update
× Source reliability
× PIT validity
÷ Latency / Compute / Cost / Operational risk

を基本思想として優先順位をつける。

source qualityが不十分なら、
PREDICT_NOW
ではなく
ACQUIRE_MORE / WAIT / RECOMPUTE / FALLBACK / ABSTAIN / DEFERRED
を選択可能とする。

情報取得に失敗した場合、成功した別sourceだけで元のscopeを完全再現できないなら、usable scopeを縮小して明示する。

⸻

115. RELEASE-GATE EVIDENCE BUNDLE — 2026-10-04

release gateは単一metricではなくbundle consistencyを確認する。

minimum evidence:

* PIT contract
* prediction ledger PIT
* leakage/meta-leakage
* universe/survivorship
* market calendar
* chronological OOS/WFO
* candidate versus incumbent on same observations
* calibration
* ablation
* robustness
* frozen holdout
* feature/schema identity
* analysis/evidence SHA lineage
* reproducibility
* artifact integrity
* monitoring/recovery
* rollback target
* state consistency

一部だけPASSしてもbundle全体が揃わなければPRODUCTIONへ進めない。

⸻

116. OPERATIONAL STATUS IS NOT PERFORMANCE STATUS — 2026-10-04

以下を明確に分離する:

EXECUTED
= workflow / commandが実行された

VERIFIED
= intended contractが検証された

PERFORMANCE_VERIFIED
= metric evidenceが検証された

PROMOTION_CANDIDATE
= release criteriaを満たす候補として比較可能

ADOPTED
= release gateを通過して採用された

PRODUCTION
= frozen runtimeとして稼働中

STABLE
= monitoring evidenceで継続安定性が確認された

queued / in_progress / completed / green Actionは、単独では上記statusを意味しない。

⸻

117. CROSS-PROJECT FRONTIER PRIORITY — 2026-10-04

他repoの最新研究からStockへ移植候補として優先するのは、

1. PIT maturity and knowledge-time firewalls
2. evidence freshness and long-run OOS continuity
3. partial failure propagation / degraded scope
4. runtime feature governance
5. concurrency-safe durable state
6. experience-derived research prioritization
7. uncertainty / case-risk routing
8. statistical dependence-aware comparison

とする。

ただしpriorityが高いことはadoptionを意味しない。

⸻

118. FINAL STOCK RESEARCH PRINCIPLE — 2026-10-04

Stock-Daily-Prediction-3000は「より複雑なmodel」を作るprojectではない。

最終systemは、

PIT-safe population
→ PIT-safe information
→ deterministic feature contract
→ prequential model ecology
→ calibrated probability
→ return/price distribution
→ uncertainty/predictability
→ selective information acquisition
→ case-level decision
→ robust OOS/WFO
→ frozen evidence
→ safe release
→ monitored production
→ exact reconciliation
→ failure learning
→ next research

を一つのcausal chainとして維持する。

特に、
「良い予測を出す」だけではなく、
「その予測がその時点で本当に作れたか」
「情報は本当にその時点で存在したか」
「失敗した場合にscopeを正しく縮小したか」
「後から得た結果を学習へ遡及させていないか」
「どのcode/config/sourceでそのevidenceが生成されたか」
「その候補をproductionへ入れてよいだけの独立証拠があるか」
までを一体で管理する。

これをFuture Generalization × Case-Level Correctness × Calibration × Predictability × Uncertainty × Robustness × PIT Integrity × Information Efficiency × Operational Reliability × Reproducibilityの基準とする。


⸻

119. NESTED-RANKING COMPUTE REUSE / EXECUTION INTEGRITY — 2026-10-04

Nested model × training-window research may reuse an already-materialized fold-local calibrated probability vector when the later global window scoring requests the identical fold, training window, and selected classifier.

This is a compute-only optimization. It must satisfy all of the following:

* the probability vector was generated from the same fold-local pre-test core and calibration slice;
* the vector shape exactly matches the later test slice;
* no current-fold outcome is introduced into model, window, calibration, or ranking selection;
* fit_rows remains available for diagnostics even when prediction reuse is taken;
* non-reusable cases fall back to the original fit/calibrate/predict path;
* reused fold/window pairs are recorded in research evidence for auditability.

The optimization changes neither PIT boundaries nor the nested prior-fold evidence contract. It is not performance evidence and cannot by itself justify promotion. Reuse is valid only when fold, training-window, model identity, calibration path, and output shape all match exactly; otherwise the original computation path is required.


⸻

120. FRONTIER PATTERN SEARCH MATRIX — 2026-10-04

The research layer must test mechanism-level pattern families rather than only hyperparameter variants.

Active families include probability-space transforms, robust/tail aggregation, prior-quality and recency weighting, minimax and diversity-aware expert weighting, calibration-before-blending, blend-level calibration, rank/probability hybrids, case-level shrinkage, selective prediction, and direction voting.

The matrix is research-only. Candidate breadth is not evidence by itself and does not alter the production release gate.

⸻

121. EXTREME PATTERN ECOLOGY / EXTERNAL-METHOD TRANSFER — 2026-10-04

External forecasting research is used for mechanism discovery only. Ideas related to forecastability-aware sparse routing, expert-loss routing and time-series dependence-aware uncertainty evaluation may be translated into small local candidates, but external reported performance never substitutes for local PIT/WFO/OOS evidence.

Required sequence:

DISCOVER → VERIFY → ABSTRACT → PIT/COST CHECK → LOCAL TEST → CHRONOLOGICAL OOS → ROBUSTNESS → FROZEN HOLDOUT → ADOPT/HOLD/REJECT

All tuned components remain prior-only, all candidates share locked OOS observations, and production promotion remains blocked until the existing independent release gate is satisfied.


⸻

122. PREQUENTIAL FRONTIER WINNER SELECTION — 2026-10-04

The final research winner is the candidate selected at the last prequential development decision before that decision fold is scored. The outcome of that decision fold is therefore excluded from the winner choice. Any candidate-level aggregate of folds on which the candidate happened to be selected is diagnostic only and cannot define the winner.

For broad frontier matrices, locked OOS may be used for final diagnostic comparison but never for winner selection.

The winner-selection path is:

1. For each development fold t, inspect only folds < t.
2. Select the candidate with the best prior development LogLoss.
3. Evaluate that selected candidate on fold t.
4. Record the decision, selected pattern, fold and score.
5. Aggregate only these prequential development decisions to choose the research winner.
6. Evaluate the frozen locked suffix only after the candidate is fixed.

This preserves a one-step-ahead selection contract inside development and makes adaptive selection visible in the evidence artifact.

A large candidate matrix does not relax multiple-testing discipline. Locked OOS remains untouched and the existing production release gate remains mandatory.


⸻

123. FRONTIER CONTRACT ID / FAILURE PRESERVATION — 2026-10-04

Each frontier research artifact must expose an explicit research_contract_id. The contract ID is part of artifact verification and prevents a result generated under one research schema from being silently interpreted as another schema.

For candidate matrices, execution failures are evidence:

* candidate exceptions must be retained with fold, candidate/family scope, error type and bounded error text;
* insufficient-valid-fold candidates must be represented explicitly;
* failures must not be silently dropped;
* a matrix with recorded candidate failures is not a clean EXECUTED success state;
* CI must require the pinned contract ID and zero execution failures before treating the matrix as successfully executed.

This control is independent of performance. It prevents execution incompleteness from being mistaken for evidence of candidate superiority.


⸻

124. RISK-OOD MISSINGNESS FAIL-SAFE — 2026-10-04

Risk-derived OOD evidence must remain finite under missingness.

* Historical risk vectors with no finite evidence produce a neutral OOD score of zero rather than NaN.
* When only some risk dimensions are observed, the OOD score uses only dimensions with finite historical and current evidence; unavailable dimensions contribute no fabricated signal.
* Entirely unavailable current risk evidence produces a finite neutral score.
* This is a fail-safe data-quality behavior, not a performance assumption and not permission to treat unknown risk as low risk.
* Regression tests must cover all-missing historical/current risk as well as mixed missingness, and execution warnings must not be used to silently downgrade research evidence.

The control preserves the distinction between unavailable, unknown and observed risk while preventing numerical NaN propagation from contaminating difficulty, routing or frontier pattern diagnostics.


⸻

125. RESEARCH EVIDENCE FRESHNESS — 2026-10-04

Research validation results are evidence artifacts, not current-state truth by default.

The exact Research validation execution SHA must be preserved with the evidence. The status/snapshot branch must also record the current main SHA used when the snapshot is persisted. The execution must additionally persist an evidence-affecting fingerprint covering runtime code, configuration, workflows and relevant dependency manifests.

Evidence freshness is determined first from the evidence-affecting fingerprint. Matching fingerprints mean evidence_freshness = FRESH even when commit SHAs differ because of non-evidence-affecting control-plane changes. Fingerprint mismatch means evidence_freshness = STALE and requires fresh OOS/WFO. If the fingerprint is absent or unverifiable, SHA equality is only a legacy fallback; otherwise evidence_freshness = UNKNOWN and release decisions fail closed.

This is a governance layer only. It does not retune frozen holdout evidence and does not mutate Production.


⸻

126. RISK-OOD MISSINGNESS-AWARE SHRINK RESEARCH CONTROL — 2026-10-04

Risk-derived OOD remains finite-neutral when historical/current risk evidence is unavailable. That neutral numeric value is not evidence that the case is fully observed or low-risk.

The frontier research layer now includes a research-only candidate that combines prior-risk OOD distance with current risk completeness without imputing unavailable dimensions.

When risk is fully observed, the candidate behaves as ordinary OOD shrinkage. When risk evidence is missing, the OOD component remains neutral while a conservative completeness factor still applies shrinkage toward 0.5.

This candidate does not alter production prediction and remains subject to chronological OOS/WFO, calibration, robustness, PIT and frozen-holdout evidence gates.


⸻

127. RESEARCH STATUS TERMINAL-EVENT STALE GUARD — 2026-10-04

Research validation status events can arrive out of order. A completed/cancelled event for an older run must not overwrite a newer active Research validation run.

The status resolver therefore keeps the event-provided run context and, when that event is terminal, checks the current active Research validation set. The resolver compares the verifiable Research run_number values:

* newer active run > terminal event → persist the active run context;
* terminal event >= active run → preserve the terminal event context;
* missing/unverifiable terminal run number or SHA with active context unavailable → BLOCKED / fail-closed.

Active requested/pending/queued/waiting/in_progress event contexts remain directly usable because the stale-event write guard in status persistence also prevents an older run from overwriting a newer persisted status.

This control is operational status integrity only. It does not alter prediction, OOS metrics, calibration, frozen holdout or production state.


⸻

128. GITHUB-SIDE RESEARCH AUTOPILOT — 2026-10-04

Research validation must be runnable without ChatGPT intervention.

The GitHub-side Research autopilot controller:

* runs daily at a canonical Asia/Tokyo schedule;
* wakes when current-main Repository verification completes;
* requires Repository verification = PASS for the exact current main SHA before dispatching expensive chronological OOS;
* dispatches Research validation explicitly on `main`;
* suppresses duplicate queued/running Research validation for the current main SHA;
* cancels queued/pending Research entries that target superseded SHAs, while leaving active runs to the evidence-fingerprint/stale-run watchdog;
* skips a new run when the current main SHA already has a successful Research validation within the last 24 hours.

The Research validation workflow therefore does not need a direct push trigger. Main-branch changes wake the controller, verification completion wakes it again, and the daily schedule provides a missed-trigger safety net.

This is an operational scheduling/control-plane contract only. It does not establish performance verification, promotion readiness, or Production status. All existing PIT, OOS/WFO, calibration, robustness, frozen-holdout and release-gate requirements remain mandatory.


⸻

129. GITHUB QUEUE SELF-RECOVERY — 2026-10-04

The autonomous Research control plane must not depend on a human noticing a queued verification or autopilot run.

The Actions reliability watchdog therefore monitors:

* Repository verification
* Research autopilot
* Research validation
* critical production/monitoring workflows

For Repository verification and Research autopilot, a queued/pending entry that remains stale beyond the bounded queue threshold is cancelled and a fresh current-main run is dispatched. Active Research validation is handled separately by the evidence-fingerprint and stale-run guards.

The recovery controller must suppress duplicate current-main executions and must never treat queue recovery as evidence of successful validation. Queue recovery is an operational reliability state only.


⸻

130. OBSOLETE RESEARCH QUEUE CONTAINMENT — 2026-10-04

Long-running research lanes must not allow obsolete queued work to consume GitHub runner capacity indefinitely.

The Actions reliability watchdog therefore applies the bounded queue recovery policy to the 24H Research Marathon as well as Repository verification and Research autopilot:

* queued/pending obsolete-SHA entries are eligible for cancellation;
* stale current-SHA queued entries are also bounded by the queue-age threshold;
* a fresh current-main dispatch is created only after stale queue cleanup;
* duplicate current-main active/queued work is suppressed;
* cancellation/recovery itself never counts as OOS evidence or performance verification.

This is operational resource protection. It does not relax research correctness or promotion gates.


⸻

131. CONTROL-PLANE FAILURE RETRY BOUND — 2026-10-04

The GitHub control plane may encounter transient runner/API/setup failures.

For current-main Repository verification and Research autopilot, the reliability watchdog may rerun a first-attempt workflow failure once. This retry:

* uses the same current-main SHA;
* does not alter research evidence;
* does not convert failure into success;
* is bounded to one retry;
* leaves a second failure explicitly FAILED.

Research/OOS execution itself remains governed by its separate stale-run and evidence-integrity policy; expensive research failures are not indiscriminately retried as if they were transient.
