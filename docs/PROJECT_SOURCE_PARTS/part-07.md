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



## 104. Nested ranking: initialization and compute-only prediction reuse

The candidate training-window tuple `(252, 504, 756, 0)` is declared before the nested window-conditioned prediction bank is constructed. This ordering is part of the executable research contract; the bank must never depend on a later local definition.

After the nested bank has materialized, the later global classifier-window scoring step reuses the exact fold-local calibrated probability vector for the globally selected classifier on non-zero windows instead of refitting the same model/window/calibration path a second time. This is compute-only reuse. It does not add current outcomes to selection history and does not alter the nested selector's prior-fold-only contract.

The research artifact records the reused fold/window pairs. PIT boundaries, model × window LogLoss evidence, contiguous prior-fold requirement, production identity audit, and promotion gate are unchanged.
