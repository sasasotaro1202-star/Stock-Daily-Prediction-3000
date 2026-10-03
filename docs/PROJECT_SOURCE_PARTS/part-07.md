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