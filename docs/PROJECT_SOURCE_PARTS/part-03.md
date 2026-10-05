
single-period gainではなく、

* overall
* recent
* regime
* market
* asset class
* volatility
* liquidity

でincremental valueを確認する。

⸻

26. BASELINE MODELS

Current baseline families:

* Logistic Regression
* ExtraTrees
* HistGradientBoosting

baselineを維持し、complex model追加時の比較基準とする。

⸻

27. CHALLENGER MODELS

候補:

* LightGBM
* XGBoost
* CatBoost
* PatchTST
* appropriate time-series models
* cross-sectional models
* ensemble models
* regime-aware models
* uncertainty-aware models
* selective models

model sophisticationをperformance superiorityと同一視しない。

⸻

28. MODEL ECOLOGY

候補role:

* global generalist
* JP expert
* US expert
* stock expert
* ETF expert
* REIT expert
* recent-data expert
* long-memory expert
* regime expert
* volatility expert
* event expert
* calibration expert
* uncertainty expert
* selective expert
* fallback expert

model registryにscopeを明示する。

⸻

29. ROUTING POLICY

Current routing concept:

asset class + market regime
→ asset class
→ regime
→ global

historical validationが不足するroutingをproductionで使用しない。

routing条件:

* minimum observations
* sufficient folds
* both classes where applicable
* calibration evidence
* recent stability

を満たさない場合は上位scopeへfallbackする。

⸻

30. ROUTING SELECTION SCORE

Current research convention:

selection score =
mean OOS LogLoss
+
0.25 × fold LogLoss standard deviation

を候補評価軸として使用可能。

ただしsingle scoreだけで決定せず、

* sample size
* latest holdout
* worst fold
* calibration
* robustness
* complexity
* routing instability

を併記する。

⸻

31. REGIME INTELLIGENCE

candidate regime:

* bull/bear/sideways
* volatility regime
* liquidity regime
* rates regime
* macro regime
* correlation regime
* sector rotation
* event regime
* market shock

regime detector自体もPIT検証する。

future labelsを利用してregimeを定義しない。

⸻

32. DRIFT

監視:

* feature drift
* target drift
* price distribution shift
* volatility shift
* liquidity shift
* correlation shift
* sector distribution shift
* source drift
* calibration drift
* error drift

drift detected時は、

recalibration
→ reroute
→ retrain
→ research
→ fallback

等を検討する。

⸻

33. PREDICTABILITY

instrument-day単位でpredictabilityを評価する。

signals:

* model disagreement
* data completeness
* OOD
* regime ambiguity
* volatility
* liquidity
* event risk
* calibration instability
* cross-model entropy

confidenceとpredictabilityを別概念とする。

⸻

34. UNCERTAINTY DECOMPOSITION

可能な範囲で:

* aleatoric
* epistemic
* data
* source
* temporal
* regime
* OOD
* model disagreement
* event risk

を分離する。

prediction marginだけをuncertaintyとしない。

⸻

35. MODEL DISAGREEMENT

保存候補:

* probability variance
* entropy gap
* KL divergence
* JS divergence
* prediction rank disagreement
* expected-return disagreement
* quantile disagreement

disagreement spikeはreview / additional information / fallback / abstention candidateとする。

⸻

36. EVENT INTELLIGENCE

イベント候補:

* earnings
* guidance
* filings
* dividends
* corporate actions
* macro announcements
* central-bank meetings
* major economic releases
* index changes
* market-wide stress

event情報は必ずPIT validationする。

event existenceとevent publication timeを混同しない。

⸻

37. INFORMATION VALUE

追加source / feature / updateの価値:

* ΔLogLoss
* ΔBrier
* ΔAccuracy
* Δreturn metric
* ranking improvement
* calibration improvement
* uncertainty reduction
* failure avoidance
* OOD detection
* latency
* cost

source/feature数ではなくincremental information valueで判断する。

⸻

38. INFORMATION ACQUISITION POLICY

action:

PREDICT_NOW
ACQUIRE_MORE
WAIT
RECOMPUTE
FALLBACK
ABSTAIN

判断:

Expected Information Value
× update probability
× source reliability
÷ latency/cost/risk

情報を増やすこと自体を目的としない。

⸻

39. FORECAST LIFETIME

prediction state:

FRESH
AGING
STALE
UNKNOWN
SHOCKED
REQUIRES_RECALC
FALLBACK
ABSTAIN
INVALIDATED

major market move、new filing、earnings release、corporate event、source revision等でstateを更新する。

⸻

40. SELECTIVE PREDICTION

prediction coverageを100%固定しない。

candidate policy:

* predict
* fallback
* abstain
* defer
* acquire-more

評価:

* coverage
* selective risk
* calibration
* utility
* stability
* false-abstention cost

⸻

41. CONFIDENCE-RISK LAYER

wrong-direction riskを推定してpredictionをpriorへshrinkするcandidateを研究できる。

単独でdirection flipを行わない。

production導入には独立OOS/holdout evidenceを要求する。

⸻

42. CONFORMAL / RISK CONTROL

research:

* conformal prediction
* adaptive conformal
* local conformal
* online conformal
* risk-controlled prediction
* quantile calibration

評価:

* coverage
* interval width