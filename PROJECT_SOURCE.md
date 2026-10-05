Stock-Daily-Prediction-3000 — Project Source

TARGET:
https://github.com/sasasotaro1202-star/Stock-Daily-Prediction-3000

ROLE:
本SourceをStock-Daily-Prediction-3000の技術・データ・研究・検証・運用仕様の正本とする。
Project Instructionsは常時遵守する高優先ルール、本Sourceはその詳細な実装規約・評価規約・研究規約・状態管理規約を保持する。

現行GitHubのHEAD、branch、code、config、tests、workflows、Actions、artifacts、registry、実測値を最優先する。
過去文書や会話と矛盾しても、過去結果を改変して整合させてはならない。

⸻

1. SYSTEM MISSION

単純なhistorical backtest accuracyではなく、未知の将来営業日に対するFuture Generalizationを最大化する。

主要目的:

* Case-Level Correctness
* Probabilistic Quality
* Calibration
* Expected Return Quality
* Price Forecast Quality
* Cross-Sectional Ranking Quality
* Uncertainty Quality
* Predictability Awareness
* Robustness
* PIT Integrity
* Survivorship Integrity
* Information Value
* Selective Prediction
* Operational Reliability
* Recovery
* Reproducibility

System:

Universe
→ Market Calendar
→ Data Acquisition
→ PIT/Snapshot
→ Feature Lineage
→ Candidate Models
→ Routing/Ensemble
→ Calibration
→ Uncertainty/Predictability
→ OOS/WFO
→ Robustness
→ Frozen Holdout
→ Release Gate
→ Production
→ Reconciliation
→ Failure Analysis
→ Research
→ Adoption/Rollback
→ Monitor

の閉ループとして扱う。

⸻

2. CURRENT REPOSITORY INTEGRATION

既存repoには、

* 24h autonomous research
* watchdog
* bounded recovery
* daily prediction
* market cycle
* production monitoring
* on-demand production
* experience review
* repository verification
* research validation
* freeze holdout
* BOJ research
* FINRA short-sale research
* SEC filings
* Treasury curve
* free-data discovery
* temporal confidence/risk
* conformal research
* event intelligence
* drift/regime
* online ensemble
* routing
* sequential research

等のWorkflow/implementation系統が存在する。

作業時は既存implementationを先に監査し、

REUSE
→ REPAIR
→ INTEGRATE
→ TEST
→ VERIFY

を優先する。

同機能の重複実装を避ける。

Workflowの存在、green CI、artifact生成だけでは性能・production完成の証拠としない。

⸻

3. UNIVERSE MASTER CONTRACT

Universeは固定銘柄リストではなくtime-dependent populationとする。

現在対象product family:

* JP stock
* JP ETF
* JP REIT
* US stock
* US ETF

別pipeline:

* investment trusts
* Japan stock CFD
* 10x CFD
* iDeCo

Current universe acquisitionでは、

* official list
* access timestamp
* snapshot hash
* source
* effective date
* additions
* removals
* ticker/name changes

を保存する。

⸻

4. SURVIVORSHIP-BIAS CONTRACT

過去の市場populationを現在の銘柄一覧だけで再構成しない。

必ず可能な限り、

* listing date
* delisting date
* ticker change
* name change
* merger
* spin-off
* market transfer
* corporate restructuring
* historical membership

を時系列管理する。

「現在も存在する銘柄だけ」で過去OOSを作らない。

survivorship auditをrelease gateに含める。

⸻

5. CORPORATE ACTIONS

処理対象:

* stock split
* reverse split
* dividend
* special dividend
* merger
* acquisition
* spin-off
* share issuance
* ticker change
* symbol change

adjusted priceとraw priceを混同しない。

return targetとexpected close targetでprice conventionを固定し、version管理する。

corporate action情報自体もavailable_at/PITを監査する。

⸻

6. MARKET CALENDAR

日本株・米国株を独立市場として管理する。

必須:

* local timezone
* session open
* session close
* holiday
* weekend
* early close
* special session
* next business day
* market-specific calendar

Japan/USのcalendarを単一generic weekdayで処理しない。

「翌日」を単純date+1で定義しない。

⸻

7. PREDICTION CUTOFF

Predictionは明確なcutoffを持つ。

基本:

information available at cutoff
→ feature snapshot
→ prediction

を維持する。

各predictionで可能な限り:

* instrument_id
* market
* product_family
* prediction_date
* target_date
* cutoff
* source snapshot
* feature version
* model version
* calibration version
* prediction state

を保存する。

⸻

8. TIME MODEL

区別:

* market_date
* market_open
* market_close
* prediction_time
* prediction_cutoff
* source_available_at
* published_at
* retrieved_at
* effective_at
* event_time
* filing_time
* revision_time

特に、

retrieved_at ≠ published_at ≠ available_at

を原則とする。

後から取得できたことを、その時点で利用可能だった証拠とみなさない。

⸻

9. PIT CONTRACT

feature/sourceがprediction cutoff以前に利用可能であることを要求する。

基本:

available_at <= prediction_cutoff

禁止:

* future price
* future return
* future close
* future volume
* later filing
* later earnings update
* future macro release
* later revised fundamental
* future universe membership
* future corporate action knowledge
* future market state

historical availabilityが証明できない場合は:

UNKNOWN
または
UNVERIFIABLE

とする。

production-quality OOSへの投入を禁止する。

⸻

10. FINANCIAL STATEMENT / SEC / DISCLOSURE TIME

financial dataでは、

period_end
filing_date
publication_time
effective_date
retrieval_time

を分離する。

企業業績値が「対象四半期のもの」であることと、「prediction時点で公開されていた」ことを同一視しない。

SEC等のfiling-based informationも公開時点をPITへ使用する。

revision後のhistorical financial valueを、過去predictionへ無条件backfillしない。

⸻

11. MACRO / CENTRAL BANK / TREASURY

候補:

* BOJ
* Treasury
* macro indicators
* yield curve
* policy events

について、

announcement_time
effective_time
measurement_period
revision_time
retrieved_at

を区別する。

「その月の経済データ」であっても、発表前ならprediction featureに使用不可。

⸻

12. FINRA / SHORT INTEREST / MARKET STRUCTURE

short-sale、short-interest、market structure dataは、

* observation date
* publication date
* availability
* reporting lag
* revision

を区別する。

reporting delayがあるデータを当日値として扱わない。

⸻

13. DATA SOURCE REGISTRY

各source:

* source_id
* owner
* upstream
* endpoint
* data type
* market
* coverage
* historical depth
* freshness
* available_at support
* published_at support
* revision behavior
* schema
* parser
* license
* cost
* reliability
* last success
* last failure
* incremental value
* production status

を記録する。

⸻

14. SOURCE INDEPENDENCE

以下は独立sourceとして重複加算しない:

* mirror
* wrapper
* copied dataset
* republished CSV
* derived archive
* scraper from same upstream
* same upstream under different endpoint

source graphを保持し、independenceをensemble / evidence評価に反映する。

⸻

15. DATA QUALITY CONTRACT

quality dimensions:

* universe coverage
* instrument coverage
* price completeness
* volume completeness
* corporate action integrity
* financial data completeness
* timestamp completeness
* availability completeness
* revision integrity
* identity integrity
* schema stability
* source freshness
* source reliability
* market-calendar integrity

row countだけでquality completeと判断しない。

⸻

16. MISSING DATA

missing != zero。

state:

UNAVAILABLE
UNKNOWN
DELAYED
NOT_APPLICABLE
NOT_YET_PUBLIC
SOURCE_FAILED
MALFORMED
REVISED
STRUCTURALLY_ABSENT

critical feature failureでは、

FALLBACK
ABSTAIN
DEFERRED
FAIL

を使用する。

silent zero-fillは禁止。

⸻

17. INSTRUMENT IDENTITY

stable instrument_idを使用する。

保持:

* canonical symbol
* raw symbol
* issuer
* exchange
* market
* product family
* effective dates
* source identifiers

ticker/name変更を同一時点へ無条件collapseしない。

ETF/REIT/stockをsemantic上混同しない。

⸻

18. TARGET DEFINITIONS

Current production targets:

Direction

next-business-day UP probability

target definitionを明示的にversion管理する。

Expected Return

next-business-day expected return。

return convention:

* arithmetic return
* log return
* adjusted/raw price convention
* corporate action treatment

を固定する。

Expected Close

next-business-day expected real-price close。

price basis、adjustment、market sessionを明示する。

Conditional Range

q10
q50
q90

等のconditional predictive range。

Cross-Sectional Rank

同一prediction date / market / eligible universe内でrelative rankingを生成する。

Model Uncertainty

uncertainty componentを独立保存する。

⸻

19. DIRECTION TARGET CONTRACT

UP/DOWN labelの境界を明確化する。

flat/near-zero return handlingが必要な場合はtarget_versionを分離する。

future adjusted priceのbackfillをPIT violationとして監査する。

⸻

20. RETURN / PRICE CONSISTENCY

Expected ReturnとExpected Closeは互いに矛盾しないように検査する。

必要条件例:

Expected Close
≈
Current Reference Price × (1 + Expected Return)

price convention、corporate action、currency、session differenceを考慮する。

内部整合性auditを行う。

⸻

21. CROSS-SECTIONAL RANKING

Rankingは時間系列予測と別評価する。

対象:

* top-k hit
* rank correlation
* NDCG / ranking metrics
* spread between top/bottom buckets
* stability
* turnover-aware usefulness

同一時点populationをPIT-safeに固定する。

future universe membershipをrankingへ混入させない。

⸻

22. FEATURE ECOSYSTEM

候補:

Market

* price
* return
* volume
* volatility
* turnover
* gap
* momentum
* reversal

Technical

* moving averages
* trend
* breakout
* oscillator
* volatility structure

Cross-Sectional

* relative momentum
* sector rank
* size rank
* valuation rank
* volume rank
* quality rank

Fundamental

* revenue
* earnings
* margins
* growth
* balance sheet
* cash flow
* valuation

Event

* earnings
* filing
* dividend
* split
* guidance
* corporate event

Macro

* BOJ
* Treasury/yield curve
* FX
* inflation
* rates
* macro regime

Market Structure

* short-sale
* liquidity
* market breadth
* spreads
* volume concentration

Meta

* data quality
* source reliability
* uncertainty
* OOD
* regime
* disagreement

candidate数増加自体は成果ではない。

⸻

23. ROLLING FEATURE CONTRACT

rolling/window featureは、

window_end <= prediction_cutoff

を満たす。

future observations、future revisions、future universe valuesを混入させない。

lookback windowはversion管理する。

⸻

24. FEATURE LINEAGE

各featureについて可能な限り:

* feature_id
* source_id
* raw fields
* transformation
* window
* available_at
* cutoff
* revision policy
* quality status
* version
* code commit

を保持する。

⸻

25. FEATURE ABLATION

新feature familyは、

FULL
vs
WITHOUT_BLOCK

でchronological OOS比較する。

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
+0.25 × fold LogLoss standard deviation

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
* selective risk
* temporal stability
* regime robustness

⸻

43. OOS / WFO STANDARD

基本:

Train
→ Validation
→ Chronological OOS/WFO
→ Robustness
→ Frozen Holdout

random split禁止。

future rowsをtrainingへ混入させない。

candidate selectionとfinal OOSを分離する。

⸻

44. OOS RECORD CONTRACT

各prediction rowで可能な限り:

* instrument_id
* prediction date
* cutoff
* target date
* universe snapshot
* data snapshot
* feature version
* model version
* calibration
* probability
* expected return
* expected close
* quantiles
* rank
* uncertainty
* regime
* quality state
* PIT status
* actual
* error classification

を保持する。

⸻

45. DIRECTION METRICS

Primary:

LogLoss

Secondary:

* Accuracy
* Brier
* ECE
* class-wise metrics
* calibration slope
* calibration intercept

必要に応じMatthews-type / AUC等を補助的に使用するが、primary objectiveを置き換えない。

⸻

46. RETURN METRICS

候補:

* MAE
* RMSE
* median absolute error
* directional accuracy
* rank correlation
* tail error
* quantile loss

distribution qualityとpoint forecast qualityを分離する。

⸻

47. PRICE METRICS

Expected closeについて:

* MAE
* RMSE
* relative price error
* tail error

を評価する。

price levelとreturn predictionの双方が必要な場合は別metricで管理する。

⸻

48. QUANTILE METRICS

q10/q50/q90について:

* pinball loss
* empirical coverage
* interval width
* interval calibration
* tail violation rate

を評価する。

⸻

49. RANKING METRICS

Cross-sectional ranking:

* Spearman
* Kendall
* NDCG
* top-k hit
* bucket spread
* turnover-aware metrics

等をcandidateとして用いる。

⸻

50. ECONOMIC USEFULNESS

Statistical improvementとeconomic usefulnessを分離する。

必要に応じ:

* transaction cost
* slippage
* spread
* turnover
* liquidity
* capacity
* fees
* execution delay
* market impact

を考慮する。

予測精度向上だけでeconomic superiorityを主張しない。

⸻

51. CALIBRATION

candidate:

* none
* temperature
* sigmoid
* beta-style
* isotonic
* suitable temporal calibrator

chronological validationで選択する。

holdout tuning禁止。

評価:

* LogLoss
* Brier
* ECE
* calibration slope
* intercept
* temporal stability

⸻

52. MULTI-MARKET EVALUATION

Japan / USを必要に応じて別評価する。

さらに:

* stock
* ETF
* REIT
* sector
* size
* liquidity
* volatility
* price range

で分解する。

一方の市場での改善だけでglobal promotionしない。

⸻

53. EXPERIENCE MEMORY

matured predictionはcanonical instrument-date-cutoff単位で保存する。

同一instrumentについてduplicate snapshots/revisionsを無制限にexperience learningへ投入しない。

experience record:

* instrument
* date
* cutoff
* prediction
* actual
* model
* calibration
* uncertainty
* regime
* data state
* source state
* error type

⸻

54. RECONCILIATION

maturity後:

prediction
→ actual
→ metric
→ error
→ state

へreconcileする。

current predictionとhistorical prediction artifactを混同しない。

⸻

55. FAILURE TAXONOMY

最低限:

DATA_FAILURE
PIT_FAILURE
TIMESTAMP_FAILURE
UNIVERSE_FAILURE
SURVIVORSHIP_FAILURE
CORPORATE_ACTION_FAILURE
IDENTITY_FAILURE
SOURCE_FAILURE
FEATURE_FAILURE
MODEL_FAILURE
ROUTING_FAILURE
CALIBRATION_FAILURE
REGIME_FAILURE
OOD_FAILURE
UNCERTAINTY_FAILURE
TIMING_FAILURE
RANKING_FAILURE
RETURN_FAILURE
AUTOMATION_FAILURE
RECOVERY_FAILURE
REPORTING_FAILURE

へ分類する。

⸻

56. FAILURE MEMORY

failure record:

* failure_id
* instrument
* date
* component
* type
* severity
* prediction
* actual
* state
* expected
* observed
* root_cause
* counterfactual
* repair
* verification
* recurrence

を保存する。

⸻

57. COUNTERFACTUAL FAILURE ANALYSIS

caseごとに:

* better data
* later valid information
* earlier information
* alternate model
* alternate routing
* calibration
* additional source
* abstention
* fallback

が改善可能だったか検証する。

実測evidenceと仮説を分離する。

⸻

58. COVERAGE DIGITAL TWIN

coverageを:

* market
* exchange
* universe
* instrument
* date
* source
* feature
* event
* PIT metadata
* target
* regime
* horizon

へ分解する。

Coverage Debtを明示する。

⸻

59. UNIVERSE FRONTIER

定期的に、

* newly listed
* recently delisted
* ticker changed
* newly eligible ETF/REIT
* unusual product
* market transfer

を検出する。

scope拡張前にdata/PIT/OOS/operational readinessを確認する。

⸻

60. RESEARCH FRONTIERS

定期scan:

Data Frontier
Source Frontier
Model Frontier
Feature Frontier
Failure Frontier
Scope Frontier
Unknown Frontier

目的は無制限拡張ではなくsystem limitation発見。

⸻

61. RESEARCH ROUTER

Research:

DIRECT_SEARCH
METHOD_SEARCH
FAILURE_SEARCH
COUNTEREXAMPLE_SEARCH
IMPLEMENTATION_SEARCH
BENCHMARK_SEARCH
NEGATIVE_EVIDENCE
FRONTIER_SEARCH
CROSS_DOMAIN_TRANSFER
UNKNOWN_UNKNOWN_SEARCH

⸻

62. RESEARCH PORTFOLIO

research allocation:

* exploit
* adjacent
* frontier
* replication
* ablation
* adversarial
* recovery
* meta-research

既知手法の反復だけにresearch budgetを集中させない。

⸻

63. EXTERNAL RESEARCH INGESTION

External evidence:

DISCOVERED
→ SOURCE_VERIFIED
→ METHOD_ABSTRACTED
→ RELEVANCE_CHECKED
→ COST_CHECKED
→ PIT_CHECKED
→ LOCAL_IMPLEMENTATION
→ LOCAL_REPRODUCTION
→ OOS
→ ROBUSTNESS
→ HOLDOUT
→ DECISION

paper/GitHub/AIのperformance claimはproduction evidenceではない。

⸻

64. EVIDENCE LEVEL

E0 = idea
E1 = external claim
E2 = external implementation
E3 = local reproduction
E4 = local OOS
E5 = robustness
E6 = frozen holdout
E7 = production evidence

⸻

65. NEGATIVE KNOWLEDGE

保存対象:

* rejected model
* rejected feature
* rejected source
* rejected routing
* rejected calibration
* rejected universe policy
* rejected event feature
* failed PIT method
* robustness failure
* excessive computation
* reason for rejection

⸻

66. CROSS-PROJECT TRANSFER

他project知見:

DISCOVER
→ ABSTRACT_MECHANISM
→ COMPATIBILITY
→ ADAPT
→ LOCAL_PIT
→ LOCAL_OOS
→ LOCAL_HOLDOUT
→ SHADOW
→ PROMOTE

他domain成功をそのままstock productionへコピーしない。

⸻

67. SOURCE VALUE / EXPERIMENT VALUE

research候補は、

Expected Impact
× Generalization Potential
× Failure Relevance
× Evidence Gap
× Information Value
÷ Cost

を基本思想として優先順位付けする。

単なる新規性をpriorityにしない。

⸻

68. STATISTICAL INTEGRITY

必要に応じ:

* paired comparison
* time-block bootstrap
* instrument-cluster bootstrap
* date-block bootstrap
* confidence interval
* permutation
* forecast comparison
* multiple-comparison correction

を用いる。

cross-sectional dependence、serial correlation、market clusteringを考慮する。

⸻

69. ROBUSTNESS

candidateで最低限:

* latest period
* recent market regime
* JP/US
* stock/ETF/REIT
* low/high volatility
* low/high liquidity
* earnings periods
* event-heavy periods
* missingness
* source removal
* feature deletion
* time shift
* distribution shift
* OOD
* new listings
* delistings
* corporate actions
* market stress

を確認する。

⸻

70. ADVERSARIAL VALIDATION

定期的に:

* future timestamp injection
* future price injection
* future filing injection
* revision attack
* universe/survivorship attack
* corporate-action timing attack
* source removal
* feature deletion* time shift
* distribution shift
* regime transition
* stale-data attack

を実施する。

⸻

71. FROZEN HOLDOUT FIREWALL

holdoutはconfig freeze後のfinal evidence専用。

禁止:

* model tuning
* feature tuning
* source selection
* universe selection
* routing tuning
* threshold tuning
* calibration tuning
* repeated holdout optimization

汚染が疑われる場合はholdout statusをinvalidateし、再freezeする。

⸻

72. ADOPTION GATE

candidateはincumbentと同一OOS observationsで比較。

参考gate:

* primary relative OOS LogLoss improvement ≥3%
* auxiliary improvement ≥1%
* ≥70% evaluation periods without worsening
* newest holdout no worsening
* calibration no material degradation
* PIT violations = 0

ただしsample、variance、confidence interval、cross-sectional dependence、serial dependence、turnover、transaction cost、operational riskを考慮する。

thresholdだけで自動採用しない。

⸻

73. FREEZE / RELEASE

Release configをfreezeした後、

model
feature
routing
calibration
universe
source
target

のversionを固定する。

frozen releaseの変更はnew candidateとして扱う。

⸻

74. PRODUCTION REGISTRY

production model:

* model_id
* version
* target_version
* feature_version
* calibration_version
* routing version
* universe policy version
* source snapshot
* OOS status
* robustness status
* holdout status
* release time
* rollback pointer

を記録する。

⸻

75. PRODUCTION OUTPUT INTEGRITY

production output:

* prediction date
* target date
* instrument
* direction probability
* expected return
* expected close
* q10/q50/q90
* rank
* uncertainty
* model
* calibration
* routing
* state
* source/data status
* generated_at

を可能な範囲で保存する。

⸻

76. CURRENT-DATA SAFETY

Current predictionでは毎回:

* current date/time
* current market calendar
* latest valid universe
* latest valid schedule/session state
* required source freshness
* production registry
* model validity

を確認する。

過去artifactをcurrent predictionとして表示しない。

⸻

77. FAIL-CLOSED

以下ではpredictionを無理に生成しない:

* invalid model
* missing required data
* stale required source
* PIT unknown
* universe inconsistency
* identity mismatch
* broken feature contract
* invalid calibration
* corrupted artifact
* target mismatch
* market calendar failure

必要ならfallback/abstain/deferred。

⸻

78. FALLBACK CHAIN

基本:

Champion
→ Scoped Expert
→ Global Generalist
→ Baseline
→ Abstain

fallback理由をprediction logへ保存する。

fallbackをChampionと同等品質とは扱わない。

⸻

79. AUTOMATION RELIABILITY

GitHub Actionsは:

* checkpoint
* resume
* idempotency
* retry
* exponential backoff
* watchdog
* heartbeat
* stale-run detection
* bounded runtime
* artifact preservation
* deterministic writes
* concurrency control
* recovery
* rollback
* long-research queue recovery
* bounded first-failure retry
* critical workflow failure reconciliation

を実装・検証する。

長時間Research/Data Collectorは正当なactive OOSを中断しない一方、stale queueと初回failureはGitHub側で限定回復する。

⸻

80. SINGLE-WRITER

critical state:

* production registry
* experiment registry
* source registry
* universe state
* promotion state
* rollback state
* experience ledger

はsingle-writer semanticsを優先する。

parallel research結果のmergeはdeterministicに行う。

⸻

81. CACHE / EFFICIENCY

最適化順:

cache
→ exact snapshot reuse
→ incremental update
→ deduplication
→ vectorization
→ parallel I/O
→ selective recomputation
→ retraining optimization
→ algorithm optimization

同一fingerprintを重複計算しない。

⸻

82. AUTOMATION QUALITY

metrics:

* false success
* false recovery
* repeated failure
* recovery time
* checkpoint recovery
* duplicate runs
* stale artifacts
* wasted compute
* blocked queue
* orphaned state

green Action countをsystem qualityのproxyにしない。

⸻

83. COST FIREWALL

優先:

1. verified free
2. free quota
3. OSS/local
4. cached snapshot
5. lightweight computation

paid-only
billing-risk
unknown-cost
trial auto-renew
quota overage

は自動利用禁止。

cost不明 = HOLD / UNCONFIRMED。

⸻

84. SECURITY / GOVERNANCE

secret/API key/tokenを:

* source
* log
* artifact
* report
* commit

へ出力しない。

data sourceについてlicense、rate limit、redistribution、commercial restriction、retentionを確認する。

不明sourceをproduction dependencyにしない。

⸻

85. REPRODUCIBILITY

可能な限り、

Universe snapshot
→ Data snapshot
→ Feature
→ Model
→ Calibration
→ Prediction
→ Decision

を再現できるようにする。

experiment fingerprint:

* git commit
* universe hash
* dataset hash
* source snapshots
* feature version
* target version
* model config
* calibration config
* routing
* seed
* environment

⸻

86. ARTIFACT INTEGRITY

artifactに可能な限り:

* hash
* experiment id
* model id
* dataset hash
* universe hash
* source snapshot
* git commit
* schema version
* generation time

を付与する。

⸻

87. NO-FAKE-SUCCESS

禁止:

* fabricated metrics
* missing→zero
* silent exception
* hidden partial completion
* skipped test→passed
* failed job→success
* unknown PIT→valid
* incomplete universe→complete
* invalid holdout→clean
* failed recovery→recovered

実測・推定・仮説・未検証を明確に区別する。

⸻

88. STATUS TAXONOMY

厳密に区別:

IMPLEMENTED
EXECUTED
VERIFIED
PERFORMANCE_VERIFIED
PROMOTION_CANDIDATE
ADOPTED
PRODUCTION
STABLE
HOLD
REJECTED
FAILED
BLOCKED
DEFERRED
ROLLED_BACK
UNKNOWN
UNVERIFIABLE
SUPERSEDED
RETIRED

⸻

89. PERFORMANCE CHANGE REPORT

performance変化があれば自動表示:

* Current Champion
* Prior Champion
* Candidate
* ΔLogLoss
* ΔBrier
* ΔAccuracy
* ΔECE
* return metric Δ
* ranking metric Δ
* quantile metric Δ
* Latest Holdout Δ
* Robustness Δ
* PIT status
* sample size
* evaluation period
* decision
* production status

統計的改善とeconomic improvementを区別する。

⸻

90. CHANGE INTERPRETATION

改善報告には可能な限り:

* absolute delta
* relative delta
* sample size
* fold count
* evaluation interval
* confidence interval
* dependence structure
* benchmark
* cost

を含める。

単一fold、大幅なsmall-sample gainを無条件に採用しない。

⸻

91. POLICY REGRET

後から:

* Model Regret
* Timing Regret
* Information Regret
* Routing Regret
* Universe Regret
* Scope Regret
* Policy Regret
* Research Regret

を分析する。

例:

* cutoff変更で改善したか
* filing待ちに価値があったか
* source追加が有効だったか
* specialist routeが有効だったか
* abstainすべきだったか

⸻

92. RESEARCH STOPPING

停止候補:

* repeated zero incremental value
* insufficient PIT evidence
* insufficient sample
* robustness failure
* duplicated mechanism
* source instability
* excessive cost
* frontier saturation

停止結果はNegative Knowledgeへ保存する。

⸻

93. SELF-EVOLUTION

検出:

* recurring failure
* obsolete rule
* contradictory documentation
* stale source
* inefficient workflow
* inefficient computation
* newly validated method
* universe gap
* source gap
* unexplained metric shift

flow:

PROPOSE
→ CONSISTENCY CHECK
→ HISTORICAL IMPACT CHECK
→ IMPLEMENT
→ TEST
→ PIT
→ OOS
→ ROBUSTNESS
→ HOLDOUT/RELEASE
→ ADOPT/REJECT

過去metric、holdout、failure historyを書き換えて改善を演出しない。

⸻

94. COVERAGE / RESEARCH DEBT

Debtを、

* historical universe gap
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
