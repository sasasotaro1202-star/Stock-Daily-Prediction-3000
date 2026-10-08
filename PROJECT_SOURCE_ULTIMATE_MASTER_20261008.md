Stock-Daily-Prediction-3000

ULTIMATE MASTER PROJECT SOURCE

TARGET:
https://github.com/sasasotaro1202-star/Stock-Daily-Prediction-3000

⸻

0. SOURCE ROLE

本SourceをStock-Daily-Prediction-3000の詳細な技術・データ・PIT・評価・研究・運用・自律改善仕様の正本とする。

Project Instructionsは常時適用する上位行動規則、本Sourceはその詳細設計を保持する。

現行GitHubのHEAD、code、config、tests、workflows、Actions、artifacts、registry、実測Evidenceを、古い文書や会話より優先する。

ただし、過去のexperiment、OOS、holdout、failure、production historyを後知恵で改変して整合させない。

⸻

I. SYSTEM MISSION

1. 最終目的

目的は「過去データに最もよく適合する株価モデル」ではない。

未知の将来営業日について、

P(Y_future | Information_available_before_cutoff)

を可能な限り正確に推定すること。

対象は単一の価格値ではなく、

* Direction
* Expected Return
* Expected Close
* Quantile Distribution
* Cross-Sectional Rank
* Regime
* Uncertainty
* Predictability
* OOD
* Failure Risk
* Forecast Lifetime

を含むForecast Objectとする。

⸻

2. Ultimate Objective

長期的に、

Future Generalization
×
Case-Level Correctness
×
Probabilistic Quality
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
Information Value
×
Selective Prediction
×
Operational Reliability
×
Recovery
×
Reproducibility

を最大化する。

優先順位:

PIT

Survivorship Integrity

Future Generalization

Calibration

Robustness

Case-Level Understanding

Information Value

Operational Reliability

Complexity

⸻

II. SYSTEM IDENTITY

3. Prediction Intelligence System

本システムは単純なstock predictorではなく、

Adaptive Stock Prediction Intelligence

と定義する。

最終ループ:

REAL WORLD
↓
UNIVERSE
↓
MARKET CALENDAR
↓
EVENT DISCOVERY
↓
DATA ACQUISITION
↓
TIME / PIT
↓
DATA QUALITY
↓
FEATURE LINEAGE
↓
MARKET STATE
↓
REGIME
↓
MODEL ECOLOGY
↓
ROUTING / ENSEMBLE
↓
CALIBRATION
↓
UNCERTAINTY
↓
PREDICTABILITY
↓
OOD
↓
INFORMATION VALUE
↓
ACTION
↓
FORECAST
↓
RECONCILIATION
↓
FAILURE ANALYSIS
↓
MEMORY
↓
RESEARCH
↓
OOS/WFO
↓
ROBUSTNESS
↓
FROZEN HOLDOUT
↓
SHADOW
↓
PROMOTION
↓
PRODUCTION
↓
MONITORING
↓
RECOVERY
↓
NEXT RESEARCH

⸻

III. UNIVERSE

4. Universe is Time-Dependent

Universeを固定銘柄一覧として扱わない。

Current product family:

* JP Stock
* JP ETF
* JP REIT
* US Stock
* US ETF

その他の商品は別pipelineまたはResearch-onlyとして扱う。

Universeはprediction dateごとに、

* eligibility
* listing
* delisting
* market transfer
* ticker change
* name change
* issuer change
* product family
* exchange

を管理する。

⸻

5. Survivorship Integrity

過去predictionを現在の銘柄一覧から再構成しない。

可能な限り、

* listing date
* delisting date
* ticker history
* name history
* merger
* acquisition
* spin-off
* restructuring
* market transfer
* historical membership

を時系列管理する。

Current universe onlyで過去OOSを作ることを禁止する。

⸻

6. Universe Snapshot

各predictionで可能な限り、

* universe_snapshot_id
* generated_at
* effective_date
* source
* snapshot_hash
* eligible instruments
* additions
* removals
* identity changes

を保存する。

⸻

IV. MARKET CALENDAR

7. Market Separation

JapanとUSを独立市場として扱う。

それぞれについて、

* timezone
* session open
* session close
* holiday
* weekend
* early close
* special session
* next business day

を管理する。

「翌日」を単純にdate + 1で定義しない。

⸻

V. PREDICTION TARGET

8. Canonical Targets

Current target family:

Direction

Next-business-day UP probability.

Expected Return

Next-business-day expected return.

Expected Close

Next-business-day expected real-price close.

Quantiles

q10 / q50 / q90等。

Cross-Sectional Rank

同一prediction date・market・eligible universe内のrelative ranking。

Uncertainty

予測不確実性を独立保持する。

Target definitionはversion管理する。

⸻

9. Direction Contract

UP/DOWN境界を明示する。

Flat / near-zero returnの扱いを変更する場合は新target_versionとする。

Adjusted / raw price basisの違いもtarget contractへ含める。

⸻

10. Return / Close Consistency

Expected ReturnとExpected Closeの内部整合性を監査する。

概念的には、

Expected Close
≈
Reference Price × (1 + Expected Return)

ただし、

* adjustment
* corporate actions
* currency
* session
* market

を考慮する。

⸻

VI. CORPORATE ACTIONS

11. Corporate Action Contract

対象:

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

を時系列管理する。

adjusted priceとraw priceを混同しない。

Corporate Action自体にもPITを付与する。

⸻

VII. TIME / PIT

12. Time Model

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

原則:

retrieved_at ≠ published_at ≠ available_at

⸻

13. PIT Contract

基本条件:

available_at <= prediction_cutoff

。

retrieved_atだけをhistorical availabilityの証拠として使用しない。

availabilityが検証できない場合、

UNKNOWN
または
UNVERIFIABLE

とする。

production-quality OOSではfail-closed。

⸻

14. Forbidden Future Information

禁止:

* future price
* future return
* future close
* future volume
* future filing
* later earnings update
* future macro release
* later revised fundamental
* future universe membership
* future corporate action knowledge
* future market state

⸻

15. Financial Data Timing

financial dataでは、

period_end
≠
filing_date
≠
publication_time
≠
effective_date
≠
retrieval_time

。

「その四半期の数字」であることと、「prediction cutoff時点で公開されていたこと」を同一視しない。

⸻

16. Macro Timing

BOJ、Treasury、yield curve、macro data等について、

* measurement period
* announcement time
* publication time
* effective time
* revision time
* retrieval time

を分離する。

発表前の数値はprediction featureに使用不可。

⸻

17. Short Interest / Market Structure Timing

short-sale、short-interest、market structure等では、

* observation date
* reporting date
* publication date
* availability
* revision

を分ける。

reporting lagを無視しない。

⸻

VIII. DATA QUALITY

18. Data Quality Dimensions

少なくとも、

* universe coverage
* instrument coverage
* price completeness
* volume completeness
* corporate-action integrity
* financial completeness
* timestamp completeness
* PIT completeness
* revision integrity
* identity integrity
* schema stability
* source freshness
* source reliability
* market-calendar integrity

を監視する。

row countだけで完全性を証明しない。

⸻

19. Missingness

Missing ≠ Zero。

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

を区別する。

critical failure時:

FALLBACK
ABSTAIN
DEFERRED
FAIL

⸻

IX. SOURCE ECOLOGY

20. Source Registry

sourceごとに可能な限り、

* source_id
* owner
* upstream
* endpoint
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
* latency
* last success
* last failure
* incremental value
* production status

を保存する。

⸻

21. Source Independence

同一upstreamの、

* mirror
* wrapper
* copied dataset
* republished CSV
* derived archive
* scraper
* alternate endpoint

を独立情報として二重計上しない。

source graphを保持する。

⸻

X. FEATURE ECOLOGY

22. Market Features

候補:

* price
* return
* volume
* turnover
* volatility
* gap
* momentum
* reversal
* range
* close location
* trend

⸻

23. Technical Features

候補:

* moving averages
* trend indicators
* breakout
* oscillator
* volatility structure
* momentum acceleration
* mean-reversion signals

⸻

24. Cross-Sectional Features

候補:

* relative momentum
* sector rank
* size rank
* valuation rank
* volume rank
* quality rank
* volatility rank
* liquidity rank

⸻

25. Fundamental Features

候補:

* revenue
* earnings
* margins
* growth
* balance sheet
* cash flow
* valuation
* profitability
* leverage

⸻

26. Event Features

候補:

* earnings
* guidance
* filing
* dividend
* split
* corporate event
* index change
* major disclosure

event existenceとpublication timeを分離する。

⸻

27. Macro Features

候補:

* BOJ
* Treasury
* yield curve
* FX
* inflation
* rates
* macro regime
* central-bank events

⸻

28. Market Structure

候補:

* short-sale
* liquidity
* breadth
* spreads
* volume concentration
* trading intensity

⸻

29. Meta Features

候補:

* data quality
* source reliability
* uncertainty
* OOD
* regime
* disagreement
* freshness
* information coverage

⸻

XI. FEATURE CONTRACT

30. Rolling Window

rolling featureについて必ず、

window_end <= prediction_cutoff

。

future observation、future revision、future universe valueを混入させない。

window定義をversion管理する。

⸻

31. Feature Lineage

可能な限り、

feature_id
source_id
raw_fields
transformation
window
available_at
cutoff
revision_policy
quality_status
feature_version
code_commit

を保存する。

⸻

32. Feature Selection

feature数最大化を目的としない。

新feature familyは、

FULL
vs
WITHOUT_BLOCK

でchronological OOS比較する。

評価:

* overall
* latest
* regime
* JP
* US
* stock
* ETF
* REIT
* volatility
* liquidity

⸻

33. Independent Information Axes

「3000銘柄」「大量feature」「大量source」を成果とみなさない。

情報拡張は、

Market
+
Fundamental
+
Event
+
Macro
+
Market Structure
+
Cross-Sectional
+
Temporal State

のような独立information axisを増やす方向を優先する。

⸻

XII. TEMPORAL STATE

34. Temporal State Research

Research-only Challengerとして、

* recent returns
* path returns
* direction persistence
* return acceleration
* volatility transition
* volume pressure
* intraday-range change
* close-location change

等を研究可能にする。

strict PIT prefixだけを使用する。

⸻

35. State Transition

state transitionは、

prediction timeより前に観測されたtransitionだけで推定する。

future stateを過去stateの説明へ逆流させない。

⸻

XIII. REGIME / DRIFT

36. Regime

candidate:

* bull
* bear
* sideways
* volatility regime
* liquidity regime
* rates regime
* macro regime
* correlation regime
* sector rotation
* event regime
* shock regime

regime detector自体もPIT検証する。

⸻

37. Drift

monitor:

* feature drift
* target drift
* price distribution shift
* volatility shift
* liquidity shift
* correlation shift
* sector distribution
* source drift
* calibration drift
* error drift

drift時には、

recalibration
→ reroute
→ retrain
→ research
→ fallback

を検討する。

⸻

XIV. MODEL ECOLOGY

38. Baselines

current baseline families:

* Logistic Regression
* ExtraTrees
* HistGradientBoosting

これらを比較anchorとして維持する。

⸻

39. Challenger Models

候補:

* LightGBM
* XGBoost
* CatBoost
* appropriate time-series models
* cross-sectional models
* PatchTST
* ensemble models
* regime-aware models
* uncertainty-aware models
* selective models

高度なmodelを自動的にsuperiorとみなさない。

⸻

40. Model Roles

候補:

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

⸻

XV. ROUTING

41. Routing Hierarchy

基本concept:

asset class + regime
→ asset class
→ regime
→ global

specialist使用条件:

* minimum observations
* sufficient folds
* class coverage where applicable
* calibration
* recent stability

条件不足時は上位scopeへfallback。

⸻

42. Routing Score

research conventionとして、

selection_score

mean OOS LogLoss
+
0.25 × fold LogLoss std

を利用可能。

ただし最終判断では、

* sample size
* latest period
* worst fold
* calibration
* robustness
* complexity
* routing instability

を併記する。

⸻

XVI. CALIBRATION

43. Calibration Methods
candidate:

* none
* temperature
* sigmoid
* beta-style
* isotonic
* temporal calibrator

holdout tuningは禁止。

⸻

44. Calibration Metrics

* LogLoss
* Brier
* ECE
* calibration slope
* calibration intercept
* reliability
* temporal stability

を評価する。

⸻

XVII. UNCERTAINTY / PREDICTABILITY

45. Strict Separation

Probability
≠
Confidence

Confidence
≠
Predictability

Predictability
≠
Accuracy

⸻

46. Predictability

instrument-day単位で、

* model disagreement
* data completeness
* OOD
* regime ambiguity
* volatility
* liquidity
* event risk
* calibration instability
* cross-model entropy

等から研究する。

⸻

47. Uncertainty Decomposition

可能な限り、

* aleatoric
* epistemic
* data
* source
* temporal
* regime
* OOD
* disagreement
* event risk
* volatility uncertainty

を分離する。

⸻

48. Model Disagreement

保存候補:

* probability variance
* entropy gap
* KL divergence
* JS divergence
* rank disagreement
* expected-return disagreement
* quantile disagreement

disagreement spikeは追加調査、recompute、fallback、abstention候補。

⸻

XVIII. OOD / FAILURE RISK

49. OOD

候補:

* new listing
* delisting
* sparse history
* unseen regime
* extreme volatility
* liquidity shock
* unusual event
* source conflict
* feature distribution shift

OODをconfidenceへ直接潰さない。

⸻

50. Confidence-Risk Layer

wrong-direction riskを別モデルで推定しpriorへshrinkする研究が可能。

単独でdirection flipを行わない。

production導入には独立OOS / holdout Evidenceが必要。

⸻

XIX. INFORMATION ACQUISITION

51. Action Layer

候補:

PREDICT_NOW
ACQUIRE_MORE
WAIT
RECOMPUTE
FALLBACK
ABSTAIN

⸻

52. Information Value

追加情報の価値を、

Expected Information Value
×
Update Probability
×
Source Reliability

と、

Latency
+
Cost
+
Failure Risk

から評価する。

情報量の増加自体を成功としない。

⸻

XX. FORECAST LIFETIME

53. Prediction State

標準:

FRESH
AGING
STALE
UNKNOWN
SHOCKED
REQUIRES_RECALC
FALLBACK
ABSTAIN
INVALIDATED

major market move、new filing、earnings、corporate event、source revisionなどで更新する。

⸻

XXI. SELECTIVE PREDICTION

54. Selective Policy

100% coverageを目的としない。

predict
fallback
abstain
defer
acquire_more

を比較する。

metrics:

* coverage
* selective risk
* calibration
* utility
* stability
* false-abstention cost

⸻

XXII. QUANTILE / CONFORMAL

55. Quantile Evaluation

q10/q50/q90について、

* pinball loss
* empirical coverage
* interval width
* interval calibration
* tail violation

を評価する。

⸻

56. Conformal

Research candidates:

* split conformal
* adaptive conformal
* local conformal
* online conformal
* risk-controlled prediction
* quantile calibration

evaluation:

coverage
+
interval width
+
selective risk
+
temporal stability
+
regime robustness

⸻

XXIII. CROSS-SECTIONAL RANKING

57. Ranking Contract

Rankingはtime-series forecastとは別targetとして扱う。

評価候補:

* Spearman
* Kendall
* NDCG
* top-k hit
* bucket spread
* turnover-aware metrics

universeは同一prediction dateでPIT-safeに固定する。

future membershipをrankingへ入れない。

⸻

XXIV. ECONOMIC USEFULNESS

58. Statistical vs Economic Performance

statistical improvementとeconomic usefulnessを分離する。

必要に応じ、

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

prediction improvementだけでinvestment/economic superiorityを主張しない。

⸻

XXV. OOS / WFO

59. Evaluation Structure

TRAIN
→ VALIDATION
→ CHRONOLOGICAL OOS/WFO
→ ROBUSTNESS
→ FROZEN HOLDOUT

random splitをproduction evidenceとしない。

⸻

60. Candidate / Evaluation Separation

candidate selectionとfinal evaluationを完全に分離する。

fold tのselectionにfold t自身のresultを使用しない。

prequential selectionを優先する。

⸻

61. OOS Record

可能な限り各rowに、

* instrument
* prediction date
* target date
* cutoff
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
* predictability
* regime
* quality
* PIT
* actual
* error type

を保存する。

⸻

XXVI. METRICS

62. Direction

Primary:

LogLoss

Secondary:

* Accuracy
* Brier
* ECE
* class-wise metrics
* calibration slope
* calibration intercept

⸻

63. Return

候補:

* MAE
* RMSE
* median absolute error
* directional accuracy
* rank correlation
* tail error
* quantile loss

⸻

64. Price

Expected Close:

* MAE
* RMSE
* relative error
* tail error

⸻

65. Ranking

* Spearman
* Kendall
* NDCG
* top-k
* bucket spread
* turnover-aware usefulness

⸻

XXVII. STATISTICAL INTEGRITY

66. Comparison

必要に応じ、

* paired comparison
* time-block bootstrap
* instrument-cluster bootstrap
* date-block bootstrap
* confidence interval
* permutation
* forecast comparison
* multiple-comparison correction

を使用する。

serial correlation、cross-sectional dependence、market clusteringを考慮する。

⸻

XXVIII. FEATURE / MODEL ABLATION

67. Ablation

新機構では、

FULL
vs
WITHOUT_FEATURE
vs
WITHOUT_MODEL
vs
WITHOUT_ROUTER
vs
WITHOUT_SOURCE

等を比較する。

目的は「追加したら良かった」ではなく、

その機構そのものにincremental valueがあることを証明すること。

⸻

XXIX. ROBUSTNESS

68. Robustness Matrix

最低限、

* latest period
* recent regime
* JP
* US
* stock
* ETF
* REIT
* high volatility
* low volatility
* high liquidity
* low liquidity
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

を評価する。

⸻

XXX. ADVERSARIAL VALIDATION

69. Attack Tests

意図的に、

* future timestamp injection
* future price injection
* future filing injection
* future membership
* later revision
* corporate-action timing attack
* stale source
* missingness
* feature removal
* source removal
* time shift
* regime transition
* distribution shift

を攻撃する。

failした候補はproduction不可。

⸻

XXXI. FROZEN HOLDOUT

70. Holdout Firewall

Holdoutを、

* model tuning
* feature tuning
* source selection
* universe selection
* routing tuning
* threshold tuning
* calibration tuning
* research prioritization

へ使用しない。

⸻

71. Holdout Contamination

汚染が疑われる場合、

HOLDOUT = INVALID

として再freezeする。

⸻

XXXII. ADOPTION

72. Incumbent Comparison

CandidateとIncumbentを同一OOS observationsで比較する。

⸻

73. Reference Gate

参考benchmark:

relative primary OOS LogLoss improvement >= 3%

auxiliary improvement >= 1%

evaluation periods without worsening >= 70%

さらに、

* newest holdout no worsening
* calibration no material degradation
* PIT violations = 0
* robustness no major regression
* reproducibility
* operational safety
* sufficient sample

を要求する。

threshold単独でautomatic promotionしない。

⸻

XXXIII. PRODUCTION

74. Production Bundle

Productionはmodel fileではない。

Bundle:

* model
* feature schema
* target version
* calibration
* routing
* universe policy
* source policy
* PIT policy
* uncertainty policy
* output schema
* monitoring
* rollback target
* manifest
* hashes

⸻

75. Production Registry

保存:

* model_id
* version
* target_version
* feature_version
* calibration_version
* routing_version
* universe_version
* source snapshot
* OOS status
* robustness status
* holdout status
* release time
* rollback pointer

⸻

76. Runtime Verification

起動時:

registry
↔ metadata
↔ artifact
↔ feature schema

を検証。

さらに、

* classes
* probability shape
* finite values
* feature schema
* hashes
* target compatibility
* calibration compatibility

を確認する。

⸻

77. Runtime Candidate Isolation

Production Runtimeが最新Research Candidateを勝手に選択しない。

Production Registryの明示されたbundleだけを使用する。

⸻

XXXIV. FAIL-CLOSED / FALLBACK

78. Production Fail-Closed

以下ではforced predictionをしない。

* model invalid
* required data missing
* stale required source
* PIT unknown
* universe inconsistency
* identity mismatch
* feature contract failure
* invalid calibration
* corrupted artifact
* target mismatch
* market-calendar failure

⸻

79. Fallback Chain

基本:

Champion
→ Scoped Expert
→ Global Generalist
→ Baseline
→ Abstain

fallback reasonをprediction logへ保存する。

Fallback結果をChampionと同等品質とは表現しない。

⸻

XXXV. EXPERIENCE / RECONCILIATION

80. Canonical Experience Unit

canonical unit:

instrument-date-cutoff

同一銘柄の複数snapshot/revisionでexperience countを水増ししない。

⸻

81. Prediction Lifecycle

CREATED
→ PIT_CHECK
→ DATA_CHECK
→ FEATURE_CHECK
→ MODEL_CHECK
→ CALIBRATION_CHECK
→ PREDICTED
→ MONITORED
→ MATURED
→ RECONCILED

⸻

82. Reconciliation

prediction
→ actual
→ metric
→ error
→ state

へ変換する。

current predictionとhistorical experienceを混同しない。

⸻

XXXVI. FAILURE INTELLIGENCE

83. Failure Taxonomy

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

⸻

84. Failure Record

保存:

* failure_id
* instrument
* date
* cutoff
* component
* type
* severity
* prediction
* actual
* expected
* observed
* root cause
* counterfactual
* repair
* verification
* recurrence

⸻

85. Counterfactual Failure Analysis

failureについて、

* better data
* additional source
* earlier valid information
* alternate model
* alternate routing
* calibration
* fallback
* abstention

で救えたかを検証する。

measured factとhypothesisを明確に分ける。

⸻

86. High-Confidence Wrong

特に、

high confidence
+
wrong outcome

を高優先研究対象とする。

候補原因:

* calibration failure
* regime shift
* OOD
* source error
* stale data
* model blind spot

⸻

XXXVII. NEGATIVE KNOWLEDGE

87. Permanent Memory

保存:

* rejected model
* rejected feature
* rejected source
* rejected routing
* rejected calibration
* rejected universe policy
* rejected event feature
* failed PIT method
* robustness failure
* computational failure
* reason for rejection

失敗を次の研究の入力にする。

⸻

XXXVIII. RESEARCH ROUTER

88. Research Categories

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

89. Research Portfolio

配分:

* exploit
* adjacent
* frontier
* replication
* ablation
* adversarial
* recovery
* meta-research

既知手法の反復だけにresearchを集中させない。

⸻

XXXIX. EXTERNAL RESEARCH

90. Research Ingestion Contract

外部論文・GitHub・研究実装・AI-generated ideas等:

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

外部performance claimはlocal evidenceではない。

⸻

XL. EVIDENCE LEVEL

91. Evidence

E0 = idea
E1 = external claim
E2 = external implementation
E3 = local reproduction
E4 = local OOS
E5 = robustness
E6 = frozen holdout
E7 = production evidence

⸻

XLI. CROSS-PROJECT TRANSFER

92. Five-Repository Transfer Firewall

他project:

BTC
7-Sport
Soccer
Baseball
Stock

間の成功を直接移植しない。

Transfer:

DISCOVER
→ ABSTRACT_MECHANISM
→ COMPATIBILITY
→ ADAPT
→ LOCAL_PIT
→ LOCAL_OOS
→ LOCAL_ROBUSTNESS
→ LOCAL_HOLDOUT
→ SHADOW
→ PROMOTE

performance resultではなくmechanismを移植する。

⸻

XLII. PLUGIN / CONNECTOR INTELLIGENCE

93. Routing

利用可能なplugin / connectorは用途別に使う。

GitHub:
code/config/Actions/artifact/current state

Web/Search:
current public information/source discovery

Academic research:
papers/methodology

Structured data:
current structured datasets

Project/File sources:
local knowledge/specification/evidence

⸻

94. Plugin Evidence Firewall

plugin由来情報を、

DISCOVERY
→ SOURCE VERIFICATION
→ TIME VERIFICATION
→ PIT
→ COST
→ SECURITY
→ LOCAL REPRODUCTION
→ OOS
→ ROBUSTNESS
→ HOLDOUT

なしにproduction evidenceへ入れない。

⸻

95. Plugin Cost / Security

優先:

verified free
→ free quota
→ OSS/local
→ cache
→ lightweight
paid-only、billing-risk、unknown-cost、secret-dependent automationは自動利用しない。

secret/API key/tokenをcode、log、artifact、report、commitへ出さない。

⸻

XLIII. AUTOMATION

96. GitHub Actions

必要:

* checkpoint
* resume
* idempotency
* retry
* exponential backoff
* watchdog
* heartbeat
* stale-run detection
* bounded runtime
* deterministic writes
* artifact preservation
* concurrency control
* recovery
* rollback

長時間Researchを一発jobへ依存させない。

⸻

97. Multi-Layer Control Plane

制御系として、

Automation Supervisor
→ Actions Reliability Watchdog
→ Research / Failure Controllers
→ Heartbeat Recovery

の複数監視層を維持する。

ただしcontrol-plane automationはproduction/model/evidenceを勝手に変更しない。

⸻

98. Long-Running OOS Continuity

active chronological OOSは通常のcontrol-plane-only commitで不必要に中断しない。

ただし、

* runtime
* data
* feature
* target
* PIT
* scoring
* routing
* calibration
* selection
* candidate identity

へ影響する変更はevidence-affectingとし、fresh OOSを要求する。

⸻

99. Evidence Freshness

evidence freshnessは可能な限り、

evidence-affecting fingerprint

で判定する。

control-plane-only変更だけなら既存Evidenceを不必要に無効化しない。

evidence ancestryを確認できなければUNKNOWNとしてfail-closed。

⸻

100. Recovery

stale run、cancelled run、queue gap等を無条件で無限rerunしない。

必要:

* current SHA
* active run
* queue state
* checkpoint
* artifact
* evidence ancestry
* duplicate-run risk
* cooldown

を確認する。

⸻

XLIV. SINGLE WRITER

101. Critical State

single-writer semanticsを優先:

* production registry
* experiment registry
* source registry
* universe state
* promotion state
* rollback state
* experience ledger
* research status

parallel researchのmergeはdeterministicにする。

⸻

XLV. EFFICIENCY

102. Compute Optimization

順序:

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

XLVI. COMPLEXITY BUDGET

103. Complexity

毎回確認:

* feature count
* source count
* model count
* router complexity
* calibration layers
* workflow count
* dependencies
* latency
* maintenance burden
* failure surface

最適化対象:

Future Generalization Gain
/
Added Complexity

複雑化そのものを成果としない。

⸻

XLVII. FAILURE SURFACE

104. New Mechanism Audit

新機能追加時は、

* new dependency
* new failure mode
* new data assumption
* new fallback path
* new state transition
* new maintenance burden

を確認する。

⸻

XLVIII. POLICY REGRET

105. Decision Retrospective

後から、

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

filingを待つ価値、
cutoffを変更する価値、
source追加の価値、
specialist routingの価値、
abstentionの価値。

⸻

XLIX. COVERAGE / RESEARCH DEBT

106. Coverage Debt

Debtを、

* historical universe gap
* delisted instrument gap
* source gap
* PIT metadata gap
* filing timestamp gap
* event gap
* calendar gap
* feature gap
* regime gap
* OOS gap

へ分解する。

model complexityでcoverage debtを隠さない。

⸻

L. UNKNOWN FRONTIER

107. Unknown Frontier

高優先研究:

* high disagreement
* high-confidence wrong
* low predictability
* regime transition
* new listings
* delistings
* event shocks
* source conflict
* sparse history
* OOD
* extreme volatility
* unexplained ranking failure
* unexplained calibration drift

UNKNOWNを既知カテゴリへ強制しない。

⸻

LI. RESEARCH STOPPING

108. Stop Conditions

以下ではHOLD/STOP可能:

* repeated zero incremental value
* insufficient PIT
* insufficient sample
* unresolved source quality
* robustness failure
* duplicated mechanism
* excessive compute
* maintenance burden
* frontier saturation

停止理由をNegative Knowledgeへ保存する。

⸻

LII. PRODUCTION STATE MACHINE

109. Release Lifecycle

IDEA
→ CANDIDATE
→ PIT_VERIFIED
→ OOS_VERIFIED
→ ROBUSTNESS_VERIFIED
→ HOLDOUT_VERIFIED
→ PROMOTION_CANDIDATE
→ SHADOW
→ PRODUCTION
→ STABLE

failure:

REJECTED
ROLLED_BACK
RETIRED

⸻

LIII. PRODUCTION FAILURE STATE

110. Degradation

PRODUCTION
→ DEGRADING
→ INVESTIGATING
→ RECALIBRATION / RETRAIN
→ SHADOW
→ REPLACE / ROLLBACK

⸻

LIV. ONLINE / OFFLINE PARITY

111. Replay Parity

同じsnapshotで、

Historical Pipeline
vs
Production-like Pipeline

を比較し、

* inputs
* features
* missingness
* timestamps
* probability
* calibration
* routing
* state

が一致するか検証する。

⸻

LV. DETERMINISTIC REPLAY

112. Reproduction Contract

必要情報:

* Git SHA
* Environment
* Dependencies
* Config
* Seed
* Universe Snapshot
* Dataset Hash
* Source Snapshots
* Feature Schema
* Model Artifact
* Calibration
* Routing
* Target Definition

⸻

LVI. CHAOS TESTING

113. Failure Injection

故意に、

* source timeout
* API outage
* schema change
* duplicate
* corrupt timestamp
* stale source
* bad artifact
* cancelled workflow
* dependency failure
* resource exhaustion

を発生させる。

理想:

FULL
→ REDUCED
→ FALLBACK
→ SELECTIVE
→ ABSTAIN
→ RECOVERY

⸻

LVII. OUTPUT

114. Final Prediction Object

概念的に、

{
“instrument_id”: “…”,
“market”: “…”,
“product_family”: “…”,
“prediction_date”: “…”,
“target_date”: “…”,
“prediction_time”: “…”,
“prediction_cutoff”: “…”,
“universe_snapshot_id”: “…”,

“prob_up”: 0.62,
“prob_down”: 0.38,

“expected_return”: “…”,
“expected_close”: “…”,

“q10”: “…”,
“q50”: “…”,
“q90”: “…”,

“rank”: “…”,

“regime”: “…”,
“uncertainty”: {
“aleatoric”: “…”,
“epistemic”: “…”,
“data”: “…”,
“source”: “…”,
“temporal”: “…”,
“regime”: “…”,
“ood”: “…”,
“event”: “…”
},

“predictability”: “…”,
“model_disagreement”: “…”,
“data_quality”: “…”,
“pit_status”: “…”,

“model_version”: “…”,
“feature_version”: “…”,
“calibration_version”: “…”,
“routing_version”: “…”,

“prediction_state”: “…”,
“generation_status”: “…”,

“git_sha”: “…”,
“dataset_hash”: “…”,
“source_snapshot”: “…”,
“request_id”: “…”
}

⸻

LVIII. EXPLANATION GRAPH

115. Provenance

可能な限り、

Raw Data
→ Source
→ Feature
→ State
→ Regime
→ Model
→ Calibration
→ Prediction
→ Decision

を追跡可能にする。

⸻

LIX. SOURCE CONFLICT

116. Evidence Conflict

source間矛盾では単純平均しない。

評価:

* availability
* reliability
* independence
* freshness
* coverage
* historical error
* identity confidence

⸻

LX. MODEL BLIND SPOT

117. Diversity

model architectureだけでなく、

* information source diversity
* feature diversity
* representation diversity
* assumption diversity

を評価する。

多数のmodelが同じ情報だけを見ている場合、真のensemble diversityとはみなさない。

⸻

LXI. COMPLETION

118. Completion Definition

以下だけではcompletionではない。

* code exists
* CI green
* workflow completed
* artifact exists
* prediction generated

completion evidenceは、

tests
+
PIT audit
+
leakage/meta-leakage audit
+
survivorship/universe audit
+
calendar audit
+
chronological OOS/WFO
+
calibration
+
ablation
+
robustness
+
frozen holdout
+
artifact integrity
+
reproducibility
+
recovery
+
release gate
+
monitoring
+
rollback
+
report

である。

⸻

LXII. STATUS

119. Status Taxonomy

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

code exists ≠ adopted

green CI ≠ performance verified

artifact exists ≠ production evidence

⸻

LXIII. NO-FAKE-SUCCESS

120. Forbidden

禁止:

* fabricated metrics
* silent exception
* missing→zero
* skipped test→passed
* failed workflow→success
* unknown PIT→valid
* incomplete universe→complete
* invalid holdout→clean
* failed recovery→recovered
* stale evidence→current evidence
* external result→local evidence
* research output→production output

⸻

LXIV. PERFORMANCE REPORT

121. Change Report

可能な限り、

* Current Champion
* Prior Champion
* Candidate
* ΔLogLoss
* ΔBrier
* ΔAccuracy
* ΔECE
* return metric Δ
* price metric Δ
* ranking metric Δ
* quantile metric Δ
* latest holdout Δ
* robustness Δ
* sample size
* fold count
* confidence interval
* PIT status
* decision
* production state

を表示する。

⸻

LXV. SELF-EVOLUTION

122. Continuous Improvement

detect:

* recurring failure
* obsolete rule
* source contradiction
* stale source
* inefficient workflow
* inefficient computation
* validated new method
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

過去metric、failure、holdoutを改変して改善を演出しない。

⸻

LXVI. RESEARCH PRIORITY

123. ResearchNext

概念的に、

ResearchNext

argmax(
Expected Future Value

Complexity

Operational Risk
)

。

研究priorityは単純な新規性やbacktest gainだけで決めない。

⸻

LXVII. FINAL LOOP

124. Permanent Autonomous Loop

MONITOR
→ DETECT
→ TRIAGE
→ UNDERSTAND
→ RESEARCH
→ HYPOTHESIS
→ IMPLEMENT
→ TEST
→ PIT
→ OOS/WFO
→ CALIBRATION
→ ROBUSTNESS
→ FROZEN HOLDOUT
→ SHADOW
→ ADOPT/HOLD/REJECT
→ RELEASE
→ PRODUCTION
→ RECONCILE
→ FAILURE ANALYSIS
→ MEMORY
→ NEXT RESEARCH

⸻

LXVIII. FINAL DEFINITION

Stock-Daily-Prediction-3000とは、

「未来営業日の市場について、prediction cutoff時点で本当に利用可能だった情報だけを用い、時点依存Universe・market calendar・survivorship・corporate actions・source provenance・PIT・feature lineage・market state・temporal state・regime・model ecology・dynamic routing・ensemble・calibration・uncertainty・predictability・OOD・information acquisition・selective prediction・return/price/quantile/ranking forecast・chronological OOS/WFO・robustness・frozen holdout・production governance・failure memory・recovery・自己改善を統合し、未知の将来ケースへの一般化性能を継続的に改善するAdaptive Stock Prediction Intelligence」

として定義する。

最終的な予測対象は単なる、

P(UP)

ではない。

Prediction Intelligence

Prediction
+
Return
+
Price
+
Quantile
+
Ranking
+
Calibration
+
Uncertainty
+
Predictability
+
OOD
+
Information Acquisition
+
Decision
+
Failure Learning
+
Self Improvement
+
Operational Reliability

である。

⸻

FINAL LAWS

NO PIT PROOF, NO HISTORICAL TRUST.

NO SURVIVORSHIP INTEGRITY, NO HISTORICAL UNIVERSE TRUST.

NO EVIDENCE, NO CLAIM.

FUTURE GENERALIZATION > HISTORICAL FIT.

CALIBRATION > RAW CONFIDENCE.

ROBUSTNESS > SINGLE-FOLD IMPROVEMENT.

INFORMATION VALUE > FEATURE COUNT.

INDEPENDENT EVIDENCE > SOURCE COUNT.

MISSING ≠ ZERO.

RETRIEVAL ≠ AVAILABILITY.

PROBABILITY ≠ CONFIDENCE.

CONFIDENCE ≠ PREDICTABILITY.

OOD ≠ LOW CONFIDENCE.

OOS ≠ HOLDOUT.

GREEN CI ≠ PERFORMANCE VERIFICATION.

ARTIFACT ≠ PRODUCTION VALIDATION.

COMPLEXITY ≠ SUPERIORITY.

SAFE ABSTENTION > FORCED PREDICTION.

FAILURE ≠ USELESSNESS.

FAILURE MEMORY > REPEATED FAILURE.

PRODUCTION IS A BUNDLE.

RUNTIME MUST NOT INVENT PRODUCTION.

CROSS-PROJECT SUCCESS IS NOT LOCAL EVIDENCE.

HISTORICAL TRUTH MUST REMAIN IMMUTABLE.

UNKNOWN MUST REMAIN UNKNOWN.

NO SAFE RECOVERY, NO AUTONOMOUS OPERATION.

NO CONTINUOUS MONITORING, NO CONTINUOUS SELF-IMPROVEMENT.
