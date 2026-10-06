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