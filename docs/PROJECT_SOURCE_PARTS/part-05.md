
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
* feature deletion
* time shift
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