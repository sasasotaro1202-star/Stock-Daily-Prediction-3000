
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