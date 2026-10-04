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
