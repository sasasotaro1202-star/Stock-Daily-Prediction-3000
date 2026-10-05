=== COPY START ===

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