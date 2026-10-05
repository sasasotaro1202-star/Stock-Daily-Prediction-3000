
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

を実装・検証する。

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