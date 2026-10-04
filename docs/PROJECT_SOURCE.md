# Stock-Daily-Prediction-3000 — Project Source

Canonical Source is preserved losslessly in seven ordered parts because the source exceeds the single-file transfer limit used by the connected GitHub file API.

## Canonical source parts
1. [part-01](./PROJECT_SOURCE_PARTS/part-01.md) — lines 1–350
2. [part-02](./PROJECT_SOURCE_PARTS/part-02.md) — lines 351–700
3. [part-03](./PROJECT_SOURCE_PARTS/part-03.md) — lines 701–1050
4. [part-04](./PROJECT_SOURCE_PARTS/part-04.md) — lines 1051–1400
5. [part-05](./PROJECT_SOURCE_PARTS/part-05.md) — lines 1401–1750
6. [part-06](./PROJECT_SOURCE_PARTS/part-06.md) — lines 1751–2100
7. [part-07](./PROJECT_SOURCE_PARTS/part-07.md) — lines 2101–2262

## 2026-10-03 repository alignment
Audit baseline observed on 2026-10-03: `045507049f7f472a06db25f0b1da0b6ba179fcc0` (historical audit reference only; not a live HEAD pin).
Recent observed work includes watchdog stale-run prioritization, post-cutoff retrieval guards, and prequential-fold stability test coverage. The README states the production universe is dynamically derived from PayPay Securities official Japan/US lists rather than a fixed 3,000-symbol population.

Current GitHub code/config/tests/workflows/actions/artifacts/registries/measurements remain authoritative. This index and the source parts do not rewrite historical metrics, holdout evidence, failure history, or prior decisions. Workflow green/artifact existence is execution evidence only, not automatic proof of performance verification or production adoption.

Future changes must preserve PIT, survivorship integrity, JP/US market-calendar separation, explicit data states, chronological OOS/WFO, calibration, robustness, frozen-holdout firewall, artifact integrity, reproducibility, fail-closed behavior, recovery, rollback, and evidence/state taxonomy.

## Cross-project governance alignment — 2026-10-03

The five-repository research set is:
- Baseball-Prediction-System
- BTC-Prediction-Research
- 7-Sport-Prediction-Research
- Soccer-Prediction-Research
- Stock-Daily-Prediction-3000

Cross-project transfer is mechanism-level only: DISCOVER → ABSTRACT_MECHANISM → COMPATIBILITY → ADAPT → LOCAL_PIT → LOCAL_OOS/WFO → ROBUSTNESS → LOCAL_FROZEN_HOLDOUT → SHADOW → PROMOTE.


A green workflow, artifact existence, model-file existence or external performance claim is not performance verification. Failures/cancellations/skips remain failures/cancellations/skips unless independently rerun and verified. Historical results and holdouts are not rewritten. Cost-unknown, billing-risk or paid-only sources remain HOLD/UNCONFIRMED.
## Recent PIT hardening
`src/research/learned_case_risk_oos.py` resolves an explicit `prediction_cutoff` when present and remains backward-compatible with legacy `prediction_time` as its cutoff alias; when both are present, generation time must represent the same instant as the declared cutoff. `available_at` is the causal PIT boundary and must not be later than the resolved cutoff. `retrieved_at` is retrieval metadata and may be later than the prediction cutoff; it must not precede `available_at`, and when `published_at` is present it must not precede publication. Dedicated regression tests cover explicit-cutoff leakage, valid late retrieval, and invalid ordering.
## Recent PIT defense-in-depth hardening — 2026-10-03 (continued)
`src/research/learned_case_risk_oos.py` now additionally honors optional case-feature provenance fields when supplied: `feature_pit_status` must be `PASS`; `feature_snapshot_cutoff` and `feature_max_available_at` must parse successfully and must not be later than the resolved prediction cutoff. This supplements the row-level `available_at` / `published_at` / `retrieved_at` checks without inventing absent metadata.

Regression coverage was added to `tests/test_learned_case_risk_oos.py` for a future feature snapshot and non-PIT feature lineage. The existing `Research validation` workflow already executes this test module, so these guards are inside the repository's normal research validation lane.

## Recent operational hardening — 2026-10-04
Research status heartbeat treats requested/queued/pending/waiting/in_progress as active transient states. The status workflow also subscribes directly to the `requested` workflow_run event, so newly requested research runs become visible immediately. When the Research validation job is not yet visible through the Actions API, status persistence retries boundedly for pending/waiting job-creation races rather than reporting a false lookup failure. This is operational evidence only and does not alter prediction, OOS, calibration, holdout, or promotion state.


## 2026-10-04 repository alignment
Current GitHub code/config/tests/workflows/actions/artifacts/registries/measurements remain authoritative. The release gate requires the generated research PIT contract audit to exist, to contain no FAIL state, and to contain a PASS prediction-ledger audit before production approval can be emitted. This requirement is a safety gate only; it does not promote research artifacts automatically.


## 2026-10-04 chronological ranking evidence hardening
The research runner no longer reuses the globally selected calibration method when evaluating ranking-weight candidates on the same chronological OOS folds. Ranking-fold calibration now comes from the prior-fold temporal calibration router, with the current fold scored only after that pre-test selection. The final global calibration method remains a separate configuration-selection artifact and is still subject to the frozen release evidence gate. This is research/evaluation hardening only; no production model, frozen holdout, or live routing state is mutated by the change.


## 2026-10-04 challenger ecology expansion
The configured research candidate set already included XGBoost and CatBoost, but the model factory previously exposed neither and the research extra dependencies did not install them. The research environment now installs both packages and the factory exposes conservative deterministic XGBoost/CatBoost challengers. They remain optional research candidates: chronological OOS/WFO selection, PIT checks, calibration, robustness and frozen-holdout gates remain unchanged, and factory availability alone never promotes a challenger.
