# Cross-Project Transfer Ledger

This document records reusable engineering patterns transferred into Stock-Daily-Prediction-3000. It is a research/operations record and does not authorize production promotion.

## Reference snapshots (verified 2026-09-28)

- Stock-Daily-Prediction-3000 recovery base: 419dad9ffeed85ebbbfa054b8ee0b8c0ebb14b0f
- Stock-Daily-Prediction-3000 regressed main head observed: b64792ec98e4d91a89fdb27182be6439d428cfb9
- BTC-Prediction-Research: c7bf9605d4b3e0d5143387fd67e887abac5aeaff
- 7-Sport-Prediction-Research: ca01f9345b88f81ba009b98514b08cf7ef0f672c
- Soccer-Prediction-Research: 6209fd98e1f50eb5dc1de9495efe3ae1f17ee325
- Baseball-Prediction-System: a3dc89b4fb06f511230d85c8bc4e3949360c4d32

## Transfers applied / retained

### Reliability and recovery
- Dedicated lightweight watchdogs remain outside heavy research jobs.
- Watchdogs compare active work against the current main SHA and exclude their own run ID.
- Recovery is bounded and failure-aware; three recent failures trigger a cooldown instead of blind dispatch loops.
- Long research jobs use checkpoint/final evidence reconciliation rather than treating configuration as proof of successful completion.

### Evidence integrity
- 24H research closeout remains research_only=true and production_changed=false.
- Final closeout requires required evidence artifacts and fails closed when the reconciliation is incomplete.
- Research, verification, adoption, and production remain distinct states.

### PIT / OOS
- PIT/OOS remains an independent audit stream.
- Unknown temporal provenance is not promoted to verified evidence.
- Cross-project mechanisms are revalidated against Stock-specific data, timing, and production constraints.

## Cross-project sources

BTC contributes watchdog/circuit-breaker and strict PIT cadence patterns.
7-Sport contributes checkpointed staged recovery and evidence reconciliation.
Soccer contributes immutable run manifests, scope/data discovery separation, and recovery isolation.
Baseball demonstrates an explicit transfer ledger so imported mechanisms remain auditable and are not mistaken for target-domain performance evidence.

## Non-transfer rule

No metric or performance result from another repository is treated as Stock-Daily-Prediction-3000 evidence. Only mechanisms, tests, architecture patterns, and failure-handling procedures are imported for independent revalidation.
