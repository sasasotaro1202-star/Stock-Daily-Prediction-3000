# Research automation policy

This project uses bounded, resumable stages rather than one long-running job.
Stages are safe to rerun and must preserve artifacts on failure.

Required production gates:
1. Universe integrity and delisted/survivorship-bias checks.
2. Source health and provenance.
3. PIT / available_at audit.
4. Chronological walk-forward OOS.
5. Model-selection leakage audit.
6. Probability calibration on data independent of model selection.
7. Frozen holdout evaluated once after the research configuration is frozen.
8. Independent release gate.
9. Reproducibility manifest.
10. Only then may a production prediction artifact be marked approved.

Free-first policy:
- Prefer official/public/free sources.
- Paid credentials are optional and must never be required for the baseline pipeline.
- Missing or stale critical data causes DEFERRED, not silent substitution or fabricated values.

Efficiency order:
cache -> incremental -> duplicate reduction -> vectorization -> parallelization -> retraining optimization -> algorithm optimization.

Model routing:
- Regime routing is a hypothesis generator, not proof of superiority.
- Router rules and ensemble weights must be selected using training/OOS data only.
- Frozen holdout results must never feed model selection.

7. Statistical model-selection evidence.
A new global production model is not frozen from the lowest point-estimate LogLoss alone. The selector records fold identity and compares the selected model with the strongest comparator on common chronological OOS folds. Freezing requires >=5 common folds, >=3% relative LogLoss improvement, and a Bonferroni-adjusted 95% paired confidence bound whose lower endpoint clears the required effect size. Insufficient evidence causes DEFERRED; frozen holdout remains excluded.
