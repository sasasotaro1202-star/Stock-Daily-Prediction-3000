# Model routing

The production classifier is selected only from chronological walk-forward OOS evidence.

Routing precedence:

1. asset class + market regime, when at least two valid OOS folds exist;
2. asset class, when at least three valid OOS folds exist;
3. market regime, when at least three valid OOS folds exist;
4. global OOS-selected model.

The stability-aware selection score is mean LogLoss plus 0.25 times fold LogLoss standard deviation. The frozen holdout is never used to select the model.

Production fitting is hierarchical. A scoped model is trained only when the scoped dataset has enough rows and both target classes; otherwise the fit falls back to the global training population.

Available product-model families are jp_stock, jp_etf, jp_reit, us_stock, and us_etf. Prediction output records the selected model, training scope, and routing reason for every row.

All production snapshots require available_at <= prediction_time. Japan and U.S. latest sessions are resolved independently because their market clocks differ.

Before a new production freeze, the selected global classifier must have paired chronological OOS evidence against the strongest comparator: at least 5 common folds, a Bonferroni-adjusted two-sided 95% confidence bound, and at least 3% relative LogLoss improvement. If the confidence bound does not clear the required effect, model freezing is DEFERRED rather than silently selecting the top point estimate.


## Temporal confidence-risk challenger

The research pipeline also evaluates a research-only temporal confidence-risk layer. It predicts the probability that the directional decision is wrong from prior chronological OOS predictions, model disagreement, and prediction-time market state. High-risk cases are only shrunk toward the prior directional base rate; the layer never flips the direction and is never imported by the production predictor.

Acceptance remains evidence-driven: the challenger is considered research-positive only when it clears the configured relative OOS LogLoss threshold, positive-fold stability threshold, bootstrap stability threshold, and calibration/high-risk-case safeguards. No confidence-risk result can freeze or replace the production artifact by itself.
