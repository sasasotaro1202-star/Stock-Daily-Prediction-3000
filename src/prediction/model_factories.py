from __future__ import annotations

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

try:
    from lightgbm import LGBMClassifier
except ImportError:  # optional research dependency
    LGBMClassifier = None



class SoftBlendClassifier:
    """Simple probability blend of two causal tree-based classifiers."""

    def __init__(self, left_model, right_model, left_weight: float = 0.5):
        self.left_model = left_model
        self.right_model = right_model
        self.left_weight = float(left_weight)
        if not 0.0 <= self.left_weight <= 1.0:
            raise ValueError("left_weight must be between 0 and 1")
        self.right_weight = 1.0 - self.left_weight
        self._fitted_left_model = None
        self._fitted_right_model = None
        self.classes_ = None

    @staticmethod
    def _fit_one(model, X, y, sample_weight=None):
        if sample_weight is None:
            model.fit(X, y)
            return model
        if hasattr(model, "steps"):
            final_step = model.steps[-1][0]
            model.fit(X, y, **{f"{final_step}__sample_weight": sample_weight})
        else:
            model.fit(X, y, sample_weight=sample_weight)
        return model

    def fit(self, X, y, sample_weight=None):
        self._fitted_left_model = self._fit_one(
            clone(self.left_model), X, y, sample_weight
        )
        self._fitted_right_model = self._fit_one(
            clone(self.right_model), X, y, sample_weight
        )
        self.classes_ = getattr(self._fitted_left_model, "classes_", None)
        if self.classes_ is None:
            self.classes_ = getattr(self._fitted_right_model, "classes_", None)
        return self

    def predict_proba(self, X):
        if self._fitted_left_model is None or self._fitted_right_model is None:
            raise RuntimeError("blend classifier is not fitted")
        left = self._fitted_left_model.predict_proba(X)
        right = self._fitted_right_model.predict_proba(X)
        return self.left_weight * left + self.right_weight * right

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def models():
    out = {
        "logistic": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            LogisticRegression(max_iter=1000, C=0.5),
        ),
        "extra_trees": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        "hgb": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.04,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
        "hgb_conservative": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.02,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=3.0,
                random_state=42,
            ),
        ),
        "hgb_conservative_recent": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.02,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=3.0,
                random_state=42,
            ),
        ),
    }
    if LGBMClassifier is not None:
        def make_lightgbm():
            return make_pipeline(
                SimpleImputer(strategy="median"),
                LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.03,
                    num_leaves=31,
                    min_child_samples=50,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            )

        def make_lightgbm_regularized():
            return make_pipeline(
                SimpleImputer(strategy="median"),
                LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.02,
                    num_leaves=15,
                    min_child_samples=80,
                    subsample=0.9,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=2.0,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            )

        def make_lightgbm_conservative():
            return make_pipeline(
                SimpleImputer(strategy="median"),
                LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.02,
                    num_leaves=7,
                    min_child_samples=120,
                    subsample=0.85,
                    subsample_freq=1,
                    colsample_bytree=0.75,
                    reg_alpha=0.2,
                    reg_lambda=4.0,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            )

        out["lightgbm"] = make_lightgbm
        out["lightgbm_recent"] = make_lightgbm
        out["lightgbm_regularized"] = make_lightgbm_regularized
        out["lightgbm_regularized_recent"] = make_lightgbm_regularized
        out["lightgbm_conservative"] = make_lightgbm_conservative
        out["lightgbm_conservative_recent"] = make_lightgbm_conservative

        def make_hgb_lgbm_blend_recent():
            return SoftBlendClassifier(
                out["hgb"](),
                out["lightgbm_regularized"](),
                left_weight=0.5,
            )

        out["blend_hgb_lgbm_regularized_recent"] = make_hgb_lgbm_blend_recent
    return out
