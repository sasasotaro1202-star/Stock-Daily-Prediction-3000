from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def models():
    return {
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
    }
