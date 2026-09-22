from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

try:
    from lightgbm import LGBMClassifier
except ImportError:  # optional research dependency
    LGBMClassifier = None


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

        out["lightgbm"] = make_lightgbm
        out["lightgbm_recent"] = make_lightgbm
    return out



def return_models():
    from sklearn.ensemble import HistGradientBoostingRegressor

    out = {
        "mean": lambda: HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            loss="squared_error",
            random_state=42,
        )
    }
    if LGBMClassifier is not None:
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            LGBMRegressor = None
        if LGBMRegressor is not None:
            def make_lgbm_return():
                return make_pipeline(
                    SimpleImputer(strategy="median"),
                    LGBMRegressor(
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
            out["lightgbm_return"] = make_lgbm_return
            out["lightgbm_return_recent"] = make_lgbm_return
    return out
