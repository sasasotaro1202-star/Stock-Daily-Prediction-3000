from __future__ import annotations
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

def make_return_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=250,learning_rate=0.04,max_leaf_nodes=31,
            l2_regularization=1.0,loss="squared_error",random_state=42,
        ),
    )
