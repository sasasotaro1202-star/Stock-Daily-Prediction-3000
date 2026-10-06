from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

def fit_direction_models(X:np.ndarray,y:np.ndarray):
    return {
        "logistic":LogisticRegression(max_iter=1000,class_weight="balanced").fit(X,y),
        "global_hgb":HistGradientBoostingClassifier(
            max_iter=150,learning_rate=0.05,max_leaf_nodes=15,l2_regularization=1.0
        ).fit(X,y),
    }
