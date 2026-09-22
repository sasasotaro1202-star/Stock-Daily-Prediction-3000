from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    method = "platt"

    def __init__(self):
        self.model = LogisticRegression(C=1e6, solver="lbfgs")
        self.fitted = False
        self.constant = 0.5

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, p_calibration, y_calibration):
        x = self._logit(p_calibration).reshape(-1, 1)
        y = np.asarray(y_calibration, dtype=int)
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean()) if len(y) else 0.5
            return self
        self.model.fit(x, y)
        self.fitted = True
        return self

    def predict(self, p):
        values = np.asarray(p, dtype=float)
        if not self.fitted:
            return np.full(len(values), self.constant, dtype=float)
        return self.model.predict_proba(
            self._logit(values).reshape(-1, 1)
        )[:, 1]


class IsotonicCalibrator:
    method = "isotonic"

    def __init__(self):
        self.model = IsotonicRegression(
            y_min=1e-5,
            y_max=1 - 1e-5,
            out_of_bounds="clip",
        )
        self.fitted = False
        self.constant = 0.5

    def fit(self, p_calibration, y_calibration):
        x = np.clip(np.asarray(p_calibration, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(y_calibration, dtype=float)
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean()) if len(y) else 0.5
            return self
        self.model.fit(x, y)
        self.fitted = True
        return self

    def predict(self, p):
        values = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        if not self.fitted:
            return np.full(len(values), self.constant, dtype=float)
        return np.asarray(self.model.predict(values), dtype=float)


def make_calibrator(method: str):
    normalized = str(method).strip().lower()
    if normalized == "platt":
        return PlattCalibrator()
    if normalized == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"unsupported calibration method: {method}")
