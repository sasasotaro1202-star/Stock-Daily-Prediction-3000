from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
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
        if not self.fitted:
            return np.full(len(np.asarray(p)), self.constant, dtype=float)
        return self.model.predict_proba(
            self._logit(p).reshape(-1, 1)
        )[:, 1]


class BetaCalibrator:
    """Beta calibration using the log p / log(1-p) basis."""

    method = "beta"

    def __init__(self):
        self.model = LogisticRegression(
            C=10.0,
            solver="lbfgs",
            max_iter=1000,
        )
        self.fitted = False
        self.constant = 0.5

    @staticmethod
    def _features(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.column_stack([np.log(p), np.log1p(-p)])

    def fit(self, p_calibration, y_calibration):
        x = self._features(p_calibration)
        y = np.asarray(y_calibration, dtype=int)
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean()) if len(y) else 0.5
            return self
        self.model.fit(x, y)
        self.fitted = True
        return self

    def predict(self, p):
        if not self.fitted:
            return np.full(len(np.asarray(p)), self.constant, dtype=float)
        return self.model.predict_proba(self._features(p))[:, 1]


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
        x = np.asarray(p_calibration, dtype=float)
        y = np.asarray(y_calibration, dtype=int)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean()) if len(y) else 0.5
            return self
        if len(x) < 40:
            # Small calibration sets are too easy to overfit with isotonic
            # regression; use the robust Platt fallback deterministically.
            fallback = PlattCalibrator().fit(x, y)
            self._fallback = fallback
            return self
        self.model.fit(np.clip(x, 1e-6, 1 - 1e-6), y)
        self.fitted = True
        return self

    def predict(self, p):
        if getattr(self, "_fallback", None) is not None:
            return self._fallback.predict(p)
        if not self.fitted:
            return np.full(len(np.asarray(p)), self.constant, dtype=float)
        return np.asarray(
            self.model.predict(
                np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
            ),
            dtype=float,
        )


class TemperatureCalibrator:
    """One-parameter logit-temperature calibration selected by chronological OOS."""

    method = "temperature"

    def __init__(self, min_temperature: float = 0.25, max_temperature: float = 4.0):
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        if not 0.0 < self.min_temperature < self.max_temperature:
            raise ValueError("temperature bounds must satisfy 0 < min < max")
        self.temperature = 1.0
        self.fitted = False
        self.constant = 0.5

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    @staticmethod
    def _sigmoid(x):
        x = np.clip(np.asarray(x, dtype=float), -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, p_calibration, y_calibration):
        p = np.asarray(p_calibration, dtype=float)
        y = np.asarray(y_calibration, dtype=int)
        finite = np.isfinite(p) & np.isfinite(y)
        p = np.clip(p[finite], 1e-6, 1 - 1e-6)
        y = y[finite]
        if len(y) == 0 or len(np.unique(y)) < 2:
            self.constant = float(y.mean()) if len(y) else 0.5
            return self
        logits = self._logit(p)

        def objective(temperature: float) -> float:
            scaled = np.clip(logits / float(temperature), -50.0, 50.0)
            probs = self._sigmoid(scaled)
            loss = -(
                y * np.log(np.clip(probs, 1e-12, 1.0))
                + (1 - y) * np.log(np.clip(1.0 - probs, 1e-12, 1.0))
            )
            return float(np.mean(loss))

        try:
            result = minimize_scalar(
                objective,
                bounds=(self.min_temperature, self.max_temperature),
                method="bounded",
                options={"xatol": 1e-3, "maxiter": 80},
            )
            candidate = float(result.x)
            if not np.isfinite(candidate) or candidate <= 0.0:
                candidate = 1.0
            self.temperature = float(
                np.clip(candidate, self.min_temperature, self.max_temperature)
            )
            self.fitted = True
        except (TypeError, ValueError, FloatingPointError):
            self.temperature = 1.0
            self.fitted = True
        return self

    def predict(self, p):
        if not self.fitted:
            return np.full(len(np.asarray(p)), self.constant, dtype=float)
        return np.clip(
            self._sigmoid(self._logit(p) / self.temperature),
            1e-5,
            1 - 1e-5,
        )


def make_calibrator(method: str):
    normalized = str(method).strip().lower()
    factories = {
        "platt": PlattCalibrator,
        "beta": BetaCalibrator,
        "isotonic": IsotonicCalibrator,
        "temperature": TemperatureCalibrator,
    }
    try:
        return factories[normalized]()
    except KeyError as exc:
        raise ValueError(
            f"unsupported calibration method: {normalized}"
        ) from exc


CALIBRATION_METHODS = ("platt", "beta", "isotonic", "temperature")
