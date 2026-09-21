from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

class PlattCalibrator:
    def __init__(self):
        self.model=LogisticRegression(C=1e6,solver="lbfgs")
        self.fitted=False
        self.constant=0.5
    @staticmethod
    def _logit(p):
        p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
        return np.log(p/(1-p))
    def fit(self,p_calibration,y_calibration):
        x=self._logit(p_calibration).reshape(-1,1); y=np.asarray(y_calibration,dtype=int)
        if len(np.unique(y))<2:
            self.constant=float(y.mean()) if len(y) else 0.5
            return self
        self.model.fit(x,y); self.fitted=True; return self
    def predict(self,p):
        if not self.fitted:return np.full(len(np.asarray(p)),self.constant,dtype=float)
        return self.model.predict_proba(self._logit(p).reshape(-1,1))[:,1]
