from __future__ import annotations
import numpy as np
from src.prediction.baseline import fit_direction_models

def main()->None:
    rng=np.random.default_rng(42); X=rng.normal(size=(600,8))
    y=(X[:,0]+0.2*X[:,1]>0).astype(int)
    models=fit_direction_models(X[:400],y[:400])
    for name,model in models.items():
        print(f"{name}: smoke_mean_probability={model.predict_proba(X[400:])[:,1].mean():.6f}")

if __name__=="__main__": main()
