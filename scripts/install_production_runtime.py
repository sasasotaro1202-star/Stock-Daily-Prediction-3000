from __future__ import annotations

import json
import subprocess
from pathlib import Path

META=Path("data/research/production_model_artifact.meta.json")


def main():
    if not META.exists():
        raise SystemExit("DEFERRED: production artifact metadata missing")
    meta=json.loads(META.read_text(encoding="utf-8"))
    versions=meta.get("runtime_dependency_versions") or {}
    required=(
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "pyarrow",
        "yfinance",
        "PyYAML",
        "curl_cffi",
    )
    specs=[]
    for name in required:
        version=versions.get(name)
        if not version:
            raise SystemExit(f"DEFERRED: runtime version missing: {name}")
        specs.append(f"{name}=={version}")
    lightgbm_version=versions.get("lightgbm")
    required_models=set(meta.get("required_classifiers") or [])
    if "lightgbm" in required_models:
        if not lightgbm_version:
            raise SystemExit("DEFERRED: LightGBM runtime version missing")
        specs.append(f"lightgbm=={lightgbm_version}")
    subprocess.check_call(["python","-m","pip","install","--disable-pip-version-check",*specs])
    print("production-runtime-restored:", ", ".join(specs))


if __name__=="__main__":
    main()
