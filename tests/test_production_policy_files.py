from pathlib import Path

def test_production_policy_files_exist():
    required=[
        ".github/workflows/market-cycle.yml",
        ".github/workflows/us-close-prediction.yml",
        ".github/workflows/bounded-production-recovery.yml",
        ".github/workflows/prediction-monitoring.yml",
        "scripts/production_state_compatibility.py",
    ]
    for path in required:
        assert Path(path).exists()
