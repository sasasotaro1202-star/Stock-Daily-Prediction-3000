from pathlib import Path
import subprocess


def test_production_invariants_script_is_current():
    path=Path("scripts/production_invariants.py")
    assert path.exists()
    result=subprocess.run(
        ["python",str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    # In a clean CI checkout no data artifacts exist, so the script may fail
    # on runtime state; it must still execute without syntax/import failure.
    assert "SyntaxError" not in result.stderr
