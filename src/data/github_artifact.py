from __future__ import annotations

import os
from pathlib import Path
import subprocess


def download_workflow_artifact(
    repo: str,
    token: str,
    artifact: dict,
    destination: Path,
    *,
    timeout_seconds: int = 120,
) -> Path:
    """Download one GitHub Actions artifact using the GitHub CLI.

    gh run download resolves the short-lived GitHub artifact redirect itself.
    The token is kept in the process environment and is never placed in argv.
    """
    workflow_run = artifact.get("workflow_run") or {}
    run_id = workflow_run.get("id")
    name = str(artifact.get("name", ""))
    if not run_id:
        raise ValueError("artifact workflow_run.id is missing")
    if not name:
        raise ValueError("artifact name is missing")

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["GH_TOKEN"] = token
    env["GITHUB_TOKEN"] = token
    env["GH_PROMPT_DISABLED"] = "1"

    command = [
        "gh",
        "run",
        "download",
        str(run_id),
        "--repo",
        repo,
        "--name",
        name,
        "--dir",
        str(destination),
    ]
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"gh run download failed for artifact {name} run {run_id}: "
            f"{detail[:500]}"
        )
    return destination


def validate_extracted_tree(root: Path) -> None:
    """Reject symlinks escaping the temporary artifact root."""
    root = Path(root).resolve()
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current).resolve()
        if root not in (current_path, *current_path.parents):
            raise RuntimeError("artifact extraction escaped restore root")
        for entry in [*dirnames, *filenames]:
            path = Path(current) / entry
            if path.is_symlink():
                resolved = path.resolve()
                if resolved != root and root not in resolved.parents:
                    raise RuntimeError(
                        f"artifact symlink escapes restore root: {path.name}"
                    )
