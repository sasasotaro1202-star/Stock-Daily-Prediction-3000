from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from src.data.github_artifact import download_workflow_artifact, validate_extracted_tree


def _download(url: str, token: str, timeout: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _find_frozen_lock(root: Path) -> Path:
    matches = [
        p for p in root.rglob("config/frozen_holdout.json")
        if p.is_file()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one frozen holdout lock in artifact, found {len(matches)}"
        )
    return matches[0].resolve()

def _find_research_root(root: Path) -> Path:
    matches = [p for p in root.rglob("production_model_artifact.pkl") if p.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one production_model_artifact.pkl in artifact, found {len(matches)}"
        )
    return matches[0].parent.resolve()


def _has_approved_production_state(
    src: Path,
    *,
    expected_holdout_generation: object | None = None,
    expected_research_fingerprint: str | None = None,
    expected_full_fingerprint: str | None = None,
) -> bool:
    if not src.exists():
        return False
    artifact = src / "production_model_artifact.pkl"
    if not artifact.exists():
        matches = [
            p for p in src.rglob("production_model_artifact.pkl")
            if p.is_file()
        ]
        if len(matches) != 1:
            return False
        src = matches[0].parent
        artifact = src / "production_model_artifact.pkl"
    metadata = src / "production_model_artifact.meta.json"
    gate = src / "release_gate.json"
    if not artifact.exists() or not metadata.exists() or not gate.exists():
        return False
    try:
        metadata_payload = json.loads(
            metadata.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    if (
        expected_holdout_generation is not None
        and metadata_payload.get("holdout_generation") != expected_holdout_generation
    ):
        return False
    if (
        expected_research_fingerprint is not None
        and metadata_payload.get("research_code_fingerprint_sha256")
        != expected_research_fingerprint
    ):
        return False
    if (
        expected_full_fingerprint is not None
        and metadata_payload.get("code_fingerprint_sha256")
        != expected_full_fingerprint
    ):
        return False
    try:
        payload = json.loads(gate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("approved") is True


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    from src.validation.code_fingerprint import research_fingerprint_sha256
    expected_research_fingerprint = research_fingerprint_sha256()

    query = f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100"
    payload = json.loads(_download(query, token, timeout=25))

    candidates = sorted(
        (
            a
            for a in payload.get("artifacts", [])
            if str(a.get("name", "")).startswith("research-cycle-")
            and not a.get("expired")
        ),
        key=lambda a: a.get("created_at", ""),
        reverse=True,
    )
    if not candidates:
        raise SystemExit("DEFERRED: no previous research-cycle artifact")

    skipped = 0
    errors = []
    for artifact in candidates:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                extract = Path(tmp) / "artifact"
                download_workflow_artifact(repo, token, artifact, extract)
                validate_extracted_tree(extract)
                src = _find_research_root(extract)
                lock_src = _find_frozen_lock(extract)
                lock_payload = json.loads(
                    lock_src.read_text(encoding="utf-8")
                )
                if lock_payload.get("status") != "FROZEN":
                    skipped += 1
                    continue
                expected_holdout_generation = lock_payload.get("holdout_generation")
                if expected_holdout_generation is None:
                    skipped += 1
                    continue

                if not _has_approved_production_state(
                    src,
                    expected_holdout_generation=expected_holdout_generation,
                    expected_research_fingerprint=expected_research_fingerprint,
                ):
                    skipped += 1
                    continue

                dst = Path("data/research")
                dst.mkdir(parents=True, exist_ok=True)
                for item in src.iterdir():
                    target = dst / item.name
                    if item.is_file():
                        shutil.copy2(item, target)
                config_dst = Path("config/frozen_holdout.json")
                config_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lock_src, config_dst)

            print(
                f"research-state: restored approved artifact_id={artifact['id']} "
                f"created_at={artifact['created_at']} skipped_unapproved={skipped}"
            )
            return
        except Exception as exc:
            skipped += 1
            errors.append(
                f"artifact_id={artifact.get('id')}:{type(exc).__name__}:{exc}"
            )

    raise SystemExit(
        "DEFERRED: no retained research-cycle artifact contains an approved "
        "release gate and immutable production model"
    )


if __name__ == "__main__":
    main()
