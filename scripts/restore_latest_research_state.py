from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def _download(url: str, token: str, timeout: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _has_approved_production_state(
    extract: Path,
    *,
    expected_holdout_generation: object,
    expected_research_fingerprint: str,
) -> bool:
    src = extract / "data" / "research"
    if not src.exists():
        return False
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
    if metadata_payload.get("holdout_generation") != expected_holdout_generation:
        return False
    if metadata_payload.get(
        "research_code_fingerprint_sha256"
    ) != expected_research_fingerprint:
        return False
    try:
        payload = json.loads(gate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("approved") is True


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    lock_path = Path("config/frozen_holdout.json")
    if not lock_path.exists():
        raise SystemExit("DEFERRED: current frozen holdout lock is absent")
    try:
        lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("DEFERRED: current frozen holdout lock is unreadable") from exc
    if lock_payload.get("status") != "FROZEN":
        raise SystemExit("DEFERRED: current frozen holdout is not locked")
    from src.validation.code_fingerprint import research_fingerprint_sha256
    expected_research_fingerprint = research_fingerprint_sha256()
    expected_holdout_generation = lock_payload.get("holdout_generation")
    if expected_holdout_generation is None:
        raise SystemExit("DEFERRED: current holdout generation is absent")
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
    for artifact in candidates:
        blob = _download(
            artifact["archive_download_url"],
            token,
            timeout=60,
        )

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "research.zip"
            archive.write_bytes(blob)
            extract = Path(tmp) / "extract"
            extract.mkdir()
            with zipfile.ZipFile(archive) as z:
                z.extractall(extract)

            if not _has_approved_production_state(
                extract,
                expected_holdout_generation=expected_holdout_generation,
                expected_research_fingerprint=expected_research_fingerprint,
            ):
                skipped += 1
                continue

            src = extract / "data" / "research"
            dst = Path("data/research")
            dst.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                target = dst / item.name
                if item.is_file():
                    shutil.copy2(item, target)

        print(
            f"research-state: restored approved artifact_id={artifact['id']} "
            f"created_at={artifact['created_at']} skipped_unapproved={skipped}"
        )
        return

    raise SystemExit(
        "DEFERRED: no retained research-cycle artifact contains an approved "
        "release gate and immutable production model"
    )


if __name__ == "__main__":
    main()
