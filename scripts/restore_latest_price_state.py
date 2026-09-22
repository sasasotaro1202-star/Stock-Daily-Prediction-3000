from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse


class _CrossHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward GitHub API credentials to an external artifact host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        source_host = urlparse(req.full_url).netloc
        target_host = urlparse(newurl).netloc
        if source_host != target_host:
            for key in ("Authorization", "Accept", "X-GitHub-Api-Version"):
                redirected.headers.pop(key, None)
        return redirected


def _github_opener():
    return urllib.request.build_opener(_CrossHostRedirectHandler())


def _get_json(url: str, token: str):
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with _github_opener().open(req, timeout=25) as resp:
        return json.load(resp)


def _get_blob(url: str, token: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with _github_opener().open(req, timeout=60) as resp:
        return resp.read()


def _safe_extract(zf: zipfile.ZipFile, destination: str) -> None:
    """Extract only regular files/directories under destination."""
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for member in zf.infolist():
        target = (root / member.filename).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(
                f"unsafe artifact member path outside restore root: {member.filename}"
            )
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member, "r") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def main():
    idx = int(os.environ.get("PRICE_SHARD_INDEX", "0"))
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    name = f"price-state-shard-{idx}"

    query = f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100"
    payload = _get_json(query, token)

    candidates = [
        a for a in payload.get("artifacts", [])
        if a.get("name") == name and not a.get("expired")
    ]
    if not candidates:
        print("price-state: no previous artifact; initial 5y fetch will run")
        return

    artifact = max(candidates, key=lambda a: a.get("created_at", ""))
    blob = _get_blob(artifact["archive_download_url"], token)

    Path("data/prices").mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        f.write(blob)
        tmp = f.name
    try:
        with zipfile.ZipFile(tmp) as z:
            _safe_extract(z, "data/prices")
    finally:
        Path(tmp).unlink(missing_ok=True)

    print(
        f"price-state: restored artifact_id={artifact['id']} "
        f"created_at={artifact['created_at']}"
    )


if __name__ == "__main__":
    main()
