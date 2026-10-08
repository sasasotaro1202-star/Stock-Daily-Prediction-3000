from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY = Path("config/external_prediction_system_oss_top1000_registry.json")
DEFAULT_OUTPUT = Path("data/research/external_oss_top1000_snapshot.json")
API_ROOT = "https://api.github.com"
USER_AGENT = "Stock-Daily-Prediction-3000/external-oss-top1000"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fetch(repo: str, token: str | None, retries: int = 3) -> tuple[int, dict[str, Any]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{API_ROOT}/repos/{repo}"
    delay = 1.0
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return int(response.status), payload
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = {"message": body[:500]}
            if exc.code in {403, 429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            return int(exc.code), payload
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            return 0, {"message": str(exc)}
    return 0, {"message": "unreachable"}


def read_previous(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def delta(current: int | None, previous: int | None) -> int | None:
    if current is None or previous is None:
        return None
    return int(current) - int(previous)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--previous", default="")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    source_rows = [
        row for row in registry.get("rows", [])
        if row.get("entry_type") == "EXPLICIT_REPOSITORY_REF"
    ]
    if not source_rows:
        raise SystemExit("FAIL: no explicit repository refs in top1000 registry")

    previous = read_previous(Path(args.previous)) if args.previous else {}
    previous_map = {
        row.get("canonical_ref"): row
        for row in previous.get("results", [])
        if row.get("canonical_ref")
    }
    token = os.environ.get("GITHUB_TOKEN")
    observed_at = now_iso()

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 16))) as pool:
        futures = {
            pool.submit(fetch, row["canonical_ref"], token): row
            for row in source_rows
        }
        for future in as_completed(futures):
            row = futures[future]
            repo = row["canonical_ref"]
            try:
                status, payload = future.result()
            except Exception as exc:  # pragma: no cover
                status, payload = 0, {"message": str(exc)}

            prior = previous_map.get(repo, {})
            result: dict[str, Any] = {
                "source_rank": row["source_rank"],
                "source_entry": row["source_entry"],
                "canonical_ref": repo,
                "observed_at": observed_at,
                "status": "UNVERIFIABLE",
                "evidence_level": "E1_EXTERNAL_CLAIM",
                "research_only": True,
                "production_changed": False,
                "promotion_allowed": False,
                "frozen_holdout_used": False,
                "stars_delta": None,
                "forks_delta": None,
            }

            if status == 200 and isinstance(payload, dict):
                stars = int(payload.get("stargazers_count", 0) or 0)
                forks = int(payload.get("forks_count", 0) or 0)
                license_payload = payload.get("license")
                license_spdx = None
                if isinstance(license_payload, dict):
                    license_spdx = license_payload.get("spdx_id")
                result.update({
                    "status": "VERIFIED_SOURCE_METADATA",
                    "stars": stars,
                    "forks": forks,
                    "open_issues": int(payload.get("open_issues_count", 0) or 0),
                    "archived": bool(payload.get("archived", False)),
                    "disabled": bool(payload.get("disabled", False)),
                    "default_branch": payload.get("default_branch"),
                    "language": payload.get("language"),
                    "license_spdx": license_spdx,
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "pushed_at": payload.get("pushed_at"),
                    "html_url": payload.get("html_url"),
                    "description": payload.get("description"),
                    "stars_delta": delta(stars, prior.get("stars")),
                    "forks_delta": delta(forks, prior.get("forks")),
                    "prior_observation_at": prior.get("observed_at"),
                    "momentum_status": "MEASURED" if prior.get("stars") is not None else "BASELINE_ONLY",
                    "cost_status": "UNKNOWN_NOT_INFERRED",
                    "security_status": "UNVERIFIED_RESEARCH_ONLY",
                })
            elif status == 404:
                result.update({
                    "status": "SOURCE_NOT_FOUND",
                    "error": "github_404",
                    "prior_observation_at": prior.get("observed_at"),
                    "momentum_status": "UNKNOWN",
                })
            elif status in {403, 429}:
                result.update({
                    "status": "VERIFICATION_RATE_LIMITED",
                    "error": f"github_http_{status}",
                    "momentum_status": "UNKNOWN",
                })
            elif status == 0:
                result.update({
                    "status": "VERIFICATION_NETWORK_FAILED",
                    "error": payload.get("message"),
                    "momentum_status": "UNKNOWN",
                })
            else:
                result.update({
                    "status": "VERIFICATION_FAILED",
                    "error": payload.get("message", f"github_http_{status}"),
                    "momentum_status": "UNKNOWN",
                })
            results.append(result)

    results.sort(key=lambda x: (int(x["source_rank"]), x["canonical_ref"]))
    verified = [r for r in results if r["status"] == "VERIFIED_SOURCE_METADATA"]

    summary = {
        "schema_version": 1,
        "snapshot_type": "external_oss_top1000_live_metadata",
        "observed_at": observed_at,
        "source_registry": str(REGISTRY),
        "source_claimed_count": 1000,
        "explicit_repo_count": len(source_rows),
        "results_count": len(results),
        "verified_source_count": len(verified),
        "not_found_count": sum(r["status"] == "SOURCE_NOT_FOUND" for r in results),
        "rate_limited_count": sum(r["status"] == "VERIFICATION_RATE_LIMITED" for r in results),
        "network_failed_count": sum(r["status"] == "VERIFICATION_NETWORK_FAILED" for r in results),
        "momentum_measured_count": sum(r.get("momentum_status") == "MEASURED" for r in results),
        "baseline_only_count": sum(r.get("momentum_status") == "BASELINE_ONLY" for r in results),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        k: summary[k]
        for k in (
            "source_claimed_count",
            "explicit_repo_count",
            "results_count",
            "verified_source_count",
            "not_found_count",
            "rate_limited_count",
            "momentum_measured_count",
            "baseline_only_count",
            "promotion_allowed",
        )
    }, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
