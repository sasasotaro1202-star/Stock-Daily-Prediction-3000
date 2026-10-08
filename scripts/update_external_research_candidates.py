from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY = Path("config/external_research_candidates.json")
USER_AGENT = "Stock-Daily-Prediction-3000/external-oss-research"
API_ROOT = "https://api.github.com"


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _days_since(value: str | None, now: datetime) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (now - parsed).total_seconds() / 86400.0)


def _get_json(url: str, token: str | None) -> tuple[int, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return int(response.status), json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"message": body[:500]}
        return int(exc.code), payload
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"message": str(exc)}


def _activity_status(days_since_push: float | None) -> str:
    if days_since_push is None:
        return "UNKNOWN"
    if days_since_push <= 30:
        return "RECENT_0_30D"
    if days_since_push <= 90:
        return "ACTIVE_31_90D"
    if days_since_push <= 180:
        return "AGING_91_180D"
    return "STALE_180D_PLUS"


def _license_status(payload: dict[str, Any]) -> str:
    license_payload = payload.get("license")
    if not isinstance(license_payload, dict):
        return "UNKNOWN"
    spdx = license_payload.get("spdx_id")
    if spdx in (None, "", "NOASSERTION"):
        return "UNKNOWN"
    return str(spdx)


def _validate_candidate(candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    repo = str(candidate.get("repo", "")).strip()
    score = candidate.get("initial_score")
    rank = candidate.get("rank")
    if "/" not in repo:
        errors.append("invalid_repo_ref")
    if not isinstance(rank, int) or rank < 1:
        errors.append("invalid_rank")
    if not isinstance(score, (int, float)) or not 0 <= float(score) <= 100:
        errors.append("invalid_initial_score")
    if candidate.get("decision") != "HOLD_RESEARCH_ONLY":
        errors.append("unsafe_initial_decision")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/external_candidate_registry.json")
    parser.add_argument("--fail-on-invalid-input", action="store_true")
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    candidates = registry.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise SystemExit("FAIL: candidate registry is missing candidates")

    token = os.environ.get("GITHUB_TOKEN")
    now = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    invalid_count = 0

    for candidate in candidates:
        errors = _validate_candidate(candidate)
        repo = str(candidate.get("repo", "")).strip()
        rank = candidate.get("rank")
        if repo in seen:
            errors.append(f"duplicate_repo_of_rank_{seen[repo]}")
        else:
            seen[repo] = int(rank) if isinstance(rank, int) else -1

        result: dict[str, Any] = {
            **candidate,
            "observed_at": _iso_now(),
            "evidence_level": "E1_EXTERNAL_CLAIM",
            "source_verified": False,
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "frozen_holdout_used": False,
            "local_reproduction_required": True,
            "momentum_status": "UNKNOWN_REQUIRES_PRIOR_SNAPSHOT",
            "status": "UNVERIFIABLE",
            "verification_errors": errors,
        }

        if errors:
            invalid_count += 1
            results.append(result)
            continue

        status, payload = _get_json(f"{API_ROOT}/repos/{repo}", token)
        if status == 200 and isinstance(payload, dict):
            pushed_days = _days_since(payload.get("pushed_at"), now)
            updated_days = _days_since(payload.get("updated_at"), now)
            result.update(
                {
                    "status": "VERIFIED_SOURCE_METADATA",
                    "source_verified": True,
                    "default_branch": payload.get("default_branch"),
                    "archived": bool(payload.get("archived", False)),
                    "disabled": bool(payload.get("disabled", False)),
                    "license": _license_status(payload),
                    "language": payload.get("language"),
                    "stars": int(payload.get("stargazers_count", 0) or 0),
                    "forks": int(payload.get("forks_count", 0) or 0),
                    "open_issues": int(payload.get("open_issues_count", 0) or 0),
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "pushed_at": payload.get("pushed_at"),
                    "days_since_push": pushed_days,
                    "days_since_repo_update": updated_days,
                    "activity_status": _activity_status(pushed_days),
                    "topics": payload.get("topics") or [],
                    "homepage": payload.get("homepage"),
                    "html_url": payload.get("html_url"),
                    "description": payload.get("description"),
                }
            )
            if result["archived"]:
                result["safety_status"] = "ARCHIVED_RESEARCH_ONLY"
            elif result["license"] == "UNKNOWN":
                result["safety_status"] = "LICENSE_UNKNOWN_RESEARCH_ONLY"
            else:
                result["safety_status"] = "METADATA_ONLY_RESEARCH_ONLY"
        elif status == 404:
            result["status"] = "SOURCE_NOT_FOUND"
            result["verification_errors"].append("github_repo_404")
        elif status == 403:
            result["status"] = "VERIFICATION_RATE_LIMIT_OR_FORBIDDEN"
            result["verification_errors"].append("github_api_403")
        elif status == 0:
            result["status"] = "VERIFICATION_NETWORK_UNAVAILABLE"
            result["verification_errors"].append(str(payload.get("message", "network_error")))
        else:
            result["status"] = "VERIFICATION_FAILED"
            result["verification_errors"].append(str(payload.get("message", f"github_http_{status}")))

        results.append(result)

    summary = {
        "schema_version": 1,
        "registry_id": registry.get("registry_id"),
        "observed_at": _iso_now(),
        "input_candidate_count": len(candidates),
        "unique_candidate_count": len(seen),
        "duplicate_count": len(candidates) - len(seen),
        "verified_source_count": sum(r.get("source_verified") is True for r in results),
        "source_not_found_count": sum(r.get("status") == "SOURCE_NOT_FOUND" for r in results),
        "unverifiable_count": sum(r.get("status") not in {"VERIFIED_SOURCE_METADATA", "SOURCE_NOT_FOUND"} for r in results),
        "evidence_contract": registry.get("evidence_contract", {}),
        "promotion_allowed": False,
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        k: summary[k]
        for k in (
            "input_candidate_count",
            "unique_candidate_count",
            "duplicate_count",
            "verified_source_count",
            "source_not_found_count",
            "unverifiable_count",
            "promotion_allowed",
        )
    }, ensure_ascii=False))

    if args.fail_on_invalid_input and invalid_count:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
