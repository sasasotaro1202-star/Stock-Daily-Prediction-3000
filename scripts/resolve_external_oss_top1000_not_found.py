from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("artifacts/external_oss_top1000_snapshot.json")
DEFAULT_OUTPUT = Path("artifacts/external_oss_top1000_identity_resolution.json")
API_ROOT = "https://api.github.com"
USER_AGENT = "Stock-Daily-Prediction-3000/external-oss-identity-resolution"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def request_json(url: str, token: str | None, retries: int = 2) -> tuple[int, dict[str, Any]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as response:
                body = json.loads(response.read().decode("utf-8"))
                return int(response.status), body
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


def resolve(row: dict[str, Any], token: str | None) -> dict[str, Any]:
    original = str(row["canonical_ref"])
    owner, name = original.split("/", 1)
    candidates_by_name: dict[str, dict[str, Any]] = {}
    queries = [f"{name} user:{owner}", f"{name} org:{owner}"]
    for query in queries:
        q = urllib.parse.quote(query, safe="")
        status, payload = request_json(f"{API_ROOT}/search/repositories?q={q}&per_page=10", token)
        if status != 200 or not isinstance(payload, dict):
            continue
        for item in payload.get("items", [])[:10]:
            if not isinstance(item, dict):
                continue
            full_name = item.get("full_name")
            if not full_name:
                continue
            exact = str(full_name).casefold() == original.casefold()
            owner_exact = str(full_name).split("/", 1)[0].casefold() == owner.casefold()
            name_exact = str(full_name).split("/", 1)[-1].casefold() == name.casefold()
            score = (100 if exact else 0) + (25 if owner_exact else 0) + (50 if name_exact else 0)
            candidates_by_name[str(full_name).casefold()] = {
                "full_name": full_name,
                "html_url": item.get("html_url"),
                "default_branch": item.get("default_branch"),
                "archived": bool(item.get("archived", False)),
                "fork": bool(item.get("fork", False)),
                "stars": int(item.get("stargazers_count", 0) or 0),
                "exact_match": exact,
                "owner_exact": owner_exact,
                "name_exact": name_exact,
                "match_score": score,
            }

    candidates: list[dict[str, Any]] = list(candidates_by_name.values())
    candidates.sort(key=lambda x: (-int(x["match_score"]), str(x["full_name"])))
    exact = [x for x in candidates if x["exact_match"]]
    owner_name = [x for x in candidates if x["owner_exact"] and x["name_exact"]]

    if exact:
        resolution = "EXACT_MATCH_REQUIRES_RETRY"
    elif len(owner_name) == 1:
        resolution = "OWNER_NAME_MATCH_UNCONFIRMED"
    elif len(candidates) == 0:
        resolution = "NO_SEARCH_MATCH"
    else:
        resolution = "AMBIGUOUS_ALTERNATIVES"

    return {
        "source_rank": row["source_rank"],
        "source_entry": row["source_entry"],
        "canonical_ref": original,
        "original_status": row["status"],
        "observed_at": now_iso(),
        "resolution": resolution,
        "candidates": candidates,
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
        "auto_rewrite_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = [row for row in source.get("results", []) if row.get("status") == "SOURCE_NOT_FOUND"]
    token = os.environ.get("GITHUB_TOKEN")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as pool:
        futures = {pool.submit(resolve, row, token): row for row in rows}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: (int(x["source_rank"]), x["canonical_ref"]))
    output = {
        "schema_version": 1,
        "resolution_id": "external-oss-top1000-identity-resolution-v1",
        "status": "EXECUTED_IDENTITY_RESOLUTION",
        "observed_at": now_iso(),
        "input_not_found_count": len(rows),
        "exact_match_requires_retry_count": sum(x["resolution"] == "EXACT_MATCH_REQUIRES_RETRY" for x in results),
        "owner_name_match_unconfirmed_count": sum(x["resolution"] == "OWNER_NAME_MATCH_UNCONFIRMED" for x in results),
        "ambiguous_alternatives_count": sum(x["resolution"] == "AMBIGUOUS_ALTERNATIVES" for x in results),
        "no_search_match_count": sum(x["resolution"] == "NO_SEARCH_MATCH" for x in results),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
        "auto_rewrite_allowed": False,
        "results": results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": output["status"],
                "input_not_found_count": output["input_not_found_count"],
                "exact_match_requires_retry_count": output["exact_match_requires_retry_count"],
                "owner_name_match_unconfirmed_count": output["owner_name_match_unconfirmed_count"],
                "ambiguous_alternatives_count": output["ambiguous_alternatives_count"],
                "no_search_match_count": output["no_search_match_count"],
                "promotion_allowed": output["promotion_allowed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
