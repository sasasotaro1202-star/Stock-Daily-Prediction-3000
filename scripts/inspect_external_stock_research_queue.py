from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_QUEUE = Path("data/research/external_stock_research_queue.json")
DEFAULT_OUTPUT = Path("data/research/external_stock_source_inspection.json")

PATHS = (
    "README.md",
    "README.rst",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "package.json",
    "environment.yml",
)

MECHANISM_TERMS = (
    "forecast",
    "forecasting",
    "time series",
    "probabilistic",
    "quantile",
    "calibration",
    "conformal",
    "uncertainty",
    "anomaly",
    "change point",
    "regime",
    "backtest",
    "walk forward",
    "cross-sectional",
    "ranking",
    "market data",
    "financial data",
    "pit",
    "point in time",
    "provenance",
    "lineage",
    "drift",
    "monitoring",
    "experiment",
    "research",
)


def _get_json(url: str, token: str | None) -> dict:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "Stock-Daily-Prediction-3000/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(repo: str, branch: str, path: str, token: str | None) -> tuple[str | None, str]:
    api = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    try:
        payload = _get_json(api, token)
    except HTTPError as exc:
        return None, f"http_{exc.code}"
    except URLError:
        return None, "network_error"
    except TimeoutError:
        return None, "timeout"
    except Exception as exc:
        return None, f"error_{type(exc).__name__}"

    if not isinstance(payload, dict) or payload.get("type") != "file":
        return None, "not_file"
    encoded = payload.get("content")
    if not encoded:
        return None, "missing_content"
    try:
        text = base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception:
        return None, "decode_error"
    return text, "ok"


def _signals(text: str) -> tuple[list[str], list[str]]:
    lower = text.lower()
    matched = [term for term in MECHANISM_TERMS if term in lower]
    snippets = []
    for term in matched[:8]:
        idx = lower.find(term)
        start = max(0, idx - 100)
        end = min(len(text), idx + 220)
        snippets.append(re.sub(r"\s+", " ", text[start:end]).strip())
    return matched[:12], snippets


def inspect(queue: dict, max_candidates: int, token: str | None) -> dict:
    if queue.get("status") != "EXECUTED_RESEARCH_QUEUE":
        return {
            "schema_version": 1,
            "status": "DEFERRED_NO_EXECUTABLE_QUEUE",
            "static_only": True,
            "predictive_performance_evidence": "NONE",
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "frozen_holdout_used": False,
            "candidates": [],
        }

    selected = list(queue.get("candidates") or [])[:max_candidates]
    results = []
    for rank, candidate in enumerate(selected, start=1):
        repo = str(candidate.get("canonical_ref") or "").strip()
        metadata = candidate.get("metadata") or {}
        branch = str(metadata.get("default_branch") or "main")
        files = []
        matched_terms: list[str] = []
        snippets: list[str] = []
        for path in PATHS:
            text, status = _fetch_text(repo, branch, path, token)
            files.append({"path": path, "status": status, "bytes": len(text.encode("utf-8")) if text else 0})
            if text:
                terms, local_snippets = _signals(text)
                matched_terms.extend(terms)
                snippets.extend(local_snippets)
            # Do not execute or install any external repository code.
            time.sleep(0.03)

        dedup_terms = list(dict.fromkeys(matched_terms))[:20]
        dedup_snippets = list(dict.fromkeys(snippets))[:8]
        retrieved = sum(row["status"] == "ok" for row in files)
        status = "SOURCE_CODE_INSPECTED" if retrieved else "UNVERIFIABLE_SOURCE_CONTENT"
        results.append(
            {
                "inspection_rank": rank,
                "canonical_ref": repo,
                "source_ranks": candidate.get("source_ranks"),
                "tier": candidate.get("tier"),
                "lane": candidate.get("lane"),
                "discovery_priority_score": candidate.get("discovery_priority_score"),
                "selection_basis": "queue_discovery_priority_only",
                "source_inspection_status": status,
                "static_only": True,
                "external_code_executed": False,
                "external_code_installed": False,
                "files": files,
                "mechanism_terms_found": dedup_terms,
                "evidence_snippets": dedup_snippets,
                "predictive_performance_evidence": "NONE",
                "required_next_stage": "LOCAL_IMPLEMENTATION_AND_REPRODUCTION",
                "research_only": True,
                "production_changed": False,
                "promotion_allowed": False,
                "frozen_holdout_used": False,
                "evidence_level": "E2_EXTERNAL_SOURCE_INSPECTION",
            }
        )

    return {
        "schema_version": 1,
        "inspection_id": "external-stock-source-inspection-v1",
        "status": "EXECUTED_STATIC_SOURCE_INSPECTION",
        "static_only": True,
        "external_code_executed": False,
        "external_code_installed": False,
        "selected_candidate_count": len(results),
        "source_inspected_count": sum(x["source_inspection_status"] == "SOURCE_CODE_INSPECTED" for x in results),
        "unverifiable_source_count": sum(x["source_inspection_status"] != "SOURCE_CODE_INSPECTED" for x in results),
        "predictive_performance_evidence": "NONE",
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
        "candidates": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-candidates", type=int, default=25)
    args = parser.parse_args()

    if args.max_candidates < 1 or args.max_candidates > 50:
        raise SystemExit("FAIL: --max-candidates must be between 1 and 50")

    if not args.queue.exists():
        result = {
            "schema_version": 1,
            "inspection_id": "external-stock-source-inspection-v1",
            "status": "DEFERRED_NO_EXECUTABLE_QUEUE",
            "static_only": True,
            "predictive_performance_evidence": "NONE",
            **{
                "research_only": True,
                "production_changed": False,
                "promotion_allowed": False,
                "frozen_holdout_used": False,
            },
            "candidates": [],
        }
    else:
        queue = json.loads(args.queue.read_text(encoding="utf-8"))
        result = inspect(queue, args.max_candidates, os.environ.get("GITHUB_TOKEN"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_candidate_count": result.get("selected_candidate_count", 0),
                "source_inspected_count": result.get("source_inspected_count", 0),
                "unverifiable_source_count": result.get("unverifiable_source_count", 0),
                "predictive_performance_evidence": result.get("predictive_performance_evidence"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
