from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml


CONFIG = Path("config/research_data_sources.yml")
OUT = Path("data/research/estat_dashboard_research.json")
USER_AGENT = "Stock-Daily-Prediction-3000/e-stat-research"
DEFAULT_TIMEOUT = int(os.getenv("ESTAT_REQUEST_TIMEOUT_SECONDS", "30"))
MAX_ATTEMPTS = max(1, int(os.getenv("ESTAT_MAX_REQUEST_ATTEMPTS", "3")))


def _request_json(base_url: str, params: dict[str, str]) -> dict:
    url = base_url + "?" + urlencode(params, doseq=True)
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                body = response.read()
            payload = json.loads(body.decode("utf-8-sig"))
            if not isinstance(payload, dict):
                raise ValueError("e-Stat response root is not an object")
            return payload
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= MAX_ATTEMPTS:
                raise
            time.sleep(min(8.0, 2 ** (attempt - 1)))
    raise RuntimeError(f"e-Stat request failed: {last_error!r}")


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _indicator_candidates(payload: dict, keyword: str) -> list[dict[str, str]]:
    root = payload.get("GET_META_INDICATOR_INF") or {}
    metadata = root.get("METADATA_INF") or {}
    class_inf = metadata.get("CLASS_INF") or {}
    objects = _as_list(class_inf.get("CLASS_OBJ"))

    rows: list[dict[str, str]] = []
    for obj in objects:
        code = str(obj.get("@code", obj.get("code", ""))).strip()
        name = str(obj.get("@name", obj.get("name", ""))).strip()
        if not code:
            continue
        classes = _as_list(obj.get("CLASS"))
        if classes:
            for cls in classes:
                element_name = str(cls.get("@name", cls.get("name", ""))).strip()
                element_code = str(cls.get("@code", cls.get("code", ""))).strip()
                rows.append({
                    "keyword": keyword,
                    "indicator_code": code,
                    "indicator_name": name,
                    "element_code": element_code,
                    "element_name": element_name,
                })
        else:
            rows.append({
                "keyword": keyword,
                "indicator_code": code,
                "indicator_name": name,
                "element_code": "",
                "element_name": "",
            })
    return rows


def collect(config: dict) -> dict:
    cfg = config.get("e_stat_dashboard") or {}
    if not bool(cfg.get("enabled", True)):
        return {
            "status": "DISABLED",
            "research_only": True,
            "production_changed": False,
        }

    metadata_url = str(cfg["base_metadata_url"])
    data_url = str(cfg["base_data_url"])
    keywords = [str(x) for x in (cfg.get("keywords") or []) if str(x).strip()]
    per_keyword = max(1, int(cfg.get("max_indicators_per_keyword", 2)))
    max_total = max(1, int(cfg.get("max_total_indicators", 24)))
    checked_at = datetime.now(timezone.utc).isoformat()

    searches = []
    indicator_index: dict[str, dict] = {}
    for keyword in keywords:
        try:
            payload = _request_json(
                metadata_url,
                {
                    "Lang": "JP",
                    "SearchIndicatorWord": keyword,
                },
            )
            candidates = _indicator_candidates(payload, keyword)
            candidates = candidates[:per_keyword]
            searches.append({
                "keyword": keyword,
                "status": "SUCCESS",
                "candidate_count": len(candidates),
            })
            for row in candidates:
                indicator_index.setdefault(row["indicator_code"], row)
        except Exception as exc:
            searches.append({
                "keyword": keyword,
                "status": "FAILED",
                "error": f"{type(exc).__name__}:{exc}",
            })

    selected = list(indicator_index.values())[:max_total]
    codes = [row["indicator_code"] for row in selected]

    responses = []
    batches = [codes[i:i + 5] for i in range(0, len(codes), 5)]
    for batch in batches:
        try:
            payload = _request_json(
                data_url,
                {
                    "Lang": "JP",
                    "IndicatorCode": ",".join(batch),
                    "MetaGetFlg": "Y",
                    "SectionHeaderFlg": "Y",
                },
            )
            responses.append({
                "indicator_codes": batch,
                "status": "SUCCESS",
                "payload": payload,
            })
        except Exception as exc:
            responses.append({
                "indicator_codes": batch,
                "status": "FAILED",
                "error": f"{type(exc).__name__}:{exc}",
            })

    success_batches = sum(row["status"] == "SUCCESS" for row in responses)
    failed_batches = sum(row["status"] == "FAILED" for row in responses)

    report = {
        "schema_version": 1,
        "status": (
            "EVALUATED"
            if success_batches and not failed_batches
            else "DEGRADED"
            if success_batches
            else "FAILED"
        ),
        "checked_at_utc": checked_at,
        "source": "Statistics Dashboard (e-Stat)",
        "source_url": "https://dashboard.e-stat.go.jp/",
        "access": {
            "api_no_registration_required": True,
            "automated_collection": True,
        },
        "research_only": True,
        "production_changed": False,
        "pit_status": "UNVERIFIED",
        "pit_policy": {
            "retrieved_at_recorded": True,
            "indicator_metadata_update_dates_preserved": True,
            "historical_available_at_not_inferred": True,
            "no_production_feature_adoption": True,
        },
        "searches": searches,
        "selected_indicators": selected,
        "data_batches": responses,
        "summary": {
            "keywords": len(keywords),
            "selected_indicator_count": len(selected),
            "successful_batches": success_batches,
            "failed_batches": failed_batches,
            "raw_observations_available_for_local_pit_research": success_batches > 0,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    if not CONFIG.exists():
        raise SystemExit("FAIL: research data-source config is missing")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    report = collect(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
