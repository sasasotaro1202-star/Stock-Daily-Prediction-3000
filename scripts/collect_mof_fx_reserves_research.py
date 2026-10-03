from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

CONFIG = Path("config/research_data_sources.yml")
UA = "Stock-Daily-Prediction-3000/1.0 mof-fx-reserves-research"
DEFAULTS = {
    "mof_fx_intervention": (
        "https://www.mof.go.jp/policy/international_policy/reference/feio/"
        "foreign_exchange_intervention_operations.csv",
        Path("data/research/mof_fx_intervention.csv"),
    ),
    "mof_official_reserves": (
        "https://www.mof.go.jp/policy/international_policy/reference/official_reserve_assets/"
        "historical.csv",
        Path("data/research/mof_official_reserves.csv"),
    ),
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch(url: str) -> tuple[bytes, dict[str, str]]:
    request = Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    with urlopen(request, timeout=45) as response:
        raw = response.read()
        headers = {
            "content_type": str(response.headers.get("Content-Type") or ""),
            "etag": str(response.headers.get("ETag") or ""),
            "last_modified": str(response.headers.get("Last-Modified") or ""),
        }
    return raw, headers


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp932", "shift_jis"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("FAIL: MOF CSV is not decodable as UTF-8/CP932/Shift-JIS")


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    reports = []
    failed = 0

    for key, (default_url, default_out) in DEFAULTS.items():
        section = cfg.get(key) or {}
        if section.get("enabled") is False:
            reports.append({"source": key, "status": "DEFERRED_DISABLED"})
            continue

        url = str(section.get("url") or default_url)
        out = Path(section.get("output") or default_out)
        try:
            raw, headers = _fetch(url)
            text = _decode(raw)
            lines = [line for line in text.splitlines() if line.strip()]
            digest = hashlib.sha256(raw).hexdigest()

            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
            meta = out.with_suffix(out.suffix + ".meta.json")
            meta.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source": key,
                        "url": url,
                        "retrieved_at": _utc(),
                        "bytes": len(raw),
                        "sha256": digest,
                        "line_count_hint": len(lines),
                        "encoding": "utf-8-sig_or_cp932",
                        "headers": headers,
                        "research_only": True,
                        "production_changed": False,
                        "promotion_allowed": False,
                        "pit_status": "UNVERIFIED",
                        "pit_policy": {
                            "publication_schedule_verified": bool(
                                section.get("pit_policy", {}).get(
                                    "publication_schedule_verified", False
                                )
                            ),
                            "row_level_available_at_verified": False,
                            "source_available_at_inferred": False,
                            "revision_history_must_be_preserved": bool(
                                section.get("pit_policy", {}).get(
                                    "revision_history_must_be_preserved", False
                                )
                            ),
                            "fail_closed_for_production": True,
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            reports.append(
                {
                    "source": key,
                    "status": "EVALUATED",
                    "rows_hint": max(0, len(lines) - 1),
                    "sha256": digest,
                    "retrieved_at": _utc(),
                    "pit_status": "UNVERIFIED",
                }
            )
        except Exception as exc:
            failed += 1
            reports.append(
                {
                    "source": key,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )

    manifest = {
        "schema_version": 1,
        "status": "EVALUATED" if reports and failed == 0 else ("DEGRADED" if reports else "FAILED"),
        "retrieved_at": _utc(),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "sources": reports,
    }
    Path("data/research").mkdir(parents=True, exist_ok=True)
    Path("data/research/mof_fx_reserves_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    required = [r for r in reports if r.get("status") != "DEFERRED_DISABLED"]
    return 0 if required and all(r.get("status") == "EVALUATED" for r in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
