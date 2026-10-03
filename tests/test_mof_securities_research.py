from __future__ import annotations

import json

from scripts import collect_mof_securities_research as collector


def test_mof_collection_preserves_raw_csv_and_hashes(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "research" / "mof_securities"
    monkeypatch.setattr(collector, "DEFAULT_TIMEOUT", 1)

    class FakeResponse:
        status = 200
        headers = {
            "Content-Type": "text/csv",
            "Last-Modified": "Wed, 07 Oct 2026 00:00:00 GMT",
            "ETag": '"demo"',
        }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return "date,category,value\n2026/09/01,stocks,100\n".encode("utf-8")

        def geturl(self):
            return "https://example.invalid/file.csv"

    monkeypatch.setattr(collector, "urlopen", lambda request, timeout: FakeResponse())

    report = collector.collect(
        {
            "mof_securities": {
                "enabled": True,
                "output_dir": str(output_dir),
                "sources": {
                    "monthly_total": {
                        "url": "https://example.invalid/montha1.csv",
                        "kind": "cross_border_securities_flows",
                    }
                },
            }
        }
    )

    assert report["status"] == "EVALUATED"
    assert report["research_only"] is True
    assert report["production_changed"] is False
    assert report["pit_status"] == "UNVERIFIED"
    assert report["summary"]["successful_sources"] == 1
    raw_path = output_dir / "monthly_total.csv"
    assert raw_path.exists()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    result = manifest["results"][0]
    assert result["bytes"] > 0
    assert len(result["sha256"]) == 64
    assert result["last_modified"] is not None
