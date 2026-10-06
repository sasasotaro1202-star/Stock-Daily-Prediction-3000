from __future__ import annotations

import json

from scripts import collect_estat_dashboard_research as collector


def test_estat_research_collection_is_research_only_and_preserves_raw_payload(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        collector,
        "OUT",
        tmp_path / "data" / "research" / "estat_dashboard_research.json",
    )

    def fake_request(base_url, params):
        if "getIndicatorInfo" in base_url:
            return {
                "GET_META_INDICATOR_INF": {
                    "METADATA_INF": {
                        "CLASS_INF": {
                            "CLASS_OBJ": [
                                {
                                    "@code": "1111111111111111111",
                                    "@name": "消費者物価指数",
                                    "CLASS": [
                                        {"@code": "1", "@name": "総合"}
                                    ],
                                },
                                {
                                    "@code": "2222222222222222222",
                                    "@name": "実質GDP",
                                },
                            ]
                        }
                    }
                }
            }
        return {
            "GET_STATS": {
                "RESULT": {"status": "0"},
                "STATISTICAL_DATA": {
                    "DATA_INF": {
                        "DATA_OBJ": [
                            {
                                "VALUE": {
                                    "@indicator": "1111111111111111111",
                                    "@unit": "1",
                                    "@stat": "stat",
                                    "@regionCode": "00000",
                                    "@time": "20260100",
                                    "@cycle": "1",
                                    "$": "105.2",
                                }
                            }
                        ]
                    }
                },
            }
        }

    monkeypatch.setattr(collector, "_request_json", fake_request)
    report = collector.collect(
        {
            "e_stat_dashboard": {
                "enabled": True,
                "base_metadata_url": "https://example/getIndicatorInfo",
                "base_data_url": "https://example/getData",
                "max_indicators_per_keyword": 2,
                "max_total_indicators": 2,
                "keywords": ["消費者物価"],
            }
        }
    )

    assert report["status"] == "EVALUATED"
    assert report["research_only"] is True
    assert report["production_changed"] is False
    assert report["pit_status"] == "UNVERIFIED"
    assert report["summary"]["selected_indicator_count"] == 2
    assert report["summary"]["raw_observations_available_for_local_pit_research"] is True
    assert (
        report["data_batches"][0]["payload"]["GET_STATS"]["STATISTICAL_DATA"]
        ["DATA_INF"]["DATA_OBJ"][0]["VALUE"]["$"]
        == "105.2"
    )
    assert collector.OUT.exists()
    json.loads(collector.OUT.read_text(encoding="utf-8"))
