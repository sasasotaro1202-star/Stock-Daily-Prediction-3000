from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path("scripts/resolve_external_oss_top1000_not_found.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("identity_resolution", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolution_is_research_only_and_non_mutating():
    module = _load_module()

    def fake_request(url, token, retries=2):
        return 200, {
            "items": [
                {
                    "full_name": "owner/repo",
                    "html_url": "https://github.com/owner/repo",
                    "default_branch": "main",
                    "archived": False,
                    "fork": False,
                    "stargazers_count": 123,
                }
            ]
        }

    original = module.request_json
    module.request_json = fake_request
    try:
        row = {
            "source_rank": 1,
            "source_entry": "owner/repo",
            "canonical_ref": "owner/repo",
            "status": "SOURCE_NOT_FOUND",
        }
        result = module.resolve(row, None)
    finally:
        module.request_json = original

    assert result["resolution"] == "EXACT_MATCH_REQUIRES_RETRY"
    assert result["auto_rewrite_allowed"] is False
    assert result["promotion_allowed"] is False
    assert result["production_changed"] is False


def test_main_accepts_empty_not_found_set(tmp_path, monkeypatch):
    module = _load_module()
    source = tmp_path / "snapshot.json"
    output = tmp_path / "resolution.json"
    source.write_text(json.dumps({"results": []}), encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    import sys
    original_argv = sys.argv
    sys.argv = ["resolver", "--input", str(source), "--output", str(output)]
    try:
        rc = module.main()
    finally:
        sys.argv = original_argv
    assert rc == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["status"] == "EXECUTED_IDENTITY_RESOLUTION"
    assert data["input_not_found_count"] == 0
    assert data["promotion_allowed"] is False
