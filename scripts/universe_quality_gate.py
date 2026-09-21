from __future__ import annotations
import json
from pathlib import Path

ALLOWED={"jp_stock","jp_etf","jp_reit","us_stock","us_etf"}

def main():
    path=Path("data/universe/latest.json")
    if not path.exists():
        raise SystemExit("FAIL: universe snapshot missing")
    snap=json.loads(path.read_text(encoding="utf-8"))
    rows=snap.get("records",[])
    reasons=[]
    if int(snap.get("record_count",-1)) != len(rows):
        reasons.append("record_count_mismatch")
    keys=[(row.get("asset_class"),row.get("symbol")) for row in rows]
    if len(keys) != len(set(keys)):
        reasons.append("duplicate_security_keys")
    if any(row.get("asset_class") not in ALLOWED for row in rows):
        reasons.append("unsupported_asset_class")
    if any(row.get("tradeable") is not True for row in rows):
        reasons.append("non_tradeable_in_current_universe")
    if any("paypay-sec.co.jp" not in str(row.get("source_url","")) for row in rows):
        reasons.append("non_paypay_source")
    result={"status":"PASS" if not reasons else "FAIL","record_count":len(rows),"reasons":reasons}
    out=Path("data/research/universe_quality.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if reasons:
        raise SystemExit("FAIL: universe quality gate")

if __name__=="__main__":
    main()
