from __future__ import annotations

import json
from pathlib import Path

import yaml

CATALOG=Path("config/paypay_catalog.yml")
OUT=Path("data/research/catalog_quality.json")


def main():
    if not CATALOG.exists():
        raise SystemExit("FAIL: PayPay master catalog is absent")
    cfg=yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    families=cfg.get("catalog",{}).get("product_families",[])
    ids=[f.get("id") for f in families]
    required={"jp_equities","us_equities","mutual_funds","jp_stock_cfd","leveraged_cfd","ideco"}
    reasons=[]
    if not required.issubset(set(ids)):
        reasons.append(f"missing_product_families:{sorted(required-set(ids))}")
    if len(ids)!=len(set(ids)):
        reasons.append("duplicate_product_family_ids")
    for family in families:
        if not str(family.get("source_url","")).startswith("https://www.paypay-sec.co.jp/"):
            reasons.append(f"non_paypay_source:{family.get('id')}")
        if not family.get("prediction_mode"):
            reasons.append(f"missing_prediction_mode:{family.get('id')}")
    result={
        "status":"PASS" if not reasons else "FAIL",
        "families":ids,
        "reasons":reasons,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if reasons:
        raise SystemExit("FAIL: PayPay catalog integrity")


if __name__=="__main__":
    main()
