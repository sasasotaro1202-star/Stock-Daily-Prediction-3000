from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

PAYPAY_APP_TOKENS=("trade_on","mini_on")

class CellParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_cell=False; self.buf=[]; self.row=[]; self.rows=[]
    def handle_starttag(self,tag,attrs):
        if tag=="tr": self.row=[]
        elif tag in {"td","th"}: self.in_cell=True; self.buf=[]
    def handle_endtag(self,tag):
        if tag in {"td","th"} and self.in_cell:
            value=" ".join("".join(self.buf).split())
            if value:self.row.append(value)
            self.in_cell=False
        elif tag=="tr" and self.row:self.rows.append(self.row)
    def handle_data(self,data):
        if self.in_cell:self.buf.append(data)

def fetch(url:str)->bytes:
    req=urllib.request.Request(url,headers={"User-Agent":"Stock-Daily-Prediction/1.0"})
    with urllib.request.urlopen(req,timeout=30) as response:
        return response.read()

def _extract_code(row:list[str],market:str)->str|None:
    if market=="japan":
        return next((c for c in row[:4] if re.fullmatch(r"[0-9]{4}[A-Z]?",c)),None)
    return next((c for c in row[:4] if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}",c)),None)

def parse_rows(raw:bytes,market:str,url:str)->list[dict]:
    parser=CellParser()
    parser.feed(raw.decode("utf-8","ignore"))
    records=[]
    for row in parser.rows:
        code=_extract_code(row,market)
        if not code: continue
        tradeable=any(token in " ".join(row) for token in PAYPAY_APP_TOKENS)
        if not tradeable: continue
        code_i=row.index(code)
        name=next((c for c in row[code_i+1:] if c not in {"コード","ticker"} and not c.startswith("trade_") and not c.startswith("mini_") and not c.startswith("cfd_")),"")
        if not name: continue
        if market=="japan":
            asset_class="jp_reit" if ("REIT" in name or "投資法人" in name) else ("jp_etf" if "ETF" in name else "jp_stock")
        else:
            asset_class="us_etf" if "ETF" in name else "us_stock"
        records.append({
            "symbol":code,"name":name,"asset_class":asset_class,
            "source_url":url,"tradeable":True,"trade_channels":[x for x in PAYPAY_APP_TOKENS if x in " ".join(row)],
        })
    return list({(x["asset_class"],x["symbol"]):x for x in records}.values())

def build_snapshot(out_path:str)->dict:
    sources=[
        ("japan","https://www.paypay-sec.co.jp/stock/list/"),
        ("us","https://www.paypay-sec.co.jp/us-stock/list/"),
    ]
    records=[]; hashes={}; raw_lengths={}
    for market,url in sources:
        raw=fetch(url)
        hashes[market]=hashlib.sha256(raw).hexdigest()
        raw_lengths[market]=len(raw)
        records.extend(parse_rows(raw,market,url))
    if len(records)<100: raise RuntimeError(f"fail-closed: parsed universe unexpectedly small ({len(records)})")
    asset_counts={}
    for row in records: asset_counts[row["asset_class"]]=asset_counts.get(row["asset_class"],0)+1
    snap={
        "retrieved_at":datetime.now(timezone.utc).isoformat(),
        "source_hashes":hashes,
        "raw_lengths":raw_lengths,
        "record_count":len(records),
        "asset_class_counts":asset_counts,
        "records":sorted(records,key=lambda x:(x["asset_class"],x["symbol"])),
    }
    path=Path(out_path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(snap,ensure_ascii=False,indent=2),encoding="utf-8")
    return snap

if __name__=="__main__":
    snap=build_snapshot("data/universe/latest.json")
    print(f"paypay-universe: {snap['record_count']} classes={snap['asset_class_counts']}")
