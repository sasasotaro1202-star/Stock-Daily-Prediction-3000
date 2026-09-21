from __future__ import annotations

import hashlib, json, re, urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

class CellParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.in_cell=False; self.buf=[]; self.row=[]; self.rows=[]
    def handle_starttag(self,tag,attrs):
        if tag=="tr": self.row=[]
        elif tag in {"td","th"}: self.in_cell=True; self.buf=[]
    def handle_endtag(self,tag):
        if tag in {"td","th"} and self.in_cell:
            v=" ".join("".join(self.buf).split())
            if v:self.row.append(v)
            self.in_cell=False
        elif tag=="tr" and self.row:self.rows.append(self.row)
    def handle_data(self,data):
        if self.in_cell:self.buf.append(data)

def fetch(url:str)->bytes:
    req=urllib.request.Request(url,headers={"User-Agent":"Stock-Daily-Prediction/1.0"})
    with urllib.request.urlopen(req,timeout=25) as r:return r.read()

def parse_rows(raw:bytes,market:str,url:str)->list[dict]:
    p=CellParser(); p.feed(raw.decode("utf-8","ignore")); out=[]
    for row in p.rows:
        code=None
        if market=="japan":
            code=next((c for c in row[:3] if re.fullmatch(r"[0-9]{4}",c)),None)
        else:
            code=next((c for c in row[:3] if re.fullmatch(r"[A-Z][A-Z0-9.-]{0,5}",c)),None)
        if not code:continue
        name=next((c for c in row if c!=code),"")
        if not name or name in {"コード","ticker"}:continue
        if market=="japan":
            asset_class="jp_reit" if ("REIT" in name or "投資法人" in name) else ("jp_etf" if "ETF" in name else "jp_stock")
        else:
            asset_class="us_etf" if "ETF" in name else "us_stock"
        out.append({"symbol":code,"name":name,"asset_class":asset_class,"source_url":url,"tradeable":True})
    return list({(x["asset_class"],x["symbol"]):x for x in out}.values())

def build_snapshot(out_path:str)->dict:
    sources=[
        ("japan","https://www.paypay-sec.co.jp/stock/list/"),
        ("us","https://www.paypay-sec.co.jp/us-stock/list/"),
    ]
    records=[]; hashes={}
    for market,url in sources:
        raw=fetch(url); hashes[market]=hashlib.sha256(raw).hexdigest()
        records.extend(parse_rows(raw,market,url))
    if len(records)<100:raise RuntimeError(f"fail-closed: parsed universe unexpectedly small ({len(records)})")
    snap={"retrieved_at":datetime.now(timezone.utc).isoformat(),"source_hashes":hashes,
          "record_count":len(records),"records":sorted(records,key=lambda x:(x["asset_class"],x["symbol"]))}
    path=Path(out_path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(snap,ensure_ascii=False,indent=2),encoding="utf-8")
    return snap
