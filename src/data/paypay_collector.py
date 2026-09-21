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
        self.in_cell=False
        self.buf=[]
        self.row=[]
        self.rows=[]
        self.section=""
        self.in_heading=False
        self.heading_buf=[]

    def handle_starttag(self,tag,attrs):
        if tag=="tr":
            self.row=[]
            self.row_section=self.section
        elif tag in {"td","th"}:
            self.in_cell=True
            self.buf=[]
        elif tag in {"h2","h3","h4"}:
            self.in_heading=True
            self.heading_buf=[]

    def handle_endtag(self,tag):
        if tag in {"td","th"} and self.in_cell:
            value=" ".join("".join(self.buf).split())
            if value:
                self.row.append(value)
            self.in_cell=False
        elif tag=="tr" and self.row:
            self.rows.append((self.row,self.row_section))
        elif tag in {"h2","h3","h4"} and self.in_heading:
            value=" ".join("".join(self.heading_buf).split())
            if value:
                self.section=value
            self.in_heading=False

    def handle_data(self,data):
        if self.in_cell:
            self.buf.append(data)
        if self.in_heading:
            self.heading_buf.append(data)


def fetch(url:str)->bytes:
    req=urllib.request.Request(
        url,
        headers={"User-Agent":"Stock-Daily-Prediction/1.0"},
    )
    with urllib.request.urlopen(req,timeout=30) as response:
        return response.read()


def _extract_code(row:list[str],market:str)->str|None:
    if market=="japan":
        return next(
            (c for c in row[:4] if re.fullmatch(r"[0-9]{4}[A-Z]?",c)),
            None,
        )
    return next(
        (c for c in row[:4] if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}",c)),
        None,
    )


def _channels(row:list[str])->list[str]:
    found=[]
    for cell in row:
        tokens=[
            token.strip().lower()
            for token in re.split(r"[,\s]+",cell)
            if token.strip()
        ]
        found.extend(token for token in PAYPAY_APP_TOKENS if token in tokens)
    return list(dict.fromkeys(found))


def _asset_class(market:str,section:str,name:str)->str|None:
    text=f"{section} {name}".upper()
    if market=="japan":
        if "REIT" in text or "投資法人" in text:
            return "jp_reit"
        if "ETF" in text or "上場投資信託" in text:
            return "jp_etf"
        if "個別" in text or "日本株" in text:
            return "jp_stock"
        return None

    if "ETF" in text:
        return "us_etf"
    if "米国株" in text:
        return "us_stock"
    return None


def parse_rows(raw:bytes,market:str,url:str)->list[dict]:
    parser=CellParser()
    parser.feed(raw.decode("utf-8","ignore"))
    records=[]
    for row,section in parser.rows:
        code=_extract_code(row,market)
        if not code:
            continue
        channels=_channels(row)
        if not channels:
            continue

        code_i=row.index(code)
        name=next(
            (
                c for c in row[code_i+1:]
                if c not in {"コード","ticker"}
                and not c.startswith("trade_")
                and not c.startswith("mini_")
                and not c.startswith("cfd_")
            ),
            "",
        )
        if not name:
            continue

        asset_class=_asset_class(market,section,name)
        if asset_class is None:
            continue

        records.append({
            "symbol":code,
            "name":name,
            "asset_class":asset_class,
            "source_url":url,
            "tradeable":True,
            "trade_channels":channels,
            "paypay_section":section,
        })

    return list({
        (x["asset_class"],x["symbol"]):x
        for x in records
    }.values())


def build_snapshot(out_path:str)->dict:
    sources=[
        ("japan","https://www.paypay-sec.co.jp/stock/list/"),
        ("us","https://www.paypay-sec.co.jp/us-stock/list/"),
    ]
    records=[]
    hashes={}
    raw_lengths={}

    for market,url in sources:
        raw=fetch(url)
        hashes[market]=hashlib.sha256(raw).hexdigest()
        raw_lengths[market]=len(raw)
        records.extend(parse_rows(raw,market,url))

    if len(records)<100:
        raise RuntimeError(
            "fail-closed: parsed universe unexpectedly small "
            f"({len(records)})"
        )

    asset_counts={}
    for row in records:
        asset_counts[row["asset_class"]]=(
            asset_counts.get(row["asset_class"],0)+1
        )

    snap={
        "retrieved_at":datetime.now(timezone.utc).isoformat(),
        "source_hashes":hashes,
        "raw_lengths":raw_lengths,
        "record_count":len(records),
        "asset_class_counts":asset_counts,
        "records":sorted(
            records,
            key=lambda x:(x["asset_class"],x["symbol"]),
        ),
    }
    path=Path(out_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(
        json.dumps(snap,ensure_ascii=False,indent=2),
        encoding="utf-8",
    )
    return snap


if __name__=="__main__":
    snap=build_snapshot("data/universe/latest.json")
    print(
        f"paypay-universe: {snap['record_count']} "
        f"classes={snap['asset_class_counts']}"
    )
