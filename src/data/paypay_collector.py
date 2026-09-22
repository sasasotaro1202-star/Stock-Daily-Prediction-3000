from __future__ import annotations

import hashlib
import json
import re
import urllib.request

from curl_cffi import requests as curl_requests
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




class VisibleTextParser(HTMLParser):
    """Extract block-oriented visible text from non-table HTML layouts."""

    BLOCK_TAGS = {
        "article", "br", "div", "h1", "h2", "h3", "h4", "h5",
        "label", "li", "p", "section", "td", "th", "tr",
    }

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def _visible_lines(raw: bytes) -> list[str]:
    parser = VisibleTextParser()
    parser.feed(raw.decode("utf-8", "ignore"))
    return [
        " ".join(line.split()).strip()
        for line in "".join(parser.parts).splitlines()
        if " ".join(line.split()).strip()
    ]


def _visible_name_noise(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized or len(normalized) < 2:
        return True
    if normalized in {
        "code", "ticker", "銘柄", "コード", "nisa", "nisa対象",
        "アプリ", "取扱いアプリ", "paypay証券アプリ",
        "paypay証券ミニアプリ", "日本株cfd", "すべて",
        "usd", "jpy", "成長投資",
        "a-", "d-", "g-", "j-", "m-", "p-", "s-", "v-",
    }:
        return True
    return normalized.startswith(("trade_", "mini_", "cfd_"))


def _section_from_line(line: str, market: str) -> str | None:
    upper = line.upper()
    if market == "japan":
        if "REIT" in upper or "不動産投資信託" in line:
            return "REIT"
        if "国内ETF" in line or "上場投資信託" in line:
            return "国内ETF"
        if "日本株" in line and ("個別" in line or "銘柄一覧" in line):
            return "日本株 個別銘柄"
        if line.strip() == "日本株":
            return "日本株"
        return None
    if "米国ETF" in line or "US ETF" in upper:
        return "米国ETF"
    if "米国株" in line and "ETF" not in upper:
        return "米国株"
    return None


def parse_visible_text(raw: bytes, market: str, url: str) -> list[dict]:
    """Parse rendered code/name/channel records from div/list layouts."""
    lines = _visible_lines(raw)
    section = ""
    records: list[dict] = []
    known_us_noise = {
        "A-", "D-", "G-", "J-", "M-", "P-", "S-", "V-",
        "ETF", "NISA", "CFD",
    }

    for i, line in enumerate(lines):
        detected = _section_from_line(line, market)
        if detected:
            section = detected

        if market == "japan":
            codes = re.findall(r"(?<!\d)(\d{4}[A-Z]?)(?!\d)", line)
        else:
            codes = re.findall(
                r"(?<![A-Z0-9])([A-Z][A-Z0-9.\-]{0,7})(?![A-Z0-9])",
                line,
            )

        for code in codes:
            if market == "us" and code in known_us_noise:
                continue

            window = lines[i : min(len(lines), i + 8)]
            window_text = " ".join(window)
            if not any(token in window_text.lower() for token in PAYPAY_APP_TOKENS):
                continue

            fields: list[str] = []
            for candidate in window:
                fields.extend(
                    value.strip(" |")
                    for value in re.split(r"\s*\|\s*", candidate)
                    if value.strip(" |")
                )

            try:
                code_position = fields.index(code)
            except ValueError:
                code_position = 0

            name = ""
            for candidate in fields[code_position + 1 :]:
                candidate = candidate.strip()
                if candidate == code or _visible_name_noise(candidate):
                    continue
                if market == "us" and candidate in known_us_noise:
                    continue
                if any(token in candidate.lower() for token in PAYPAY_APP_TOKENS):
                    continue
                if len(candidate) > 80:
                    continue
                name = candidate
                break

            if not name:
                remainder = re.sub(
                    rf"(?<!\d){re.escape(code)}(?!\d)",
                    "",
                    line,
                ).strip(" |:-")
                if remainder and not _visible_name_noise(remainder):
                    name = remainder.split("|", 1)[0].strip()

            if not name:
                continue

            asset_class = _asset_class(market, section, name)
            if not asset_class:
                continue

            channels = [
                token
                for token in PAYPAY_APP_TOKENS
                if token in window_text.lower()
            ]
            records.append({
                "symbol": code,
                "name": name,
                "asset_class": asset_class,
                "source_url": url,
                "tradeable": True,
                "trade_channels": list(dict.fromkeys(channels)),
                "paypay_section": section,
            })

    return list({
        (x["asset_class"], x["symbol"]): x
        for x in records
    }.values())


def _jina_reader(url: str) -> bytes:
    reader_url = "https://r.jina.ai/" + url
    req = urllib.request.Request(
        reader_url,
        headers={
            "User-Agent": "Stock-Daily-Prediction-PayPay/1.0",
            "Accept": "text/markdown,text/plain;q=0.9,*/*;q=0.8",
            "X-Target-Selector": "main",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        body = response.read()
    if len(body) < 1000:
        raise RuntimeError("Jina Reader response unexpectedly small")
    return body

def fetch(url: str) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
    }
    last_exc: Exception | None = None
    for attempt in range(3):
        candidate = url if attempt < 2 else url.rstrip("/")
        try:
            response = curl_requests.get(
                candidate,
                headers=headers,
                timeout=30,
                impersonate="chrome",
                allow_redirects=True,
            )
            response.raise_for_status()
            body = response.content
            if len(body) < 10_000:
                raise RuntimeError(
                    f"PayPay response unexpectedly small: {len(body)} bytes"
                )
            return body
        except Exception as exc:
            last_exc = exc

    # Conservative urllib fallback for transient TLS/client failures.
    req = urllib.request.Request(candidate, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read()
            if len(body) < 10_000:
                raise RuntimeError(
                    f"PayPay fallback response unexpectedly small: {len(body)} bytes"
                )
            return body
    except Exception as exc:
        raise RuntimeError(
            f"PayPay source fetch failed after retries: {last_exc or exc}"
        ) from exc


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



def parse_reader_text(raw: bytes, market: str, url: str) -> list[dict]:
    text = raw.decode("utf-8", "ignore")
    lines = [" ".join(line.split()).strip() for line in text.splitlines()]
    section = ""
    records = []
    for i, line in enumerate(lines):
        if not line:
            continue
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        if "trade_on" not in line.lower() and "mini_on" not in line.lower():
            continue
        window = lines[max(0, i - 4): i + 1]
        code = None
        name = None
        for candidate in window:
            fields = [x.strip(" |") for x in re.split(r"\s*\|\s*", candidate)]
            for field in fields:
                if not field:
                    continue
                if market == "japan" and re.fullmatch(r"[0-9]{4}[A-Z]?", field):
                    code = field
                elif market == "us" and re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}", field):
                    if field not in {"ETF", "NISA", "CFD", "USD", "JPY"}:
                        code = field
                elif name is None and len(field) >= 2:
                    low = field.lower()
                    if not low.startswith(("trade_", "mini_", "cfd_")) and field not in {"コード", "銘柄", "ticker"}:
                        name = field
        if not code or not name:
            continue
        asset_class = _asset_class(market, section, name)
        if not asset_class:
            continue
        channels = []
        if "trade_on" in line.lower():
            channels.append("trade_on")
        if "mini_on" in line.lower():
            channels.append("mini_on")
        records.append({
            "symbol": code,
            "name": name,
            "asset_class": asset_class,
            "source_url": url,
            "tradeable": True,
            "trade_channels": channels,
            "paypay_section": section,
        })
    return list({(x["asset_class"], x["symbol"]): x for x in records}.values())

def build_snapshot(out_path:str)->dict:
    sources=[
        ("japan","https://www.paypay-sec.co.jp/stock/list/"),
        ("us","https://www.paypay-sec.co.jp/us-stock/list/"),
    ]
    records=[]
    hashes={}
    raw_lengths={}

    retrieval_methods={}
    for market,url in sources:
        raw=fetch(url)
        parsed=parse_rows(raw,market,url)
        method="direct"

        # PayPay can expose the current catalog in div/list markup rather
        # than table rows. Parse rendered text before using any external
        # free reader fallback.
        visible_parsed=parse_visible_text(raw,market,url)
        if len(visible_parsed) > len(parsed):
            parsed=visible_parsed
            method="direct_visible_text"

        if len(parsed) < 50:
            try:
                reader_raw=_jina_reader(url)
                reader_candidates=[
                    ("jina_reader",parse_reader_text(reader_raw,market,url)),
                    ("jina_reader_visible_text",parse_visible_text(reader_raw,market,url)),
                ]
                reader_method,reader_parsed=max(
                    reader_candidates,
                    key=lambda item: len(item[1]),
                )
                if len(reader_parsed) > len(parsed):
                    raw=reader_raw
                    parsed=reader_parsed
                    method=reader_method
            except Exception as exc:
                method=f"direct_parse_weak:{type(exc).__name__}"
        hashes[market]=hashlib.sha256(raw).hexdigest()
        raw_lengths[market]=len(raw)
        retrieval_methods[market]=method
        records.extend(parsed)

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
        "retrieval_methods":retrieval_methods,
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
