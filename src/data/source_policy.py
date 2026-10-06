from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SourcePolicy:
    name:str
    priority:int
    supports_prices:bool=False
    supports_news:bool=False
    supports_macro:bool=False
    requires_key:bool=False
    authoritative:bool=False

DEFAULT_SOURCES=(
    SourcePolicy("paypay_official",1,authoritative=True),
    SourcePolicy("jpx",2,supports_prices=True),
    SourcePolicy("yfinance",3,supports_prices=True),
    SourcePolicy("web_news",1,supports_news=True),
    SourcePolicy("alpaca",4,supports_prices=True,requires_key=True),
    SourcePolicy("massive",5,supports_prices=True,requires_key=True),
    SourcePolicy("fred",2,supports_macro=True,requires_key=False),
)
