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

DEFAULT_SOURCES=(
    SourcePolicy("jpx",1,supports_prices=True),
    SourcePolicy("yfinance",2,supports_prices=True),
    SourcePolicy("alpaca",3,supports_prices=True,requires_key=True),
    SourcePolicy("massive",4,supports_prices=True,requires_key=True),
    SourcePolicy("web_news",1,supports_news=True),
    SourcePolicy("fred",1,supports_macro=True,requires_key=True),
)
