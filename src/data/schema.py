from __future__ import annotations
from dataclasses import dataclass
from datetime import date,datetime

@dataclass(frozen=True)
class DailyBar:
    # Preserve the original positional API; new metadata fields are appended.
    symbol:str
    session_date:date | str
    available_at:datetime | str
    open:float
    high:float
    low:float
    close:float
    volume:float
    asset_class:str="unknown"
    source:str="unknown"

    def validate(self)->None:
        if not self.symbol: raise ValueError("symbol is required")
        if not self.asset_class: raise ValueError("asset_class is required")
        if not self.source: raise ValueError("source is required")
        values=(self.open,self.high,self.low,self.close)
        if any(v<=0 for v in values): raise ValueError("OHLC values must be positive")
        if self.high<max(self.open,self.close) or self.low>min(self.open,self.close) or self.high<self.low:
            raise ValueError("invalid OHLC relationship")
        if self.volume<0: raise ValueError("volume must be non-negative")
