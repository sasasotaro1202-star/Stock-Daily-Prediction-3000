from __future__ import annotations
from dataclasses import dataclass
from datetime import date,datetime

@dataclass(frozen=True)
class DailyBar:
    symbol:str
    session_date:date
    available_at:datetime
    open:float
    high:float
    low:float
    close:float
    volume:float
    asset_class:str="unknown"
    source:str="unknown"

    def validate(self)->None:
        if not self.symbol:
            raise ValueError("symbol is required")
        if any(v is None for v in (self.open,self.high,self.low,self.close,self.volume)):
            raise ValueError("OHLCV must be present")
        if any(v<=0 for v in (self.open,self.high,self.low,self.close)):
            raise ValueError("OHLC values must be positive")
        if self.high<max(self.open,self.close) or self.low>min(self.open,self.close) or self.high<self.low:
            raise ValueError("invalid OHLC relationship")
        if self.volume<0:
            raise ValueError("volume must be non-negative")
        if not self.source:
            raise ValueError("source is required")
