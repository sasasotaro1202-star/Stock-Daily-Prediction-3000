from __future__ import annotations
from dataclasses import dataclass
from datetime import date,datetime

@dataclass(frozen=True)
class DailyBar:
    symbol:str
    asset_class:str
    session_date:date
    available_at:datetime
    open:float
    high:float
    low:float
    close:float
    volume:float
    source:str

    def validate(self)->None:
        if not self.symbol or not self.asset_class or not self.source: raise ValueError("identity fields are required")
        values=(self.open,self.high,self.low,self.close)
        if any(v<=0 for v in values): raise ValueError("OHLC values must be positive")
        if self.high<max(self.open,self.close) or self.low>min(self.open,self.close) or self.high<self.low:
            raise ValueError("invalid OHLC relationship")
        if self.volume<0: raise ValueError("volume must be non-negative")
