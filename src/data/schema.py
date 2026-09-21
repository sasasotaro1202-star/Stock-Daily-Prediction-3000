from dataclasses import dataclass

@dataclass(frozen=True)
class DailyBar:
    symbol: str
    session_date: str
    available_at: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def validate(self) -> None:
        if self.high < max(self.open, self.close): raise ValueError("invalid high")
        if self.low > min(self.open, self.close): raise ValueError("invalid low")
        if self.volume < 0: raise ValueError("negative volume")
