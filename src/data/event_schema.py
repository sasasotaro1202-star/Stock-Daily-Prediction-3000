from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class MarketEvent:
    security_id:str
    event_type:str
    published_at:datetime
    available_at:datetime
    source:str
    title:str
    importance:float=0.0
    sentiment:float|None=None
    source_url:str=""

    def is_visible_at(self,prediction_time:datetime)->bool:
        return self.available_at<=prediction_time
