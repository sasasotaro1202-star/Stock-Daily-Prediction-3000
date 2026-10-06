from __future__ import annotations
from dataclasses import asdict,dataclass
from datetime import datetime
import hashlib,json

@dataclass(frozen=True)
class Provenance:
    source:str
    source_url:str
    retrieved_at:datetime
    available_at:datetime
    raw_hash:str
    revision_id:str|None=None
    license_note:str|None=None

def content_hash(payload:bytes)->str:
    return hashlib.sha256(payload).hexdigest()

def to_json(p:Provenance)->str:
    return json.dumps(asdict(p),default=lambda x:x.isoformat() if hasattr(x,"isoformat") else x,ensure_ascii=False)
