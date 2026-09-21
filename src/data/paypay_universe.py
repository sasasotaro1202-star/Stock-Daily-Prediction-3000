from __future__ import annotations

"""PayPay Securities universe policy.

The official PayPay lists are the authority for current tradability.  The
collector that materializes the snapshot must preserve the source timestamp,
retrieval timestamp, and raw-source hash so that historical universe states
are reproducible and survivorship-bias checks remain possible.
"""

from dataclasses import dataclass
from enum import Enum


class AssetClass(str, Enum):
    JP_STOCK = "jp_stock"
    JP_ETF = "jp_etf"
    JP_REIT = "jp_reit"
    US_STOCK = "us_stock"
    US_ETF = "us_etf"


@dataclass(frozen=True)
class UniverseRule:
    asset_classes: tuple[AssetClass, ...] = tuple(AssetClass)
    official_only: bool = True
    current_tradeable_only: bool = True
    preserve_history: bool = True


PAYPAY_UNIVERSE_RULE = UniverseRule()


def is_prediction_eligible(asset_class: str, tradeable: bool) -> bool:
    """Return eligibility for the exchange-traded daily-price pipeline.

    Mutual funds, CFDs, leveraged CFDs, and iDeCo are intentionally outside
    this pipeline because their pricing/execution clocks differ.
    """
    try:
        AssetClass(asset_class)
    except ValueError:
        return False
    return bool(tradeable)
