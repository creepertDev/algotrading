"""Shared data model dataclasses passed between feed → strategies."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    timeframe: str = "1Min"


@dataclass
class Quote:
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float

    @property
    def mid(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0


@dataclass
class Trade:
    symbol: str
    timestamp: datetime
    price: float
    size: float
    conditions: list[str] = field(default_factory=list)
