"""
Moving-average crossover strategy.
Uses a pandas Series to maintain a rolling window; ready for ML features.

Params (set via config):
    fast_period  - fast EMA period (default 9)
    slow_period  - slow EMA period (default 21)
    qty          - shares per signal (default 1)
    bar_timeframe - timeframe string used for warmup (default "1Min")
"""
from __future__ import annotations
import logging
from collections import deque

import numpy as np
import pandas as pd

from .base import Strategy
from ..data.models import Bar

log = logging.getLogger(__name__)


class MACrossover(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self._closes: dict[str, deque] = {}
        self._prev_signal: dict[str, int] = {}   # +1 / -1 / 0

    def _init_symbol(self, symbol: str) -> None:
        slow = self.params.get("slow_period", 21)
        self._closes[symbol]      = deque(maxlen=slow * 3)
        self._prev_signal[symbol] = 0

    async def on_bar(self, bar: Bar) -> None:
        sym = bar.symbol
        if sym not in self._closes:
            self._init_symbol(sym)

        self._closes[sym].append(bar.close)

        fast_p = self.params.get("fast_period", 9)
        slow_p = self.params.get("slow_period", 21)

        if len(self._closes[sym]) < slow_p:
            return   # not enough data yet

        closes   = pd.Series(list(self._closes[sym]))
        fast_ema = closes.ewm(span=fast_p, adjust=False).mean().iloc[-1]
        slow_ema = closes.ewm(span=slow_p, adjust=False).mean().iloc[-1]

        signal   = 1 if fast_ema > slow_ema else -1
        prev     = self._prev_signal[sym]

        if signal != prev:
            qty = self.params.get("qty", 1)
            if signal == 1:
                log.info("[%s] Bullish crossover on %s @ %.2f", self.name, sym, bar.close)
                await self.buy(sym, qty=qty)
            elif signal == -1 and prev == 1:
                log.info("[%s] Bearish crossover on %s @ %.2f", self.name, sym, bar.close)
                await self.sell(sym, qty=qty)

            self._prev_signal[sym] = signal
