"""
RSI mean-reversion strategy with trend filter, stop-loss, and tighter entry.

Params:
    rsi_period      - RSI lookback (default 14)
    oversold        - RSI buy threshold (default 25)
    overbought      - RSI exit threshold (default 70)
    trend_ma_period - Only buy when close > this MA (default 50). Set 0 to disable.
    stop_loss_pct   - Exit if price drops this % below entry (default 2.0)
    qty             - shares per signal (default 1)
    bar_timeframe   - timeframe used for warmup (default "5Min")
"""
from __future__ import annotations
import logging
from collections import deque

import numpy as np
import pandas as pd

from .base import Strategy
from ..data.models import Bar

log = logging.getLogger(__name__)


def _calc_rsi(closes: list[float], period: int) -> float:
    if len(closes) < period + 1:
        return 50.0
    s     = pd.Series(closes)
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean().iloc[-1]
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean().iloc[-1]
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    return float(100 - (100 / (1 + gain / loss)))


class RSIMeanReversion(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self._closes: dict[str, deque]        = {}
        self._in_position: dict[str, bool]    = {}
        self._entry_price: dict[str, float]   = {}

    def _init_symbol(self, symbol: str) -> None:
        ma_period = self.params.get("trend_ma_period", 50)
        buf       = max(self.params.get("rsi_period", 14) * 5, ma_period * 2)
        self._closes[symbol]       = deque(maxlen=buf)
        self._in_position[symbol]  = False
        self._entry_price[symbol]  = 0.0

    def predict_signal(self, symbol: str, rsi: float,
                       closes: list[float]) -> str | None:
        """
        Override this to use an ML model.
        Return "buy", "sell", or None (no action).
        Default: threshold-based rules with trend filter.
        """
        oversold   = self.params.get("oversold",   25)
        overbought = self.params.get("overbought", 70)

        # Trend filter — price must be above the MA to go long
        ma_period = self.params.get("trend_ma_period", 50)
        if ma_period and len(closes) >= ma_period:
            ma = float(pd.Series(closes).rolling(ma_period).mean().iloc[-1])
            above_trend = closes[-1] > ma
        else:
            above_trend = True

        if rsi < oversold and above_trend:
            return "buy"
        if rsi > overbought:
            return "sell"
        return None

    async def on_bar(self, bar: Bar) -> None:
        sym = bar.symbol
        if sym not in self._closes:
            self._init_symbol(sym)

        self._closes[sym].append(bar.close)
        period = self.params.get("rsi_period", 14)

        if len(self._closes[sym]) < period + 1:
            return

        closes = list(self._closes[sym])
        rsi    = _calc_rsi(closes, period)
        qty    = self.params.get("qty", 1)

        # Stop-loss check — exit before evaluating new signals
        if self._in_position[sym]:
            stop_pct   = self.params.get("stop_loss_pct", 2.0)
            entry      = self._entry_price[sym]
            drop_pct   = (entry - bar.close) / entry * 100 if entry else 0
            if drop_pct >= stop_pct:
                log.info("[%s] Stop-loss hit on %s — down %.2f%% from entry $%.2f",
                         self.name, sym, drop_pct, entry)
                resp = await self.sell(sym, qty=qty)
                if resp and not resp.get("error_message"):
                    self._in_position[sym] = False
                    self._entry_price[sym] = 0.0
                return

        signal = self.predict_signal(sym, rsi, closes)

        if signal == "buy" and not self._in_position[sym]:
            log.info("[%s] RSI=%.1f oversold + uptrend confirmed — buying %s @ $%.2f",
                     self.name, rsi, sym, bar.close)
            resp = await self.buy(sym, qty=qty)
            if resp and not resp.get("error_message"):
                self._in_position[sym] = True
                self._entry_price[sym] = bar.close

        elif signal == "sell" and self._in_position[sym]:
            log.info("[%s] RSI=%.1f overbought — selling %s @ $%.2f",
                     self.name, rsi, sym, bar.close)
            resp = await self.sell(sym, qty=qty)
            if resp and not resp.get("error_message"):
                self._in_position[sym] = False
                self._entry_price[sym] = 0.0
