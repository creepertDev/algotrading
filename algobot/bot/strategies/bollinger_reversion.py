"""
Bollinger Band mean-reversion strategy (long only).

Entry:  close drops below the lower band (MA − ndev·σ)
Exit:   close recovers to the middle band (MA), stop-loss hit,
        or the session is about to close (flatten by 15:55 ET)

Backtested on QQQ 5Min bars Jun 9 – Jul 8 2026:
33 trades, 60.6% win rate, PF 1.61.

Params:
    bb_period       - band lookback (default 20)
    bb_ndev         - standard deviations for the lower band (default 2.0)
    stop_loss_pct   - exit if down this % from entry (default 0.4)
    max_trades_per_session - entry cap per day (default 3)
    qty             - shares per signal (default 1)
    bar_timeframe   - timeframe used for warmup (default "5Min")
"""
from __future__ import annotations
import logging
from collections import deque
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .base import Strategy
from ..data.models import Bar

log = logging.getLogger(__name__)

EASTERN       = ZoneInfo("America/New_York")
FLATTEN_AFTER = time(15, 55)          # exit any position before the close


class BollingerReversion(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self._closes: dict[str, deque] = {}
        self._in_position: dict[str, bool] = {}
        self._entry_price: dict[str, float] = {}
        self._session_trades: dict[str, int] = {}
        self._session_date: dict[str, object] = {}

    def _init_symbol(self, symbol: str) -> None:
        period = self.params.get("bb_period", 20)
        self._closes[symbol]         = deque(maxlen=period * 3)
        self._in_position[symbol]    = False
        self._entry_price[symbol]    = 0.0
        self._session_trades[symbol] = 0
        self._session_date[symbol]   = None

    def _et_time(self, bar: Bar) -> datetime:
        ts = bar.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(EASTERN)

    def _reset_session_if_new(self, symbol: str, et_now: datetime) -> None:
        today = et_now.date()
        if self._session_date.get(symbol) != today:
            self._session_trades[symbol] = 0
            self._session_date[symbol]   = today

    async def on_bar(self, bar: Bar) -> None:
        sym = bar.symbol
        if sym not in self._closes:
            self._init_symbol(sym)

        self._closes[sym].append(bar.close)

        period     = self.params.get("bb_period", 20)
        ndev       = self.params.get("bb_ndev", 2.0)
        stop_pct   = self.params.get("stop_loss_pct", 0.4)
        max_trades = self.params.get("max_trades_per_session", 3)
        qty        = self.params.get("qty", 1)

        if len(self._closes[sym]) < period:
            return

        closes = pd.Series(list(self._closes[sym]))
        mid    = float(closes.rolling(period).mean().iloc[-1])
        sd     = float(closes.rolling(period).std().iloc[-1])
        lower  = mid - ndev * sd

        et_now = self._et_time(bar)
        self._reset_session_if_new(sym, et_now)

        if self._in_position[sym]:
            entry    = self._entry_price[sym]
            stop     = entry * (1 - stop_pct / 100)
            hit_stop = bar.close <= stop
            at_mid   = bar.close >= mid
            eod      = et_now.time() >= FLATTEN_AFTER

            if hit_stop or at_mid or eod:
                reason = ("stop-loss" if hit_stop else
                          "middle band reached" if at_mid else "session close")
                log.info("[%s] Selling %s @ %.2f (%s, entry %.2f)",
                         self.name, sym, bar.close, reason, entry)
                resp = await self.sell(sym, qty=qty, price_hint=bar.close)
                if resp and not resp.get("error_message"):
                    self._in_position[sym] = False
                    self._entry_price[sym] = 0.0
            return

        # Entry — only during regular hours, before the flatten window
        if et_now.time() >= FLATTEN_AFTER:
            return
        if self._session_trades[sym] >= max_trades:
            return
        if bar.close < lower:
            log.info("[%s] %s closed %.2f below lower band %.2f — buying",
                     self.name, sym, bar.close, lower)
            resp = await self.buy(sym, qty=qty, price_hint=bar.close)
            if resp and not resp.get("error_message"):
                self._in_position[sym]  = True
                self._entry_price[sym]  = bar.close
                self._session_trades[sym] += 1
