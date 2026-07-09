"""
Moving-average crossover strategy.

Rules:
  1. Only enter long if price is above 20-period MA (trend filter)
  2. Hold minimum 30 minutes before selling
  3. Max 3 trades per session
  4. No entries in first 15 minutes of session (9:30–9:45 AM ET)
"""
from __future__ import annotations
import logging
from collections import deque
from datetime import datetime, time, timezone, timedelta

import pandas as pd

from .base import Strategy
from ..data.models import Bar

log = logging.getLogger(__name__)

ET_OFFSET  = timedelta(hours=-4)   # EDT; switch to -5 in winter
SESSION_OPEN = time(9, 30)
NO_ENTRY_UNTIL = time(9, 45)       # rule 4: skip first 15 min


class MACrossover(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self._closes: dict[str, deque] = {}
        self._prev_signal: dict[str, int] = {}
        self._entry_time: dict[str, datetime | None] = {}
        self._session_trades: dict[str, int] = {}
        self._session_date: dict[str, object] = {}

    def _init_symbol(self, symbol: str) -> None:
        slow = self.params.get("slow_period", 21)
        self._closes[symbol]        = deque(maxlen=max(slow * 3, 60))
        self._prev_signal[symbol]   = 0
        self._entry_time[symbol]    = None
        self._session_trades[symbol] = 0
        self._session_date[symbol]  = None

    def _et_time(self, bar: Bar) -> datetime:
        if bar.timestamp.tzinfo is None:
            ts = bar.timestamp.replace(tzinfo=timezone.utc)
        else:
            ts = bar.timestamp
        return ts + ET_OFFSET

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

        fast_p  = self.params.get("fast_period", 9)
        slow_p  = self.params.get("slow_period", 21)
        ma20_p  = self.params.get("trend_ma_period", 20)
        min_hold = self.params.get("min_hold_minutes", 30)
        max_trades = self.params.get("max_trades_per_session", 3)

        if len(self._closes[sym]) < slow_p:
            return

        closes   = pd.Series(list(self._closes[sym]))
        fast_ema = closes.ewm(span=fast_p, adjust=False).mean().iloc[-1]
        slow_ema = closes.ewm(span=slow_p, adjust=False).mean().iloc[-1]
        ma20     = closes.rolling(ma20_p).mean().iloc[-1] if len(closes) >= ma20_p else None

        signal = 1 if fast_ema > slow_ema else -1
        prev   = self._prev_signal[sym]

        if signal == prev:
            return

        self._prev_signal[sym] = signal
        et_now = self._et_time(bar)
        self._reset_session_if_new(sym, et_now)

        if signal == 1:
            # ── Rule 4: no entry in first 15 min ──────────────────────────
            if et_now.time() < NO_ENTRY_UNTIL:
                log.info("[%s] Skipping buy — within opening 15 min (%s ET)",
                         self.name, et_now.strftime("%H:%M"))
                return

            # ── Rule 3: max trades per session ────────────────────────────
            if self._session_trades[sym] >= max_trades:
                log.info("[%s] Skipping buy — max %d trades reached today",
                         self.name, max_trades)
                return

            # ── Rule 1: price must be above 20-period MA ──────────────────
            if ma20 is not None and bar.close <= ma20:
                log.info("[%s] Skipping buy — price %.2f below MA20 %.2f",
                         self.name, bar.close, ma20)
                return

            qty = self.params.get("qty", 1)
            log.info("[%s] Bullish crossover on %s @ %.2f", self.name, sym, bar.close)
            resp = await self.buy(sym, qty=qty, price_hint=bar.close)
            if resp and not resp.get("error_message"):
                self._entry_time[sym] = et_now
                self._session_trades[sym] += 1

        elif signal == -1 and prev == 1:
            entry = self._entry_time[sym]

            # Only sell if we actually hold a position — a bullish cross whose
            # buy was filtered out must not produce a sell (opens a short).
            if entry is None:
                return

            # ── Rule 2: minimum hold time ─────────────────────────────────
            held = (et_now - entry).total_seconds() / 60
            if held < min_hold:
                log.info("[%s] Skipping sell — only held %.0f min (min %d)",
                         self.name, held, min_hold)
                # Stay in "bullish" state so the next bearish bar retries the exit
                self._prev_signal[sym] = 1
                return

            qty = self.params.get("qty", 1)
            log.info("[%s] Bearish crossover on %s @ %.2f", self.name, sym, bar.close)
            resp = await self.sell(sym, qty=qty, price_hint=bar.close)
            if resp and not resp.get("error_message"):
                self._entry_time[sym] = None
