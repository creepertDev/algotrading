"""
Volume-exhaustion (capitulation) reversion strategy — long only.

When a bar prints several times its average volume while dropping sharply,
the selling pressure is often exhausted and price snaps back. Buy the panic
bar, exit on the bounce.

Entry:  bar volume >= rvol_threshold x rolling average volume
        AND bar return <= drop_threshold (i.e. a sharp down bar)
Exit:   hold_minutes elapsed, stop-loss hit, or session close (15:55 ET)

Backtested on NVDA 5Min bars Jun 9 – Jul 8 2026:
rvol 2.5 / drop -0.3%: 10 trades, 70% win, PF 2.73.

Params:
    rvol_threshold  - volume multiple vs rolling avg to trigger (default 2.5)
    drop_threshold  - bar %-return that counts as a panic bar (default -0.3)
    vol_lookback    - bars in the rolling volume average (default 40)
    hold_minutes    - max hold time (default 30)
    stop_loss_pct   - exit if down this % from entry (default 0.6)
    max_trades_per_session - entry cap per day (default 2)
    qty             - shares per signal (default 1)
    bar_timeframe   - timeframe used for warmup (default "5Min")
"""
from __future__ import annotations
import logging
from collections import deque
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from .base import Strategy
from ..data.models import Bar

log = logging.getLogger(__name__)

EASTERN        = ZoneInfo("America/New_York")
FLATTEN_AFTER  = time(15, 55)
NO_ENTRY_AFTER = time(15, 0)   # leave room for the bounce before the close


class VolumeExhaustion(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self._volumes: dict[str, deque] = {}
        self._prev_close: dict[str, float] = {}
        self._in_position: dict[str, bool] = {}
        self._entry_price: dict[str, float] = {}
        self._entry_time: dict[str, datetime | None] = {}
        self._session_trades: dict[str, int] = {}
        self._session_date: dict[str, object] = {}

    def _init_symbol(self, symbol: str) -> None:
        lookback = self.params.get("vol_lookback", 40)
        self._volumes[symbol]        = deque(maxlen=lookback)
        self._prev_close[symbol]     = 0.0
        self._in_position[symbol]    = False
        self._entry_price[symbol]    = 0.0
        self._entry_time[symbol]     = None
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
        if sym not in self._volumes:
            self._init_symbol(sym)

        rvol_th    = self.params.get("rvol_threshold", 2.5)
        drop_th    = self.params.get("drop_threshold", -0.3)
        hold_min   = self.params.get("hold_minutes", 30)
        stop_pct   = self.params.get("stop_loss_pct", 0.6)
        max_trades = self.params.get("max_trades_per_session", 2)
        qty        = self.params.get("qty", 1)

        vols       = self._volumes[sym]
        prev_close = self._prev_close[sym]

        # Compute signal inputs *before* adding this bar to the average —
        # the spike must be measured against normal volume, not itself.
        avg_vol = sum(vols) / len(vols) if len(vols) >= vols.maxlen else 0.0
        rvol    = bar.volume / avg_vol if avg_vol > 0 else 0.0
        bar_ret = ((bar.close - prev_close) / prev_close * 100
                   if prev_close > 0 else 0.0)

        vols.append(bar.volume)
        self._prev_close[sym] = bar.close

        et_now = self._et_time(bar)
        self._reset_session_if_new(sym, et_now)

        # ── Manage open position ──────────────────────────────────────────
        if self._in_position[sym]:
            entry    = self._entry_price[sym]
            held_min = ((et_now - self._entry_time[sym]).total_seconds() / 60
                        if self._entry_time[sym] else 0)
            hit_stop = bar.close <= entry * (1 - stop_pct / 100)
            time_up  = held_min >= hold_min
            eod      = et_now.time() >= FLATTEN_AFTER

            if hit_stop or time_up or eod:
                reason = ("stop-loss" if hit_stop else
                          "hold expired" if time_up else "session close")
                log.info("[%s] Selling %s @ %.2f (%s, entry %.2f)",
                         self.name, sym, bar.close, reason, entry)
                resp = await self.sell(sym, qty=qty, price_hint=bar.close)
                if resp and not resp.get("error_message"):
                    self._in_position[sym] = False
                    self._entry_price[sym] = 0.0
                    self._entry_time[sym]  = None
            return

        # ── Entry on volume-spike panic bar ───────────────────────────────
        if et_now.time() >= NO_ENTRY_AFTER:
            return
        if self._session_trades[sym] >= max_trades:
            return
        if rvol >= rvol_th and bar_ret <= drop_th:
            log.info("[%s] Capitulation bar on %s: %.1fx volume, %.2f%% drop "
                     "— buying @ %.2f", self.name, sym, rvol, bar_ret, bar.close)
            resp = await self.buy(sym, qty=qty, price_hint=bar.close)
            if resp and not resp.get("error_message"):
                self._in_position[sym]  = True
                self._entry_price[sym]  = bar.close
                self._entry_time[sym]   = et_now
                self._session_trades[sym] += 1
