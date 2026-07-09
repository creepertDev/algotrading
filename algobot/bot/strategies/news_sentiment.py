"""
News sentiment strategy (long only).

On each bar, fetches fresh headlines for the symbol from the Alpaca News API
and scores them with VADER (plus a finance-specific lexicon). A strongly
positive headline triggers a buy; the position is exited after a fixed hold,
on a stop-loss, on a strongly negative headline, or at the session close.

Backtested on TSLA Jun 9 – Jul 8 2026 (350 headlines):
event-study corr(score, next-30min return) = +0.09; positive headlines
preceded +0.37% avg 30-min moves vs +0.13% baseline. Sim at threshold 0.6
with 60-min hold: 49 trades, PF 1.57. Negative headlines showed no
predictive power, hence long-only.

Params:
    buy_threshold   - min compound score to enter (default 0.6)
    exit_threshold  - negative score that forces an early exit (default -0.6)
    hold_minutes    - max hold time (default 60)
    stop_loss_pct   - exit if down this % from entry (default 0.5)
    max_trades_per_session - entry cap per day (default 3)
    qty             - shares per signal (default 1)
    bar_timeframe   - bar cadence, drives the news poll rate (default "5Min")
"""
from __future__ import annotations
import asyncio
import logging
import os
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .base import Strategy
from ..data.models import Bar

log = logging.getLogger(__name__)

EASTERN       = ZoneInfo("America/New_York")
FLATTEN_AFTER = time(15, 55)
NO_ENTRY_AFTER = time(15, 30)   # leave room for a full hold before the close

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

# Finance terms VADER doesn't know (or under-weights)
FINANCE_LEXICON = {
    "upgrade": 2.0, "upgrades": 2.0, "downgrade": -2.0, "downgrades": -2.0,
    "beats": 1.5, "misses": -1.5, "surges": 1.5, "soars": 1.5,
    "plunges": -1.5, "tumbles": -1.5, "sinks": -1.2, "rallies": 1.2,
    "record": 0.8, "recall": -1.5, "lawsuit": -1.2, "probe": -1.0,
    "investigation": -1.0, "delivery": 0.5, "collapse": -1.8,
    "bankruptcy": -2.5, "outperform": 1.5, "underperform": -1.5,
}


class NewsSentiment(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self._analyzer = SentimentIntensityAnalyzer()
        self._analyzer.lexicon.update(FINANCE_LEXICON)
        self._seen_ids: set[int] = set()
        self._in_position: dict[str, bool] = {}
        self._entry_price: dict[str, float] = {}
        self._entry_time: dict[str, datetime | None] = {}
        self._session_trades: dict[str, int] = {}
        self._session_date: dict[str, object] = {}
        self._last_poll: datetime | None = None

    def _init_symbol(self, symbol: str) -> None:
        self._in_position[symbol]    = False
        self._entry_price[symbol]    = 0.0
        self._entry_time[symbol]     = None
        self._session_trades[symbol] = 0
        self._session_date[symbol]   = None

    async def warmup(self) -> None:
        # No indicators to seed — sentiment reacts to fresh headlines only.
        log.info("[%s] warmup complete (no indicator state)", self.name)

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

    def _fetch_news(self, symbol: str, since: datetime) -> list[dict]:
        headers = {
            "APCA-API-KEY-ID":     os.getenv("ALPACA_API_KEY", ""),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
        }
        params = {
            "symbols": symbol,
            "start":   since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit":   50,
        }
        r = requests.get(NEWS_URL, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("news", [])

    def score_headline(self, item: dict) -> float:
        """Compound sentiment in [-1, 1]. Override to plug in an ML model."""
        text = item.get("headline", "") + ". " + (item.get("summary") or "")
        return self._analyzer.polarity_scores(text)["compound"]

    async def on_bar(self, bar: Bar) -> None:
        if self._warming_up:
            return

        sym = bar.symbol
        if sym not in self._in_position:
            self._init_symbol(sym)

        buy_th   = self.params.get("buy_threshold", 0.6)
        exit_th  = self.params.get("exit_threshold", -0.6)
        hold_min = self.params.get("hold_minutes", 60)
        stop_pct = self.params.get("stop_loss_pct", 0.5)
        max_trades = self.params.get("max_trades_per_session", 3)
        qty      = self.params.get("qty", 1)

        et_now = self._et_time(bar)
        self._reset_session_if_new(sym, et_now)

        # ── Poll fresh headlines since the previous bar ───────────────────
        since = self._last_poll or et_now
        self._last_poll = et_now
        best, worst = 0.0, 0.0
        try:
            items = await asyncio.to_thread(self._fetch_news, sym, since)
            for item in items:
                if item["id"] in self._seen_ids:
                    continue
                self._seen_ids.add(item["id"])
                score = self.score_headline(item)
                log.info("[%s] %s %+0.2f  \"%s\"", self.name, sym, score,
                         item.get("headline", "")[:80])
                best  = max(best, score)
                worst = min(worst, score)
        except Exception as exc:
            log.warning("[%s] news fetch failed: %s", self.name, exc)

        # ── Manage open position ──────────────────────────────────────────
        if self._in_position[sym]:
            entry     = self._entry_price[sym]
            held_min  = ((et_now - self._entry_time[sym]).total_seconds() / 60
                         if self._entry_time[sym] else 0)
            hit_stop  = bar.close <= entry * (1 - stop_pct / 100)
            time_up   = held_min >= hold_min
            bad_news  = worst <= exit_th
            eod       = et_now.time() >= FLATTEN_AFTER

            if hit_stop or time_up or bad_news or eod:
                reason = ("stop-loss" if hit_stop else
                          "hold expired" if time_up else
                          "negative headline" if bad_news else "session close")
                log.info("[%s] Selling %s @ %.2f (%s, entry %.2f)",
                         self.name, sym, bar.close, reason, entry)
                resp = await self.sell(sym, qty=qty, price_hint=bar.close)
                if resp and not resp.get("error_message"):
                    self._in_position[sym] = False
                    self._entry_price[sym] = 0.0
                    self._entry_time[sym]  = None
            return

        # ── Entry on strongly positive fresh news ─────────────────────────
        if et_now.time() >= NO_ENTRY_AFTER:
            return
        if self._session_trades[sym] >= max_trades:
            return
        if best >= buy_th:
            log.info("[%s] Positive news (%.2f >= %.2f) — buying %s @ %.2f",
                     self.name, best, buy_th, sym, bar.close)
            resp = await self.buy(sym, qty=qty, price_hint=bar.close)
            if resp and not resp.get("error_message"):
                self._in_position[sym]  = True
                self._entry_price[sym]  = bar.close
                self._entry_time[sym]   = et_now
                self._session_trades[sym] += 1
