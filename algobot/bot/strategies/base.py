"""
Strategy base class.

To implement a new strategy:
  1. Subclass Strategy
  2. Override warmup() to load historical data and fit any ML model
  3. Override on_bar() / on_quote() to emit signals
  4. Call self.buy() / self.sell() to route orders through the risk gate

The engine wires up the execution client and portfolio manager automatically.
"""
from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from ..data.models import Bar, Quote, Trade
from ..data.historical import fetch_bars

if TYPE_CHECKING:
    from ..execution.client import ExecutionClient
    from ..portfolio.manager import PortfolioManager

log = logging.getLogger(__name__)


class Strategy(ABC):
    """
    Base class for all trading strategies.

    Attributes set by the engine before start():
        name            - unique string identifier
        symbols         - list of symbols this strategy trades
        params          - dict of strategy-specific hyperparameters
        execution       - ExecutionClient (shared across strategies)
        portfolio       - PortfolioManager (shared across strategies)
    """

    def __init__(self) -> None:
        self.name:      str  = self.__class__.__name__
        self.symbols:   list[str] = []
        self.params:    dict = {}
        self.execution: Optional["ExecutionClient"]    = None
        self.portfolio: Optional["PortfolioManager"]   = None
        self._running     = False
        self._warming_up  = False

    # -- lifecycle (override if needed) -----------------------------------

    async def warmup(self) -> None:
        """
        Called once before live data starts.
        Default: fetch historical bars and call on_bar() for each so
        indicators are seeded.  Override to add ML model loading/training.
        Orders are suppressed during warmup — only indicator state is built.
        """
        self._warming_up = True
        timeframe  = self.params.get("bar_timeframe", "1Min")
        lookback   = self.params.get("warmup_bars", 200)
        for symbol in self.symbols:
            bars = fetch_bars(symbol, timeframe, lookback)
            for bar in bars:
                await self.on_bar(bar)
        self._warming_up = False
        log.info("[%s] warmup complete", self.name)

    async def start(self) -> None:
        """Called after warmup, just before live data is routed here."""
        self._running = True

    async def stop(self) -> None:
        self._running = False

    # -- data callbacks (override at least on_bar) -----------------------

    @abstractmethod
    async def on_bar(self, bar: Bar) -> None:
        """Receives every bar for subscribed symbols."""

    async def on_quote(self, quote: Quote) -> None:
        pass

    async def on_trade(self, trade: Trade) -> None:
        pass

    # -- order helpers ---------------------------------------------------

    async def buy(
        self,
        symbol: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        price_hint: Optional[float] = None,
    ) -> Optional[dict]:
        return await self._place_order(
            symbol, "buy", qty, notional, order_type, limit_price,
            time_in_force, price_hint,
        )

    async def sell(
        self,
        symbol: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        price_hint: Optional[float] = None,
    ) -> Optional[dict]:
        return await self._place_order(
            symbol, "sell", qty, notional, order_type, limit_price,
            time_in_force, price_hint,
        )

    async def _place_order(
        self,
        symbol: str,
        side: str,
        qty: Optional[float],
        notional: Optional[float],
        order_type: str,
        limit_price: Optional[float],
        time_in_force: str,
        price_hint: Optional[float] = None,
    ) -> Optional[dict]:
        if not self._running or self._warming_up:
            return None

        # Risk gate — price_hint (usually the bar close) lets market orders
        # be checked against USD caps and position tracking.
        price_est = limit_price or price_hint or 0.0
        allowed, reason = self.portfolio.check_order(symbol, side, qty, price_est or None)
        if not allowed:
            log.warning("[%s] Order blocked: %s", self.name, reason)
            return None

        try:
            resp = await self.execution.submit_order(
                symbol=symbol, side=side, order_type=order_type,
                qty=qty, notional=notional, limit_price=limit_price,
                time_in_force=time_in_force,
            )
            log.info("[%s] Order submitted: %s", self.name, resp)
            # Optimistically record the fill so the portfolio manager
            # tracks position and doesn't block the matching close order.
            if qty and resp and isinstance(resp, dict):
                fill_price = float(
                    resp.get("filled_avg_price") or
                    resp.get("limit_price") or
                    limit_price or price_hint or 0
                )
                if fill_price:
                    self.portfolio.record_fill(symbol, side, qty, fill_price)
            return resp
        except Exception as exc:
            log.error("[%s] Order submission failed: %s", self.name, exc)
            return None
