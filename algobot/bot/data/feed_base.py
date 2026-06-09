"""Abstract base class for all market data feeds."""
from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Awaitable

from .models import Bar, Quote, Trade

BarHandler   = Callable[[Bar],   Awaitable[None]]
QuoteHandler = Callable[[Quote], Awaitable[None]]
TradeHandler = Callable[[Trade], Awaitable[None]]


class DataFeed(ABC):
    """
    Subclass this for each data provider (Alpaca, Polygon, …).
    Consumers register async callbacks; the feed calls them on every event.
    """

    def __init__(self) -> None:
        self._bar_handlers:   list[BarHandler]   = []
        self._quote_handlers: list[QuoteHandler] = []
        self._trade_handlers: list[TradeHandler] = []

    # -- subscription management ------------------------------------------

    def subscribe_bars(self, handler: BarHandler) -> None:
        self._bar_handlers.append(handler)

    def subscribe_quotes(self, handler: QuoteHandler) -> None:
        self._quote_handlers.append(handler)

    def subscribe_trades(self, handler: TradeHandler) -> None:
        self._trade_handlers.append(handler)

    # -- feed lifecycle ----------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Open connection / start polling."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean shutdown."""

    @abstractmethod
    async def subscribe_symbols(self, symbols: list[str]) -> None:
        """Tell the feed which symbols to stream."""

    # -- dispatch helpers --------------------------------------------------

    async def _emit_bar(self, bar: Bar) -> None:
        await asyncio.gather(*[h(bar) for h in self._bar_handlers],
                             return_exceptions=True)

    async def _emit_quote(self, quote: Quote) -> None:
        await asyncio.gather(*[h(quote) for h in self._quote_handlers],
                             return_exceptions=True)

    async def _emit_trade(self, trade: Trade) -> None:
        await asyncio.gather(*[h(trade) for h in self._trade_handlers],
                             return_exceptions=True)
