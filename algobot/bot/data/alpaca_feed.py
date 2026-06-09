"""
Alpaca market data WebSocket feed.
Streams bars, quotes, and trades via the Alpaca Data Stream v2 API.
Swap data_ws_url to Polygon's URL later without touching anything else.
"""
from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime, timezone

import websockets

from .feed_base import DataFeed
from .models import Bar, Quote, Trade

log = logging.getLogger(__name__)


class AlpacaDataFeed(DataFeed):
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        ws_url: str = "wss://stream.data.alpaca.markets/v2/iex",
        reconnect_delay: float = 5.0,
    ) -> None:
        super().__init__()
        self._api_key      = api_key
        self._secret_key   = secret_key
        self._ws_url       = ws_url
        self._reconnect_delay = reconnect_delay
        self._symbols: list[str] = []
        self._ws = None
        self._running = False

    async def connect(self) -> None:
        self._running = True
        asyncio.ensure_future(self._run_loop())

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()

    async def subscribe_symbols(self, symbols: list[str]) -> None:
        new = [s for s in symbols if s not in self._symbols]
        self._symbols.extend(new)
        if self._ws and new:
            await self._send_subscribe(new)

    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        while self._running:
            try:
                async with websockets.connect(self._ws_url) as ws:
                    self._ws = ws
                    await self._authenticate()
                    if self._symbols:
                        await self._send_subscribe(self._symbols)
                    await self._listen()
            except Exception as exc:
                log.warning("Alpaca feed disconnected: %s — reconnecting in %ss",
                            exc, self._reconnect_delay)
                if self._running:
                    await asyncio.sleep(self._reconnect_delay)
        self._ws = None

    async def _authenticate(self) -> None:
        msg = await self._ws.recv()
        data = json.loads(msg)
        log.debug("Alpaca feed connected: %s", data)

        await self._ws.send(json.dumps({
            "action": "auth",
            "key": self._api_key,
            "secret": self._secret_key,
        }))
        resp = json.loads(await self._ws.recv())
        if any(m.get("msg") == "authenticated" for m in resp):
            log.info("Alpaca data feed authenticated")
        else:
            raise RuntimeError(f"Alpaca auth failed: {resp}")

    async def _send_subscribe(self, symbols: list[str]) -> None:
        await self._ws.send(json.dumps({
            "action": "subscribe",
            "bars":   symbols,
            "quotes": symbols,
            "trades": symbols,
        }))
        log.info("Subscribed to Alpaca symbols: %s", symbols)

    async def _listen(self) -> None:
        async for raw in self._ws:
            messages = json.loads(raw)
            if not isinstance(messages, list):
                messages = [messages]
            for m in messages:
                t = m.get("T")
                if t == "b":
                    await self._emit_bar(self._parse_bar(m))
                elif t == "q":
                    await self._emit_quote(self._parse_quote(m))
                elif t == "t":
                    await self._emit_trade(self._parse_trade(m))
                elif t == "error":
                    log.error("Alpaca stream error: %s", m)

    # -- parsers -----------------------------------------------------------

    @staticmethod
    def _parse_bar(m: dict) -> Bar:
        return Bar(
            symbol=m["S"],
            timestamp=datetime.fromisoformat(m["t"].replace("Z", "+00:00")),
            open=m["o"], high=m["h"], low=m["l"], close=m["c"],
            volume=m["v"], vwap=m.get("vw"), trade_count=m.get("n"),
        )

    @staticmethod
    def _parse_quote(m: dict) -> Quote:
        return Quote(
            symbol=m["S"],
            timestamp=datetime.fromisoformat(m["t"].replace("Z", "+00:00")),
            bid_price=m["bp"], bid_size=m["bs"],
            ask_price=m["ap"], ask_size=m["as"],
        )

    @staticmethod
    def _parse_trade(m: dict) -> Trade:
        return Trade(
            symbol=m["S"],
            timestamp=datetime.fromisoformat(m["t"].replace("Z", "+00:00")),
            price=m["p"], size=m["s"],
            conditions=m.get("c", []),
        )
