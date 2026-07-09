"""
Async WebSocket client for the C++ execution service.
Handles auth, reconnection, and exposes high-level order methods.
"""
from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, Optional

import websockets

log = logging.getLogger(__name__)


class ExecutionClient:
    def __init__(self, ws_url: str, auth_token: str,
                 reconnect_delay: float = 5.0) -> None:
        self._url   = ws_url
        self._token = auth_token
        self._reconnect_delay = reconnect_delay
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._recv_queue: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()   # one request in-flight at a time
        self._running = False
        self._ready   = asyncio.Event()

    # -- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        self._running = True
        asyncio.ensure_future(self._run_loop())
        await self._ready.wait()   # block until first successful auth

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()

    # -- public API -------------------------------------------------------

    async def submit_order(
        self,
        symbol: str,
        side: str,                    # "buy" | "sell"
        order_type: str,              # "market" | "limit" | …
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_price: Optional[float] = None,
        trail_percent: Optional[float] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> dict:
        order: dict[str, Any] = {
            "symbol": symbol,
            "side":   side,
            "type":   order_type,
            "time_in_force": time_in_force,
        }
        if qty       is not None: order["qty"]           = qty
        if notional  is not None: order["notional"]      = notional
        if limit_price  is not None: order["limit_price"]  = limit_price
        if stop_price   is not None: order["stop_price"]   = stop_price
        if trail_price  is not None: order["trail_price"]  = trail_price
        if trail_percent is not None: order["trail_percent"] = trail_percent
        if client_order_id: order["client_order_id"] = client_order_id

        return await self._call({"action": "submit_order", "order": order})

    async def query_account(self) -> dict:
        return await self._call({"action": "query_account"})

    async def query_positions(self) -> dict:
        return await self._call({"action": "query_positions"})

    async def query_order(self, order_id: str) -> dict:
        return await self._call({"action": "query_order", "order_id": order_id})

    async def ping(self) -> dict:
        return await self._call({"action": "ping"})

    # -- internals --------------------------------------------------------

    async def _call(self, payload: dict, timeout: float = 10.0) -> dict:
        await self._ready.wait()
        async with self._lock:
            # Drain any stale responses left behind by a previous timeout —
            # otherwise a late reply gets consumed by this request and every
            # response after it is off by one.
            while not self._recv_queue.empty():
                stale = self._recv_queue.get_nowait()
                log.warning("Discarding stale execution-service response: %s", stale)
            await self._ws.send(json.dumps(payload))
            return await asyncio.wait_for(self._recv_queue.get(), timeout=timeout)

    async def _run_loop(self) -> None:
        while self._running:
            try:
                async with websockets.connect(self._url) as ws:
                    self._ws = ws
                    await self._authenticate()
                    self._ready.set()
                    log.info("Execution service connected")
                    await self._listen()
            except Exception as exc:
                self._ready.clear()
                log.warning("Execution service disconnected: %s — retry in %ss",
                            exc, self._reconnect_delay)
                if self._running:
                    await asyncio.sleep(self._reconnect_delay)
        self._ws = None

    async def _authenticate(self) -> None:
        await self._ws.send(json.dumps({"action": "auth", "token": self._token}))
        raw  = await asyncio.wait_for(self._ws.recv(), timeout=10)
        resp = json.loads(raw)
        if resp.get("status") != "ok":
            raise RuntimeError(f"Execution service auth failed: {resp}")

    async def _listen(self) -> None:
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            await self._recv_queue.put(msg)
