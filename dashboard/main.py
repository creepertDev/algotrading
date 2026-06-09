"""
AlgoBot dashboard — serves the UI and streams live logs via WebSocket.
Also proxies account/position queries to the execution service.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiofiles
import websockets
import yfinance as yf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

LOG_FILE    = os.getenv("LOG_FILE",     "/logs/algobot.log")
EXEC_WS_URL = os.getenv("EXEC_WS_URL",  "ws://execution-service:8765")
EXEC_TOKEN  = os.getenv("WS_AUTH_TOKEN", "")
STATIC_DIR  = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

log = logging.getLogger("dashboard")


# ── Persistent execution service connection ───────────────────────────────────

class ExecConnection:
    """Single persistent WS connection to the execution service, shared by all API calls."""
    def __init__(self):
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._lock   = asyncio.Lock()
        self._ready  = asyncio.Event()
        self._queue  = asyncio.Queue()

    async def start(self):
        asyncio.ensure_future(self._run())
        await asyncio.wait_for(self._ready.wait(), timeout=15)

    async def _run(self):
        while True:
            try:
                async with websockets.connect(EXEC_WS_URL) as ws:
                    self._ws = ws
                    await ws.send(json.dumps({"action": "auth", "token": EXEC_TOKEN}))
                    await ws.recv()  # auth response
                    self._ready.set()
                    log.info("Exec connection ready")
                    async for raw in ws:
                        await self._queue.put(json.loads(raw))
            except Exception as exc:
                log.warning("Exec connection lost: %s — retrying", exc)
                self._ready.clear()
                await asyncio.sleep(3)

    async def call(self, action: str, timeout: float = 8.0) -> dict:
        async with self._lock:
            await self._ws.send(json.dumps({"action": action}))
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)


exec_conn = ExecConnection()


@app.on_event("startup")
async def startup():
    try:
        await exec_conn.start()
    except Exception as exc:
        log.warning("Could not connect to execution service on startup: %s", exc)


# ── HTML shell ────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ── Live log stream ───────────────────────────────────────────────────────────

@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    await ws.accept()
    try:
        try:
            async with aiofiles.open(LOG_FILE, "r") as f:
                lines = await f.readlines()
            for line in lines[-200:]:
                await ws.send_text(line.rstrip())
        except FileNotFoundError:
            await ws.send_text("[dashboard] waiting for log file...")

        async with aiofiles.open(LOG_FILE, "r") as f:
            await f.seek(0, 2)
            while True:
                line = await f.readline()
                if line:
                    await ws.send_text(line.rstrip())
                else:
                    await asyncio.sleep(0.25)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass


# ── Execution service proxy ───────────────────────────────────────────────────

@app.get("/api/account")
async def get_account():
    try:
        return await exec_conn.call("query_account")
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/positions")
async def get_positions():
    try:
        return await exec_conn.call("query_positions")
    except Exception as exc:
        return {"error": str(exc)}


# ── Chart data ────────────────────────────────────────────────────────────────

# Log line patterns:
# buy:  "[ma_crossover_SPY] Bullish crossover on SPY @ 738.30"
# buy:  "[rsi_mean_reversion_AAPL] RSI=24.9 oversold ... buying AAPL @ $307.80"
# sell: "[ma_crossover_SPY] Bearish crossover on SPY @ 744.41"
# sell: "[rsi_mean_reversion_AAPL] RSI=72.1 overbought — selling AAPL @ $312.00"
# stop: "[rsi_mean_reversion_AAPL] Stop-loss hit on AAPL ..."
_SIG_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[INFO\].*"
    r"\[(\w+)\] .*([Bb]ullish|[Bb]uy|oversold|[Bb]earish|[Ss]ell|overbought|[Ss]top-loss)"
    r".+?(\w+) @ \$?([\d.]+)"
)

def _parse_signals(log_text: str, symbol: str) -> list[dict]:
    signals = []
    for line in log_text.splitlines():
        if symbol not in line:
            continue
        m = _SIG_RE.search(line)
        if not m:
            continue
        ts_str, strat, action, sym, price = m.groups()
        if sym != symbol:
            continue
        action_lower = action.lower()
        if any(w in action_lower for w in ("bullish", "buy", "oversold")):
            side = "buy"
        else:
            side = "sell"
        try:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            signals.append({"t": ts.isoformat(), "side": side, "price": float(price)})
        except ValueError:
            continue
    return signals


@app.get("/api/chart/{symbol}")
async def get_chart(symbol: str, timeframe: str = "5m", bars: int = 390):
    # bars
    tf_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1m": "1m", "5m": "5m"}
    yf_tf  = tf_map.get(timeframe, "5m")
    period = "5d" if yf_tf in ("1m", "5m") else "30d"

    try:
        ticker = yf.Ticker(symbol)
        df     = ticker.history(period=period, interval=yf_tf, auto_adjust=True)
        df     = df.tail(bars)
        price_bars = [
            {
                "t": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                "o": round(float(row["Open"]),  2),
                "h": round(float(row["High"]),  2),
                "l": round(float(row["Low"]),   2),
                "c": round(float(row["Close"]), 2),
                "v": int(row["Volume"]),
            }
            for idx, row in df.iterrows()
        ]
    except Exception as exc:
        price_bars = []
        log.warning("yfinance error for %s: %s", symbol, exc)

    # signals from log
    signals: list[dict] = []
    try:
        async with aiofiles.open(LOG_FILE, "r") as f:
            log_text = await f.read()
        signals = _parse_signals(log_text, symbol)
    except FileNotFoundError:
        pass

    return {"symbol": symbol, "bars": price_bars, "signals": signals}
