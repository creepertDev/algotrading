"""
BotEngine — orchestrates data feeds, strategies, execution, and portfolio.

Flow:
  1. connect execution service
  2. load + warmup strategies (in parallel)
  3. connect data feed and subscribe symbols
  4. route incoming bars/quotes to each strategy that wants them
  5. run until interrupted
"""
from __future__ import annotations
import asyncio
import importlib
import logging
import os
from typing import Any

import yaml

from .data.alpaca_feed import AlpacaDataFeed
from .data.models import Bar, Quote, Trade
from .execution.client import ExecutionClient
from .portfolio.manager import PortfolioManager
from .strategies.base import Strategy

log = logging.getLogger(__name__)

# Registry: config "class" key → fully-qualified module path
STRATEGY_REGISTRY: dict[str, str] = {
    "MACrossover":        "bot.strategies.ma_crossover.MACrossover",
    "RSIMeanReversion":   "bot.strategies.rsi_mean_reversion.RSIMeanReversion",
    "BollingerReversion": "bot.strategies.bollinger_reversion.BollingerReversion",
}


def _load_strategy_class(class_name: str) -> type:
    fqn = STRATEGY_REGISTRY.get(class_name)
    if fqn is None:
        raise ValueError(f"Unknown strategy class '{class_name}'. "
                         f"Register it in engine.STRATEGY_REGISTRY.")
    module_path, cls_name = fqn.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


class BotEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config

        # Execution service
        exec_cfg = config["execution_service"]
        self._exec = ExecutionClient(
            ws_url=exec_cfg["ws_url"],
            auth_token=exec_cfg["auth_token"],
            reconnect_delay=exec_cfg.get("reconnect_delay", 5),
        )

        # Portfolio manager
        port_cfg = config.get("portfolio", {})
        self._portfolio = PortfolioManager(
            max_position_usd=port_cfg.get("max_position_usd", 10_000),
            max_total_exposure_usd=port_cfg.get("max_total_exposure_usd", 50_000),
            max_drawdown_pct=port_cfg.get("max_drawdown_pct", 5.0),
        )

        # Data feed
        alpaca = config["alpaca"]
        self._feed = AlpacaDataFeed(
            api_key=alpaca["api_key"],
            secret_key=alpaca["secret_key"],
            ws_url=alpaca.get("data_ws_url",
                              "wss://stream.data.alpaca.markets/v2/iex"),
            reconnect_delay=exec_cfg.get("reconnect_delay", 5),
        )

        # Strategies
        self._strategies: list[Strategy] = []
        self._symbol_to_strategies: dict[str, list[Strategy]] = {}

        for s_cfg in config.get("strategies", []):
            cls    = _load_strategy_class(s_cfg["class"])
            strat: Strategy = cls()
            strat.name      = s_cfg.get("name", cls.__name__)
            strat.symbols   = s_cfg.get("symbols", [])
            strat.params    = s_cfg.get("params", {})
            strat.execution = self._exec
            strat.portfolio = self._portfolio
            self._strategies.append(strat)

            for sym in strat.symbols:
                self._symbol_to_strategies.setdefault(sym, []).append(strat)

    # -- entry point ------------------------------------------------------

    async def run(self) -> None:
        log.info("BotEngine starting …")

        # Connect execution service first
        await self._exec.connect()
        log.info("Execution service ready")

        # Seed portfolio equity
        try:
            account = await self._exec.query_account()
            equity  = float(account.get("equity", 0))
            self._portfolio.set_equity(equity)
            log.info("Account equity: $%.2f", equity)
        except Exception as exc:
            log.warning("Could not fetch account equity: %s", exc)

        # Warmup all strategies in parallel
        log.info("Warming up %d strategies …", len(self._strategies))
        await asyncio.gather(*[s.warmup() for s in self._strategies])

        # Start strategies
        await asyncio.gather(*[s.start() for s in self._strategies])

        # Wire data callbacks
        self._feed.subscribe_bars(self._on_bar)
        self._feed.subscribe_quotes(self._on_quote)
        self._feed.subscribe_trades(self._on_trade)

        # Connect feed and subscribe all needed symbols
        await self._feed.connect()
        all_symbols = list(self._symbol_to_strategies.keys())
        await self._feed.subscribe_symbols(all_symbols)
        log.info("Live data feed running for symbols: %s", all_symbols)

        # Keep alive until cancelled
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            log.info("BotEngine shutting down …")
            await asyncio.gather(*[s.stop() for s in self._strategies])
            await self._feed.disconnect()
            await self._exec.disconnect()

    # -- data routing -----------------------------------------------------

    async def _on_bar(self, bar: Bar) -> None:
        for strat in self._symbol_to_strategies.get(bar.symbol, []):
            try:
                await strat.on_bar(bar)
            except Exception as exc:
                log.error("[%s] on_bar error: %s", strat.name, exc, exc_info=True)

    async def _on_quote(self, quote: Quote) -> None:
        for strat in self._symbol_to_strategies.get(quote.symbol, []):
            try:
                await strat.on_quote(quote)
            except Exception as exc:
                log.error("[%s] on_quote error: %s", strat.name, exc, exc_info=True)

    async def _on_trade(self, trade: Trade) -> None:
        for strat in self._symbol_to_strategies.get(trade.symbol, []):
            try:
                await strat.on_trade(trade)
            except Exception as exc:
                log.error("[%s] on_trade error: %s", strat.name, exc, exc_info=True)


# -- Config loader --------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        raw = os.path.expandvars(f.read())
    return yaml.safe_load(raw)
