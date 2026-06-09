"""
Backtester — replays historical bars through strategies and tracks P&L.
Uses the exact same Strategy subclasses as live trading; no code changes needed.
"""
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from bot.data.historical import fetch_bars
from bot.data.models import Bar
from bot.strategies.base import Strategy

log = logging.getLogger(__name__)


@dataclass
class Fill:
    timestamp: datetime
    symbol: str
    side: str
    qty: float
    price: float
    strategy: str


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    fills: list[Fill] = field(default_factory=list)
    starting_equity: float = 10_000.0

    # computed by .summary()
    def summary(self) -> dict:
        equity    = self.starting_equity
        position  = 0.0
        entry_px  = 0.0
        trades    = []
        gross_pnl = 0.0
        wins = losses = 0

        for f in self.fills:
            if f.side == "buy" and position == 0:
                position = f.qty
                entry_px = f.price
                equity  -= f.qty * f.price
            elif f.side == "sell" and position > 0:
                pnl      = (f.price - entry_px) * min(f.qty, position)
                gross_pnl += pnl
                equity   += f.qty * f.price
                trades.append(pnl)
                if pnl > 0: wins += 1
                else:        losses += 1
                position = max(0.0, position - f.qty)
                entry_px = 0.0

        final_equity  = self.starting_equity + gross_pnl
        total_trades  = len(trades)
        win_rate      = wins / total_trades * 100 if total_trades else 0
        avg_win       = sum(p for p in trades if p > 0) / wins   if wins   else 0
        avg_loss      = sum(p for p in trades if p < 0) / losses if losses else 0
        profit_factor = (abs(sum(p for p in trades if p > 0)) /
                         abs(sum(p for p in trades if p < 0))) if losses else float("inf")

        # Max drawdown
        peak  = self.starting_equity
        mdd   = 0.0
        eq    = self.starting_equity
        for pnl in trades:
            eq  += pnl
            peak = max(peak, eq)
            mdd  = max(mdd, (peak - eq) / peak * 100)

        return {
            "strategy":      self.strategy_name,
            "symbol":        self.symbol,
            "total_trades":  total_trades,
            "wins":          wins,
            "losses":        losses,
            "win_rate_pct":  round(win_rate, 1),
            "gross_pnl":     round(gross_pnl, 2),
            "final_equity":  round(final_equity, 2),
            "return_pct":    round((final_equity - self.starting_equity) / self.starting_equity * 100, 2),
            "avg_win":       round(avg_win, 2),
            "avg_loss":      round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(mdd, 2),
            "fills":         self.fills,
        }


class PaperPortfolio:
    """Minimal portfolio that records fills instead of sending orders."""
    def check_order(self, *args, **kwargs):
        return True, ""

    def record_fill(self, *args, **kwargs):
        pass

    def get_position(self, symbol):
        from bot.portfolio.manager import Position
        return Position(symbol)

    def set_equity(self, v):
        pass

    @property
    def state(self):
        class _S:
            daily_drawdown_pct = 0.0
        return _S()


class PaperExecution:
    """Intercepts buy/sell calls and records fills at bar close price."""
    def __init__(self, result: BacktestResult, strategy_name: str):
        self._result   = result
        self._strat    = strategy_name
        self._last_bar: Optional[Bar] = None

    def set_bar(self, bar: Bar):
        self._last_bar = bar

    async def submit_order(self, symbol, side, order_type=None,
                           qty=None, **kwargs) -> dict:
        if self._last_bar is None:
            return {"error_message": "no bar"}
        price = self._last_bar.close
        self._result.fills.append(Fill(
            timestamp=self._last_bar.timestamp,
            symbol=symbol, side=side,
            qty=qty or 1, price=price,
            strategy=self._strat,
        ))
        return {"id": "bt", "status": "filled", "error_message": ""}

    async def query_account(self):   return {"equity": 10000}
    async def query_positions(self): return []


async def run_backtest(
    strategy: Strategy,
    symbol: str,
    timeframe: str = "1Min",
    lookback_bars: int = 500,
    starting_equity: float = 10_000.0,
) -> BacktestResult:
    result  = BacktestResult(strategy.name, symbol, starting_equity=starting_equity)
    exec_   = PaperExecution(result, strategy.name)
    port_   = PaperPortfolio()

    strategy.symbols   = [symbol]
    strategy.execution = exec_
    strategy.portfolio = port_
    strategy.params.setdefault("bar_timeframe", timeframe)

    bars = fetch_bars(symbol, timeframe, lookback_bars)
    if not bars:
        log.warning("No bars returned for %s %s", symbol, timeframe)
        return result

    log.info("Backtesting %s on %s bars of %s [%s]",
             strategy.name, len(bars), symbol, timeframe)

    # Warmup on first 30% of bars, trade on remaining 70%
    split      = max(50, int(len(bars) * 0.3))
    warmup_bars = bars[:split]
    live_bars   = bars[split:]

    # Warmup pass — no orders
    strategy._warming_up = True
    strategy._running    = False
    for bar in warmup_bars:
        await strategy.on_bar(bar)
    strategy._warming_up = False
    strategy._running    = True

    # Live pass — orders recorded
    for bar in live_bars:
        exec_.set_bar(bar)
        await strategy.on_bar(bar)

    return result
