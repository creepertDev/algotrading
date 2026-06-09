"""
Parameter sweep optimizer.

Usage:
    python -m backtest.optimize --strategy MACrossover --symbol SPY --timeframe 5Min
    python -m backtest.optimize --strategy RSIMeanReversion --symbol AAPL --timeframe 5Min
    python -m backtest.optimize --strategy MACrossover --symbol SPY --timeframe 5Min --top 10
"""
from __future__ import annotations
import argparse
import asyncio
import itertools
import logging
import sys
from dataclasses import dataclass
from typing import Any

from .engine import run_backtest

logging.basicConfig(level=logging.WARNING,          # quiet during sweep
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── Parameter grids ──────────────────────────────────────────────────────────

GRIDS: dict[str, list[dict]] = {
    "MACrossover": list(
        {"fast_period": f, "slow_period": s, "qty": 1}
        for f, s in itertools.product(
            [5, 9, 12, 15, 20],        # fast EMA
            [21, 30, 50, 100],         # slow EMA
        )
        if f < s
    ),
    "RSIMeanReversion": list(
        {
            "rsi_period":     rp,
            "oversold":       ob,
            "overbought":     70,
            "trend_ma_period": ma,
            "stop_loss_pct":  sl,
            "qty": 1,
        }
        for rp, ob, ma, sl in itertools.product(
            [10, 14, 21],              # RSI period
            [20, 25, 30],              # oversold threshold
            [0, 20, 50],               # trend MA (0 = disabled)
            [1.5, 2.0, 3.0],           # stop loss %
        )
    ),
}

STRATEGY_MODULES = {
    "MACrossover":      ("bot.strategies.ma_crossover",       "MACrossover"),
    "RSIMeanReversion": ("bot.strategies.rsi_mean_reversion",  "RSIMeanReversion"),
}


def _make_strategy(name: str, params: dict):
    import importlib
    mod_path, cls_name = STRATEGY_MODULES[name]
    cls   = getattr(importlib.import_module(mod_path), cls_name)
    strat = cls()
    strat.name   = name
    strat.params = params.copy()
    return strat


def _param_str(params: dict) -> str:
    skip = {"qty"}
    return "  ".join(f"{k}={v}" for k, v in params.items() if k not in skip)


def _print_table(rows: list[dict], top: int, symbol: str, timeframe: str):
    rows = rows[:top]
    col_w = 52

    print(f"\n  Top {len(rows)} parameter sets — {rows[0]['strategy']} on {symbol} [{timeframe}]")
    print(f"  {'─' * 90}")
    print(f"  {'Parameters':<{col_w}}  {'Trades':>6}  {'Win%':>5}  {'P&L':>9}  {'Ret%':>6}  {'PF':>5}  {'MaxDD%':>6}")
    print(f"  {'─' * 90}")

    for r in rows:
        pnl_s = f"+${r['gross_pnl']:.2f}" if r['gross_pnl'] >= 0 else f"-${abs(r['gross_pnl']):.2f}"
        ret_s = f"+{r['return_pct']:.2f}%" if r['return_pct'] >= 0 else f"{r['return_pct']:.2f}%"
        pf_s  = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "  ∞"
        ps    = _param_str(r['params'])
        if len(ps) > col_w:
            ps = ps[:col_w - 1] + "…"
        print(f"  {ps:<{col_w}}  {r['total_trades']:>6}  {r['win_rate_pct']:>4.1f}%  "
              f"{pnl_s:>9}  {ret_s:>6}  {pf_s:>5}  {r['max_drawdown_pct']:>5.1f}%")

    print(f"  {'─' * 90}")
    best = rows[0]
    print(f"\n  Best params:  {_param_str(best['params'])}")
    print(f"  P&L: {'+'if best['gross_pnl']>=0 else ''}${best['gross_pnl']:.2f}  "
          f"Return: {'+' if best['return_pct']>=0 else ''}{best['return_pct']:.2f}%  "
          f"Win rate: {best['win_rate_pct']:.1f}%  "
          f"Profit factor: {best['profit_factor']:.2f}\n")


async def sweep(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    bars: int,
    top: int,
    min_trades: int,
):
    grid = GRIDS[strategy_name]
    total = len(grid)
    print(f"\n  Sweeping {total} parameter combinations for {strategy_name} "
          f"on {symbol} [{timeframe}] …")

    # Pre-fetch bars once so yfinance isn't hammered
    from bot.data.historical import fetch_bars
    all_bars = fetch_bars(symbol, timeframe, bars)
    if not all_bars:
        print("  ERROR: no historical data returned")
        return

    results = []
    for i, params in enumerate(grid, 1):
        strat  = _make_strategy(strategy_name, params)
        result = await run_backtest(strat, symbol, timeframe,
                                    lookback_bars=bars,
                                    starting_equity=10_000)
        s = result.summary()
        s["params"] = params
        results.append(s)

        # Progress indicator
        pct = i / total * 100
        bar_w = 30
        filled = int(bar_w * i / total)
        bar_str = "█" * filled + "░" * (bar_w - filled)
        print(f"\r  [{bar_str}] {pct:5.1f}%  ({i}/{total})", end="", flush=True)

    print()  # newline after progress bar

    # Filter: require a minimum number of trades to avoid overfitting flukes
    valid = [r for r in results if r["total_trades"] >= min_trades]
    if not valid:
        print(f"  No combinations had >= {min_trades} trades. "
              f"Try --min-trades 1 or a larger --bars value.")
        return

    # Sort by: profit factor first (capped at 10 to avoid division-by-zero noise),
    # then return %, then win rate
    def sort_key(r):
        pf = min(r["profit_factor"], 10.0)
        return (pf, r["return_pct"], r["win_rate_pct"])

    valid.sort(key=sort_key, reverse=True)
    _print_table(valid, top, symbol, timeframe)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy",   choices=list(GRIDS), default="MACrossover")
    parser.add_argument("--symbol",     default="SPY")
    parser.add_argument("--timeframe",  default="5Min")
    parser.add_argument("--bars",       type=int, default=1000)
    parser.add_argument("--top",        type=int, default=10,
                        help="Show top N results (default 10)")
    parser.add_argument("--min-trades", type=int, default=5,
                        help="Ignore combos with fewer trades (default 5)")
    args = parser.parse_args()

    await sweep(args.strategy, args.symbol, args.timeframe,
                args.bars, args.top, args.min_trades)


if __name__ == "__main__":
    asyncio.run(main())
