"""
Backtest runner.

Usage:
    python -m backtest.run --strategy RSIMeanReversion --symbol AAPL --timeframe 5Min
    python -m backtest.run --strategy MACrossover --symbol SPY --timeframe 1Min --bars 1000
    python -m backtest.run --all
"""
from __future__ import annotations
import argparse
import asyncio
import importlib
import logging
import sys

from .engine import run_backtest

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

STRATEGY_REGISTRY = {
    "MACrossover":        ("bot.strategies.ma_crossover",       "MACrossover"),
    "RSIMeanReversion":   ("bot.strategies.rsi_mean_reversion",  "RSIMeanReversion"),
    "BollingerReversion": ("bot.strategies.bollinger_reversion", "BollingerReversion"),
    "VolumeExhaustion":   ("bot.strategies.volume_exhaustion",   "VolumeExhaustion"),
}

DEFAULT_PARAMS = {
    "MACrossover": {
        "fast_period": 9, "slow_period": 21, "qty": 1,
    },
    "RSIMeanReversion": {
        "rsi_period": 14, "oversold": 25, "overbought": 70,
        "trend_ma_period": 50, "stop_loss_pct": 2.0, "qty": 1,
    },
    "BollingerReversion": {
        "bb_period": 20, "bb_ndev": 2.0, "stop_loss_pct": 0.4,
        "max_trades_per_session": 3, "qty": 1,
    },
    "VolumeExhaustion": {
        "rvol_threshold": 2.5, "drop_threshold": -0.3, "vol_lookback": 40,
        "hold_minutes": 30, "stop_loss_pct": 0.6,
        "max_trades_per_session": 2, "qty": 1,
    },
}

PRESETS = [
    ("MACrossover",      "SPY",  "1Min",  1000),
    ("MACrossover",      "SPY",  "5Min",  1000),
    ("RSIMeanReversion", "AAPL", "5Min",  1000),
    ("RSIMeanReversion", "AAPL", "15Min", 500),
]


def _load_strategy(name: str):
    mod_path, cls_name = STRATEGY_REGISTRY[name]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    strat = cls()
    strat.name   = name
    strat.params = DEFAULT_PARAMS[name].copy()
    return strat


def _print_report(s: dict):
    pnl_sign  = "+" if s["gross_pnl"] >= 0 else ""
    ret_sign  = "+" if s["return_pct"] >= 0 else ""
    pnl_color = "\033[92m" if s["gross_pnl"] >= 0 else "\033[91m"
    reset     = "\033[0m"

    print(f"""
┌─────────────────────────────────────────────────┐
│  {s['strategy']:<20}  {s['symbol']:>6}               │
├─────────────────────────────────────────────────┤
│  Trades       {s['total_trades']:>6}   Win rate   {s['win_rate_pct']:>5.1f}%  │
│  Wins         {s['wins']:>6}   Losses     {s['losses']:>6}   │
│  Avg win    ${s['avg_win']:>7.2f}   Avg loss  ${s['avg_loss']:>7.2f}  │
│  Profit fac   {s['profit_factor']:>6.2f}   Max DD     {s['max_drawdown_pct']:>5.1f}%  │
├─────────────────────────────────────────────────┤
│  Gross P&L  {pnl_color}{pnl_sign}${s['gross_pnl']:>8.2f}{reset}                       │
│  Return     {pnl_color}{ret_sign}{s['return_pct']:>7.2f}%{reset}                        │
│  Final EQ   ${s['final_equity']:>9.2f}                      │
└─────────────────────────────────────────────────┘""")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGY_REGISTRY),
                        default="RSIMeanReversion")
    parser.add_argument("--symbol",    default="AAPL")
    parser.add_argument("--timeframe", default="5Min")
    parser.add_argument("--bars",      type=int, default=1000)
    parser.add_argument("--equity",    type=float, default=10_000)
    parser.add_argument("--all",       action="store_true",
                        help="Run all preset combinations")
    args = parser.parse_args()

    if args.all:
        for strat_name, symbol, tf, bars in PRESETS:
            strat  = _load_strategy(strat_name)
            result = await run_backtest(strat, symbol, tf, bars, args.equity)
            _print_report(result.summary())
    else:
        strat  = _load_strategy(args.strategy)
        result = await run_backtest(strat, args.symbol, args.timeframe,
                                    args.bars, args.equity)
        _print_report(result.summary())


if __name__ == "__main__":
    asyncio.run(main())
