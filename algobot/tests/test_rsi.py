"""Unit tests for RSI calculation and signal generation (no network needed)."""
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.data.models import Bar
from bot.strategies.rsi_mean_reversion import RSIMeanReversion, _calc_rsi


def make_bar(sym: str, close: float) -> Bar:
    return Bar(sym, datetime.now(timezone.utc), close, close, close, close, 1000.0)


def test_calc_rsi_neutral():
    closes = [100.0] * 20
    assert abs(_calc_rsi(closes, 14) - 50.0) < 1.0


def test_calc_rsi_uptrend():
    closes = [float(i) for i in range(1, 21)]  # strictly rising
    rsi = _calc_rsi(closes, 14)
    assert rsi > 70, f"Expected overbought RSI, got {rsi:.1f}"


def test_calc_rsi_downtrend():
    closes = [float(20 - i) for i in range(20)]  # strictly falling
    rsi = _calc_rsi(closes, 14)
    assert rsi < 30, f"Expected oversold RSI, got {rsi:.1f}"


@pytest.mark.asyncio
async def test_strategy_buys_on_oversold():
    strat = RSIMeanReversion()
    strat.name    = "test_rsi"
    strat.symbols = ["AAPL"]
    strat.params  = {"rsi_period": 14, "oversold": 25, "overbought": 70,
                     "trend_ma_period": 0, "stop_loss_pct": 2.0, "qty": 1}
    strat._running = True

    strat.execution = AsyncMock()
    strat.execution.submit_order = AsyncMock(return_value={"id": "x", "status": "accepted"})
    strat.portfolio = MagicMock()
    strat.portfolio.check_order = MagicMock(return_value=(True, ""))

    # Feed 20 declining bars to force RSI < 30
    for i in range(20, 0, -1):
        await strat.on_bar(make_bar("AAPL", float(i)))

    strat.execution.submit_order.assert_called()
    first_call_kwargs = strat.execution.submit_order.call_args_list[0].kwargs
    assert first_call_kwargs["side"] == "buy"
