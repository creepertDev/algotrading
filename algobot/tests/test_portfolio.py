"""Unit tests for PortfolioManager risk gates."""
from bot.portfolio.manager import PortfolioManager


def test_allows_normal_order():
    pm = PortfolioManager(max_position_usd=10_000, max_total_exposure_usd=50_000)
    ok, reason = pm.check_order("SPY", "buy", 10, 500.0)
    assert ok
    assert reason == ""


def test_blocks_position_cap():
    pm = PortfolioManager(max_position_usd=1_000)
    ok, reason = pm.check_order("SPY", "buy", 10, 500.0)
    assert not ok
    assert "position" in reason.lower()


def test_blocks_total_exposure():
    pm = PortfolioManager(max_total_exposure_usd=2_000)
    pm.record_fill("SPY", "buy", 3, 500.0)
    ok, reason = pm.check_order("AAPL", "buy", 3, 500.0)
    assert not ok
    assert "exposure" in reason.lower()


def test_drawdown_halt():
    pm = PortfolioManager(max_drawdown_pct=5.0)
    pm.set_equity(100_000)
    pm.set_equity(94_000)   # -6%, exceeds 5%
    ok, reason = pm.check_order("SPY", "buy", 1, 500.0)
    assert not ok
    assert "drawdown" in reason.lower()


def test_fill_tracking():
    pm = PortfolioManager()
    pm.record_fill("AAPL", "buy", 5, 200.0)
    pos = pm.get_position("AAPL")
    assert pos.qty == 5
    assert abs(pos.avg_entry_price - 200.0) < 0.01

    pm.record_fill("AAPL", "sell", 3, 210.0)
    pos = pm.get_position("AAPL")
    assert pos.qty == 2
    assert abs(pos.realized_pnl - 30.0) < 0.01
