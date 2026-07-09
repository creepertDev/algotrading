"""
Portfolio manager — tracks positions, enforces per-symbol and aggregate
risk limits before any order reaches the execution service.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.qty * self.avg_entry_price   # approx; updated on fills

    def update_fill(self, side: str, qty: float, price: float) -> None:
        if side == "buy":
            total_cost = self.avg_entry_price * self.qty + price * qty
            self.qty  += qty
            self.avg_entry_price = total_cost / self.qty if self.qty else 0.0
        else:
            if self.qty > 0:
                self.realized_pnl += (price - self.avg_entry_price) * min(qty, self.qty)
            self.qty = max(0.0, self.qty - qty)
            if self.qty == 0:
                self.avg_entry_price = 0.0


@dataclass
class PortfolioState:
    positions: dict[str, Position] = field(default_factory=dict)
    starting_equity: float = 0.0
    current_equity: float  = 0.0

    @property
    def total_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def daily_drawdown_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return (self.starting_equity - self.current_equity) / self.starting_equity * 100


class PortfolioManager:
    def __init__(
        self,
        max_position_usd: float   = 10_000,
        max_total_exposure_usd: float = 50_000,
        max_drawdown_pct: float   = 5.0,
    ) -> None:
        self._max_pos    = max_position_usd
        self._max_exp    = max_total_exposure_usd
        self._max_dd_pct = max_drawdown_pct
        self.state       = PortfolioState()
        self._halted     = False

    # -- risk gate --------------------------------------------------------

    def check_order(
        self,
        symbol: str,
        side: str,
        qty: Optional[float],
        price: Optional[float],
    ) -> tuple[bool, str]:
        """Return (allowed, reason). Reason is empty string when allowed."""
        if self._halted:
            return False, "trading halted due to drawdown limit"

        if self.state.daily_drawdown_pct >= self._max_dd_pct:
            self._halted = True
            log.warning("Drawdown %.2f%% exceeded limit %.2f%% — halting",
                        self.state.daily_drawdown_pct, self._max_dd_pct)
            return False, "daily drawdown limit breached"

        if side == "sell":
            pos = self.state.positions.get(symbol)
            held = pos.qty if pos else 0.0
            sell_qty = qty or 0.0
            if held <= 0:
                return False, f"no long position in {symbol} — sell would open a short"
            if sell_qty > held:
                return False, (f"cannot sell {sell_qty} {symbol} — only holding {held}")

        if side == "buy" and qty and price:
            order_usd = qty * price
            pos       = self.state.positions.get(symbol)
            pos_usd   = (pos.market_value if pos else 0.0) + order_usd

            if pos_usd > self._max_pos:
                return False, (f"order would bring {symbol} position to "
                               f"${pos_usd:.0f}, exceeding limit ${self._max_pos:.0f}")

            if self.state.total_exposure + order_usd > self._max_exp:
                return False, (f"order would push total exposure to "
                               f"${self.state.total_exposure + order_usd:.0f}, "
                               f"exceeding limit ${self._max_exp:.0f}")

        return True, ""

    # -- fill tracking ----------------------------------------------------

    def record_fill(self, symbol: str, side: str,
                    qty: float, price: float) -> None:
        if symbol not in self.state.positions:
            self.state.positions[symbol] = Position(symbol)
        self.state.positions[symbol].update_fill(side, qty, price)
        log.info("Fill recorded: %s %s %g @ $%.2f", side, symbol, qty, price)

    def set_equity(self, equity: float) -> None:
        if self.state.starting_equity == 0:
            self.state.starting_equity = equity
        self.state.current_equity = equity

    def get_position(self, symbol: str) -> Position:
        return self.state.positions.get(symbol, Position(symbol))
