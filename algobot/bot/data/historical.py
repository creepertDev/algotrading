"""
Historical bar fetching via yfinance.
Used by strategies to warm up indicators before live trading begins.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from .models import Bar

log = logging.getLogger(__name__)

_TF_MAP = {
    "1Min": "1m", "5Min": "5m", "15Min": "15m",
    "30Min": "30m", "1H": "1h", "1D": "1d",
}


def fetch_bars(
    symbol: str,
    timeframe: str = "1Min",
    lookback_bars: int = 200,
) -> list[Bar]:
    """
    Return up to `lookback_bars` bars for `symbol`.
    Raises ValueError for unsupported timeframes.
    """
    yf_tf = _TF_MAP.get(timeframe)
    if yf_tf is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. "
                         f"Choose from: {list(_TF_MAP)}")

    # yfinance limits intraday history; pick a safe period
    period = "60d" if timeframe == "1D" else "7d"

    ticker = yf.Ticker(symbol)
    df: pd.DataFrame = ticker.history(period=period, interval=yf_tf,
                                       auto_adjust=True)
    if df.empty:
        log.warning("yfinance returned no data for %s %s", symbol, timeframe)
        return []

    df = df.tail(lookback_bars)
    bars: list[Bar] = []
    for ts, row in df.iterrows():
        if isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc)

        bars.append(Bar(
            symbol=symbol,
            timestamp=dt,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
            timeframe=timeframe,
        ))
    log.info("Fetched %d historical bars for %s [%s]", len(bars), symbol, timeframe)
    return bars
