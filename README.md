# algotrading

Python algorithmic trading bot with a web dashboard, backtesting engine, and parameter optimizer. Connects to a C++ execution service for order routing via WebSocket, and uses Alpaca for live market data and order execution.

## Architecture

```
Alpaca Data WS ──► algobot (Python)
                       │
                       ├── MACrossover (SPY, 5Min bars)
                       └── RSIMeanReversion (AAPL, 5Min bars)
                                │
                         execution-service (C++)
                                │
                         Alpaca Paper API
```

- **algobot/** — async trading bot (strategies, risk manager, backtest engine)
- **dashboard/** — FastAPI web UI with live logs, account info, and Chart.js price charts
- **docker-compose.yml** — orchestrates all three services on a shared Docker network

## Strategies

### MA Crossover (SPY)
EMA 12/21 crossover with filters:
- Only enter when price is above 50-bar MA (trend filter)
- Minimum 60-minute hold before selling
- Max 2 trades per session

### RSI Mean Reversion (AAPL)
- Buy when RSI < 25 **and** price is above 50-bar MA
- Sell when RSI > 70
- 2% stop-loss

## Setup

```bash
cp algobot/config/config.example.yaml algobot/config/config.yaml
# set ALPACA_API_KEY, ALPACA_SECRET_KEY, WS_AUTH_TOKEN in .env
docker compose up -d
```

Dashboard available at `http://localhost:8889`.

## Backtesting

```bash
cd algobot
python -m backtest.run --strategy MACrossover --symbol SPY --timeframe 5Min
python -m backtest.optimize --strategy MACrossover --symbol SPY --top 10
```

## Requirements

- Docker + Docker Compose
- Alpaca paper trading account
- [execution-service](https://github.com/creepertDev/execution-service) running
