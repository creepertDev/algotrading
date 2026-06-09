"""
Entry point.
Usage:
    python main.py --config config/config.yaml
"""
import argparse
import asyncio
import logging
import os
import signal
import sys

from bot.engine import BotEngine, load_config


def setup_logging(level: str, log_file: str | None) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers)


async def main(config_path: str) -> None:
    cfg = load_config(config_path)

    log_cfg = cfg.get("logging", {})
    setup_logging(log_cfg.get("level", "INFO"), log_cfg.get("file"))

    engine = BotEngine(cfg)

    loop = asyncio.get_running_loop()
    task = loop.create_task(engine.run())

    def _shutdown(*_):
        task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    try:
        await task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlgoBot")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to config YAML (default: config/config.yaml)")
    args = parser.parse_args()
    asyncio.run(main(args.config))
