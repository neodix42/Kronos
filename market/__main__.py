"""Run the same archived research workflow without opening a browser."""

import argparse
import json

from market.binance import SUPPORTED_INTERVALS
from market.forecast import ForecastEngine
from market.service import MarketService, settings


def main():
    parser = argparse.ArgumentParser(description="Binance candles and Kronos research; no orders are placed.")
    parser.add_argument("command", choices=["forecast", "backtest"])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", choices=SUPPORTED_INTERVALS, default="1d")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--fee-bps", type=float, default=10)
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--batch-size", type=int, default=8, help="Reduce for machines with limited memory.")
    args = parser.parse_args()
    service = MarketService(engine=ForecastEngine(device=args.device, batch_size=args.batch_size))
    try:
        config = settings(vars(args))
        progress = lambda fraction, message: print(f"{fraction:.0%} {message}", flush=True)
        if args.command == "forecast":
            result = service.forecast(config, progress)
            summary = {key: result[key] for key in ("id", "symbol", "interval", "as_of", "sample_count", "analysis", "plan")}
        else:
            result = service.backtest(config, days=args.days, progress=progress)
            summary = {key: result[key] for key in ("id", "symbol", "interval", "coverage", "summary")}
        print(json.dumps(summary, indent=2, allow_nan=False))
        kind = "forecasts" if args.command == "forecast" else "backtests"
        print("Saved:", service.directory / kind / (result["id"] + ".json"))
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
