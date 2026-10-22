"""Public spot observations for local paper accounts; no trading endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import math

import pandas as pd

from .binance import BinanceClient
from .bybit import BybitClient


def utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


class PublicMarketFeed:
    """Observe a last-price snapshot and completed OHLC bars from one market."""

    def __init__(self, market, *, clock=None, client=None):
        self.market = dict(market)
        if self.market.get("exchange") not in ("binance", "bybit") or self.market.get("category", "spot") != "spot":
            raise ValueError("Automatic paper trading currently supports Binance/Bybit spot markets only.")
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.client = client or (BinanceClient(timeout=8, max_retries=1) if self.market["exchange"] == "binance"
                                 else BybitClient(category="spot", timeout=8, max_retries=1))

    def quote(self):
        symbol = self.market["symbol"]
        if self.market["exchange"] == "bybit":
            payload = self.client._get("/v5/market/tickers", {"category": "spot", "symbol": symbol})
            result = payload.get("result", {}) if isinstance(payload, dict) else {}
            if not isinstance(result, dict):
                raise ValueError("Bybit returned an invalid public ticker result.")
            rows = result.get("list", [])
            if (result.get("category") != "spot" or not isinstance(rows, list) or len(rows) != 1
                    or not isinstance(rows[0], dict) or rows[0].get("symbol") != symbol):
                raise ValueError("Bybit returned a mismatched public spot ticker.")
            price = rows[0].get("lastPrice")
            stamp = payload.get("time")
        else:
            payload = self.client._get("/api/v3/ticker/price", {"symbol": symbol})
            if not isinstance(payload, dict) or payload.get("symbol") != symbol:
                raise ValueError("Binance returned a mismatched public spot ticker.")
            price = payload.get("price")
            # Binance's price ticker has no timestamp. Its public server clock
            # timestamps this observation; it is not a last-trade timestamp.
            stamp = self.client.server_time()
        try:
            price = float(price)
        except (TypeError, ValueError) as exc:
            raise ValueError("The public ticker has an invalid last price.") from exc
        if not math.isfinite(price) or price <= 0 or isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0:
            raise ValueError("The public ticker has an invalid price or server timestamp.")
        return {"price": price, "timestamp": pd.Timestamp(stamp, unit="ms", tz="UTC").isoformat(),
                "observed_at": utc(self.clock()).isoformat(), "timestamp_kind": "exchange_snapshot_observation",
                "source": "public_last_price", **self.market}

    def candles(self, *, start, end):
        """Fetch completed bars for exit recovery without filling missing slots."""
        start, end = utc(start), utc(end)
        if start >= end:
            return pd.DataFrame(columns=["timestamps", "open", "high", "low", "close", "volume", "amount", "close_time"])
        if self.market["exchange"] == "binance":
            from .download import binance_pages

            pages = list(binance_pages(self.client, self.market["symbol"], self.market["interval"], start, end))
            return (pd.concat(pages, ignore_index=True).sort_values("timestamps").reset_index(drop=True)
                    if pages else pd.DataFrame(columns=["timestamps", "open", "high", "low", "close", "volume", "amount", "close_time"]))
        return self.client.fetch_candles(self.market["symbol"], self.market["interval"], start=start, end=end)
