"""Unauthenticated Binance spot candles with UTC, closed-candle semantics.

History uses Binance's public market-data host. Prices are Binance spot prices;
another exchange or TradingView's synthetic feeds can have different candles.
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

import numpy as np
import pandas as pd
import requests


SUPPORTED_INTERVALS = (
    "1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h",
    "8h", "12h", "1d", "3d", "1w", "1M",
)
_FIXED_INTERVALS = {
    interval: pd.Timedelta(interval)
    for interval in SUPPORTED_INTERVALS if interval != "1M"
}
_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount", "close_time"]
_VALUE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
MAX_CANDLES = 1_000_000


class BinanceError(RuntimeError):
    """Invalid input, unavailable data, or a public Binance API failure."""


def _validate_interval(interval: str) -> None:
    if interval not in SUPPORTED_INTERVALS:
        raise BinanceError(f"Unsupported Binance interval {interval!r}; choose {', '.join(SUPPORTED_INTERVALS)}.")


def to_milliseconds(value: pd.Timestamp | datetime | int | str) -> int:
    """Convert a date or integer milliseconds to UTC; naive dates mean UTC."""
    try:
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            result = int(value)
            pd.Timestamp(result, unit="ms", tz="UTC")
        else:
            if isinstance(value, (float, bool)):
                raise ValueError("Use integer milliseconds or an ISO date.")
            stamp = pd.Timestamp(value)
            if pd.isna(stamp):
                raise ValueError("Timestamp is missing.")
            stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
            result = stamp.value // 1_000_000
        if result < 0:
            raise ValueError("Timestamp precedes 1970.")
        return result
    except (TypeError, ValueError, OverflowError) as exc:
        raise BinanceError(f"Invalid UTC date {value!r}: {exc}") from exc


def next_open(timestamp: pd.Timestamp | datetime | int | str, interval: str) -> pd.Timestamp:
    """Return the next UTC candle open, preserving calendar months for 1M."""
    _validate_interval(interval)
    stamp = pd.Timestamp(to_milliseconds(timestamp), unit="ms", tz="UTC")
    return stamp + (pd.DateOffset(months=1) if interval == "1M" else _FIXED_INTERVALS[interval])


def future_timestamps(
    last_open: pd.Timestamp | datetime | int | str, interval: str, count: int,
) -> pd.Series:
    """UTC opens after the final observed candle, excluding that candle itself."""
    _validate_interval(interval)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_CANDLES:
        raise BinanceError(f"count must be an integer between 1 and {MAX_CANDLES:,}.")
    first = next_open(last_open, interval)
    frequency = pd.DateOffset(months=1) if interval == "1M" else _FIXED_INTERVALS[interval]
    return pd.Series(pd.date_range(first, periods=count, freq=frequency), name="timestamps")


class BinanceClient:
    """Requests client restricted to Binance's account-free, public data API.

    ``fetch_candles(..., limit=512)`` returns exactly the latest 512 *closed*
    candles or fails if the market has insufficient history. ``start`` selects
    forward-paginated history; ``end`` is an exclusive UTC cutoff and defaults
    to Binance server time. Candles still open at the cutoff are excluded.
    Historical results can be shorter for newly listed assets; gaps are errors.
    A limit with start returns at most that many earliest candles in the range.
    """

    BASE_URL = "https://data-api.binance.vision"

    def __init__(
        self, session: requests.Session | None = None, timeout: float = 20, max_retries: int = 3,
    ) -> None:
        self.session = session if session is not None else requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(self.BASE_URL + path, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise BinanceError(f"Binance public API connection failed: {exc}") from exc
            if response.status_code in (418, 429) or 500 <= response.status_code < 600:
                delay = min(2 ** attempt, 8)
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        try:
                            delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - time.time())
                        except (TypeError, ValueError, OverflowError):
                            pass
                if attempt < self.max_retries and math.isfinite(delay) and delay <= 60:
                    time.sleep(delay)
                    continue
                raise BinanceError(
                    f"Binance public API HTTP {response.status_code}; retry after {retry_after or str(delay) + ' seconds'}."
                )
            try:
                payload = response.json()
            except (ValueError, requests.JSONDecodeError) as exc:
                raise BinanceError(f"Binance returned invalid JSON (HTTP {response.status_code}).") from exc
            if response.status_code != 200 or isinstance(payload, dict) and "code" in payload:
                detail = payload.get("msg", "Request failed") if isinstance(payload, dict) else "Request failed"
                raise BinanceError(f"Binance public API HTTP {response.status_code}: {detail}")
            return payload
        raise BinanceError("Binance public API retry limit reached.")

    def server_time(self) -> int:
        payload = self._get("/api/v3/time")
        if not isinstance(payload, dict) or not isinstance(payload.get("serverTime"), int):
            raise BinanceError("Binance returned an invalid server time.")
        return to_milliseconds(payload["serverTime"])

    def symbols(self) -> list[str]:
        payload = self._get("/api/v3/exchangeInfo")
        if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
            raise BinanceError("Binance returned invalid symbol information.")
        return sorted(
            item["symbol"] for item in payload["symbols"]
            if isinstance(item, dict) and item.get("status") == "TRADING"
            and item.get("isSpotTradingAllowed", False) and isinstance(item.get("symbol"), str)
        )

    def fetch_candles(
        self, symbol: str, interval: str, *,
        start: pd.Timestamp | datetime | int | str | None = None,
        end: pd.Timestamp | datetime | int | str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        _validate_interval(interval)
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]{2,30}", symbol.strip().upper()):
            raise BinanceError("symbol must be a Binance spot symbol such as BTCUSDT.")
        symbol = symbol.strip().upper()
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_CANDLES):
            raise BinanceError(f"limit must be an integer between 1 and {MAX_CANDLES:,}.")
        start_ms = to_milliseconds(start) if start is not None else None
        end_ms = to_milliseconds(end) if end is not None else None
        if start_ms is not None and end_ms is not None and start_ms >= end_ms:
            raise BinanceError("start must be earlier than the exclusive end date.")
        now = self.server_time()
        cutoff = min(end_ms, now) if end_ms is not None else now
        if start_ms is not None and start_ms >= cutoff:
            raise BinanceError("start must be earlier than Binance server time and the end date.")
        if start_ms is not None and limit is None:
            minimum_step = pd.Timedelta(days=28) if interval == "1M" else _FIXED_INTERVALS[interval]
            if math.ceil((cutoff - start_ms) / (minimum_step.value / 1_000_000)) > MAX_CANDLES:
                raise BinanceError(f"Requested range exceeds {MAX_CANDLES:,} candles; choose a shorter range or coarser interval.")

        rows = self._latest(symbol, interval, cutoff, limit or 512) if start_ms is None else self._history(
            symbol, interval, start_ms, cutoff, limit,
        )
        frame = self._frame(rows, interval)
        frame = frame.loc[(frame["close_time"] < pd.Timestamp(cutoff, unit="ms", tz="UTC"))]
        if start_ms is not None:
            frame = frame.loc[frame["timestamps"] >= pd.Timestamp(start_ms, unit="ms", tz="UTC")]
            if limit is not None:
                frame = frame.head(limit)
        else:
            wanted = limit or 512
            frame = frame.tail(wanted)
            if len(frame) != wanted:
                raise BinanceError(f"{symbol} has only {len(frame)} closed {interval} candles before the cutoff; {wanted} required.")
        return frame.reset_index(drop=True)

    def _latest(self, symbol: str, interval: str, cutoff: int, wanted: int) -> list[list[Any]]:
        pages: list[list[list[Any]]] = []
        end_time = cutoff - 1
        closed_count = 0
        while closed_count < wanted:
            size = min(1000, wanted - closed_count + 1)
            page = self._page(symbol, interval, endTime=end_time, limit=size)
            if not page:
                break
            self._frame(page, interval)
            pages.append(page)
            closed_count += sum(int(row[6]) < cutoff for row in page)
            first = int(page[0][0])
            if first > end_time:
                raise BinanceError("Binance returned candles beyond the requested cutoff.")
            end_time = first - 1
            if len(page) < size:
                break
        return [row for page in reversed(pages) for row in page]

    def _history(
        self, symbol: str, interval: str, start: int, cutoff: int, limit: int | None,
    ) -> list[list[Any]]:
        rows: list[list[Any]] = []
        cursor = start
        maximum = limit if limit is not None else MAX_CANDLES
        while cursor < cutoff and len(rows) < maximum:
            size = min(1000, maximum - len(rows))
            page = self._page(symbol, interval, startTime=cursor, endTime=cutoff - 1, limit=size)
            if not page:
                break
            self._frame(page, interval)
            if int(page[-1][0]) < cursor:
                raise BinanceError("Binance candle pagination did not advance.")
            rows.extend(page)
            cursor = to_milliseconds(next_open(int(page[-1][0]), interval))
            if len(page) < size:
                break
        return rows

    def _page(self, symbol: str, interval: str, **params: int) -> list[list[Any]]:
        payload = self._get("/api/v3/klines", {"symbol": symbol, "interval": interval, "timeZone": "0", **params})
        if not isinstance(payload, list):
            raise BinanceError("Binance returned an invalid candle response.")
        return payload

    @staticmethod
    def _frame(rows: list[list[Any]], interval: str) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame({
                **{column: pd.Series(dtype="float64") for column in _VALUE_COLUMNS},
                "timestamps": pd.Series(dtype="datetime64[ns, UTC]"),
                "close_time": pd.Series(dtype="datetime64[ns, UTC]"),
            })[_COLUMNS]
        try:
            if any(not isinstance(row, (list, tuple)) or len(row) < 8 for row in rows):
                raise ValueError("A candle is missing required fields.")
            frame = pd.DataFrame([
                [row[0], row[1], row[2], row[3], row[4], row[5], row[7], row[6]]
                for row in rows
            ], columns=_COLUMNS)
            for column in _VALUE_COLUMNS:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
            for column in ("timestamps", "close_time"):
                values = pd.to_numeric(frame[column], errors="raise")
                if not np.isfinite(values).all() or (values % 1 != 0).any() or (values < 0).any():
                    raise ValueError("Invalid candle timestamp.")
                frame[column] = pd.to_datetime(values, unit="ms", utc=True, errors="raise")
            if not np.isfinite(frame[_VALUE_COLUMNS].to_numpy()).all():
                raise ValueError("Candle values must be finite.")
            if (frame[["open", "high", "low", "close"]] <= 0).any().any():
                raise ValueError("Candle prices must be positive.")
            if (frame[["volume", "amount"]] < 0).any().any():
                raise ValueError("Candle volumes must be nonnegative.")
            if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any() or (
                frame["low"] > frame[["open", "close", "high"]].min(axis=1)
            ).any():
                raise ValueError("Candle OHLC bounds are inconsistent.")
            if not frame["timestamps"].is_monotonic_increasing:
                raise ValueError("Candle timestamps are not sorted.")
            frame = frame.drop_duplicates().reset_index(drop=True)
            if frame["timestamps"].duplicated().any():
                raise ValueError("Conflicting duplicate candles.")
            if interval == "1M":
                expected_close = pd.Series([
                    stamp + pd.DateOffset(months=1) - pd.Timedelta(milliseconds=1)
                    for stamp in frame["timestamps"]
                ])
            else:
                expected_close = frame["timestamps"] + _FIXED_INTERVALS[interval] - pd.Timedelta(milliseconds=1)
            if not frame["close_time"].equals(expected_close.rename("close_time")):
                raise ValueError("Candle close time does not match its interval.")
            if len(frame) > 1 and not frame["timestamps"].iloc[1:].reset_index(drop=True).equals(
                (expected_close.iloc[:-1] + pd.Timedelta(milliseconds=1)).reset_index(drop=True).rename("timestamps")
            ):
                raise ValueError("Candle history has gaps; choose a contiguous, supported market history.")
            return frame
        except (TypeError, ValueError, OverflowError, IndexError) as exc:
            raise BinanceError(f"Invalid Binance candle data: {exc}") from exc
