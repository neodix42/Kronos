"""Account-free Bybit V5 candles, with UTC opens and completed candles only.

The spot market is explicit: Bybit otherwise defaults to linear contracts.
History pages backwards because the API returns newest candles first. Both
volume columns are normalized to base volume and quote turnover, including for
inverse contracts whose API units are reversed.
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import requests

from market.binance import BinanceError, next_open, to_milliseconds


_INTERVALS = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1d": "D", "1w": "W", "1M": "M",
}
SUPPORTED_INTERVALS = tuple(_INTERVALS)
_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount", "close_time"]
_VALUE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
MAX_CANDLES = 1_000_000


class BybitError(RuntimeError):
    """Invalid input, malformed data, or a public Bybit API failure."""


def _milliseconds(value: pd.Timestamp | datetime | int | str) -> int:
    try:
        return to_milliseconds(value)
    except BinanceError as exc:
        raise BybitError(str(exc)) from exc


class BybitClient:
    """Public Bybit candles for an explicit spot, linear, or inverse market.

    ``start`` is inclusive and ``end`` is an exclusive cutoff (default: server
    time). Only candles already closed at the cutoff are returned. Missing or
    prelisting history is returned as available, so callers can report coverage
    notices without fabricating candles. No authentication or account is used.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(
        self, category: str = "spot", session: requests.Session | None = None,
        timeout: float = 20, max_retries: int = 3,
    ) -> None:
        if category not in ("spot", "linear", "inverse"):
            raise BybitError("Bybit category must be spot, linear, or inverse.")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise BybitError("max_retries must be a nonnegative integer.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise BybitError("timeout must be a positive finite number.")
        self.category = category
        self.session = session if session is not None else requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def _retry_delay(response: requests.Response, attempt: int, code: int | None) -> float:
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
        # On retCode 10006, Bybit supplies an absolute millisecond reset time.
        if code == 10006 and response.headers.get("X-Bapi-Limit-Reset-Timestamp"):
            try:
                delay = max(delay, float(response.headers["X-Bapi-Limit-Reset-Timestamp"]) / 1000 - time.time())
            except ValueError:
                pass
        return delay

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(self.BASE_URL + path, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise BybitError(f"Bybit public API connection failed: {exc}") from exc

            payload = None
            try:
                payload = response.json()
            except ValueError:
                pass
            code = payload.get("retCode") if isinstance(payload, dict) else None
            if response.status_code == 429 or 500 <= response.status_code < 600 or code in (10006, 10016):
                delay = self._retry_delay(response, attempt, code)
                if attempt < self.max_retries and math.isfinite(delay) and delay <= 60:
                    time.sleep(delay)
                    continue
                raise BybitError(
                    f"Bybit public API HTTP {response.status_code}, retCode {code}; "
                    f"retry after {response.headers.get('Retry-After') or str(delay) + ' seconds'}."
                )
            if response.status_code == 403:
                raise BybitError("Bybit public API HTTP 403: access may be restricted in this region or by the exchange.")
            if response.status_code != 200:
                detail = payload.get("retMsg", "Request failed") if isinstance(payload, dict) else "Request failed"
                raise BybitError(f"Bybit public API HTTP {response.status_code}: {detail}")
            if not isinstance(payload, dict) or isinstance(code, bool) or not isinstance(code, int):
                raise BybitError("Bybit returned an invalid public API response.")
            if code != 0:
                raise BybitError(f"Bybit public API retCode {code}: {payload.get('retMsg', 'Request failed')}")
            return payload
        raise BybitError("Bybit public API retry limit reached.")

    def server_time(self) -> int:
        payload = self._get("/v5/market/time")
        value = payload.get("time")
        if isinstance(value, int) and not isinstance(value, bool):
            return _milliseconds(value)
        result = payload.get("result")
        if isinstance(result, dict):
            for key in ("timeNano", "timeSecond"):
                value = result.get(key)
                if isinstance(value, str) and value.isdecimal():
                    number = int(value)
                    return _milliseconds(number // 1_000_000 if key == "timeNano" else number * 1000)
        raise BybitError("Bybit returned an invalid server time.")

    def fetch_candles(
        self, symbol: str, interval: str, *,
        start: pd.Timestamp | datetime | int | str,
        end: pd.Timestamp | datetime | int | str | None = None,
    ) -> pd.DataFrame:
        pages = []
        count = 0
        for page in self.iter_candle_pages(symbol, interval, start=start, end=end):
            count += len(page)
            if count > MAX_CANDLES:
                raise BybitError(
                    f"Bybit history exceeds {MAX_CANDLES:,} candles in memory; "
                    "choose a shorter range or use iter_candle_pages for streaming exports."
                )
            pages.append(page)
        if not pages:
            return self._frame([], interval)
        return pd.concat(pages, ignore_index=True).sort_values("timestamps").reset_index(drop=True)

    def iter_candle_pages(
        self, symbol: str, interval: str, *,
        start: pd.Timestamp | datetime | int | str,
        end: pd.Timestamp | datetime | int | str | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Yield validated closed candles with bounded memory and no total cap.

        Each page is sorted oldest first, but pages themselves run backwards
        through history. A streaming exporter must order the final output.
        """
        if interval not in SUPPORTED_INTERVALS:
            raise BybitError(f"Unsupported Bybit interval {interval!r}; choose {', '.join(SUPPORTED_INTERVALS)}.")
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]{2,30}", symbol.strip().upper()):
            raise BybitError("symbol must be a Bybit symbol such as BTCUSDT.")
        symbol = symbol.strip().upper()
        start_ms = _milliseconds(start)
        end_ms = _milliseconds(end) if end is not None else None
        if end_ms is not None and start_ms >= end_ms:
            raise BybitError("start must be earlier than the exclusive end date.")
        now = self.server_time()
        cutoff = min(end_ms, now) if end_ms is not None else now
        if start_ms >= cutoff:
            raise BybitError("start must be earlier than Bybit server time and the end date.")
        cursor = cutoff - 1
        while cursor >= start_ms:
            page = self._page(symbol, interval, start=start_ms, end=cursor, limit=1000)
            if not page:
                break
            frame = self._frame(page, interval)
            earliest = _milliseconds(frame["timestamps"].iloc[0])
            latest = _milliseconds(frame["timestamps"].iloc[-1])
            # Bybit rounds a mid-candle start down and includes that candle.
            # Permit only this overlapping boundary candle, then exclude it.
            overlaps_start = earliest < start_ms < _milliseconds(next_open(earliest, interval))
            if (earliest < start_ms and not overlaps_start) or latest > cursor:
                raise BybitError("Bybit returned candles outside the requested time range; pagination did not advance.")
            closed = frame.loc[
                (frame["timestamps"] >= pd.Timestamp(start_ms, unit="ms", tz="UTC"))
                & (frame["close_time"] < pd.Timestamp(cutoff, unit="ms", tz="UTC"))
            ].reset_index(drop=True)
            if not closed.empty:
                yield closed
            cursor = earliest - 1
            if len(page) < 1000:
                break

    def _page(self, symbol: str, interval: str, **params: int) -> list[list[Any]]:
        payload = self._get("/v5/market/kline", {
            "category": self.category, "symbol": symbol, "interval": _INTERVALS[interval], **params,
        })
        result = payload.get("result")
        if (
            not isinstance(result, dict) or result.get("symbol") != symbol
            or result.get("category") != self.category or not isinstance(result.get("list"), list)
            or len(result["list"]) > params.get("limit", 1000)
        ):
            raise BybitError("Bybit returned an invalid candle response or the wrong market.")
        return result["list"]

    def _frame(self, rows: list[list[Any]], interval: str) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame({
                **{column: pd.Series(dtype="float64") for column in _VALUE_COLUMNS},
                "timestamps": pd.Series(dtype="datetime64[ns, UTC]"),
                "close_time": pd.Series(dtype="datetime64[ns, UTC]"),
            })[_COLUMNS]
        try:
            if any(not isinstance(row, (list, tuple)) or len(row) < 7 for row in rows):
                raise ValueError("A candle is missing required fields.")
            frame = pd.DataFrame([list(row[:7]) for row in rows], columns=_COLUMNS[:-1])
            for column in _VALUE_COLUMNS:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
            if self.category == "inverse":
                frame[["volume", "amount"]] = frame[["amount", "volume"]].to_numpy()
            values = pd.to_numeric(frame["timestamps"], errors="raise")
            if not np.isfinite(values).all() or (values % 1 != 0).any() or (values < 0).any():
                raise ValueError("Invalid candle timestamp.")
            frame["timestamps"] = pd.to_datetime(values, unit="ms", utc=True, errors="raise")
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
            if not frame["timestamps"].is_monotonic_decreasing:
                raise ValueError("Bybit candle timestamps are not in descending order.")
            frame = frame.drop_duplicates().sort_values("timestamps").reset_index(drop=True)
            if frame["timestamps"].duplicated().any():
                raise ValueError("Conflicting duplicate candles.")
            if interval == "1M":
                frame["close_time"] = [next_open(stamp, interval) - pd.Timedelta(milliseconds=1) for stamp in frame["timestamps"]]
            else:
                frame["close_time"] = frame["timestamps"] + pd.Timedelta(interval) - pd.Timedelta(milliseconds=1)
            return frame[_COLUMNS]
        except (TypeError, ValueError, OverflowError, IndexError) as exc:
            raise BybitError(f"Invalid Bybit candle data: {exc}") from exc
