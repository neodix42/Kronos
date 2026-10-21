"""Public data ingestion tests; never require a network connection or API key."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from market.binance import BinanceClient, BinanceError, future_timestamps, next_open, to_milliseconds


BASE = pd.Timestamp("2025-01-01", tz="UTC")

# Binance's real historical maintenance candle has an official nonstandard
# close time. Download exports must preserve it while forecasting stays strict.
SHORT_HISTORY_CANDLE = [
    1613012400000, "44582.07000000", "44582.07000000", "44582.07000000", "44582.07000000",
    "0.00000000", 1613014854773, "0.00000000", 0, "0.00000000", "0.00000000", "0",
]


def candle(stamp, interval="1h", **changes):
    start = to_milliseconds(stamp)
    row = [start, "100", "110", "90", "105", "4", to_milliseconds(next_open(stamp, interval)) - 1, "410", 5, "2", "205", "0"]
    indexes = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5, "close_time": 6, "amount": 7}
    for name, value in changes.items():
        row[indexes[name]] = value
    return row


def mocked_client(now, pages):
    client = BinanceClient()
    client.server_time = Mock(return_value=to_milliseconds(now))
    client._page = Mock(side_effect=pages)
    return client


def test_latest_excludes_unclosed_candle_and_uses_quote_volume():
    rows = [candle(BASE + pd.Timedelta(hours=index)) for index in range(4)]
    client = mocked_client(BASE + pd.Timedelta(hours=3, minutes=20), [rows])
    result = client.fetch_candles("btcusdt", "1h", limit=3)

    assert len(result) == 3
    assert list(result["timestamps"]) == [BASE + pd.Timedelta(hours=index) for index in range(3)]
    assert str(result.timestamps.dtype) == "datetime64[ns, UTC]"
    assert result.amount.tolist() == [410.0] * 3
    assert result.volume.tolist() == [4.0] * 3
    client._page.assert_called_once_with("BTCUSDT", "1h", endTime=to_milliseconds(BASE + pd.Timedelta(hours=3, minutes=20)) - 1, limit=4)


def test_history_paginates_inclusive_start_exclusive_end_without_duplicates():
    rows = [candle(BASE + pd.Timedelta(hours=index)) for index in range(1003)]
    cutoff = BASE + pd.Timedelta(hours=1003)
    client = mocked_client(cutoff, [rows[:1000], rows[1000:]])
    result = client.fetch_candles("BTCUSDT", "1h", start=BASE, end=cutoff)

    assert len(result) == 1003
    assert result.timestamps.iloc[-1] == cutoff - pd.Timedelta(hours=1)
    assert client._page.call_args_list[1].kwargs["startTime"] == to_milliseconds(BASE + pd.Timedelta(hours=1000))
    assert client._page.call_args_list[1].kwargs["endTime"] == to_milliseconds(cutoff) - 1


def test_latest_paginates_backwards_for_large_limit():
    rows = [candle(BASE + pd.Timedelta(hours=index)) for index in range(1200)]
    client = mocked_client(BASE + pd.Timedelta(hours=1200), [rows[200:], rows[:200]])
    result = client.fetch_candles("BTCUSDT", "1h", limit=1200)
    assert len(result) == 1200
    assert result.timestamps.iloc[0] == BASE
    assert client._page.call_args_list[1].kwargs["endTime"] == to_milliseconds(BASE + pd.Timedelta(hours=200)) - 1


def test_end_date_filters_candle_that_was_still_open_at_cutoff():
    rows = [candle(BASE), candle(BASE + pd.Timedelta(hours=1))]
    client = mocked_client(BASE + pd.Timedelta(days=1), [rows])
    result = client.fetch_candles("BTCUSDT", "1h", start=BASE, end=BASE + pd.Timedelta(hours=1, minutes=30))
    assert result.timestamps.tolist() == [BASE]


def test_monthly_intervals_respect_calendar_and_leap_years():
    future = future_timestamps("2024-01-01", "1M", 3)
    assert future.tolist() == [pd.Timestamp(value, tz="UTC") for value in ["2024-02-01", "2024-03-01", "2024-04-01"]]
    assert next_open("2024-02-01", "1M") == pd.Timestamp("2024-03-01", tz="UTC")
    client = mocked_client("2024-04-01", [[candle("2024-02-01", "1M"), candle("2024-03-01", "1M")]])
    result = client.fetch_candles("BTCUSDT", "1M", limit=2)
    assert result.close_time.iloc[0] == pd.Timestamp("2024-02-29 23:59:59.999", tz="UTC")


@pytest.mark.parametrize("kwargs", [
    {"symbol": "BTC/USDT", "interval": "1h"},
    {"symbol": "BTCUSDT", "interval": "7m"},
    {"symbol": "BTCUSDT", "interval": "1h", "limit": 0},
    {"symbol": "BTCUSDT", "interval": "1h", "limit": True},
    {"symbol": "BTCUSDT", "interval": "1h", "start": "2025-02-01", "end": "2025-01-01"},
])
def test_invalid_parameters_fail_before_network(kwargs):
    client = BinanceClient(session=Mock())
    with pytest.raises(BinanceError):
        client.fetch_candles(**kwargs)
    client.session.get.assert_not_called()


@pytest.mark.parametrize("changes", [
    {"high": 99}, {"low": 101}, {"close": 0}, {"volume": -1},
    {"amount": -1}, {"open": "nan"}, {"high": np.inf}, {"close_time": 1},
])
def test_invalid_numeric_and_ohlc_data_rejected(changes):
    client = mocked_client(BASE + pd.Timedelta(hours=1), [[candle(BASE, **changes)]])
    with pytest.raises(BinanceError, match="Invalid Binance candle data"):
        client.fetch_candles("BTCUSDT", "1h", limit=1)


def test_gap_and_unsorted_rows_rejected_instead_of_fabricated():
    for rows in [
        [candle(BASE), candle(BASE + pd.Timedelta(hours=2))],
        [candle(BASE + pd.Timedelta(hours=1)), candle(BASE)],
    ]:
        client = mocked_client(BASE + pd.Timedelta(days=1), [rows])
        with pytest.raises(BinanceError, match="gaps|not sorted"):
            client.fetch_candles("BTCUSDT", "1h", start=BASE)


def test_short_historical_candle_is_opt_in_and_preserves_official_close():
    with pytest.raises(BinanceError, match="close time"):
        BinanceClient._frame([SHORT_HISTORY_CANDLE], "1h")

    result = BinanceClient._frame([SHORT_HISTORY_CANDLE], "1h", allow_short_candles=True)
    assert result.timestamps.iloc[0] == pd.Timestamp(1613012400000, unit="ms", tz="UTC")
    assert result.close_time.iloc[0] == pd.Timestamp(1613014854773, unit="ms", tz="UTC")
    assert result.volume.tolist() == [0.0]
    assert result.amount.tolist() == [0.0]
    assert result.close.tolist() == [44582.07]


@pytest.mark.parametrize("invalid_close", [1613012399999, 1613016000000])
def test_allow_short_candles_still_rejects_close_outside_interval(invalid_close):
    row = SHORT_HISTORY_CANDLE.copy()
    row[6] = invalid_close
    with pytest.raises(BinanceError, match="close time"):
        BinanceClient._frame([row], "1h", allow_short_candles=True)


def test_identical_duplicates_are_dropped_and_conflicting_duplicates_rejected():
    rows = [candle(BASE), candle(BASE), candle(BASE + pd.Timedelta(hours=1))]
    client = mocked_client(BASE + pd.Timedelta(days=1), [rows])
    assert len(client.fetch_candles("BTCUSDT", "1h", start=BASE)) == 2
    client = mocked_client(BASE + pd.Timedelta(days=1), [[candle(BASE), candle(BASE, close=101)]])
    with pytest.raises(BinanceError, match="Conflicting duplicate"):
        client.fetch_candles("BTCUSDT", "1h", start=BASE)


def test_insufficient_latest_history_is_explicit_error():
    client = mocked_client(BASE + pd.Timedelta(days=1), [[candle(BASE)]])
    with pytest.raises(BinanceError, match="only 1 closed"):
        client.fetch_candles("NEWUSDT", "1h", limit=512)


def test_oversized_history_is_rejected_before_fetching_pages():
    client = mocked_client("2026-01-01", [])
    with pytest.raises(BinanceError, match="exceeds 1,000,000 candles"):
        client.fetch_candles("BTCUSDT", "1s", start="2025-01-01", end="2026-01-01")
    client._page.assert_not_called()


def response(status, payload, headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = payload
    return result


def test_rate_limit_retry_honors_retry_after_and_public_host():
    session = Mock()
    session.get.side_effect = [response(429, {"msg": "slow down"}, {"Retry-After": "3"}), response(200, {"serverTime": to_milliseconds(BASE)})]
    client = BinanceClient(session=session)
    with patch("market.binance.time.sleep") as sleep:
        assert client.server_time() == to_milliseconds(BASE)
    sleep.assert_called_once_with(3.0)
    assert session.get.call_args.args[0] == "https://data-api.binance.vision/api/v3/time"


def test_long_rate_limit_fails_without_retrying_before_allowed_time():
    session = Mock()
    session.get.return_value = response(418, {}, {"Retry-After": "120"})
    with patch("market.binance.time.sleep") as sleep, pytest.raises(BinanceError, match="retry after 120"):
        BinanceClient(session=session).server_time()
    sleep.assert_not_called()
    assert session.get.call_count == 1


def test_exchange_error_and_symbol_listing():
    session = Mock()
    session.get.return_value = response(400, {"code": -1121, "msg": "Invalid symbol."})
    with pytest.raises(BinanceError, match="Invalid symbol"):
        BinanceClient(session=session)._page("FAKEUSDT", "1h", limit=10)
    session.get.return_value = response(200, {"symbols": [
        {"symbol": "BTCUSDT", "status": "TRADING", "isSpotTradingAllowed": True},
        {"symbol": "ETHUSDT", "status": "BREAK", "isSpotTradingAllowed": True},
        {"symbol": "FAKEUSDT", "status": "TRADING", "isSpotTradingAllowed": False},
    ]})
    assert BinanceClient(session=session).symbols() == ["BTCUSDT"]


def test_naive_times_are_utc_and_aware_times_convert():
    assert to_milliseconds("2025-01-01") == to_milliseconds(BASE)
    assert to_milliseconds("2025-01-01T04:00:00+04:00") == to_milliseconds(BASE)
    with pytest.raises(BinanceError):
        to_milliseconds(float("nan"))
