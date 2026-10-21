"""Bybit public data ingestion; deterministic fixtures need no API keys."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from market.binance import to_milliseconds
from market.bybit import BybitClient, BybitError


BASE = pd.Timestamp("2025-01-01", tz="UTC")


def candle(stamp, **changes):
    row = [str(to_milliseconds(stamp)), "100", "110", "90", "105", "4", "410"]
    indexes = {"timestamp": 0, "open": 1, "high": 2, "low": 3, "close": 4, "volume": 5, "amount": 6}
    for name, value in changes.items():
        row[indexes[name]] = value
    return row


def mocked_client(now, pages, category="spot"):
    client = BybitClient(category=category)
    client.server_time = Mock(return_value=to_milliseconds(now))
    client._page = Mock(side_effect=pages)
    return client


def response(status, payload, headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = payload
    return result


def test_history_pages_backwards_without_skipping_middle_and_returns_ascending():
    rows = [candle(BASE + pd.Timedelta(hours=index)) for index in range(1203)]
    cutoff = BASE + pd.Timedelta(hours=1203)
    client = mocked_client(cutoff, [rows[203:][::-1], rows[:203][::-1]])
    result = client.fetch_candles("btcusdt", "1h", start=BASE)

    assert len(result) == 1203
    assert result.timestamps.tolist() == [BASE + pd.Timedelta(hours=index) for index in range(1203)]
    assert str(result.timestamps.dtype) == "datetime64[ns, UTC]"
    assert result.volume.eq(4).all() and result.amount.eq(410).all()
    assert client._page.call_args_list[0].args == ("BTCUSDT", "1h")
    assert client._page.call_args_list[1].kwargs == {
        "start": to_milliseconds(BASE),
        "end": to_milliseconds(BASE + pd.Timedelta(hours=203)) - 1,
        "limit": 1000,
    }


def test_excludes_current_candle_and_historical_candle_not_closed_at_cutoff():
    rows = [candle(BASE + pd.Timedelta(hours=index)) for index in reversed(range(3))]
    client = mocked_client(BASE + pd.Timedelta(hours=2, minutes=30), [rows])
    result = client.fetch_candles("BTCUSDT", "1h", start=BASE)
    assert result.timestamps.tolist() == [BASE, BASE + pd.Timedelta(hours=1)]
    client = mocked_client(BASE + pd.Timedelta(days=1), [[candle(BASE + pd.Timedelta(hours=1)), candle(BASE)]])
    result = client.fetch_candles("BTCUSDT", "1h", start=BASE, end=BASE + pd.Timedelta(hours=1, minutes=30))
    assert result.timestamps.tolist() == [BASE]


def test_calendar_month_close_respects_leap_year_and_exact_boundary():
    rows = [candle("2024-03-01"), candle("2024-02-01")]
    client = mocked_client("2024-03-01T00:00:00.001Z", [rows])
    result = client.fetch_candles("BTCUSDT", "1M", start="2024-02-01")
    assert result.timestamps.tolist() == [pd.Timestamp("2024-02-01", tz="UTC")]
    assert result.close_time.iloc[0] == pd.Timestamp("2024-02-29T23:59:59.999Z")


def test_api_rounding_start_down_excludes_the_partial_start_candle():
    rows = [candle(BASE + pd.Timedelta(hours=1)), candle(BASE)]
    client = mocked_client(BASE + pd.Timedelta(hours=2), [rows])
    result = client.fetch_candles("BTCUSDT", "1h", start=BASE + pd.Timedelta(minutes=30))
    assert result.timestamps.tolist() == [BASE + pd.Timedelta(hours=1)]


def test_missing_prelisting_history_and_interior_gaps_remain_available():
    client = mocked_client(BASE + pd.Timedelta(hours=4), [[candle(BASE + pd.Timedelta(hours=3)), candle(BASE + pd.Timedelta(hours=1))]])
    result = client.fetch_candles("NEWUSDT", "1h", start=BASE)
    assert result.timestamps.tolist() == [BASE + pd.Timedelta(hours=1), BASE + pd.Timedelta(hours=3)]
    client = mocked_client(BASE + pd.Timedelta(hours=4), [[]])
    result = client.fetch_candles("NEWUSDT", "1h", start=BASE)
    assert result.empty
    assert result.columns.tolist() == ["timestamps", "open", "high", "low", "close", "volume", "amount", "close_time"]


@pytest.mark.parametrize("category, volume, amount", [("spot", 4, 410), ("linear", 4, 410), ("inverse", 410, 4)])
def test_volumes_are_base_units_and_amount_quote_units(category, volume, amount):
    client = mocked_client(BASE + pd.Timedelta(hours=1), [[candle(BASE)]], category=category)
    result = client.fetch_candles("BTCUSDT", "1h", start=BASE)
    assert result.volume.tolist() == [volume]
    assert result.amount.tolist() == [amount]


def test_identical_duplicate_rows_are_removed_but_conflicting_rows_fail():
    client = mocked_client(BASE + pd.Timedelta(hours=1), [[candle(BASE), candle(BASE)]])
    assert len(client.fetch_candles("BTCUSDT", "1h", start=BASE)) == 1
    client = mocked_client(BASE + pd.Timedelta(hours=1), [[candle(BASE), candle(BASE, close="104")]])
    with pytest.raises(BybitError, match="Conflicting duplicate"):
        client.fetch_candles("BTCUSDT", "1h", start=BASE)


@pytest.mark.parametrize("row, message", [
    (candle(BASE, high="99"), "OHLC bounds"),
    (candle(BASE, low="101"), "OHLC bounds"),
    (candle(BASE, close="NaN"), "parse|finite"),
    (candle(BASE, amount="inf"), "finite"),
    (candle(BASE, open="0"), "positive"),
    (candle(BASE, volume="-1"), "nonnegative"),
    (candle(BASE, timestamp="1.5"), "timestamp"),
    (candle(BASE, timestamp="-1"), "timestamp"),
    (["missing", "fields"], "missing required"),
])
def test_invalid_candles_are_rejected(row, message):
    client = mocked_client(BASE + pd.Timedelta(hours=1), [[row]])
    with pytest.raises(BybitError, match=message):
        client.fetch_candles("BTCUSDT", "1h", start=BASE)


@pytest.mark.parametrize("rows", [
    [candle(BASE + pd.Timedelta(hours=2))],
    [candle(BASE - pd.Timedelta(hours=1))],
])
def test_out_of_range_or_nonadvancing_page_is_rejected(rows):
    client = mocked_client(BASE + pd.Timedelta(hours=2), [rows])
    with pytest.raises(BybitError, match="outside.*pagination did not advance"):
        client.fetch_candles("BTCUSDT", "1h", start=BASE)


def test_page_order_is_validated_before_pagination():
    client = mocked_client(BASE + pd.Timedelta(hours=2), [[candle(BASE), candle(BASE + pd.Timedelta(hours=1))]])
    with pytest.raises(BybitError, match="descending order"):
        client.fetch_candles("BTCUSDT", "1h", start=BASE)


@pytest.mark.parametrize("kwargs", [
    {"symbol": "BTC/USDT", "interval": "1h", "start": BASE},
    {"symbol": "BTCUSDT", "interval": "8h", "start": BASE},
    {"symbol": "BTCUSDT", "interval": "1h", "start": "bad date"},
    {"symbol": "BTCUSDT", "interval": "1h", "start": BASE, "end": BASE},
    {"symbol": "BTCUSDT", "interval": "1h", "start": BASE + pd.Timedelta(days=2)},
])
def test_invalid_input_fails_before_candle_request(kwargs):
    client = mocked_client(BASE + pd.Timedelta(days=1), [])
    with pytest.raises(BybitError):
        client.fetch_candles(**kwargs)
    client._page.assert_not_called()


def test_in_memory_limit_fails_explicitly_but_streaming_has_no_total_cap():
    rows = [candle(BASE + pd.Timedelta(hours=1)), candle(BASE)]
    client = mocked_client(BASE + pd.Timedelta(hours=2), [rows])
    with patch("market.bybit.MAX_CANDLES", 1), pytest.raises(BybitError, match="exceeds 1 candles"):
        client.fetch_candles("BTCUSDT", "1h", start=BASE)
    client = mocked_client(BASE + pd.Timedelta(hours=2), [rows])
    with patch("market.bybit.MAX_CANDLES", 1):
        pages = list(client.iter_candle_pages("BTCUSDT", "1h", start=BASE))
    assert len(pages) == 1 and len(pages[0]) == 2


@pytest.mark.parametrize("interval, api_interval", [("1m", "1"), ("1h", "60"), ("1d", "D"), ("1w", "W"), ("1M", "M")])
def test_public_endpoint_and_explicit_spot_market(interval, api_interval):
    session = Mock()
    session.get.return_value = response(200, {"retCode": 0, "result": {"category": "spot", "symbol": "BTCUSDT", "list": []}})
    client = BybitClient(session=session)
    assert client._page("BTCUSDT", interval, start=1, end=2, limit=1000) == []
    session.get.assert_called_once_with(
        "https://api.bybit.com/v5/market/kline",
        params={"category": "spot", "symbol": "BTCUSDT", "interval": api_interval, "start": 1, "end": 2, "limit": 1000},
        timeout=20,
    )


@pytest.mark.parametrize("payload", [
    {"retCode": 0, "time": to_milliseconds(BASE)},
    {"retCode": 0, "result": {"timeNano": str(to_milliseconds(BASE) * 1_000_000)}},
    {"retCode": 0, "result": {"timeSecond": str(to_milliseconds(BASE) // 1000)}},
])
def test_server_time_formats(payload):
    session = Mock()
    session.get.return_value = response(200, payload)
    assert BybitClient(session=session).server_time() == to_milliseconds(BASE)


@pytest.mark.parametrize("status, code", [(429, 10006), (200, 10006), (200, 10016), (503, None)])
def test_retryable_http_and_api_errors_honor_retry_after(status, code):
    session = Mock()
    session.get.side_effect = [
        response(status, {"retCode": code}, {"Retry-After": "3"}),
        response(200, {"retCode": 0, "time": to_milliseconds(BASE)}),
    ]
    with patch("market.bybit.time.sleep") as sleep:
        assert BybitClient(session=session).server_time() == to_milliseconds(BASE)
    sleep.assert_called_once_with(3.0)


def test_bybit_reset_timestamp_is_observed():
    session = Mock()
    session.get.side_effect = [
        response(200, {"retCode": 10006}, {"X-Bapi-Limit-Reset-Timestamp": "104000"}),
        response(200, {"retCode": 0, "time": to_milliseconds(BASE)}),
    ]
    with patch("market.bybit.time.sleep") as sleep, patch("market.bybit.time.time", return_value=100):
        assert BybitClient(session=session).server_time() == to_milliseconds(BASE)
    sleep.assert_called_once_with(4.0)


def test_long_wait_fails_without_retrying_early():
    session = Mock()
    session.get.return_value = response(429, {"retCode": 10006}, {"Retry-After": "120"})
    with patch("market.bybit.time.sleep") as sleep, pytest.raises(BybitError, match="retry after 120"):
        BybitClient(session=session).server_time()
    sleep.assert_not_called()
    assert session.get.call_count == 1


def test_transport_errors_retry_then_raise_clear_error():
    session = Mock()
    session.get.side_effect = requests.ConnectionError("unavailable")
    with patch("market.bybit.time.sleep") as sleep, pytest.raises(BybitError, match="connection failed"):
        BybitClient(session=session, max_retries=1).server_time()
    sleep.assert_called_once_with(1)
    assert session.get.call_count == 2


@pytest.mark.parametrize("status,payload,message", [
    (403, None, "restricted"),
    (400, {"retMsg": "invalid symbol"}, "invalid symbol"),
    (200, {"retCode": 10001, "retMsg": "invalid symbol"}, "invalid symbol"),
    (200, [], "invalid public API response"),
    (200, {"retCode": True}, "invalid public API response"),
    (200, {"retCode": 0, "result": {}}, "invalid server time"),
])
def test_api_and_malformed_responses_fail_clearly(status, payload, message):
    session = Mock()
    session.get.return_value = response(status, payload)
    with pytest.raises(BybitError, match=message):
        BybitClient(session=session).server_time()


def test_candle_response_rejects_wrong_market():
    session = Mock()
    session.get.return_value = response(200, {"retCode": 0, "result": {"category": "linear", "symbol": "BTCUSDT", "list": []}})
    with pytest.raises(BybitError, match="wrong market"):
        BybitClient(session=session)._page("BTCUSDT", "1h")
