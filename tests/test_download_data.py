"""CSV export and CLI tests using public-API-shaped, entirely offline data."""

import csv
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from market import download
from market.binance import BinanceClient, BinanceError, next_open


BASE = pd.Timestamp("2025-01-01", tz="UTC")


def candle(stamp, interval="1h", *, close="105"):
    stamp = pd.Timestamp(stamp)
    return [download.milliseconds(stamp), "100", "110", "90", close, "4",
            download.milliseconds(next_open(stamp, interval)) - 1, "410", 5, "2", "205", "0"]


def frame(stamps, interval="1h"):
    return BinanceClient._frame([candle(stamp, interval) for stamp in stamps], interval, allow_gaps=True)


def export(pages, output, *, interval="1h", start=BASE, end=None, **kwargs):
    return download.export_pages(
        pages, output, exchange="binance", category="spot", symbol="BTCUSDT",
        interval=interval, start=start, end=end if end is not None else BASE + pd.Timedelta(hours=3),
        **kwargs,
    )


def assert_no_temporary_files(directory):
    assert not list(directory.glob(".candle-*.tmp"))


def test_reversed_pages_export_exact_training_columns_sorted_utc(tmp_path):
    target = tmp_path / "training.csv"
    progress = Mock()
    result = export([
        frame([BASE + pd.Timedelta(hours=2)]),
        frame([BASE, BASE + pd.Timedelta(hours=1)]),
    ], target, progress=progress)

    with target.open(newline="") as source:
        rows = list(csv.reader(source))
    assert rows[0] == ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    assert [row[0] for row in rows[1:]] == [
        "2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z",
    ]
    assert all(row[1:] == ["100.0", "110.0", "90.0", "105.0", "4.0", "410.0"] for row in rows[1:])
    metadata = json.loads(target.with_suffix(".metadata.json").read_text())
    assert metadata["rows"] == 3
    assert metadata["complete_requested_closed_candles"] is True
    assert metadata["unfinished_candles_included"] is False
    assert metadata["timestamp_timezone"] == "UTC"
    assert metadata["notices"] == []
    assert result["output"] == str(target.resolve())
    assert progress.call_args_list[-1].args == (3, 2)
    assert_no_temporary_files(tmp_path)


def test_export_keeps_gaps_and_reports_missing_head_internal_and_tail(tmp_path):
    target = tmp_path / "gaps.csv"
    result = export([frame([BASE + pd.Timedelta(hours=index) for index in [2, 4]])],
                    target, end=BASE + pd.Timedelta(hours=7))

    downloaded = pd.read_csv(target)
    assert len(downloaded) == 2
    assert result["gap_count"] == 1
    assert result["missing_candles_inside_gaps"] == 1
    assert result["gap_examples"] == [{"after": "2025-01-01T02:00:00Z", "before": "2025-01-01T04:00:00Z"}]
    assert result["complete_requested_closed_candles"] is False
    assert len(result["notices"]) == 3
    assert "Earlier requested candles" in result["notices"][0]
    assert "no fabricated rows" in result["notices"][1]
    assert "later completed candles are missing" in result["notices"][2]


@pytest.mark.parametrize("interval,start,stamps,end", [
    ("1w", "2025-01-01", ["2025-01-06", "2025-01-13"], "2025-01-20"),
    ("3d", "2025-01-01", ["2025-01-03", "2025-01-06"], "2025-01-09"),
    ("1M", "2025-01-15", ["2025-02-01", "2025-03-01"], "2025-04-01"),
])
def test_nonaligned_start_does_not_falsely_report_missing_first_candle(tmp_path, interval, start, stamps, end):
    result = export([frame([pd.Timestamp(value, tz="UTC") for value in stamps], interval)],
                    tmp_path / "aligned.csv", interval=interval,
                    start=pd.Timestamp(start, tz="UTC"), end=pd.Timestamp(end, tz="UTC"))
    assert result["notices"] == []
    assert result["complete_requested_closed_candles"] is True


def test_monthly_missing_candles_count_calendar_months(tmp_path):
    stamps = [pd.Timestamp(value, tz="UTC") for value in ["2024-01-01", "2024-04-01"]]
    result = export([frame(stamps, "1M")], tmp_path / "months.csv", interval="1M",
                    start=stamps[0], end=pd.Timestamp("2024-05-01", tz="UTC"))
    assert result["missing_candles_inside_gaps"] == 2
    assert len(result["notices"]) == 1


def test_current_incomplete_candle_is_not_a_missing_tail(tmp_path):
    result = export([frame([BASE])], tmp_path / "closed.csv", end=BASE + pd.Timedelta(hours=1, minutes=40))
    assert result["notices"] == []


@pytest.mark.parametrize("end", [BASE + pd.Timedelta(minutes=30), BASE + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)])
def test_export_rejects_unfinished_candle_including_exact_cutoff_boundary(tmp_path, end):
    target = tmp_path / "existing.csv"
    target.write_text("old csv\n")
    with pytest.raises(ValueError, match="unfinished or out-of-range"):
        export([frame([BASE])], target, end=end)
    assert target.read_text() == "old csv\n"
    assert not target.with_suffix(".metadata.json").exists()
    assert_no_temporary_files(tmp_path)


def test_export_rejects_candle_before_requested_start(tmp_path):
    with pytest.raises(ValueError, match="out-of-range"):
        export([frame([BASE - pd.Timedelta(hours=1)])], tmp_path / "invalid.csv")
    assert_no_temporary_files(tmp_path)


def test_identical_overlaps_are_deduplicated_with_metadata(tmp_path):
    one = frame([BASE])
    duplicated_page = pd.concat([one, one], ignore_index=True)
    result = export([duplicated_page, one], tmp_path / "duplicates.csv", end=BASE + pd.Timedelta(hours=1))
    assert result["rows"] == 1
    assert result["duplicate_rows_removed"] == 2
    assert len(pd.read_csv(result["output"])) == 1


@pytest.mark.parametrize("within_page", [True, False])
def test_conflicting_duplicates_preserve_existing_csv_and_metadata(tmp_path, within_page):
    target = tmp_path / "existing.csv"
    metadata = target.with_suffix(".metadata.json")
    target.write_text("old csv\n")
    metadata.write_text('{"old": true}\n')
    one = frame([BASE])
    changed = one.copy()
    changed.loc[0, "close"] = 104.0
    pages = [pd.concat([one, changed], ignore_index=True)] if within_page else [one, changed]

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        export(pages, target)
    assert target.read_text() == "old csv\n"
    assert metadata.read_text() == '{"old": true}\n'
    assert_no_temporary_files(tmp_path)


def test_empty_download_explicitly_fails_without_replacing_existing_output(tmp_path):
    target = tmp_path / "existing.csv"
    target.write_text("old csv\n")
    with pytest.raises(download.NoDataError, match="No completed 1h candles"):
        export([frame([])], target)
    assert target.read_text() == "old csv\n"
    assert_no_temporary_files(tmp_path)


def test_mid_download_api_failure_preserves_old_outputs_and_cleans_stage(tmp_path):
    target = tmp_path / "existing.csv"
    target.write_text("old csv\n")

    def failed_pages():
        yield frame([BASE])
        raise BinanceError("simulated disconnect")

    with pytest.raises(BinanceError, match="simulated disconnect"):
        export(failed_pages(), target)
    assert target.read_text() == "old csv\n"
    assert_no_temporary_files(tmp_path)


def test_binance_download_paginates_past_full_and_short_pages_until_empty(monkeypatch):
    monkeypatch.setattr("market.binance.MAX_CANDLES", 1)
    client = BinanceClient()
    stamps = [BASE + pd.Timedelta(minutes=index) for index in range(1002)]
    rows = [candle(stamp, "1m") for stamp in stamps]
    client._page = Mock(side_effect=[rows[:1000], rows[1000:], []])
    end = BASE + pd.Timedelta(minutes=1010)

    pages = list(download.binance_pages(client, "BTCUSDT", "1m", BASE, end))
    assert sum(len(page) for page in pages) == 1002
    assert client._page.call_count == 3
    calls = client._page.call_args_list
    assert [call.kwargs["startTime"] for call in calls] == [
        download.milliseconds(BASE), download.milliseconds(stamps[1000]),
        download.milliseconds(BASE + pd.Timedelta(minutes=1002)),
    ]
    assert all(call.kwargs["limit"] == 1000 for call in calls)
    assert all(call.kwargs["endTime"] == download.milliseconds(end) - 1 for call in calls)


def test_binance_pages_exclude_unfinished_and_allow_internal_gaps():
    client = BinanceClient()
    client._page = Mock(return_value=[candle(BASE), candle(BASE + pd.Timedelta(hours=2))])
    pages = list(download.binance_pages(client, "BTCUSDT", "1h", BASE,
                                       BASE + pd.Timedelta(hours=2, minutes=30)))
    assert len(pages) == 1
    assert pages[0].timestamps.tolist() == [BASE]


def test_download_preserves_and_reports_official_short_historical_candle(tmp_path):
    # A real Binance BTCUSDT maintenance candle; its close occurred before the
    # nominal hourly boundary and its zero volume must remain zero.
    row = [1613012400000, "44582.07000000", "44582.07000000", "44582.07000000", "44582.07000000",
           "0.00000000", 1613014854773, "0.00000000", 0, "0.00000000", "0.00000000", "0"]
    start = pd.Timestamp(row[0], unit="ms", tz="UTC")
    end = next_open(start, "1h")
    client = BinanceClient()
    client._page = Mock(return_value=[row])
    pages = list(download.binance_pages(client, "BTCUSDT", "1h", start, end))
    assert pages[0].close_time.iloc[0] == pd.Timestamp(row[6], unit="ms", tz="UTC")

    target = tmp_path / "maintenance.csv"
    result = export(pages, target, start=start, end=end)
    downloaded = pd.read_csv(target)
    assert downloaded.volume.tolist() == [0.0]
    assert downloaded.amount.tolist() == [0.0]
    assert downloaded.close.tolist() == [44582.07]
    assert result["rows"] == 1
    assert result["short_candle_count"] == 1
    assert len(result["short_candle_examples"]) == 1
    example = result["short_candle_examples"][0]
    assert pd.Timestamp(example["open"]) == start
    assert pd.Timestamp(example["close"]) == pd.Timestamp(row[6], unit="ms", tz="UTC")
    assert pd.Timestamp(example["expected_close"]) == end - pd.Timedelta(milliseconds=1)
    assert len(result["notices"]) == 1
    assert "shorter duration" in result["notices"][0]
    assert result["complete_requested_closed_candles"] is False
    metadata = json.loads(target.with_suffix(".metadata.json").read_text())
    assert metadata["short_candle_count"] == 1
    assert metadata["short_candle_examples"] == result["short_candle_examples"]
    assert_no_temporary_files(tmp_path)


@pytest.mark.parametrize("offset", [-1, 3])
def test_binance_page_cannot_escape_requested_bounds(offset):
    client = BinanceClient()
    client._page = Mock(return_value=[candle(BASE + pd.Timedelta(hours=offset))])
    with pytest.raises(BinanceError, match="outside the requested page range"):
        list(download.binance_pages(client, "BTCUSDT", "1h", BASE, BASE + pd.Timedelta(hours=3)))


@pytest.mark.parametrize("arguments", [
    ["binance", "BTCUSDT", "1h", "2025/01/01"],
    ["binance", "BTCUSDT", "1h", "2025-02-30"],
    ["binance", "BTCUSDT", "7m", "2025-01-01"],
    ["bybit", "BTCUSDT", "1s", "2025-01-01"],
    ["binance", "BTCUSDT", "1h", "2025-01-01", "--category", "linear"],
    ["binance", "BTCUSDT", "1h", "2025-01-01", "--end-date", "2025-01-01"],
    ["bybit", "BTC;USDT", "1h", "2025-01-01"],
])
def test_cli_rejects_invalid_input_before_constructing_clients(monkeypatch, arguments):
    binance = Mock()
    bybit = Mock()
    monkeypatch.setattr(download, "BinanceClient", binance)
    monkeypatch.setattr(download, "BybitClient", bybit)
    with pytest.raises(SystemExit) as exc:
        download.main(arguments)
    assert exc.value.code == 2
    binance.assert_not_called()
    bybit.assert_not_called()


@pytest.mark.parametrize("exchange,category", [("binance", "spot"), ("bybit", "spot"), ("bybit", "linear")])
def test_cli_exports_with_selected_exchange_category_and_normalized_symbol(monkeypatch, tmp_path, capsys, exchange, category):
    client = Mock()
    client.server_time.return_value = download.milliseconds(BASE + pd.Timedelta(days=1))
    page = frame([BASE], "1d")
    factory = Mock(return_value=client)
    monkeypatch.setattr(download, "BinanceClient" if exchange == "binance" else "BybitClient", factory)
    if exchange == "binance":
        iterator = Mock(return_value=iter([page]))
        monkeypatch.setattr(download, "binance_pages", iterator)
    else:
        client.iter_candle_pages.return_value = iter([page])
    target = tmp_path / f"{exchange}.csv"
    status = download.main([exchange.upper(), "btc/usdt", "1d", "2025-01-01",
                            "--category", category, "--output", str(target)])

    assert status == 0
    assert target.exists()
    captured = capsys.readouterr()
    assert "Saved 1 candles:" in captured.out
    assert "NOTICE:" not in captured.err
    if exchange == "binance":
        factory.assert_called_once_with()
        assert iterator.call_args.args[1:3] == ("BTCUSDT", "1d")
    else:
        factory.assert_called_once_with(category=category)
        assert client.iter_candle_pages.call_args.args == ("BTCUSDT", "1d")


def test_cli_empty_exchange_result_prints_notice_and_returns_failure(monkeypatch, tmp_path, capsys):
    client = Mock()
    client.server_time.return_value = download.milliseconds(BASE + pd.Timedelta(days=1))
    client.iter_candle_pages.return_value = iter([])
    monkeypatch.setattr(download, "BybitClient", Mock(return_value=client))
    target = tmp_path / "empty.csv"
    assert download.main(["bybit", "NEWUSDT", "1h", "2025-01-01", "--output", str(target)]) == 1
    assert "NOTICE: No completed" in capsys.readouterr().err
    assert not target.exists()
    assert_no_temporary_files(tmp_path)


def test_cli_end_date_is_clamped_to_exchange_time(monkeypatch, tmp_path):
    client = Mock()
    now = BASE + pd.Timedelta(hours=1, minutes=30)
    client.server_time.return_value = download.milliseconds(now)
    client.iter_candle_pages.return_value = iter([frame([BASE])])
    monkeypatch.setattr(download, "BybitClient", Mock(return_value=client))
    assert download.main(["bybit", "BTCUSDT", "1h", "2025-01-01", "--end-date", "2026-01-01",
                          "--output", str(tmp_path / "clamped.csv")]) == 0
    assert client.iter_candle_pages.call_args.kwargs["end"] == now


def test_cli_omitted_output_uses_repo_data_directory(monkeypatch, tmp_path):
    client = Mock()
    client.server_time.return_value = download.milliseconds(BASE + pd.Timedelta(days=1))
    client.iter_candle_pages.return_value = iter([frame([BASE], "1d")])
    monkeypatch.setattr(download, "BybitClient", Mock(return_value=client))
    monkeypatch.setattr(download, "ROOT", tmp_path)
    assert download.main(["bybit", "BTCUSDT", "1d", "2025-01-01"]) == 0
    assert (tmp_path / "data" / "bybit_spot_BTCUSDT_1d_2025-01-01.csv").exists()
