"""Export complete public exchange history to the Kronos CSV format.

Pages are staged on disk so large exports and Bybit's reverse pagination do
not require holding the full dataset in memory. Existing CSVs are replaced
only after a successful download and validation.
"""

import argparse
import csv
from datetime import date, datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import time

import pandas as pd

from market.binance import BinanceClient, BinanceError, SUPPORTED_INTERVALS, next_open
from market.bybit import BybitClient, BybitError, SUPPORTED_INTERVALS as BYBIT_INTERVALS

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


class NoDataError(ValueError):
    """The API returned no completed candles for the requested period."""


def utc_date(value):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise argparse.ArgumentTypeError("Use a UTC date in YYYY-MM-DD format.")
    try:
        return pd.Timestamp(date.fromisoformat(value), tz="UTC")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}") from exc


def milliseconds(stamp):
    return int(pd.Timestamp(stamp).value // 1_000_000)


def utc_text(ms):
    precision = "milliseconds" if ms % 1000 else "seconds"
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec=precision).replace("+00:00", "Z")


def binance_pages(client, symbol, interval, start, end):
    """Forward paginate without the in-memory forecasting client's row cap."""
    cursor, cutoff = milliseconds(start), milliseconds(end)
    while cursor < cutoff:
        rows = client._page(symbol, interval, startTime=cursor, endTime=cutoff - 1, limit=1000)
        if not rows:
            return
        frame = client._frame(rows, interval, allow_gaps=True, allow_short_candles=True)
        if (frame["timestamps"] < pd.Timestamp(cursor, unit="ms", tz="UTC")).any() or (
            frame["timestamps"] >= end
        ).any():
            raise BinanceError("Binance returned candles outside the requested page range.")
        following = milliseconds(next_open(frame["timestamps"].iloc[-1], interval))
        if following <= cursor:
            raise BinanceError("Binance candle pagination did not advance.")
        closed = frame.loc[frame["close_time"] < end].reset_index(drop=True)
        if not closed.empty:
            yield closed
        cursor = following


@lru_cache(maxsize=32)
def _step_ms(interval):
    return int(pd.Timedelta(interval).value // 1_000_000)


def _advance(ms, interval, backwards=False):
    if interval == "1M":
        stamp = pd.Timestamp(ms, unit="ms", tz="UTC")
        return milliseconds(stamp + pd.DateOffset(months=-1 if backwards else 1))
    delta = _step_ms(interval)
    return ms - delta if backwards else ms + delta


def _missing_between(previous, current, interval):
    if interval != "1M":
        step = _advance(previous, interval) - previous
        return max(0, (current - previous) // step - 1)
    missing, expected = 0, _advance(previous, interval)
    while expected < current:
        missing += 1
        expected = _advance(expected, interval)
    return missing


def export_pages(pages, output, *, exchange, category, symbol, interval, start, end, progress=None):
    """Write a sorted CSV and coverage metadata from pages in either order."""
    output = Path(output).expanduser().resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if metadata_path == output:
        raise ValueError("Output filename must differ from its .metadata.json sidecar.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = []
    connection = None
    start_ms, end_ms = milliseconds(start), milliseconds(end)
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".candle-stage-", suffix=".tmp", delete=False) as stage:
            stage_path = Path(stage.name)
        temporary.append(stage_path)
        connection = sqlite3.connect(stage_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE candles (time INTEGER PRIMARY KEY, open TEXT, high TEXT, low TEXT, close TEXT, volume TEXT, amount TEXT, close_time INTEGER)")
        count = duplicates = page_count = 0
        for frame in pages:
            records = []
            for row in frame.itertuples(index=False):
                stamp = milliseconds(row.timestamps)
                if stamp < start_ms or milliseconds(row.close_time) >= end_ms:
                    raise ValueError("An exchange page included an unfinished or out-of-range candle.")
                records.append((stamp, *(str(float(getattr(row, name))) for name in COLUMNS[1:]),
                                milliseconds(row.close_time)))
            if not records:
                continue
            # Identical overlap is harmless; conflicting closed candles are not.
            values = {record[0]: record for record in records}
            if len(values) != len(records) and any(values[record[0]] != record for record in records):
                raise ValueError("Conflicting duplicate candles within a page.")
            placeholders = ",".join("?" for _ in values)
            existing = connection.execute(f"SELECT * FROM candles WHERE time IN ({placeholders})", list(values))
            if any(tuple(record) != values[record[0]] for record in existing):
                raise ValueError("Conflicting duplicate candles across exchange pages.")
            before = connection.total_changes
            connection.executemany("INSERT OR IGNORE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)", records)
            inserted = connection.total_changes - before
            count += inserted
            duplicates += len(records) - inserted
            page_count += 1
            if progress:
                progress(count, page_count)
        if count == 0:
            raise NoDataError(f"No completed {interval} candles are available for {exchange} {category} {symbol} from {start.isoformat()} to {end.isoformat()}.")
        connection.commit()
        first = last = None
        gap_count = missing_count = short_candle_count = 0
        gap_examples, short_candle_examples = [], []
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=output.parent,
                                         prefix=".candle-export-", suffix=".tmp", delete=False) as target:
            csv_temp = Path(target.name)
            temporary.append(csv_temp)
            writer = csv.writer(target)
            writer.writerow(COLUMNS)
            for row in connection.execute("SELECT * FROM candles ORDER BY time"):
                stamp = row[0]
                if first is None:
                    first = stamp
                if last is not None and stamp != _advance(last, interval):
                    gap_count += 1
                    missing_count += _missing_between(last, stamp, interval)
                    if len(gap_examples) < 10:
                        gap_examples.append(dict(after=utc_text(last), before=utc_text(stamp)))
                expected_close = _advance(stamp, interval) - 1
                if row[-1] < expected_close:
                    short_candle_count += 1
                    if len(short_candle_examples) < 10:
                        short_candle_examples.append(dict(open=utc_text(stamp), close=utc_text(row[-1]),
                                                         expected_close=utc_text(expected_close)))
                writer.writerow([utc_text(stamp), *row[1:-1]])
                last = stamp
        notices = []
        if _advance(first, interval, backwards=True) >= milliseconds(start):
            notices.append(f"Requested history from {start.isoformat()}, but the first available completed candle is {utc_text(first)}. Earlier requested candles were not returned by the exchange.")
        if gap_count:
            notices.append(f"History contains {gap_count:,} gap(s), with at least {missing_count:,} missing candle(s). Gaps were preserved, with no fabricated rows.")
        if _advance(_advance(last, interval), interval) <= milliseconds(end):
            notices.append(f"The last available completed candle opens at {utc_text(last)}; later completed candles are missing before {end.isoformat()}.")
        if short_candle_count:
            notices.append(f"The exchange returned {short_candle_count:,} candle(s) with a shorter duration than {interval}. Their OHLCV/amount values were kept unchanged; see metadata for close times.")
        result = dict(exchange=exchange, category=category, symbol=symbol, interval=interval,
                      requested_start=start.isoformat(), end_exclusive=end.isoformat(),
                      first_candle=utc_text(first), last_candle=utc_text(last), rows=count,
                      columns=COLUMNS, timestamp_timezone="UTC", unfinished_candles_included=False,
                      duplicate_rows_removed=duplicates, gap_count=gap_count,
                      missing_candles_inside_gaps=missing_count, gap_examples=gap_examples,
                      short_candle_count=short_candle_count, short_candle_examples=short_candle_examples,
                      complete_requested_closed_candles=not notices, notices=notices,
                      generated_at=datetime.now(timezone.utc).isoformat())
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".candle-metadata-", suffix=".tmp", delete=False) as metadata:
            metadata_temp = Path(metadata.name)
            temporary.append(metadata_temp)
            json.dump(result, metadata, indent=2, allow_nan=False)
            metadata.write("\n")
        csv_temp.replace(output)
        metadata_temp.replace(metadata_path)
        result.update(output=str(output), metadata=str(metadata_path))
        return result
    finally:
        if connection is not None:
            connection.close()
        for path in temporary:
            path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="download_data.sh",
        description="Download completed Binance/Bybit candles into a Kronos CSV. Dates and timestamps use UTC.")
    parser.add_argument("exchange", type=str.lower, choices=["binance", "bybit"])
    parser.add_argument("asset", help="Exchange symbol, e.g. BTCUSDT")
    parser.add_argument("period", help="Native candle interval, e.g. 1m, 1h, 4h, 1d, 1w, 1M")
    parser.add_argument("start_date", type=utc_date, help="First requested candle date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=utc_date, help="Optional exclusive UTC end date; default: exchange server time")
    parser.add_argument("--category", choices=["spot", "linear", "inverse"], default="spot",
                        help="Bybit market type; Binance supports spot here (default: spot)")
    parser.add_argument("--output", type=Path, help="CSV destination; default: this repo's data/ directory")
    args = parser.parse_args(argv)
    symbol = args.asset.strip().upper().replace("/", "")
    if not re.fullmatch(r"[A-Z0-9]{2,30}", symbol):
        parser.error("Use a symbol such as BTCUSDT, containing only letters and digits.")
    supported = SUPPORTED_INTERVALS if args.exchange == "binance" else BYBIT_INTERVALS
    if args.period not in supported:
        parser.error(f"{args.exchange} supports: {', '.join(supported)}")
    if args.exchange == "binance" and args.category != "spot":
        parser.error("Binance downloads use spot. --category linear/inverse is supported for Bybit.")
    if args.end_date is not None and args.end_date <= args.start_date:
        parser.error("--end-date must be later than start-date.")
    client = BinanceClient() if args.exchange == "binance" else BybitClient(category=args.category)
    output = args.output or ROOT / "data" / f"{args.exchange}_{args.category}_{symbol}_{args.period}_{args.start_date.date()}.csv"
    try:
        server_now = pd.Timestamp(client.server_time(), unit="ms", tz="UTC")
        end = min(args.end_date, server_now) if args.end_date is not None else server_now
        if args.start_date >= end:
            parser.error("start-date must precede exchange server time and the end date.")
        print(f"Downloading {args.exchange} {args.category} {symbol} {args.period}: {args.start_date.isoformat()} to {end.isoformat()}", file=sys.stderr, flush=True)
        last_progress = 0.0

        def progress(count, pages):
            nonlocal last_progress
            now = time.monotonic()
            if pages == 1 or now - last_progress >= 2:
                print(f"Downloaded {count:,} candles ({pages:,} pages)...", file=sys.stderr, flush=True)
                last_progress = now

        pages = binance_pages(client, symbol, args.period, args.start_date, end) if args.exchange == "binance" else client.iter_candle_pages(symbol, args.period, start=args.start_date, end=end)
        result = export_pages(pages, output, exchange=args.exchange, category=args.category,
                              symbol=symbol, interval=args.period, start=args.start_date, end=end, progress=progress)
        for notice in result["notices"]:
            print("NOTICE: " + notice, file=sys.stderr)
        print(f"Saved {result['rows']:,} candles: {result['output']}")
        print(f"Available opens: {result['first_candle']} to {result['last_candle']} (UTC)")
        print(f"Coverage metadata: {result['metadata']}")
        return 0
    except NoDataError as exc:
        print(f"NOTICE: {exc} No CSV was written.", file=sys.stderr)
        return 1
    except (BinanceError, BybitError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
