"""Local forecast archive and serialized research jobs."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import uuid

import numpy as np
import pandas as pd

from market.binance import BinanceClient, SUPPORTED_INTERVALS, next_open
from market.forecast import ForecastEngine, MODEL_ID, MODEL_REVISION, TOKENIZER_REVISION, candle_records

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def settings(body):
    if not isinstance(body, dict):
        raise ValueError("Send a JSON object.")
    symbol = str(body.get("symbol", "BTCUSDT")).upper().replace("/", "").strip()
    if not re.fullmatch(r"[A-Z0-9]{2,30}", symbol):
        raise ValueError("Use a Binance spot pair such as BTCUSDT.")
    interval = body.get("interval", "1d")
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError("Unsupported Binance candle interval.")
    count = integer_setting(body.get("sample_count", 100), "sample_count", 1, 500)
    costs = {}
    for key, default in (("fee_bps", 10), ("slippage_bps", 5)):
        value = float(body.get(key, default))
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"{key} must be between 0 and 100 basis points per side.")
        costs[key] = value
    return dict(symbol=symbol, interval=interval, sample_count=int(count), **costs)


def integer_setting(value, name, minimum, maximum):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = float("nan")
    if isinstance(value, bool) or not math.isfinite(number) or not number.is_integer() or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")
    return int(number)


class MarketService:
    def __init__(self, directory=None, client=None, engine=None):
        self.directory = Path(directory or ROOT / "outputs" / "market")
        self.client = client or BinanceClient()
        self.engine = engine or ForecastEngine()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-market")
        self.jobs = {}
        self.lock = threading.Lock()

    def _path(self, kind, ident):
        if not re.fullmatch(r"[a-f0-9]{32}", ident):
            raise ValueError("Invalid result ID.")
        return self.directory / kind / f"{ident}.json"

    def archive(self, result):
        atomic_json(self._path("forecasts", result["id"]), result)
        return result

    def forecast(self, config, progress=None):
        if progress:
            progress(.01, "Fetching 512 closed Binance candles")
        frame = self.client.fetch_candles(config["symbol"], config["interval"], limit=512)
        if progress:
            progress(.03, "Loading Kronos-small")
        result = self.engine.forecast(frame, **config,
            progress=(lambda fraction: progress(.05 + fraction * .9, f"Sampling {config['sample_count']} futures")) if progress else None)
        return self.archive(result)

    def forecasts(self, symbol=None, interval=None):
        found = []
        for path in (self.directory / "forecasts").glob("*.json"):
            data = json.loads(path.read_text())
            if symbol and data["symbol"] != symbol or interval and data["interval"] != interval:
                continue
            found.append({key: data[key] for key in ("id", "symbol", "interval", "generated_at", "as_of", "sample_count")})
        return sorted(found, key=lambda row: row["generated_at"], reverse=True)[:100]

    def read_forecast(self, ident):
        return json.loads(self._path("forecasts", ident).read_text())

    def latest_backtest(self, symbol, interval):
        results = []
        for path in (self.directory / "backtests").glob("*.json"):
            data = json.loads(path.read_text())
            if data["symbol"] == symbol and data["interval"] == interval:
                results.append(data)
        return max(results, key=lambda row: row["generated_at"]) if results else None

    def candles(self, symbol, interval, forecast_id=None):
        frame = self.client.fetch_candles(symbol, interval, limit=512)
        records = candle_records(frame)
        if forecast_id:
            saved = self.read_forecast(forecast_id)
            if saved["symbol"] != symbol or saved["interval"] != interval:
                raise ValueError("Saved forecast belongs to a different market.")
            first = pd.to_datetime(saved["quantiles"][0]["time"], unit="s", utc=True)
            end = next_open(pd.to_datetime(saved["quantiles"][-1]["time"], unit="s", utc=True), interval)
            actual = self.client.fetch_candles(symbol, interval, start=first, end=end)
            merged = {row["time"]: row for row in saved["history"] + candle_records(actual) + records}
            records = [merged[t] for t in sorted(merged)]
        return records

    def backtest(self, config, *, days=365, progress=None):
        from market.backtest import run_backtest
        if not 1 <= days <= 730:
            raise ValueError("days must be between 1 and 730.")
        if progress:
            progress(.01, "Fetching closed candles and warmup history")
        latest = self.client.fetch_candles(config["symbol"], config["interval"], limit=1)
        end = next_open(latest["timestamps"].iloc[-1], config["interval"])
        start = end - pd.Timedelta(days=days)
        if config["interval"] == "1M":
            warmup = start - pd.DateOffset(months=514)
        else:
            warmup = start - (next_open(start, config["interval"]) - start) * 514
        frame = self.client.fetch_candles(config["symbol"], config["interval"], start=warmup, end=end)
        # Save the exact source candles used by this run for audit/reproduction.
        run_id = uuid.uuid4().hex
        source = self.directory / "backtests" / f"{run_id}-candles.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(source, index=False)
        total = max(1, math.ceil(int(((frame["timestamps"] >= start) & (frame["timestamps"] < end)).sum()) / 24))
        completed = 0

        def sample(history, timestamps):
            nonlocal completed
            digest = hashlib.sha256()
            digest.update(f"{MODEL_REVISION}:{TOKENIZER_REVISION}:{config['interval']}:{config['sample_count']}:42:24:T1:p0.9:batch{self.engine.batch_size}".encode())
            digest.update(pd.util.hash_pandas_object(history[["timestamps", "open", "high", "low", "close", "volume", "amount"]], index=False).to_numpy().tobytes())
            cache = self.directory / "samples" / (digest.hexdigest() + ".npy")
            if cache.exists():
                paths = np.load(cache, allow_pickle=False)
            else:
                full = self.engine.sample(history, timestamps, sample_count=config["sample_count"], seed=42,
                    progress=(lambda f: progress(.05 + .9 * (completed + f) / total, f"Forecast {completed + 1}/{total}: {config['sample_count']} paths")) if progress else None)
                paths = full[:, :, 3]
                cache.parent.mkdir(parents=True, exist_ok=True)
                temp = cache.with_suffix(f".{uuid.uuid4().hex}.tmp")
                with temp.open("wb") as output:
                    np.save(output, paths, allow_pickle=False)
                temp.replace(cache)
            completed += 1
            return paths

        result = run_backtest(frame, sample, start=start, end=end, interval=config["interval"],
            lookback=512, horizon=24, sample_count=config["sample_count"],
            fee_bps=config["fee_bps"], slippage_bps=config["slippage_bps"],
            progress=(lambda p: progress(.05 + .9 * p["progress"], f"Evaluated {p['completed_forecasts']}/{p['total_forecasts']} forecasts")) if progress else None)
        result.update(id=run_id, symbol=config["symbol"], interval=config["interval"], days=days,
            generated_at=datetime.now(timezone.utc).isoformat(), model=MODEL_ID,
            model_revision=MODEL_REVISION, tokenizer_revision=TOKENIZER_REVISION,
            sample_count=config["sample_count"], seed=42, batch_size=self.engine.batch_size,
            temperature=1.0, top_p=.9,
            source_candles=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source))
        # The UI uses a trade list while the metrics retain a numeric trade count.
        result["trades"] = result.get("trade_log", [])
        atomic_json(self._path("backtests", run_id), result)
        return result

    def submit(self, kind, config, *, days=365):
        with self.lock:
            if sum(job["status"] in ("queued", "running") for job in self.jobs.values()) >= 3:
                raise ValueError("Three jobs are already queued/running; wait for a result.")
            ident = uuid.uuid4().hex
            self.jobs[ident] = dict(id=ident, kind=kind, status="queued", progress=0,
                                    message="Queued", config=config, result=None, error=None)
            atomic_json(self._path("jobs", ident), self.jobs[ident])

        def update(fraction, message):
            with self.lock:
                self.jobs[ident].update(status="running", progress=min(.99, max(0, fraction)), message=message)
                atomic_json(self._path("jobs", ident), self.jobs[ident])

        def work():
            try:
                update(0, "Starting")
                result = self.forecast(config, update) if kind == "forecast" else self.backtest(config, days=days, progress=update)
                with self.lock:
                    self.jobs[ident].update(status="complete", progress=1, message="Complete", result=result)
                    atomic_json(self._path("jobs", ident), self.jobs[ident])
            except Exception as exc:
                with self.lock:
                    self.jobs[ident].update(status="error", message="Job failed", error=str(exc))
                    atomic_json(self._path("jobs", ident), self.jobs[ident])
        self.executor.submit(work)
        return ident

    def job(self, ident):
        path = self._path("jobs", ident)
        with self.lock:
            if ident in self.jobs:
                return self.jobs[ident].copy()
        data = json.loads(path.read_text())
        if data["status"] in ("running", "queued"):
            data.update(status="error", error="Server stopped before this job completed. Submit the job again to reuse cached forecasts.")
        return data
