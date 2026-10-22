"""Local, fixed-market dashboard for a frozen CSV training experiment.

Live refreshes remain in memory. They never extend the CSV used to define the
training, validation, or untouched test periods.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import uuid
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd

from finetune_csv.config_loader import CustomFinetuneConfig
from finetune_csv.data_utils import load_candles
from market.binance import BinanceClient, future_timestamps, next_open
from market.bybit import BybitClient

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def candle_records(frame):
    return [dict(time=int(row.timestamps.timestamp()), **{
        key: float(getattr(row, key)) for key in FEATURES
    }) for row in frame.itertuples(index=False)]


def validated_frame(frame):
    if not {"timestamps", *FEATURES}.issubset(frame.columns):
        raise ValueError("Candle data must contain timestamps, open, high, low, close, volume, amount.")
    frame = frame.copy()
    frame["timestamps"] = pd.to_datetime(frame["timestamps"], utc=True, errors="raise")
    for key in FEATURES:
        frame[key] = pd.to_numeric(frame[key], errors="raise").astype(float)
    if frame.empty or frame["timestamps"].isna().any():
        raise ValueError("No valid timestamped candles are available.")
    if not frame["timestamps"].is_monotonic_increasing or frame["timestamps"].duplicated().any():
        raise ValueError("Candle timestamps must be unique and sorted in ascending UTC order.")
    if not np.isfinite(frame[FEATURES].to_numpy()).all():
        raise ValueError("Candle values must be finite.")
    if (frame[FEATURES[:4]] <= 0).any().any() or (frame[FEATURES[4:]] < 0).any().any():
        raise ValueError("Prices must be positive and volumes nonnegative.")
    if ((frame["high"] < frame[FEATURES[:4]].max(axis=1)) |
            (frame["low"] > frame[FEATURES[:4]].min(axis=1))).any():
        raise ValueError("Candle high/low bounds are invalid.")
    return frame.reset_index(drop=True)


class TrainingDashboard:
    def __init__(self, config_path, *, engine_factory=None, live_fetcher=None, paper_factory=None):
        self.config_path = local_path(config_path).resolve()
        self.config = CustomFinetuneConfig(str(self.config_path))
        self.settings = self.config.loader.config
        self.experiment = self.settings.get("experiment", {})
        self.evaluation = self.settings.get("evaluation", {})
        self.market = {key: self.experiment.get(key, self.evaluation.get(key, default)) for key, default in (
            ("exchange", "binance"), ("category", "spot"), ("symbol", "BTCUSDT"), ("interval", "1h"))}
        if self.market["exchange"] not in ("binance", "bybit"):
            raise ValueError("The dashboard supports Binance and Bybit experiments.")
        self.data_path = local_path(self.config.data_path)
        self.checkpoint = local_path(self.config.basemodel_best_model_path)
        self.original = local_path(self.config.pretrained_predictor_path)
        self.original_tokenizer = local_path(self.config.pretrained_tokenizer_path)
        self.trained_tokenizer = local_path(self.config.finetuned_tokenizer_path)
        self.report_directory = local_path(self.evaluation.get("output_dir") or
                                           str(local_path(self.config.base_save_path) / "evaluation"))
        self.report_path = self.report_directory / "report.json"
        self.engine_factory = engine_factory
        self.live_fetcher = live_fetcher
        self.engines = {}
        self.engine_versions = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-trained-dashboard")
        self.lock = threading.RLock()
        self.forecast_lock = threading.RLock()
        self.jobs = {}
        self.latest_forecast = None
        self.hash_cache = {}
        self.paper_factory = paper_factory
        self.paper_trader = None

    def snapshot(self):
        return validated_frame(load_candles(self.config))

    def status(self):
        return dict(market=self.market, config_path=str(self.config_path), data_path=str(self.data_path),
                    trained_checkpoint=str(self.checkpoint), original_checkpoint=str(self.original),
                    trained_ready=(self.checkpoint / "config.json").is_file(),
                    original_ready=(self.original / "config.json").is_file(),
                    report_available=self.report_path.is_file(), report_path=str(self.report_path),
                    lookback=self.config.lookback_window, horizon=self.config.predict_window,
                    max_context=self.config.max_context, sample_count=self.sampling()["sample_count"],
                    fixed_timeframe=True, live_refresh_changes_training_data=False,
                    paper_trading_supported=True)

    def paper(self):
        with self.lock:
            if self.paper_trader is None:
                factory = self.paper_factory
                if factory is None:
                    from finetune_csv.paper_trading import PaperTrader
                    factory = PaperTrader
                self.paper_trader = factory(self)
            return self.paper_trader

    def sampling(self):
        nested = self.evaluation.get("sampling", {})
        defaults = {"sample_count": 100, "seed": self.config.seed,
                    "temperature": 1.0, "top_p": .9, "top_k": 0, "batch_size": 8}
        return {key: nested.get(key, self.evaluation.get(key, default)) for key, default in defaults.items()}

    def live(self):
        if self.live_fetcher:
            return validated_frame(self.live_fetcher(dict(self.market), self.config.lookback_window))
        market = self.market
        if market["exchange"] == "binance":
            if market["category"] != "spot":
                raise ValueError("Binance dashboard refresh currently supports spot candles.")
            frame = BinanceClient().fetch_candles(market["symbol"], market["interval"],
                                                 limit=self.config.lookback_window + 24)
        else:
            client = BybitClient(category=market["category"])
            end = pd.Timestamp(client.server_time(), unit="ms", tz="UTC")
            start = end
            for _ in range(self.config.lookback_window + 26):
                start = (start - pd.DateOffset(months=1)) if market["interval"] == "1M" else start - pd.Timedelta(market["interval"])
            frame = client.fetch_candles(market["symbol"], market["interval"], start=start, end=end)
        return validated_frame(frame)

    def context(self, frame):
        history = validated_frame(frame).iloc[-self.config.lookback_window:].reset_index(drop=True)
        if len(history) != self.config.lookback_window:
            raise ValueError(f"Forecasting requires {self.config.lookback_window} completed context candles.")
        opens = history["timestamps"].tolist()
        if any(next_open(left, self.market["interval"]) != right for left, right in zip(opens, opens[1:])):
            raise ValueError("The latest context contains missing candles. Choose a complete dataset or refresh live data.")
        # CSVs exported by download_data.sh contain only completed candles. Check
        # the clock again in case a different CSV was selected for this run.
        if next_open(opens[-1], self.market["interval"]) > pd.Timestamp.now(tz="UTC"):
            raise ValueError("The context contains an unfinished candle.")
        return history

    def engine(self, kind):
        checkpoint = self.checkpoint if kind == "finetuned" else self.original
        tokenizer = self.trained_tokenizer if kind == "finetuned" else self.original_tokenizer
        version = tuple((str(item), item.stat().st_size, item.stat().st_mtime_ns)
                        for directory in (checkpoint, tokenizer)
                        for item in sorted(directory.rglob("*")) if item.is_file())
        if kind in self.engines and self.engine_versions.get(kind) != version:
            self.engines.pop(kind).close()
        if kind not in self.engines:
            factory = self.engine_factory
            if factory is None:
                from finetune_csv.inference import LocalForecastEngine
                factory = LocalForecastEngine
            device = "cpu"
            if self.config.use_cuda:
                import torch
                if torch.cuda.is_available():
                    device = f"cuda:{self.config.device_id}"
            self.engines[kind] = factory(
                str(checkpoint), str(tokenizer),
                max_context=self.config.max_context, clip=self.config.clip,
                lookback=self.config.lookback_window, batch_size=self.sampling()["batch_size"], device=device,
            )
            self.engine_versions[kind] = version
        return self.engines[kind]

    def submit(self, refresh_live=False):
        with self.lock:
            if any(job["status"] in ("queued", "running") for job in self.jobs.values()):
                raise RuntimeError("A comparison forecast is already running.")
            ident = uuid.uuid4().hex
            self.jobs[ident] = dict(id=ident, status="queued", progress=0, message="Preparing candles")
            # Bound completed jobs without losing the current result.
            for old in list(self.jobs)[:-20]:
                del self.jobs[old]
            self.executor.submit(self._run, ident, refresh_live)
            return ident

    def _run(self, ident, refresh_live):
        def progress(fraction, message):
            with self.lock:
                self.jobs[ident].update(status="running", progress=float(fraction), message=message)
        try:
            progress(.01, "Fetching live closed candles" if refresh_live else "Reading the frozen dataset")
            history = self.context(self.live() if refresh_live else self.snapshot())
            timestamps = future_timestamps(history["timestamps"].iloc[-1], self.market["interval"], self.config.predict_window)
            result = dict(market=self.market, history=candle_records(history), models={},
                          source="live" if refresh_live else "frozen_dataset", sampling=self.sampling(),
                          generated_at=datetime.now(timezone.utc).isoformat(),
                          context_length=len(history), horizon=len(timestamps),
                          origin_time=int(history["timestamps"].iloc[-1].timestamp()),
                          normalization="Each model normalizes only its historical context; future candles are not used.")
            for index, kind in enumerate(("finetuned", "original")):
                progress(.03 + index * .47, f"Sampling {kind} checkpoint")
                sample_args = {key: value for key, value in self.sampling().items() if key != "batch_size"}
                with self.forecast_lock:
                    samples = self.engine(kind).sample(history, timestamps, **sample_args,
                        progress=lambda fraction, index=index, kind=kind: progress(.03 + .47 * (index + fraction), f"Sampling {kind} checkpoint"))
                samples = np.asarray(samples, dtype=float)
                if samples.shape != (self.sampling()["sample_count"], len(timestamps), 6) or not np.isfinite(samples).all() or (samples[:, :, :4] <= 0).any():
                    raise ValueError(f"The {kind} checkpoint generated invalid forecast values.")
                times = [int(pd.Timestamp(t).timestamp()) for t in timestamps]
                closes = samples[:, :, 3]
                quantiles = np.quantile(closes, [.1, .5, .9], axis=0)
                result["models"][kind] = dict(
                    quantiles=[dict(time=t, p10=float(quantiles[0, j]), p50=float(quantiles[1, j]),
                                    p90=float(quantiles[2, j])) for j, t in enumerate(times)],
                    paths=[[dict(time=t, value=float(value)) for t, value in zip(times, path)] for path in closes],
                )
            with self.lock:
                self.latest_forecast = result
                self.jobs[ident].update(status="completed", progress=1, message="Comparison forecast ready", result=result)
        except Exception as exc:
            with self.lock:
                self.jobs[ident].update(status="failed", message=str(exc), error=str(exc))

    def job(self, ident):
        if not re.fullmatch(r"[a-f0-9]{32}", ident):
            raise ValueError("Invalid job ID.")
        with self.lock:
            return dict(self.jobs[ident])

    def report(self):
        if not self.report_path.is_file():
            return dict(available=False, message="No evaluation report yet. Complete training and untouched-test evaluation first.")
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        stale = []
        try:
            if report.get("source_csv_sha256") != self._file_hash(self.data_path):
                stale.append("dataset")
            pairs = (("finetuned", self.checkpoint, self.trained_tokenizer),
                     ("original", self.original, self.original_tokenizer))
            for kind, checkpoint, tokenizer in pairs:
                saved = report.get("models", {}).get(kind, {})
                if (saved.get("checkpoint_hash") != self._checkpoint_hash(checkpoint) or
                        saved.get("tokenizer_hash") != self._checkpoint_hash(tokenizer)):
                    stale.append(f"{kind} model/tokenizer")
            if report.get("sampling") != self.sampling():
                stale.append("sampling settings")
            if report.get("interval") != self.market["interval"]:
                stale.append("candle interval")
            rules = report.get("execution_rules", {})
            if rules.get("lookback") != self.config.lookback_window or rules.get("horizon") != self.config.predict_window:
                stale.append("context or forecast length")
        except OSError:
            stale.append("missing model or data files")
        if stale:
            return dict(available=False, stale=True,
                        message="Saved evaluation does not match the current " + ", ".join(stale) + ". Run evaluation again.")
        return dict(available=True, stale=False, report=report)

    def _file_hash(self, path):
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.hash_cache:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.hash_cache[key] = digest.hexdigest()
        return self.hash_cache[key]

    def _checkpoint_hash(self, path):
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(bytes.fromhex(self._file_hash(item)))
        return digest.hexdigest()

    def close(self):
        if self.paper_trader is not None:
            self.paper_trader.close()
        self.executor.shutdown(wait=True)
        for engine in self.engines.values():
            engine.close()


def create_app(config_path, *, engine_factory=None, live_fetcher=None, paper_factory=None, start_paper=False):
    service = TrainingDashboard(config_path, engine_factory=engine_factory, live_fetcher=live_fetcher,
                                paper_factory=paper_factory)
    app = Flask(__name__, template_folder=str(ROOT / "webui" / "templates"),
                static_folder=str(ROOT / "webui" / "static"))
    app.extensions["training_dashboard"] = service

    @app.errorhandler(ValueError)
    def invalid(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(FileNotFoundError)
    def missing(exc):
        return jsonify(error=f"Required experiment file is missing: {exc.filename or exc}"), 404

    @app.get("/")
    def index():
        return render_template("training.html")

    @app.get("/api/status")
    def status():
        return jsonify(service.status())

    @app.get("/api/data")
    def data():
        source = request.args.get("source", "snapshot")
        if source not in ("snapshot", "live"):
            raise ValueError("source must be snapshot or live.")
        try:
            frame = service.live() if source == "live" else service.snapshot()
        except (ValueError, FileNotFoundError):
            raise
        except Exception as exc:
            return jsonify(error=str(exc)), 502
        return jsonify(candles=candle_records(frame.iloc[-max(512, service.config.lookback_window):]),
                       rows=len(frame), first_open=frame["timestamps"].iloc[0].isoformat(),
                       last_open=frame["timestamps"].iloc[-1].isoformat(), source=source,
                       market=service.market, training_data_unchanged=True)

    @app.get("/api/report")
    def report():
        return jsonify(service.report())

    @app.post("/api/forecast")
    def forecast():
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc != request.host:
            return jsonify(error="Forecast requests must come from this local dashboard."), 403
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or set(body) - {"refresh_live"} or not isinstance(body.get("refresh_live", False), bool):
            raise ValueError("Send a JSON object containing only refresh_live: true or false.")
        try:
            return jsonify(job_id=service.submit(body.get("refresh_live", False))), 202
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409

    @app.get("/api/jobs/<ident>")
    def job(ident):
        try:
            return jsonify(service.job(ident))
        except KeyError:
            return jsonify(error="Forecast job not found."), 404

    @app.get("/api/latest-forecast")
    def latest():
        with service.lock:
            return jsonify(forecast=service.latest_forecast)

    @app.get("/api/paper/status")
    def paper_status():
        try:
            return jsonify(service.paper().status())
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409

    @app.post("/api/paper/<action>")
    def paper_action(action):
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc != request.host:
            return jsonify(error="Paper-trading controls must come from this local dashboard."), 403
        if action not in ("start", "pause", "close"):
            return jsonify(error="Unknown paper-trading action."), 404
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or body:
            raise ValueError("Send an empty JSON object; this account uses the experiment's frozen rules.")
        try:
            trader = service.paper()
            if action == "start":
                trader.enable()
            elif action == "pause":
                trader.pause()
            else:
                trader.close_position()
            return jsonify(trader.status())
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409

    if start_paper:
        try:
            service.paper().start()
        except (ValueError, OSError, RuntimeError) as exc:
            app.logger.error("Paper monitoring could not start: %s", exc)

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--port", type=int, default=7071)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    app = create_app(args.config, start_paper=True)
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True, use_reloader=False)
    finally:
        app.extensions["training_dashboard"].close()


if __name__ == "__main__":
    main()
