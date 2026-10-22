"""Automatic, persistent paper decisions for one trained spot checkpoint.

The quote monitor runs separately from GPU forecasting. Entries use the first
fresh observation after inference, never an earlier candle's historical open.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading

import numpy as np
import pandas as pd

from market.binance import future_timestamps, next_open
from market.public_feed import PublicMarketFeed, utc
from market.strategy import make_trade_plan, summarize_paths


def paper_identity(service):
    """Bind an account to actual model bytes, market, sampling and fixed rules."""
    from .evaluate import checkpoint_hash, file_hash

    settings = service.evaluation
    return {
        "version": 1, "market": dict(service.market),
        "model_sha256": checkpoint_hash(service.checkpoint),
        "tokenizer_sha256": checkpoint_hash(service.trained_tokenizer),
        "dataset_sha256": file_hash(service.data_path),
        "experiment_config_sha256": file_hash(service.config_path),
        "sampling": service.sampling(), "lookback": service.config.lookback_window,
        "horizon": service.config.predict_window, "max_context": service.config.max_context,
        "clip": service.config.clip,
        "rules": {key: settings.get(key, default) for key, default in (
            ("fee_bps", 10), ("slippage_bps", 5), ("initial_equity", 10000),
            ("risk_fraction", .01), ("max_position_fraction", .25),
            ("minimum_reward_risk", 1.5), ("consensus_threshold", .70))},
        "entry_schedule": "each_new_closed_candle_while_flat",
        "entry_execution": "first_observed_public_quote_after_forecast",
    }


class PaperTrader:
    def __init__(self, service, *, broker=None, feed=None, poll_seconds=15,
                 clock=None, start_worker=False):
        if not np.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive and finite.")
        self.service = service
        self.market = deepcopy(service.market)
        if self.market.get("category", "spot") != "spot":
            raise ValueError("Automatic paper trading supports spot accounts only.")
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.poll_seconds = float(poll_seconds)
        self.feed = feed or PublicMarketFeed(self.market, clock=self.clock)
        self.identity = paper_identity(service)
        self.identity_digest = self._digest(self.identity)
        self.file_version = self._model_file_version()
        rules = self.identity["rules"]
        if broker is None:
            from market.paper import PaperBroker
            broker = PaperBroker(Path(service.config.base_save_path) / "paper" / "account.json", self.identity,
                                 clock=self.clock, max_quote_age_seconds=30,
                                 **{key: rules[key] for key in ("initial_equity", "fee_bps", "slippage_bps",
                                    "risk_fraction", "max_position_fraction", "minimum_reward_risk")})
        self.broker = broker
        self.lock = threading.RLock()
        self.observation_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-paper-forecast")
        self.worker = None
        self.future = None
        self.last_poll = None
        self.last_quote = None
        self.last_error = None
        self._quote_error = None
        self.last_forecast = None
        self.forecast_origin = None
        self.forecast_progress = 0.0
        self.next_candle_at = None
        self.recovery = {"last_recovered_at": None, "notices": []}
        self._last_successful_quote = None
        self._needs_recovery = True
        self._closed = False
        self._incompatible = False
        self.forecast_path = Path(service.config.base_save_path) / "paper" / "latest-forecast.json"
        if self.forecast_path.is_file():
            try:
                saved = json.loads(self.forecast_path.read_text())
                forecast = saved.get("forecast", {})
                cursor = self.broker.status().get("last_processed_origin")
                if (saved.get("identity_digest") == self.identity_digest and cursor is not None
                        and forecast.get("source") == "paper_live"
                        and self._origin_stamp(cursor) == self._origin_stamp(forecast["origin_time"])):
                    self.last_forecast = forecast
                    with self.service.lock:
                        if self.service.latest_forecast is None:
                            self.service.latest_forecast = forecast
            except (ValueError, KeyError, OSError, TypeError):
                pass  # Display cache is optional; the account ledger is authoritative.
        if start_worker:
            self.start()

    @staticmethod
    def _digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

    def _model_file_version(self):
        files = [path for directory in (self.service.checkpoint, self.service.trained_tokenizer)
                 for path in sorted(Path(directory).rglob("*")) if path.is_file()]
        files += [Path(self.service.data_path), Path(self.service.config_path)]
        return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns)
                     for path in files)

    def _check_identity(self):
        if self._incompatible:
            raise ValueError("This paper account is incompatible with the changed experiment; entries remain paused.")
        try:
            version = self._model_file_version()
        except OSError as exc:
            self._incompatible = True
            self.broker.set_enabled(False)
            raise ValueError("Experiment model or data files became unavailable; paper entries are paused.") from exc
        rules = {key: self.service.evaluation.get(key, value) for key, value in self.identity["rules"].items()}
        settings_changed = (self.service.market != self.market or self.service.sampling() != self.identity["sampling"]
                            or rules != self.identity["rules"]
                            or self.service.config.lookback_window != self.identity["lookback"]
                            or self.service.config.predict_window != self.identity["horizon"]
                            or self.service.config.max_context != self.identity["max_context"]
                            or self.service.config.clip != self.identity["clip"])
        if settings_changed or version != self.file_version:
            if settings_changed or self._digest(paper_identity(self.service)) != self.identity_digest:
                self._incompatible = True
                self.broker.set_enabled(False)
                raise ValueError("Model, tokenizer, market or paper settings changed. This persisted account cannot trade a different experiment.")
            self.file_version = version

    def status(self):
        """Read local state only; this endpoint never calls an exchange or model."""
        with self.lock:
            state = deepcopy(self.broker.status())
            decision = state.get("last_decision")
            if decision:
                decision = dict(decision)
                decision.setdefault("action", decision.get("plan", {}).get("action"))
                decision.setdefault("reason", decision.get("execution_reason") or decision.get("plan", {}).get("reason"))
            quote = self.last_quote
            if quote is None and state.get("last_quote_price") is not None:
                quote = {"price": state["last_quote_price"], "timestamp": state.get("last_quote_time"),
                         "source": "persisted_previous_observation"}
            return {**state, "market": dict(self.market), "running": bool(self.worker and self.worker.is_alive()),
                    "forecast_running": bool(self.future and not self.future.done()),
                    "forecast_origin": self.forecast_origin, "forecast_progress": self.forecast_progress,
                    "latest_decision": decision, "last_quote": deepcopy(quote),
                    "last_error": self.last_error, "last_poll": self.last_poll,
                    "next_candle_at": self.next_candle_at, "recovery": deepcopy(self.recovery),
                    "sampling": deepcopy(self.identity["sampling"]), "rules": deepcopy(self.identity["rules"]),
                    "execution_mode": "paper_only", "history_source": "live public API",
                    "quote_currency": next((suffix for suffix in ("USDT", "USDC", "USD", "BTC", "ETH")
                                            if self.market["symbol"].endswith(suffix)), "quote units"),
                    "backtest_difference": "Paper entries use the first fresh public quote observed after inference, not the historical next open. New closed candles are considered while flat; the backtest uses nonoverlapping 24-bar origins. Exits poll public quotes; offline recovery uses only complete bars after the last observation and cannot reconstruct unobserved intrabar prices.",
                    "incompatible": self._incompatible}

    def enable(self):
        if self._closed:
            raise RuntimeError("The paper runner is closed.")
        self._check_identity()
        self.broker.set_enabled(True)
        self.start()
        self.wake_event.set()
        return self.status()

    def pause(self):
        self.broker.set_enabled(False)
        self.wake_event.set()
        return self.status()

    def _fresh_quote(self):
        try:
            quote = self._read_quote()
        except Exception as exc:
            with self.lock:
                self._quote_error = str(exc)
                self.last_error = str(exc)
            raise
        with self.lock:
            if self._quote_error is not None and self.last_error == self._quote_error:
                self.last_error = None
            self._quote_error = None
        return quote

    def _read_quote(self):
        quote = dict(self.feed.quote())
        now = utc(self.clock())
        price = float(quote.get("price", 0))
        server = utc(quote["timestamp"])
        received = utc(quote.get("observed_at", now))
        if not np.isfinite(price) or price <= 0:
            raise ValueError("Public quote price must be positive and finite.")
        if not -5 <= (now - server).total_seconds() <= 30:
            raise ValueError("Public quote is over 30 seconds old or the exchange clock is over 5 seconds ahead.")
        if not -2 <= (now - received).total_seconds() <= 30:
            raise ValueError("The quote was not recently observed by this runner.")
        for key in ("exchange", "category", "symbol"):
            if key in quote and quote[key] != self.market.get(key):
                raise ValueError("Public quote market differs from this paper account.")
        quote.update(price=price, timestamp=server.isoformat(), observed_at=received.isoformat())
        return quote

    def _save_forecast(self, forecast):
        self.forecast_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=self.forecast_path.parent,
                                         prefix=".paper-forecast-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(dict(identity_digest=self.identity_digest, forecast=forecast), stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(self.forecast_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _recover(self, quote):
        state = self.broker.status()
        position = state.get("position")
        if not position:
            self._needs_recovery = False
            return
        cursor = state.get("recovery_cursor") or position.get("last_observed_time") or position["entry_time"]
        end = min(utc(quote["observed_at"]), utc(position["horizon_end"]))
        start = utc(cursor)
        if start >= end:
            self._needs_recovery = False
            return
        frame = self.feed.candles(start=start, end=end)
        notices = ["A partial candle containing the last observed quote is excluded; intrabar prices during that interval are unknown."]
        previous = None
        for row in frame.sort_values("timestamps").to_dict("records") if not frame.empty else []:
            stamp = utc(row["timestamps"])
            if previous is not None and next_open(previous, self.market["interval"]) != stamp:
                notices.append(f"Missing closed candles between {previous.isoformat()} and {stamp.isoformat()}; no prices were fabricated.")
            previous = stamp
            if "close_time" not in row:
                row["close_time"] = next_open(stamp, self.market["interval"]) - pd.Timedelta(milliseconds=1)
            self.broker.process_bar(row)
            if not self.broker.status().get("position"):
                break
        if frame.empty:
            notices.append("No complete recovery candles were returned for the observation gap.")
        with self.lock:
            self.recovery = {"last_recovered_at": utc(self.clock()).isoformat(), "notices": notices}
        self._needs_recovery = False

    def tick(self):
        """One quote-monitor iteration, also available to deterministic tests."""
        if self._closed:
            return
        now = utc(self.clock())
        with self.lock:
            self.last_poll = now.isoformat()
        try:
            with self.observation_lock:
                quote = self._fresh_quote()
                gap = (self._last_successful_quote is not None and
                       (now - self._last_successful_quote).total_seconds() > max(45, self.poll_seconds * 2))
                if self._needs_recovery or gap:
                    try:
                        self._recover(quote)
                    except Exception as exc:
                        with self.lock:
                            self.last_error = f"Exit recovery incomplete: {exc}. Monitoring the current quote continues."
                            self.recovery = {"last_recovered_at": now.isoformat(), "notices": [str(exc)]}
                # Local receipt time records when an observation could actually
                # have executed. Exchange timestamps remain in quote metadata.
                self.broker.process_quote(quote["price"], quote["observed_at"])
                self._last_successful_quote = now
                with self.lock:
                    self.last_quote = quote
                self._attempt_pending(quote)
            state = self.broker.status()
            if (state.get("enabled") or state.get("position")) and not self._incompatible:
                self._check_identity()
            if not state.get("enabled") or state.get("position"):
                return
            if self._incompatible:
                return
            origin = state.get("last_processed_origin")
            if origin is not None:
                boundary = next_open(next_open(utc(origin) if not isinstance(origin, (int, float)) else pd.Timestamp(origin, unit="s", tz="UTC"),
                                               self.market["interval"]), self.market["interval"])
                with self.lock:
                    self.next_candle_at = boundary.isoformat()
                if now < boundary:
                    return
            with self.lock:
                if self.future is None or self.future.done():
                    self.future = self.executor.submit(self._forecast_current)
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)

    @contextmanager
    def _forecast_lock(self):
        lock = getattr(self.service, "forecast_lock", None)
        if lock is None:
            yield
            return
        acquired = False
        try:
            while not acquired:
                if self.stop_event.is_set():
                    raise RuntimeError("Paper forecasting stopped.")
                acquired = lock.acquire(timeout=.2)
            yield
        finally:
            if acquired:
                lock.release()

    def _history_is_current(self, history):
        origin = utc(history["timestamps"].iloc[-1])
        closed = next_open(origin, self.market["interval"])
        following = next_open(closed, self.market["interval"])
        now = utc(self.clock())
        if closed > now or following <= now:
            raise ValueError("The latest context is unfinished or stale; waiting for the most recent closed candle.")
        return origin, following

    @staticmethod
    def _origin_stamp(value):
        return pd.Timestamp(value, unit="s", tz="UTC") if isinstance(value, (int, float)) else utc(value)

    def _attempt_pending(self, quote):
        state = self.broker.status()
        decision = state.get("last_decision")
        if not state.get("enabled") or state.get("position") or not decision or decision.get("status") != "pending":
            return
        if utc(quote["observed_at"]) < utc(decision["decision_time"]):
            return
        self._check_identity()
        origin = self._origin_stamp(decision["origin_time"])
        latest_boundary = next_open(next_open(origin, self.market["interval"]), self.market["interval"])
        if latest_boundary <= utc(self.clock()) or self.stop_event.is_set():
            if hasattr(self.broker, "cancel_decision"):
                self.broker.cancel_decision(decision["origin_time"], "A newer candle closed before entry; no historical entry was backfilled.")
            return
        position = self.broker.open_position(decision["plan"], quote["price"], quote["observed_at"],
                                             origin_time=decision["origin_time"], horizon_end=decision["horizon_end"])
        if position:
            with self.lock:
                self.last_error = None

    def _forecast_current(self):
        try:
            if self.stop_event.is_set() or not self.broker.status().get("enabled"):
                return
            self._check_identity()
            history = self.service.context(self.service.live())
            origin, following = self._history_is_current(history)
            origin_time = int(origin.timestamp())
            if any(int(self._origin_stamp(item["origin_time"]).timestamp()) == origin_time for item in self.broker.status().get("decisions", [])):
                return
            future = future_timestamps(origin, self.market["interval"], self.service.config.predict_window)
            with self.lock:
                self.forecast_origin = origin.isoformat()
                self.next_candle_at = following.isoformat()
                self.forecast_progress = 0

            def progress(value):
                if self.stop_event.is_set():
                    raise RuntimeError("Paper forecasting stopped.")
                with self.lock:
                    self.forecast_progress = float(value)

            with self._forecast_lock():
                self._check_identity()
                self._history_is_current(history)
                args = {key: value for key, value in self.identity["sampling"].items() if key != "batch_size"}
                paths = np.asarray(self.service.engine("finetuned").sample(history, future, **args, progress=progress))
            shape = (self.identity["sampling"]["sample_count"], len(future), 6)
            if paths.shape != shape or not np.isfinite(paths).all() or (paths[:, :, :4] <= 0).any():
                raise ValueError("The paper forecast returned invalid sample paths.")
            self._check_identity()
            summary = summarize_paths(paths[:, :, 3], float(history["close"].iloc[-1]),
                                      consensus_threshold=self.identity["rules"]["consensus_threshold"])
            plan = make_trade_plan(summary, paths[:, :, 3], history,
                                   **{key: self.identity["rules"][key] for key in (
                                       "fee_bps", "slippage_bps", "risk_fraction", "max_position_fraction", "minimum_reward_risk")})
            plan["entry_rule"] = "Paper: buy at the first fresh observed public quote after inference; recheck frozen levels and net reward/risk at that quote."
            completed = utc(self.clock())
            horizon_end = next_open(future.iloc[-1], self.market["interval"])
            try:
                self._history_is_current(history)
            except ValueError:
                plan.update(action="WAIT", reason="Forecast finished after a newer candle closed; no historical entry was backfilled.")
            decision = self.broker.record_decision(origin_time, plan, summary, decision_time=completed.isoformat(),
                                                   horizon_end=horizon_end.isoformat(),
                                                   forecast_source="live_closed_candles", fill_policy="first_observed_quote_after_forecast")
            times = [int(pd.Timestamp(value).timestamp()) for value in future]
            quantiles = np.quantile(paths[:, :, 3], [.1, .5, .9], axis=0)
            result = {"market": dict(self.market), "source": "paper_live", "origin_time": origin_time,
                      "generated_at": completed.isoformat(), "sampling": deepcopy(self.identity["sampling"]),
                      "context_length": len(history), "horizon": len(future),
                      "normalization": "Each model normalizes only its historical context; future candles are not used.",
                      "history": [{"time": int(row.timestamps.timestamp()), **{name: float(getattr(row, name)) for name in ("open", "high", "low", "close", "volume", "amount")}}
                                  for row in history.itertuples(index=False)],
                      "models": {"finetuned": {
                          "quantiles": [{"time": time, "p10": float(quantiles[0, i]), "p50": float(quantiles[1, i]), "p90": float(quantiles[2, i])}
                                        for i, time in enumerate(times)],
                          "paths": [[{"time": time, "value": float(value)} for time, value in zip(times, sample)] for sample in paths[:, :, 3]],
                          "analysis": summary, "plan": plan}}, "paper_decision": decision}
            self._save_forecast(result)
            with self.service.lock:
                self.service.latest_forecast = result
            with self.lock:
                self.last_forecast = result
                self.forecast_progress = 1.0
                self.last_error = None
            if plan["action"] == "BUY" and not self.stop_event.is_set():
                with self.observation_lock:
                    quote = self._fresh_quote()
                    self._attempt_pending(quote)
                    with self.lock:
                        self.last_quote = quote
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)

    def close_position(self):
        with self.observation_lock:
            quote = self._fresh_quote()
            self.broker.close_position(quote["price"], quote["observed_at"], reason="manual")
            with self.lock:
                self.last_quote = quote
        return self.status()

    def start(self):
        with self.lock:
            if self._closed:
                raise RuntimeError("The paper runner is closed.")
            if self.worker and self.worker.is_alive():
                return
            self.worker = threading.Thread(target=self._loop, daemon=True, name="kronos-paper-quotes")
            self.worker.start()

    def _loop(self):
        while not self.stop_event.is_set():
            self.tick()
            self.wake_event.wait(self.poll_seconds)
            self.wake_event.clear()

    def close(self):
        with self.lock:
            if self._closed:
                return
            self._closed = True
        self.stop_event.set()
        self.wake_event.set()
        if self.worker and self.worker is not threading.current_thread():
            self.worker.join(timeout=30)
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.broker.close()
