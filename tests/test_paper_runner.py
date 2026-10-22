"""Public-feed and automatic-paper scheduling tests never place real orders."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from finetune_csv.paper_trading import PaperTrader
from market.public_feed import PublicMarketFeed


class Clock:
    def __init__(self, value="2026-01-02T12:05:00Z"):
        self.now = pd.Timestamp(value)

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += pd.Timedelta(**kwargs)


def history(clock, count=8):
    end = clock().floor("h") - pd.Timedelta(hours=1)
    return pd.DataFrame({"timestamps": pd.date_range(end=end, periods=count, freq="h"),
                         "open": 100., "high": 101., "low": 99., "close": 100.,
                         "volume": 1., "amount": 100.})


class Feed:
    def __init__(self, clock):
        self.clock = clock
        self.price = 100.
        self.age = 0
        self.calls = 0
        self.bar_calls = []
        self.bars = pd.DataFrame()

    def quote(self):
        self.calls += 1
        return {"price": self.price, "timestamp": (self.clock() - pd.Timedelta(seconds=self.age)).isoformat(),
                "observed_at": self.clock().isoformat(), "symbol": "BTCUSDT", "exchange": "bybit", "category": "spot"}

    def candles(self, **kwargs):
        self.bar_calls.append(kwargs)
        return self.bars.copy()


class Engine:
    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = None
        self.after = None
        self.level = 110

    def sample(self, frame, future, **kwargs):
        self.calls.append((frame.copy(), future.copy(), kwargs))
        self.entered.set()
        if self.release:
            assert self.release.wait(timeout=5)
        if self.after:
            self.after()
        kwargs["progress"](1)
        return np.full((kwargs["sample_count"], len(future), 6), float(self.level))


class Service:
    def __init__(self, tmp_path, clock):
        self.clock = clock
        self.market = dict(exchange="bybit", category="spot", symbol="BTCUSDT", interval="1h")
        self.evaluation = dict(fee_bps=10, slippage_bps=5)
        self.config = SimpleNamespace(base_save_path=str(tmp_path / "run"), lookback_window=8,
                                      predict_window=4, max_context=12, clip=5.)
        self.checkpoint = tmp_path / "model"
        self.trained_tokenizer = tmp_path / "tokenizer"
        self.data_path = tmp_path / "candles.csv"
        self.config_path = tmp_path / "config.yaml"
        self.config_path.write_text("experiment: fixture")
        history(clock).to_csv(self.data_path, index=False)
        for folder in (self.checkpoint, self.trained_tokenizer):
            folder.mkdir()
            (folder / "config.json").write_text("{}")
            (folder / "model.safetensors").write_bytes(b"fixture")
        self.model = Engine()
        self.lock = threading.RLock()
        self.forecast_lock = threading.RLock()
        self.latest_forecast = None
        self.live_calls = 0
        self.old_history = None

    def sampling(self):
        return dict(sample_count=100, batch_size=8, seed=42, temperature=1., top_p=.9, top_k=0)

    def live(self):
        self.live_calls += 1
        return history(self.clock) if self.old_history is None else self.old_history

    def context(self, frame):
        return frame.copy()

    def engine(self, kind):
        assert kind == "finetuned"
        return self.model


@pytest.fixture
def setup(tmp_path):
    clock = Clock()
    service = Service(tmp_path, clock)
    feed = Feed(clock)
    trader = PaperTrader(service, feed=feed, clock=clock)
    yield trader, service, feed, clock
    trader.close()


def complete(trader):
    assert trader.future is not None
    trader.future.result(timeout=5)


def test_status_is_local_only_and_default_paused(setup):
    trader, service, feed, clock = setup
    state = trader.status()
    assert state["enabled"] is False
    assert state["execution_mode"] == "paper_only"
    assert feed.calls == 0 and service.live_calls == 0


def test_once_per_new_origin_first_observed_fill_and_paused_exits(setup):
    trader, service, feed, clock = setup
    trader.broker.set_enabled(True)
    service.model.after = lambda: clock.advance(seconds=30)
    trader.tick()
    complete(trader)
    state = trader.status()
    position = state["position"]
    assert position and pd.Timestamp(position["entry_time"]) == clock()
    assert position["entry_quote"] == 100
    assert position["entry_time"] != "2026-01-02T12:00:00+00:00"
    assert len(service.model.calls) == 1 and service.model.calls[0][2]["sample_count"] == 100
    assert state["total_fees"] > 0 and state["total_slippage"] > 0
    assert service.latest_forecast["source"] == "paper_live"
    assert set(service.latest_forecast["models"]) == {"finetuned"}
    assert "historical context" in service.latest_forecast["normalization"]
    trader.pause()
    clock.advance(seconds=15)
    feed.price = 95
    trader.tick()
    state = trader.status()
    assert state["position"] is None
    assert len(state["trades"]) == 1 and state["trades"][0]["exit_reason"] == "stop_loss"
    assert state["enabled"] is False


def test_wait_decision_persists_and_restart_does_not_repeat_origin(setup):
    trader, service, feed, clock = setup
    service.model.level = 90
    trader.broker.set_enabled(True)
    trader.tick()
    complete(trader)
    assert trader.status()["latest_decision"]["plan"]["action"] == "WAIT"
    for _ in range(3):
        clock.advance(seconds=15)
        trader.tick()
    assert len(service.model.calls) == 1
    trader.close()
    service.latest_forecast = None
    resumed = PaperTrader(service, feed=feed, clock=clock)
    try:
        assert resumed.status()["enabled"] is True
        assert service.latest_forecast["source"] == "paper_live"
        assert len(service.latest_forecast["models"]["finetuned"]["paths"]) == 100
        resumed.tick()
        assert len(service.model.calls) == 1
        clock.advance(hours=1)
        resumed.tick()
        complete(resumed)
        assert len(service.model.calls) == 2
    finally:
        resumed.close()


def test_cached_forecast_from_another_identity_is_not_displayed(setup):
    import json
    trader, service, feed, clock = setup
    service.model.level = 90
    trader.broker.set_enabled(True)
    trader.tick()
    complete(trader)
    path = trader.forecast_path
    cache = json.loads(path.read_text())
    cache["identity_digest"] = "different experiment"
    path.write_text(json.dumps(cache))
    trader.close()
    service.latest_forecast = None
    resumed = PaperTrader(service, feed=feed, clock=clock)
    try:
        assert resumed.last_forecast is None and service.latest_forecast is None
        assert resumed.status()["latest_decision"]["action"] == "WAIT"
    finally:
        resumed.close()


def test_pause_while_inference_running_cannot_open_position(setup):
    trader, service, feed, clock = setup
    release = service.model.release = threading.Event()
    trader.broker.set_enabled(True)
    trader.tick()
    assert service.model.entered.wait(timeout=2)
    trader.pause()
    release.set()
    complete(trader)
    assert trader.status()["position"] is None
    assert trader.status()["latest_decision"]["status"] == "cancelled"


def test_quote_monitor_remains_responsive_while_gpu_forecast_is_blocked(setup):
    trader, service, feed, clock = setup
    release = service.model.release = threading.Event()
    trader.broker.set_enabled(True)
    trader.tick()
    assert service.model.entered.wait(timeout=2)
    first_calls = feed.calls
    clock.advance(seconds=15)
    trader.tick()
    assert feed.calls == first_calls + 1
    assert trader.status()["forecast_running"] is True
    release.set()
    complete(trader)


def test_forecast_crossing_new_candle_boundary_never_backfills_entry(setup):
    trader, service, feed, clock = setup
    service.model.after = lambda: clock.advance(hours=1)
    trader.broker.set_enabled(True)
    trader.tick()
    complete(trader)
    state = trader.status()
    assert state["position"] is None
    assert "newer candle" in state["latest_decision"]["plan"]["reason"]


def test_stale_context_and_stale_quotes_never_open(setup):
    trader, service, feed, clock = setup
    trader.broker.set_enabled(True)
    feed.age = 31
    trader.tick()
    assert "30 seconds" in trader.status()["last_error"]
    assert service.live_calls == 0
    feed.age = 0
    service.old_history = history(clock)
    service.old_history["timestamps"] -= pd.Timedelta(hours=1)
    trader.tick()
    complete(trader)
    assert "stale" in trader.status()["last_error"]
    assert service.model.calls == [] and trader.status()["position"] is None


def test_quote_recovery_clears_error_without_waiting_for_another_forecast(setup):
    trader, service, feed, clock = setup
    service.model.level = 90
    trader.broker.set_enabled(True)
    trader.tick()
    complete(trader)
    assert trader.status()["latest_decision"]["action"] == "WAIT"
    clock.advance(seconds=15)
    feed.age = 31
    trader.tick()
    assert "30 seconds" in trader.status()["last_error"]
    clock.advance(seconds=15)
    feed.age = 0
    trader.tick()
    assert trader.status()["last_error"] is None
    assert len(service.model.calls) == 1


def test_healthy_quotes_do_not_hide_failed_forecasts(setup):
    trader, service, feed, clock = setup

    def fail_model():
        raise RuntimeError("Forecast computation failed")

    service.model.after = fail_model
    trader.broker.set_enabled(True)
    trader.tick()
    complete(trader)
    assert trader.status()["last_error"] == "Forecast computation failed"
    trader.pause()
    clock.advance(seconds=15)
    trader.tick()
    assert trader.status()["last_error"] == "Forecast computation failed"


def test_changed_model_pauses_entries_but_keeps_existing_exit_monitor(setup):
    trader, service, feed, clock = setup
    trader.broker.set_enabled(True)
    trader.tick()
    complete(trader)
    assert trader.status()["position"]
    (service.checkpoint / "model.safetensors").write_bytes(b"different weights")
    clock.advance(seconds=15)
    trader.tick()
    assert trader.status()["incompatible"] is True and trader.status()["enabled"] is False
    clock.advance(seconds=15)
    feed.price = 95
    trader.tick()
    assert trader.status()["position"] is None
    assert len(trader.status()["trades"]) == 1
    with pytest.raises(ValueError, match="incompatible"):
        trader.enable()


def test_restart_recovers_complete_offline_bars_before_current_quote(setup):
    trader, service, feed, clock = setup
    trader.broker.set_enabled(True)
    trader.tick()
    complete(trader)
    trader.pause()
    trader.close()
    clock.advance(hours=3)
    feed.bars = pd.DataFrame([
        dict(timestamps=pd.Timestamp("2026-01-02T13:00Z"), open=100, high=101, low=90, close=95,
             volume=1, amount=95, close_time=pd.Timestamp("2026-01-02T13:59:59.999Z")),
    ])
    resumed = PaperTrader(service, feed=feed, clock=clock)
    try:
        resumed.tick()
        state = resumed.status()
        assert state["position"] is None
        assert state["trades"][0]["exit_reason"] == "stop_loss"
        assert feed.bar_calls
        assert state["recovery"]["notices"]
    finally:
        resumed.close()


def test_pending_entry_retries_next_fresh_quote_without_reforecasting(setup):
    trader, service, feed, clock = setup
    trader.broker.set_enabled(True)
    service.model.after = lambda: setattr(feed, "age", 31)
    trader.tick()
    complete(trader)
    assert trader.status()["latest_decision"]["status"] == "pending"
    assert trader.status()["position"] is None
    feed.age = 0
    clock.advance(seconds=15)
    trader.tick()
    assert trader.status()["position"]
    assert len(service.model.calls) == 1
    assert trader.status()["last_error"] is None


def test_entry_quote_fetch_crossing_boundary_cancels_pending_old_origin(setup):
    trader, service, feed, clock = setup
    trader.broker.set_enabled(True)
    original = feed.quote

    def delayed():
        if feed.calls == 1:
            clock.advance(hours=1)
        return original()

    feed.quote = delayed
    trader.tick()
    complete(trader)
    assert trader.status()["position"] is None
    assert trader.status()["latest_decision"]["status"] == "cancelled"
    assert "newer candle" in trader.status()["latest_decision"]["execution_reason"]


def test_public_bybit_quote_uses_only_public_spot_ticker_and_server_snapshot():
    clock = Clock()
    client = Mock()
    client._get.return_value = {"time": int(clock().timestamp() * 1000), "result": {
        "category": "spot", "list": [{"symbol": "BTCUSDT", "lastPrice": "100"}]}}
    feed = PublicMarketFeed(dict(exchange="bybit", category="spot", symbol="BTCUSDT", interval="1h"), client=client, clock=clock)
    quote = feed.quote()
    assert quote["price"] == 100 and quote["timestamp"] == clock().isoformat()
    client._get.assert_called_once_with("/v5/market/tickers", {"category": "spot", "symbol": "BTCUSDT"})


def test_public_binance_quote_observes_price_and_server_clock_without_order_endpoints():
    clock = Clock()
    client = Mock()
    client._get.return_value = {"symbol": "BTCUSDT", "price": "100"}
    client.server_time.return_value = int(clock().timestamp() * 1000)
    feed = PublicMarketFeed(dict(exchange="binance", category="spot", symbol="BTCUSDT", interval="1h"), client=client, clock=clock)
    quote = feed.quote()
    assert quote["price"] == 100
    client._get.assert_called_once_with("/api/v3/ticker/price", {"symbol": "BTCUSDT"})
    client.server_time.assert_called_once()


@pytest.mark.parametrize("bad", ["nan", "-1", "0"])
def test_public_quote_rejects_invalid_prices(bad):
    clock = Clock()
    client = Mock()
    client._get.return_value = {"symbol": "BTCUSDT", "price": bad}
    client.server_time.return_value = int(clock().timestamp() * 1000)
    feed = PublicMarketFeed(dict(exchange="binance", category="spot", symbol="BTCUSDT", interval="1h"), client=client, clock=clock)
    with pytest.raises(ValueError, match="invalid"):
        feed.quote()
