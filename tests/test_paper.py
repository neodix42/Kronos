from copy import deepcopy
import json

import pandas as pd
import pytest

from market.paper import PaperAccountLockedError, PaperBroker, PaperIdentityError


IDENTITY = {"exchange": "bybit", "category": "spot", "symbol": "BTCUSDT",
            "interval": "1h", "checkpoint_hash": "abc", "tokenizer_hash": "def"}
PLAN = {"action": "BUY", "stop_price": 95., "take_profit_price": 115., "reason": "Bullish"}


class Clock:
    def __init__(self):
        self.now = pd.Timestamp("2026-01-01T12:00:00Z")

    def __call__(self):
        return self.now

    def advance(self, **duration):
        self.now += pd.Timedelta(**duration)
        return self.now


@pytest.fixture
def account(tmp_path):
    clock = Clock()
    broker = PaperBroker(tmp_path / "account.json", IDENTITY, clock=clock)
    yield broker, clock
    broker.close()


def decision(broker, clock, *, plan=None, origin=None, horizon=None):
    plan = deepcopy(plan or PLAN)
    origin = origin if origin is not None else clock.now - pd.Timedelta(hours=1)
    horizon = horizon if horizon is not None else clock.now + pd.Timedelta(hours=24)
    broker.record_decision(origin, plan, {"regime": "TRENDING"},
                           decision_time=clock.now, horizon_end=horizon)
    return origin, horizon, plan


def enter(broker, clock, price=100, **kwargs):
    broker.set_enabled(True)
    origin, horizon, plan = decision(broker, clock, **kwargs)
    return broker.open_position(plan, price, clock.now, origin_time=origin, horizon_end=horizon)


def test_paused_by_default_and_no_order_before_enabled(account):
    broker, clock = account
    origin, horizon, plan = decision(broker, clock)
    assert not broker.status()["enabled"]
    assert broker.open_position(plan, 100, clock.now, origin_time=origin, horizon_end=horizon) is None
    assert broker.status()["last_decision"]["status"] == "cancelled"


def test_entry_sizing_costs_and_net_pnl_use_actual_quotes(account):
    broker, clock = account
    position = enter(broker, clock)
    fill = 100 * 1.0005
    entry_cost = fill * 1.001
    loss_per_unit = entry_cost - 95 * .9995 * .999
    quantity = min(2500 / entry_cost, 100 / loss_per_unit)
    assert position["quantity"] == pytest.approx(quantity)
    assert broker.status()["cash"] == pytest.approx(10_000 - quantity * entry_cost)
    mark = broker.status(102)
    assert mark["equity"] == pytest.approx(mark["cash"] + quantity * 102)
    assert mark["equity"] - 10_000 == pytest.approx(mark["unrealized_pnl"])
    trade = broker.close_position(110, clock.advance(minutes=10))
    expected_fee = quantity * (fill + 110 * .9995) * .001
    expected_slip = quantity * (fill - 100 + 110 - 110 * .9995)
    expected_pnl = quantity * (110 * .9995 - fill) - expected_fee
    status = broker.status()
    assert trade["fees"] == pytest.approx(expected_fee)
    assert trade["slippage"] == pytest.approx(expected_slip)
    assert trade["pnl"] == pytest.approx(expected_pnl)
    assert status["cash"] == pytest.approx(10_000 + expected_pnl)
    assert status["realized_pnl"] == pytest.approx(expected_pnl)
    assert status["total_fees"] == pytest.approx(expected_fee)
    assert status["position"] is None
    assert status["trade_count"] == 1


def test_position_cap_applies_including_entry_fees(account):
    broker, clock = account
    position = enter(broker, clock, plan={**PLAN, "stop_price": 99., "take_profit_price": 115.})
    assert position["quantity"] * position["entry_price"] + position["entry_fee"] == pytest.approx(2500)


@pytest.mark.parametrize("quote,expected_reason", [(90, "crossed"), (120, "crossed"), (110, "reward/risk")])
def test_entry_rechecks_frozen_levels_and_cost_adjusted_rr(account, quote, expected_reason):
    broker, clock = account
    broker.set_enabled(True)
    origin, horizon, plan = decision(broker, clock)
    assert broker.open_position(plan, quote, clock.now, origin_time=origin, horizon_end=horizon) is None
    record = broker.status()["last_decision"]
    assert record["status"] == "rejected"
    assert expected_reason in record["execution_reason"]
    assert broker.open_position(plan, 100, clock.now, origin_time=origin, horizon_end=horizon) is None


def test_origins_are_idempotent_and_positions_do_not_overlap(account):
    broker, clock = account
    position = enter(broker, clock)
    origin = position["forecast_origin"]
    previous = broker.record_decision(origin, PLAN, decision_time=clock.now)
    assert previous["status"] == "opened"
    assert len(broker.status()["decisions"]) == 1
    newer_origin, horizon, plan = decision(broker, clock, origin=clock.now)
    assert broker.open_position(plan, 100, clock.now, origin_time=newer_origin, horizon_end=horizon) is None
    assert len(broker.status()["decisions"]) == 2
    assert broker.status()["position"]["id"] == position["id"]
    stale = broker.record_decision(clock.now - pd.Timedelta(days=1), PLAN)
    assert stale["status"] == "duplicate_or_stale"


def test_pause_cancels_pending_entries_but_still_manages_open_position(account):
    broker, clock = account
    enter(broker, clock)
    broker.set_enabled(False)
    trade = broker.process_quote(94, clock.advance(seconds=10))
    assert trade["exit_reason"] == "stop_loss"
    assert not broker.status()["enabled"]
    broker.set_enabled(True)
    origin, horizon, plan = decision(broker, clock, origin=clock.now)
    broker.set_enabled(False)
    broker.set_enabled(True)
    assert broker.open_position(plan, 100, clock.now, origin_time=origin, horizon_end=horizon) is None
    assert broker.status()["last_decision"]["status"] == "cancelled"


def test_quote_stop_overshoot_and_target_gap_are_conservative(account):
    broker, clock = account
    enter(broker, clock)
    first = broker.process_quote(90, clock.advance(seconds=15))
    assert first["exit_base_price"] == 90
    assert first["exit_price"] == pytest.approx(90 * .9995)
    clock.advance(seconds=1)
    enter(broker, clock, origin=clock.now)
    second = broker.process_quote(130, clock.advance(seconds=15))
    assert second["exit_observed_price"] == 130
    assert second["exit_base_price"] == 115
    assert second["exit_price"] == pytest.approx(115 * .9995)


def test_pending_cancellation_and_newer_forecast_prevent_old_entries(account):
    broker, clock = account
    broker.set_enabled(True)
    origin, horizon, plan = decision(broker, clock)
    cancelled = broker.cancel_decision(origin, "Origin expired while waiting for quote")
    assert cancelled["status"] == "cancelled"
    assert broker.open_position(plan, 100, clock.now, origin_time=origin, horizon_end=horizon) is None
    origin, horizon, plan = decision(broker, clock, origin=clock.now)
    clock.advance(seconds=1)
    decision(broker, clock, origin=clock.now)
    assert broker.open_position(plan, 100, clock.now, origin_time=origin, horizon_end=horizon) is None
    assert "Superseded" in broker.status()["decisions"][-2]["execution_reason"]


def test_horizon_uses_first_actual_quote_after_expiry(account):
    broker, clock = account
    enter(broker, clock, horizon=clock.now + pd.Timedelta(minutes=1))
    assert broker.process_quote(101, clock.advance(seconds=30)) is None
    trade = broker.process_quote(103, clock.advance(seconds=31))
    assert trade["exit_reason"] == "horizon"
    assert trade["exit_base_price"] == 103


def test_expired_forecast_never_enters(account):
    broker, clock = account
    broker.set_enabled(True)
    origin, horizon, plan = decision(broker, clock, horizon=clock.now - pd.Timedelta(seconds=1))
    assert broker.status()["last_decision"]["status"] == "expired"
    assert broker.open_position(plan, 100, clock.now, origin_time=origin, horizon_end=horizon) is None


def test_quotes_are_forward_only_fresh_and_never_before_decision(account):
    broker, clock = account
    broker.set_enabled(True)
    origin, horizon, plan = decision(broker, clock)
    with pytest.raises(ValueError, match="precedes the forecast decision"):
        broker.open_position(plan, 100, clock.now - pd.Timedelta(seconds=1),
                             origin_time=origin, horizon_end=horizon)
    broker.open_position(plan, 100, clock.now, origin_time=origin, horizon_end=horizon)
    observed = clock.advance(seconds=30)
    broker.process_quote(102, observed)
    with pytest.raises(ValueError, match="precedes the last"):
        broker.process_quote(101, observed - pd.Timedelta(seconds=1))
    with pytest.raises(ValueError, match="Conflicting"):
        broker.process_quote(103, observed)
    with pytest.raises(ValueError, match="stale"):
        broker.process_quote(102, clock.advance(minutes=3) - pd.Timedelta(minutes=3))
    with pytest.raises(ValueError, match="future"):
        broker.process_quote(102, clock.now + pd.Timedelta(minutes=1))
    with pytest.raises(ValueError, match="milliseconds"):
        broker.process_quote(102, clock.now.timestamp() * 1000)


def bar(clock, *, start=None, **values):
    start = start or clock.now
    return {"timestamps": start, "close_time": start + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            "open": 100, "high": 110, "low": 96, "close": 105, **values}


def test_ohlc_recovery_stop_first_and_idempotent(account):
    broker, clock = account
    enter(broker, clock)
    candle = bar(clock, high=120, low=90)
    clock.advance(hours=1)
    trade = broker.process_bar(candle)
    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_base_price"] == 95
    assert trade["reconstruction"]
    assert broker.process_bar(candle) is None
    assert broker.status()["trade_count"] == 1


def test_ohlc_gap_stop_and_horizon(account):
    broker, clock = account
    enter(broker, clock)
    candle = bar(clock, open=90, low=85, close=92)
    clock.advance(hours=1)
    assert broker.process_bar(candle)["exit_base_price"] == 90
    enter(broker, clock, origin=clock.now, horizon=clock.now + pd.Timedelta(hours=1))
    candle = bar(clock)
    clock.advance(hours=1)
    assert broker.process_bar(candle)["exit_reason"] == "horizon"


def test_no_retroactive_stop_from_before_entry_or_prior_quote(account):
    broker, clock = account
    clock.advance(minutes=30)
    enter(broker, clock)
    overlapping = bar(clock, start=clock.now.floor("h"), low=90)
    clock.advance(minutes=30)
    assert broker.process_bar(overlapping) is None
    assert broker.status()["position"] is not None
    broker.process_quote(101, clock.advance(minutes=30))
    partial = bar(clock, start=clock.now.floor("h"), low=90)
    clock.advance(minutes=30)
    assert broker.process_bar(partial) is None
    assert broker.status()["position"] is not None
    assert len(broker.status()["notices"]) == 2


def test_recovery_cannot_use_unfinished_bar_or_post_horizon_extremes(account):
    broker, clock = account
    enter(broker, clock, horizon=clock.now + pd.Timedelta(minutes=30))
    candle = bar(clock, low=90)
    with pytest.raises(ValueError, match="completed"):
        broker.process_bar(candle)
    clock.advance(hours=1)
    assert broker.process_bar(candle) is None
    assert broker.status()["position"] is not None
    assert "forecast expiry" in broker.status()["notices"][-1]
    assert broker.process_quote(102, clock.now)["exit_reason"] == "horizon"


def test_restart_preserves_account_enabled_position_and_notices(tmp_path):
    clock = Clock()
    path = tmp_path / "ledger.json"
    first = PaperBroker(path, IDENTITY, clock=clock)
    position = enter(first, clock)
    first.close()
    clock.advance(hours=2)
    second = PaperBroker(path, IDENTITY, clock=clock)
    try:
        status = second.status()
        assert status["enabled"]
        assert status["position"] == position
        assert status["last_processed_origin"] == position["forecast_origin"]
        assert "Restart gap" in status["notices"][-1]
        assert status["cash"] < 10_000
        assert status["trade_count"] == 0
    finally:
        second.close()


def test_single_writer_lock_and_identity_protection(tmp_path):
    path = tmp_path / "ledger.json"
    broker = PaperBroker(path, IDENTITY)
    with pytest.raises(PaperAccountLockedError):
        PaperBroker(path, IDENTITY)
    broker.close()
    before = path.read_bytes()
    with pytest.raises(PaperIdentityError):
        PaperBroker(path, {**IDENTITY, "checkpoint_hash": "changed"})
    assert path.read_bytes() == before
    with pytest.raises(PaperIdentityError):
        PaperBroker(path, IDENTITY, fee_bps=0)
    with PaperBroker(path, IDENTITY) as reopened:
        assert reopened.status()["cash"] == 10_000


def test_atomic_write_failure_rolls_back_memory_and_existing_ledger(account, monkeypatch):
    broker, clock = account
    before, disk = broker.status(), broker.path.read_bytes()
    def failed(state):
        raise OSError("disk failure")
    monkeypatch.setattr(broker, "_persist", failed)
    with pytest.raises(OSError, match="disk failure"):
        broker.set_enabled(True)
    assert broker.status() == before
    assert broker.path.read_bytes() == disk


@pytest.mark.parametrize("setting,value", [("fee_bps", float("nan")), ("fee_bps", float("inf")),
    ("slippage_bps", -1), ("slippage_bps", 10000), ("risk_fraction", float("nan")),
    ("max_position_fraction", float("inf")), ("minimum_reward_risk", float("nan")),
    ("initial_equity", True), ("risk_fraction", True), ("fee_bps", False),
    ("max_quote_age_seconds", 0)])
def test_invalid_numeric_parameters_are_rejected(tmp_path, setting, value):
    with pytest.raises(ValueError):
        PaperBroker(tmp_path / "account.json", IDENTITY, **{setting: value})


def test_currency_comes_from_nested_market_symbol(tmp_path):
    with PaperBroker(tmp_path / "usdc.json", {"market": {"symbol": "BTCUSDC"}}) as broker:
        assert broker.status()["rules"]["account_currency"] == "USDC"


def test_restart_pending_entry_cannot_fill_old_quote(tmp_path):
    clock, path = Clock(), tmp_path / "account.json"
    with PaperBroker(path, IDENTITY, clock=clock) as first:
        first.set_enabled(True)
        origin, horizon, plan = decision(first, clock)
    clock.advance(minutes=5)
    with PaperBroker(path, IDENTITY, clock=clock) as second:
        with pytest.raises(ValueError, match="stale"):
            second.open_position(plan, 100, clock.now - pd.Timedelta(minutes=5),
                                 origin_time=origin, horizon_end=horizon)
        assert second.status()["position"] is None
        assert second.status()["trade_count"] == 0


def test_drawdown_marks_every_observation_and_costs_not_double_counted(account):
    broker, clock = account
    enter(broker, clock, plan={**PLAN, "stop_price": 80., "take_profit_price": 150.})
    broker.process_quote(90, clock.advance(seconds=15))
    status = broker.status()
    assert status["max_drawdown"] > 0
    assert status["equity"] == pytest.approx(10_000 + status["realized_pnl"] + status["unrealized_pnl"])
    broker.close_position(100, clock.advance(seconds=15))
    status = broker.status()
    assert status["equity"] == pytest.approx(10_000 + status["realized_pnl"])
    assert status["realized_pnl"] < 0
    assert status["costs"] == pytest.approx(status["total_fees"] + status["total_slippage"])
    persisted = json.loads(broker.path.read_text())
    assert persisted["cash"] == status["cash"]
    assert persisted["trades"] == status["trades"]
