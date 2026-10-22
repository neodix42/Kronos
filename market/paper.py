"""Durable, single-writer spot paper account. This module has no order API."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
import threading

import pandas as pd

from .strategy import _trade_economics, _validate_costs


class PaperAccountLockedError(RuntimeError):
    pass


class PaperIdentityError(ValueError):
    pass


def _timestamp(value):
    if value is None or isinstance(value, bool):
        raise ValueError("Timestamps must be UTC datetimes, ISO strings, or epoch seconds")
    if isinstance(value, Real):
        if not math.isfinite(value) or abs(value) >= 100_000_000_000:
            raise ValueError("Numeric timestamps must use epoch seconds, not milliseconds")
        value = pd.Timestamp(value, unit="s", tz="UTC")
    else:
        value = pd.Timestamp(value)
        if pd.isna(value):
            raise ValueError("Timestamp must be present")
        value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    return value


def _iso(value):
    return _timestamp(value).isoformat().replace("+00:00", "Z")


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp)):
        return _iso(value)
    raise TypeError(f"Unsupported ledger value: {type(value).__name__}")


def _plain(value):
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


class PaperBroker:
    """A persisted virtual USDT account using adverse fills and costs on both sides.

    The account starts paused. Pausing cancels pending entries while existing
    positions remain managed. Decisions are consumed once, including rejected
    entry attempts. Quotes use epoch seconds or UTC datetimes, never epoch ms.
    ``clock`` can be injected for deterministic simulations and tests.
    """

    def __init__(self, path, identity, *, initial_equity=10_000, fee_bps=10,
                 slippage_bps=5, risk_fraction=.01, max_position_fraction=.25,
                 minimum_reward_risk=1.5, clock=None, max_quote_age_seconds=120):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = _plain(identity)
        if not isinstance(self.identity, dict) or not self.identity:
            raise ValueError("Paper account identity must be a nonempty mapping")
        self.initial_equity = _positive(initial_equity, "Initial equity")
        if any(isinstance(value, bool) for value in (fee_bps, slippage_bps, risk_fraction, max_position_fraction)):
            raise ValueError("Fees, slippage, risk, and position fractions must be numbers, not booleans")
        self.fee, self.slip = _validate_costs(fee_bps, slippage_bps)
        if not 0 < risk_fraction <= 1 or not 0 < max_position_fraction <= 1:
            raise ValueError("Risk and position fractions must be greater than zero and at most one")
        self.minimum_reward_risk = _positive(minimum_reward_risk, "Minimum reward/risk")
        self.max_quote_age_seconds = _positive(max_quote_age_seconds, "Maximum quote age")
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        market = self.identity.get("market", {})
        symbol = self.identity.get("symbol") or (market.get("symbol", "") if isinstance(market, dict) else "")
        currency = self.identity.get("quote_currency") or next(
            (suffix for suffix in ("FDUSD", "USDT", "USDC", "BUSD", "TUSD", "USD", "BTC", "ETH", "EUR", "GBP")
             if symbol.endswith(suffix)), "USDT")
        self.rules = {"market_type": "spot_long_only", "fee_bps": float(fee_bps),
                      "slippage_bps": float(slippage_bps), "risk_fraction": float(risk_fraction),
                      "max_position_fraction": float(max_position_fraction),
                      "minimum_reward_risk": self.minimum_reward_risk,
                      "target_fill": "min(observed quote, target), followed by adverse slippage",
                      "stop_fill": "observed quote, or conservative full-bar stop/gap reconstruction",
                      "same_bar_ambiguity": "stop before target", "account_currency": currency}
        self._mutex = threading.RLock()
        self._closed = False
        self._lockfile = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
        try:
            fcntl.flock(self._lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lockfile.close()
            self._closed = True
            raise PaperAccountLockedError(f"Paper account already has a running writer: {self.path}") from exc
        try:
            now = _iso(self.clock())
            if self.path.exists():
                state = json.loads(self.path.read_text())
                if state.get("version") != 1:
                    raise ValueError("Unsupported paper ledger version")
                if (state.get("identity") != self.identity or state.get("rules") != self.rules or
                        state.get("initial_equity") != self.initial_equity):
                    raise PaperIdentityError("Paper ledger model, market, rules, or initial equity differ; use a separate account file")
                self._state = state
                def restarted(candidate):
                    cursor = candidate.get("recovery_cursor")
                    message = (f"Restart gap: live quotes after {cursor or candidate['updated_at']} "
                               f"until {now} were not observed. No historical entries are fabricated; "
                               "only complete bars after the last observation may reconstruct exits.")
                    self._notice(candidate, message)
                    self._event(candidate, "restart_gap", now, from_time=cursor, message=message)
                self._mutate(restarted)
            else:
                self._state = {"version": 1, "identity": self.identity, "rules": self.rules,
                               "initial_equity": self.initial_equity, "enabled": False,
                               "created_at": now, "updated_at": now, "cash": self.initial_equity,
                               "position": None, "trades": [], "decisions": [], "events": [],
                               "equity_curve": [], "notices": [], "last_processed_origin": None,
                               "last_quote_time": None, "last_quote_price": None, "last_mark_price": None,
                               "recovery_cursor": None,
                               "last_bar_end": None, "total_fees": 0., "total_slippage": 0.,
                               "realized_pnl": 0., "peak_equity": self.initial_equity,
                               "max_drawdown": 0., "next_event_id": 1, "next_trade_id": 1}
                self._event(self._state, "account_created", now)
                self._persist(self._state)
        except Exception:
            self.close()
            raise

    def _persist(self, state):
        encoded = json.dumps(state, indent=2, allow_nan=False) + "\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent,
                                             prefix=".paper-ledger-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _mutate(self, operation):
        with self._mutex:
            if self._closed:
                raise RuntimeError("Paper account is closed")
            candidate = deepcopy(self._state)
            result = operation(candidate)
            candidate["updated_at"] = _iso(self.clock())
            self._persist(candidate)
            self._state = candidate
            return deepcopy(result)

    @staticmethod
    def _notice(state, message):
        state["notices"].append(message)
        del state["notices"][:-100]

    @staticmethod
    def _event(state, kind, time, **details):
        state["events"].append({"id": state["next_event_id"], "type": kind,
                                "time": _iso(time), **_plain(details)})
        state["next_event_id"] += 1
        del state["events"][:-2000]

    def _mark(self, state, price, time):
        position = state["position"]
        equity = state["cash"] + (position["quantity"] * price if position else 0.)
        state["peak_equity"] = max(state["peak_equity"], equity)
        drawdown = max(0., 1 - equity / state["peak_equity"])
        state["max_drawdown"] = max(state["max_drawdown"], drawdown)
        point = {"time": _iso(time), "equity": equity}
        if state["equity_curve"] and state["equity_curve"][-1]["time"] == point["time"]:
            state["equity_curve"][-1] = point
        else:
            state["equity_curve"].append(point)
            del state["equity_curve"][:-2000]

    def status(self, mark_price=None):
        with self._mutex:
            if self._closed:
                raise RuntimeError("Paper account is closed")
            result = deepcopy(self._state)
        position = result["position"]
        mark = (_positive(mark_price, "Mark price") if mark_price is not None else
                result.get("last_mark_price") or result["last_quote_price"] or
                (position["entry_quote"] if position else None))
        equity = result["cash"] + (position["quantity"] * mark if position else 0.)
        unrealized = (position["quantity"] * (mark - position["entry_price"]) - position["entry_fee"]
                      if position else 0.)
        metrics = {"cash": result["cash"], "equity": equity, "unrealized_pnl": unrealized,
                   "realized_pnl": result["realized_pnl"], "net_return": equity / self.initial_equity - 1,
                   "max_drawdown": result["max_drawdown"], "trade_count": len(result["trades"]),
                   "total_fees": result["total_fees"], "total_slippage": result["total_slippage"],
                   "costs": result["total_fees"] + result["total_slippage"],
                   "win_rate": (sum(t["pnl"] > 0 for t in result["trades"]) / len(result["trades"])
                                if result["trades"] else 0.)}
        result.update(metrics)
        result["metrics"] = metrics
        result["last_decision"] = result["decisions"][-1] if result["decisions"] else None
        result["last_origin_time"] = result["last_processed_origin"]
        result["mark_price"] = mark
        result["trade_log"] = result["trades"]
        return result

    def set_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be true or false")
        def update(state):
            state["enabled"] = enabled
            if not enabled:
                for decision in state["decisions"]:
                    if decision["status"] == "pending":
                        decision.update(status="cancelled", execution_reason="Paper entries paused")
            self._event(state, "enabled" if enabled else "paused", self.clock())
            return enabled
        return self._mutate(update)

    def record_decision(self, origin_time, plan, summary=None, *, decision_time=None,
                        horizon_end=None, **metadata):
        origin = _iso(origin_time)
        at = _timestamp(decision_time if decision_time is not None else self.clock())
        if at < _timestamp(origin):
            raise ValueError("Decision cannot precede its forecast origin")
        if at > _timestamp(self.clock()) + pd.Timedelta(seconds=5):
            raise ValueError("Decision time is in the future")
        horizon = _iso(horizon_end) if horizon_end is not None else None
        clean_plan, clean_summary = _plain(plan), _plain(summary or {})
        def record(state):
            for previous in state["decisions"]:
                if previous["origin_time"] == origin:
                    return previous
            status, reason = "pending", "Awaiting a fresh observed quote"
            if state["last_processed_origin"] and _timestamp(origin) <= _timestamp(state["last_processed_origin"]):
                return {"origin_time": origin, "status": "duplicate_or_stale",
                        "execution_reason": "Origin already processed or older than the account cursor"}
            if not state["enabled"]:
                status, reason = "cancelled", "Paper entries paused"
            elif state["position"]:
                status, reason = "rejected", "An existing position prevents overlapping entries"
            elif clean_plan.get("action") != "BUY":
                status, reason = "wait", clean_plan.get("reason", "No long entry signal")
            elif horizon and _timestamp(horizon) <= at:
                status, reason = "expired", "Forecast horizon expired before decision completion"
            for previous in state["decisions"]:
                if previous["status"] == "pending":
                    previous.update(status="cancelled", execution_reason="Superseded by a newer forecast origin")
            decision = {"origin_time": origin, "decision_time": _iso(at), "horizon_end": horizon,
                        "plan": clean_plan, "summary": clean_summary, "status": status,
                        "execution_reason": reason, "metadata": _plain(metadata)}
            state["decisions"].append(decision)
            del state["decisions"][:-2000]
            state["last_processed_origin"] = origin
            self._event(state, "decision", at, origin_time=origin, action=clean_plan.get("action"),
                        status=status, reason=reason)
            return decision
        return self._mutate(record)

    def cancel_decision(self, origin_time, reason="Entry cancelled"):
        origin = _iso(origin_time)
        def cancel(state):
            decision = next((d for d in state["decisions"] if d["origin_time"] == origin), None)
            if decision is not None and decision["status"] == "pending":
                decision.update(status="cancelled", execution_reason=str(reason))
                self._event(state, "decision_cancelled", self.clock(), origin_time=origin, reason=str(reason))
            return decision
        return self._mutate(cancel)

    def _validate_quote(self, state, price, quote_time, *, require_fresh=True):
        price = _positive(price, "Quote price")
        stamp, now = _timestamp(quote_time), _timestamp(self.clock())
        if stamp > now + pd.Timedelta(seconds=5):
            raise ValueError("Quote is in the future")
        if require_fresh and now - stamp > pd.Timedelta(seconds=self.max_quote_age_seconds):
            raise ValueError("Quote is stale; obtain a fresh public quote")
        cursor = state.get("recovery_cursor")
        if cursor and stamp < _timestamp(cursor):
            raise ValueError("Quote precedes the last processed account observation")
        if (state.get("last_quote_time") and stamp == _timestamp(state["last_quote_time"]) and
                price != state["last_quote_price"]):
            raise ValueError("Conflicting prices for the same quote timestamp")
        return price, stamp

    def _observe_quote(self, state, price, stamp, *, mark=True):
        state.update(last_quote_time=_iso(stamp), last_quote_price=price,
                     last_mark_price=price, recovery_cursor=_iso(stamp))
        if state["position"]:
            state["position"]["last_observed_time"] = _iso(stamp)
        if mark:
            self._mark(state, price, stamp)

    def open_position(self, plan, quote_price, quote_time, *, origin_time, horizon_end):
        origin, horizon = _iso(origin_time), _timestamp(horizon_end)
        clean_plan = _plain(plan)
        def enter(state):
            decision = next((d for d in state["decisions"] if d["origin_time"] == origin), None)
            if decision is None:
                raise ValueError("Record the forecast decision before attempting an entry")
            if decision["status"] != "pending":
                return None
            if clean_plan != decision["plan"]:
                raise ValueError("Entry plan differs from the persisted forecast decision")
            if decision["horizon_end"] and _timestamp(decision["horizon_end"]) != horizon:
                raise ValueError("Entry horizon differs from the persisted forecast decision")
            price, stamp = self._validate_quote(state, quote_price, quote_time)
            if stamp < _timestamp(decision["decision_time"]):
                raise ValueError("Entry quote precedes the forecast decision")
            reason = None
            if not state["enabled"]:
                reason = "Paper entries paused"
            elif state["position"]:
                reason = "An existing position prevents overlapping entries"
            elif stamp >= horizon:
                reason = "Forecast horizon expired before entry"
            elif clean_plan.get("action") != "BUY":
                reason = "Forecast does not authorize a long entry"
            stop = _positive(clean_plan.get("stop_price"), "Stop price")
            target = _positive(clean_plan.get("take_profit_price"), "Target price")
            economics = _trade_economics(price, stop, target, self.fee, self.slip)
            if reason is None and not stop < price < target:
                reason = "Observed quote crossed a frozen stop or target before entry"
            if reason is None and economics["reward_risk"] < self.minimum_reward_risk:
                reason = "Observed quote no longer meets minimum net reward/risk"
            if reason:
                decision.update(status="rejected", execution_reason=reason)
                self._event(state, "entry_rejected", stamp, origin_time=origin, reason=reason)
                return None
            allocation = min(self.rules["max_position_fraction"],
                             self.rules["risk_fraction"] * economics["entry_cost"] / economics["loss_per_unit"])
            quantity = state["cash"] * allocation / economics["entry_cost"]
            fill = economics["entry_fill"]
            fee = quantity * fill * self.fee
            slippage = quantity * (fill - price)
            position = {"id": state["next_trade_id"], "forecast_origin": origin,
                        "decision_time": decision["decision_time"], "entry_time": _iso(stamp),
                        "entry_quote": price, "entry_price": fill, "quantity": quantity,
                        "stop_price": stop, "take_profit_price": target, "horizon_end": _iso(horizon),
                        "entry_fee": fee, "entry_slippage": slippage, "position_fraction": allocation,
                        "reward_risk_at_entry": economics["reward_risk"], "last_observed_time": _iso(stamp)}
            state["next_trade_id"] += 1
            state["cash"] -= quantity * fill + fee
            state["total_fees"] += fee
            state["total_slippage"] += slippage
            state["position"] = position
            decision.update(status="opened", execution_reason="Filled from a fresh observed public quote",
                            trade_id=position["id"])
            self._observe_quote(state, price, stamp)
            self._event(state, "entry", stamp, trade_id=position["id"], position=position)
            return position
        return self._mutate(enter)

    def _exit(self, state, base_price, stamp, reason, *, observed_price=None, reconstruction=None):
        position = state["position"]
        if position is None:
            return None
        if _timestamp(stamp) < _timestamp(position["entry_time"]):
            raise ValueError("Exit cannot precede the actual entry")
        fill = base_price * (1 - self.slip)
        quantity = position["quantity"]
        fee = quantity * fill * self.fee
        slippage = quantity * (base_price - fill)
        proceeds = quantity * fill - fee
        pnl = proceeds - quantity * position["entry_price"] - position["entry_fee"]
        trade = {**position, "exit_time": _iso(stamp), "exit_price": fill,
                 "exit_base_price": base_price, "exit_observed_price": observed_price,
                 "exit_reason": reason, "exit_fee": fee, "exit_slippage": slippage,
                 "fees": position["entry_fee"] + fee,
                 "slippage": position["entry_slippage"] + slippage, "pnl": pnl,
                 "return": pnl / (quantity * position["entry_price"] + position["entry_fee"]),
                 "reconstruction": reconstruction}
        state["cash"] += proceeds
        state["total_fees"] += fee
        state["total_slippage"] += slippage
        state["realized_pnl"] += pnl
        state["position"] = None
        state["trades"].append(trade)
        self._mark(state, observed_price or base_price, stamp)
        self._event(state, "exit", stamp, trade_id=trade["id"], reason=reason, pnl=pnl,
                    fill=fill, reconstruction=reconstruction)
        return trade

    def process_quote(self, price, quote_time):
        def observe(state):
            mark, stamp = self._validate_quote(state, price, quote_time)
            position = state["position"]
            if not position:
                self._observe_quote(state, mark, stamp)
                return None
            base, reason = None, None
            if stamp >= _timestamp(position["horizon_end"]):
                base, reason = mark, "horizon"
            elif mark <= position["stop_price"]:
                base, reason = mark, "stop_loss"
            elif mark >= position["take_profit_price"]:
                base, reason = min(mark, position["take_profit_price"]), "take_profit"
            self._observe_quote(state, mark, stamp, mark=base is None)
            if base is not None:
                return self._exit(state, base, stamp, reason, observed_price=mark)
            return None
        return self._mutate(observe)

    def process_bar(self, bar, *, end_time=None):
        """Recover exits from a fully unobserved, completed bar after actual entry.

        ``end_time`` is exclusive. Binance/Bybit-style ``close_time`` is inclusive
        and is advanced one millisecond. Partial bars before entry or before the
        latest observed quote are skipped because their earlier path is unknown.
        """
        start = _timestamp(bar.get("timestamps", bar.get("open_time", bar.get("time"))))
        if end_time is not None:
            end = _timestamp(end_time)
        elif bar.get("close_time") is not None:
            end = _timestamp(bar["close_time"]) + pd.Timedelta(milliseconds=1)
        else:
            raise ValueError("Closed-bar recovery requires end_time or inclusive close_time")
        if end <= start or end > _timestamp(self.clock()) or bar.get("closed") is False:
            raise ValueError("Recovery candle must be completed and have a positive duration")
        prices = {name: _positive(bar[name], f"Candle {name}") for name in ("open", "high", "low", "close")}
        if prices["high"] < max(prices.values()) or prices["low"] > min(prices.values()):
            raise ValueError("Recovery candle OHLC bounds are inconsistent")
        def recover(state):
            cursor = _timestamp(state["recovery_cursor"]) if state["recovery_cursor"] else None
            if cursor is not None and end <= cursor:
                return None
            position = state["position"]
            if not position:
                return None
            boundary = max(_timestamp(position["entry_time"]), cursor) if cursor is not None else _timestamp(position["entry_time"])
            if start < boundary:
                message = f"Skipped partial recovery candle {start.isoformat()}: its path began before the actual entry or last observed quote"
                self._notice(state, message)
                self._event(state, "recovery_bar_skipped", end, message=message)
                return None
            horizon = _timestamp(position["horizon_end"])
            if start < horizon < end:
                self._notice(state, "Recovery candle crosses the forecast expiry; later OHLC extremes cannot establish an exit at expiry")
                return None
            base, reason = None, None
            if start >= horizon:
                base, reason = prices["open"], "horizon_gap"
            elif prices["open"] <= position["stop_price"]:
                base, reason = prices["open"], "stop_gap"
            elif prices["low"] <= position["stop_price"]:
                base, reason = position["stop_price"], "stop_loss"
            elif prices["high"] >= position["take_profit_price"]:
                base, reason = position["take_profit_price"], "take_profit"
            elif end >= horizon:
                base, reason = prices["close"], "horizon"
            state["last_bar_end"] = _iso(end)
            state["recovery_cursor"] = _iso(end)
            state["last_mark_price"] = prices["close"]
            position["last_observed_time"] = _iso(end)
            if base is not None:
                return self._exit(state, base, start if reason == "horizon_gap" else end, reason,
                                  observed_price=prices["close"],
                                  reconstruction="Completed OHLC bar after last observation; stop before target; intrabar order unknown")
            self._mark(state, prices["close"], end)
            self._event(state, "recovery_bar", end, start_time=_iso(start), close=prices["close"])
            return None
        return self._mutate(recover)

    def close_position(self, price, quote_time, reason="manual"):
        def close(state):
            mark, stamp = self._validate_quote(state, price, quote_time)
            self._observe_quote(state, mark, stamp, mark=state["position"] is None)
            return self._exit(state, mark, stamp, reason, observed_price=mark)
        return self._mutate(close)

    def close(self):
        with self._mutex:
            if not self._closed:
                fcntl.flock(self._lockfile.fileno(), fcntl.LOCK_UN)
                self._lockfile.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
