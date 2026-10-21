"""Causal walk-forward evaluation of fixed Kronos sample-path rules."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from .strategy import _paths, _trade_economics, _validate_costs, make_trade_plan, summarize_paths


def _utc(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _seconds(value) -> int:
    return int(_utc(value).timestamp())


def _candles(candles: pd.DataFrame) -> pd.DataFrame:
    result = candles.copy()
    timestamp_column = next((name for name in ("timestamps", "timestamp", "open_time") if name in result), None)
    if timestamp_column is None:
        if not isinstance(result.index, pd.DatetimeIndex):
            raise ValueError("Candles need a timestamps column or DatetimeIndex.")
        result["timestamps"] = result.index
    else:
        result["timestamps"] = result[timestamp_column]
    if pd.api.types.is_numeric_dtype(result["timestamps"]):
        raise ValueError("Convert numeric candle timestamps to pandas datetimes before backtesting.")
    result["timestamps"] = pd.to_datetime(result["timestamps"], utc=True)
    if result["timestamps"].isna().any() or result["timestamps"].duplicated().any():
        raise ValueError("Candle timestamps must be present and unique.")
    result = result.sort_values("timestamps").reset_index(drop=True)
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(result):
        raise ValueError("Candles need open, high, low, close, and volume columns.")
    result[required] = result[required].astype(float)
    values = result[required].to_numpy()
    if not np.isfinite(values).all() or (result[["open", "high", "low", "close"]] <= 0).any().any() or (result["volume"] < 0).any():
        raise ValueError("Candle prices must be positive and finite; volume must be nonnegative and finite.")
    if ((result["high"] < result[["open", "close", "low"]].max(axis=1)) | (result["low"] > result[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Candle high/low must contain the open and close.")
    return result


def _future_timestamps(history: pd.DataFrame, horizon: int, interval: str | None) -> pd.Series:
    if interval is not None:
        from .binance import future_timestamps

        return pd.Series(future_timestamps(history["timestamps"].iloc[-1], interval, horizon))
    deltas = history["timestamps"].diff().dropna()
    if deltas.empty or (deltas <= pd.Timedelta(0)).any():
        raise ValueError("At least two chronological history timestamps are needed to infer an interval.")
    step = deltas.median()
    origin = history["timestamps"].iloc[-1]
    return pd.Series([origin + step * (i + 1) for i in range(horizon)])


def run_backtest(
    candles: pd.DataFrame,
    forecast_fn: Callable,
    *,
    start=None,
    end=None,
    lookback: int = 512,
    horizon: int = 24,
    sample_count: int = 100,
    fee_bps: float = 10,
    slippage_bps: float = 5,
    initial_equity: float = 10_000,
    progress: Callable | None = None,
    consensus_threshold: float = 0.70,
    risk_fraction: float = 0.01,
    max_position_fraction: float = 0.25,
    minimum_reward_risk: float = 1.5,
    stride: int | None = None,
    interval: str | None = None,
) -> dict:
    """Evaluate nonoverlapping forecasts on ``[start, end)``.

    ``forecast_fn(history, future_timestamps)`` sees only ``lookback`` completed
    candles and calendar timestamps; it must return ``[sample_count, horizon]``
    positive close-price samples. Each forecast is held at most ``horizon`` bars.
    The final window is evaluated to the end of available requested history.
    Fees apply to both filled notionals. Slippage is adverse on every fill.
    Parameters are frozen before evaluation; this function does no optimization.
    """
    frame = _candles(candles)
    fee, slip = _validate_costs(fee_bps, slippage_bps)
    if any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) or x <= 0 for x in (lookback, horizon, sample_count)) or lookback < 2:
        raise ValueError("lookback >= 2, horizon, and sample_count must be positive integers.")
    stride = horizon if stride is None else stride
    if not isinstance(stride, (int, np.integer)) or isinstance(stride, bool) or stride < horizon:
        raise ValueError("stride must be at least horizon to prevent overlapping positions.")
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial_equity must be finite and positive.")
    if not 0 < risk_fraction <= 1 or not 0 < max_position_fraction <= 1 or minimum_reward_risk <= 0:
        raise ValueError("Risk, position cap, and minimum reward/risk must be positive; fractions cannot exceed one.")
    if not 0.5 < consensus_threshold <= 1:
        raise ValueError("consensus_threshold must be greater than 0.5 and at most 1.")
    timestamps = frame["timestamps"]
    start_at = int(timestamps.searchsorted(_utc(start), side="left")) if start is not None else lookback
    end_at = int(timestamps.searchsorted(_utc(end), side="left")) if end is not None else len(frame)
    if start_at < lookback:
        raise ValueError(f"Need {lookback} completed warm-up candles before the requested start.")
    if start_at >= end_at:
        raise ValueError("No evaluation candles are available in the requested date range.")

    cash = float(initial_equity)
    equity_curve = [{"time": _seconds(timestamps.iloc[start_at - 1]), "equity": cash}]
    trade_log, forecasts = [], []
    total_fees = total_slippage = 0.0
    total_forecasts = math.ceil((end_at - start_at) / stride)
    next_index = start_at
    for completed, first in enumerate(range(start_at, end_at, stride), start=1):
        # Idle bars for explicitly wider forecast strides still count in equity.
        for idle in range(next_index, first):
            equity_curve.append({"time": _seconds(timestamps.iloc[idle]), "equity": float(cash)})
        history = frame.iloc[first - lookback:first].copy().reset_index(drop=True)
        future = _future_timestamps(history, horizon, interval)
        # Keep the reference and ATR history independent of adapter mutations.
        paths = _paths(forecast_fn(history.copy(), future.copy()))
        if paths.shape != (sample_count, horizon):
            raise ValueError(f"forecast_fn returned {paths.shape}; expected {(sample_count, horizon)}.")
        summary = summarize_paths(paths, float(history["close"].iloc[-1]), consensus_threshold=consensus_threshold)
        plan = make_trade_plan(
            summary, paths, history,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
            risk_fraction=risk_fraction, max_position_fraction=max_position_fraction,
            minimum_reward_risk=minimum_reward_risk,
        )
        forecast_record = {
            "origin_time": _seconds(history["timestamps"].iloc[-1]),
            "regime": summary["regime"], "direction": summary["direction"],
            "agreement": summary["agreement"], "action": plan["action"], "reason": plan["reason"],
        }
        forecasts.append(forecast_record)
        last = min(first + horizon, end_at)
        quantity, entry_fill, entry_fee, trade = 0.0, 0.0, 0.0, None
        if plan["action"] == "BUY":
            entry_open = float(frame.iloc[first]["open"])
            stop, target = plan["stop_price"], plan["take_profit_price"]
            economics = _trade_economics(entry_open, stop, target, fee, slip)
            if stop < entry_open < target and economics["reward_risk"] >= minimum_reward_risk:
                allocation = min(max_position_fraction, risk_fraction * economics["entry_cost"] / economics["loss_per_unit"])
                quantity = cash * allocation / economics["entry_cost"]
                entry_fill = economics["entry_fill"]
                entry_fee = quantity * entry_fill * fee
                cash -= quantity * entry_fill + entry_fee
                total_fees += entry_fee
                entry_slippage = quantity * (entry_fill - entry_open)
                total_slippage += entry_slippage
                trade = {
                    "forecast_origin": forecast_record["origin_time"],
                    "entry_time": _seconds(timestamps.iloc[first]),
                    "entry_price": float(entry_fill), "entry_open": entry_open,
                    "quantity": float(quantity), "stop_price": stop,
                    "take_profit_price": target, "entry_fee": float(entry_fee),
                    "entry_slippage": float(entry_slippage), "position_fraction": float(allocation),
                }
            else:
                forecast_record.update({"action": "WAIT", "reason": "Next open invalidated frozen levels or minimum net reward/risk."})

        for i in range(first, last):
            bar = frame.iloc[i]
            if quantity > 0:
                exit_base, exit_reason = None, None
                if float(bar["open"]) <= stop:
                    exit_base, exit_reason = float(bar["open"]), "stop_gap"
                elif float(bar["low"]) <= stop:
                    # OHLC does not establish intrabar order: assume stop first.
                    exit_base, exit_reason = stop, "stop_loss"
                elif float(bar["high"]) >= target:
                    exit_base, exit_reason = target, "take_profit"
                elif i == last - 1:
                    exit_base = float(bar["close"])
                    exit_reason = "horizon" if last - first == horizon else "end_of_history"
                if exit_base is not None:
                    exit_fill = exit_base * (1 - slip)
                    exit_fee = quantity * exit_fill * fee
                    exit_slippage = quantity * (exit_base - exit_fill)
                    cash += quantity * exit_fill - exit_fee
                    total_fees += exit_fee
                    total_slippage += exit_slippage
                    pnl = quantity * (exit_fill - entry_fill) - entry_fee - exit_fee
                    trade.update({
                        "exit_time": _seconds(bar["timestamps"]), "exit_price": float(exit_fill),
                        "exit_base_price": float(exit_base), "exit_reason": exit_reason,
                        "exit_fee": float(exit_fee), "exit_slippage": float(exit_slippage),
                        "fees": float(entry_fee + exit_fee),
                        "slippage": float(trade["entry_slippage"] + exit_slippage),
                        "pnl": float(pnl), "return": float(pnl / (quantity * entry_fill + entry_fee)),
                        "holding_bars": int(i - first + 1),
                    })
                    trade_log.append(trade)
                    quantity = 0.0
            equity_curve.append({"time": _seconds(bar["timestamps"]), "equity": float(cash + quantity * float(bar["close"]))})
        next_index = last
        if progress is not None:
            progress({
                "completed_forecasts": completed, "total_forecasts": total_forecasts,
                "progress": completed / total_forecasts,
                "forecast_origin": history["timestamps"].iloc[-1].isoformat(),
            })

    # The tail may be idle when a caller chooses stride > horizon.
    for idle in range(next_index, end_at):
        equity_curve.append({"time": _seconds(timestamps.iloc[idle]), "equity": float(cash)})
    equities = np.array([point["equity"] for point in equity_curve], dtype=float)
    drawdowns = 1 - equities / np.maximum.accumulate(equities)
    benchmark_entry = float(frame.iloc[start_at]["open"]) * (1 + slip)
    benchmark_quantity = initial_equity / (benchmark_entry * (1 + fee))
    benchmark_exit = float(frame.iloc[end_at - 1]["close"]) * (1 - slip)
    benchmark_equity = benchmark_quantity * benchmark_exit * (1 - fee)
    metrics = {
        "initial_equity": float(initial_equity), "final_equity": float(cash),
        "net_return": float(cash / initial_equity - 1),
        "max_drawdown": float(drawdowns.max()),
        "win_rate": float(sum(trade["pnl"] > 0 for trade in trade_log) / len(trade_log)) if trade_log else 0.0,
        "trades": len(trade_log), "forecasts": len(forecasts),
        "benchmark_return": float(benchmark_equity / initial_equity - 1),
        "total_fees": float(total_fees), "total_slippage": float(total_slippage),
    }
    return {
        **metrics, "summary": metrics, "trade_log": trade_log,
        "equity_curve": equity_curve, "forecast_log": forecasts,
        "coverage": {
            "requested_start": _utc(start).isoformat() if start is not None else None,
            "requested_end_exclusive": _utc(end).isoformat() if end is not None else None,
            "first_evaluated_candle": timestamps.iloc[start_at].isoformat(),
            "last_evaluated_candle": timestamps.iloc[end_at - 1].isoformat(),
            "evaluation_candles": int(end_at - start_at),
            "warmup_candles": int(lookback), "forecast_horizon": int(horizon),
            "final_window_bars": int(min(horizon, end_at - (start_at + (total_forecasts - 1) * stride))),
        },
        "rules": {
            "market_type": "spot_long_only", "fee_bps_per_side": float(fee_bps),
            "slippage_bps_per_fill": float(slippage_bps),
            "consensus_threshold": float(consensus_threshold),
            "minimum_reward_risk": float(minimum_reward_risk),
            "risk_fraction": float(risk_fraction), "max_position_fraction": float(max_position_fraction),
            "lookback": int(lookback), "horizon": int(horizon), "sample_count": int(sample_count), "stride": int(stride),
            "entry": "Next candle open with adverse slippage; frozen levels and net reward/risk rechecked at that open.",
            "stop": "Wider of 1.5 ATR14 and model terminal p10 below reference; adverse gaps fill at open.",
            "exit": "Frozen median terminal target, stop, or horizon close; final window closes at history end.",
            "same_bar_ambiguity": "Stop is filled before target when both are touched; target gaps receive no favorable price improvement.",
            "cost_assumptions": "Configurable constant spot taker fee and adverse slippage estimates; no borrowing, funding, spread model, or market impact.",
            "drawdown": "Equity marked to every evaluated candle close, including cash and any open position.",
            "model_frequency_notice": "Sample agreement and spreads are uncalibrated model outputs, not measured outcome probabilities.",
        },
    }
