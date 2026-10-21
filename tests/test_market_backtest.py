"""Small deterministic fixtures test accounting and causal trading behavior."""

import numpy as np
import pandas as pd
import pytest

from market.backtest import run_backtest
from market.strategy import make_trade_plan, summarize_paths


def candles(count=8):
    return pd.DataFrame({
        "timestamps": pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC"),
        "open": np.full(count, 100.0),
        "high": np.full(count, 101.0),
        "low": np.full(count, 99.0),
        "close": np.full(count, 100.0),
        "volume": np.ones(count),
    })


def bullish(history, future):
    return np.full((100, len(future)), 110.0)


def test_regime_counts_all_paths_including_flat_and_requires_consensus():
    paths = np.array([[110.0]] * 69 + [[90.0]] * 30 + [[100.0]])
    summary = summarize_paths(paths, 100)
    assert summary["regime"] == "CHOPPY"
    assert summary["direction"] == "MIXED"
    assert summary["up_probability"] == 0.69
    assert summary["flat_frequency"] == 0.01
    paths[69] = 110
    summary = summarize_paths(paths, 100)
    assert summary["regime"] == "TRENDING"
    assert summary["direction"] == "BULLISH"
    assert summary["agreement"] == 0.70
    assert "not calibrated" in summary["frequency_label"]


@pytest.mark.parametrize("terminal", [[110] * 50 + [90] * 50, [90] * 100])
def test_no_trade_when_choppy_or_bearish(terminal):
    history = candles(3)
    paths = np.tile(np.array(terminal)[:, None], (1, 2))
    plan = make_trade_plan(summarize_paths(paths, 100), paths, history)
    assert plan["action"] == "WAIT"
    result = run_backtest(candles(), lambda history, future: paths, lookback=3, horizon=2)
    assert result["trades"] == 0
    assert result["final_equity"] == 10_000
    assert result["total_fees"] == 0


def test_fees_and_slippage_are_charged_on_both_filled_notionals():
    frame = candles(5)
    frame.loc[4, ["close", "high"]] = [102, 103]
    result = run_backtest(frame, bullish, lookback=3, horizon=2, fee_bps=10, slippage_bps=5)
    trade = result["trade_log"][0]
    entry = 100 * 1.0005
    exit_fill = 102 * 0.9995
    quantity = min(2500 / (entry * 1.001), 100 / (entry * 1.001 - 97 * 0.9995 * 0.999))
    fees = quantity * (entry + exit_fill) * 0.001
    slippage = quantity * (100 * 0.0005 + 102 * 0.0005)
    expected_pnl = quantity * (exit_fill - entry) - fees
    assert trade["quantity"] == pytest.approx(quantity)
    assert trade["exit_reason"] == "horizon"
    assert result["total_fees"] == pytest.approx(fees)
    assert result["total_slippage"] == pytest.approx(slippage)
    assert result["final_equity"] == pytest.approx(10_000 + expected_pnl)
    assert sum(trade["pnl"] for trade in result["trade_log"]) == pytest.approx(result["final_equity"] - 10_000)
    benchmark = 102 * 0.9995 * 0.999 / (100 * 1.0005 * 1.001) - 1
    assert result["benchmark_return"] == pytest.approx(benchmark)


def test_stop_is_first_when_both_levels_touched_in_same_candle():
    frame = candles(5)
    frame.loc[3, ["high", "low"]] = [120, 95]
    result = run_backtest(frame, bullish, lookback=3, horizon=2, fee_bps=0, slippage_bps=0)
    trade = result["trade_log"][0]
    assert trade["stop_price"] == 97
    assert trade["exit_price"] == 97
    assert trade["exit_reason"] == "stop_loss"
    assert trade["holding_bars"] == 1
    assert trade["pnl"] < 0


def test_stop_gap_uses_worse_open_and_exit_slippage():
    frame = candles(5)
    frame.loc[4, ["open", "high", "low", "close"]] = [90, 92, 88, 91]
    result = run_backtest(frame, bullish, lookback=3, horizon=2, slippage_bps=5)
    trade = result["trade_log"][0]
    assert trade["exit_reason"] == "stop_gap"
    assert trade["exit_base_price"] == 90
    assert trade["exit_price"] == pytest.approx(90 * 0.9995)
    # Estimated risk is not a guarantee through an adverse gap.
    assert -trade["pnl"] > 100


def test_forecaster_receives_only_completed_history_and_calendar_timestamps():
    frame = candles(8)
    seen = []

    def forecast(history, future):
        first = 3 + 2 * len(seen)
        pd.testing.assert_frame_equal(history, frame.iloc[first - 3:first].reset_index(drop=True))
        assert history["timestamps"].iloc[-1] < future.iloc[0]
        expected = pd.date_range(history["timestamps"].iloc[-1], periods=3, freq="h")[1:]
        pd.testing.assert_index_equal(pd.DatetimeIndex(future), expected, check_names=False)
        seen.append(first)
        # Mutating this supplied copy cannot mutate the evaluation data.
        history.loc[:, "close"] = 1
        return np.full((100, 2), 90)

    result = run_backtest(frame, forecast, lookback=3, horizon=2)
    assert seen == [3, 5, 7]
    assert (frame["close"] == 100).all()
    assert result["coverage"]["evaluation_candles"] == 5
    assert result["coverage"]["final_window_bars"] == 1
    assert len(result["equity_curve"]) == 6
    assert all(item["direction"] == "BEARISH" for item in result["forecast_log"])


def test_equity_drawdown_marks_open_position_each_candle_and_tail_liquidates():
    frame = candles(5)
    frame.loc[:2, ["high", "low"]] = [110, 90]
    frame.loc[3, ["open", "high", "low", "close"]] = [100, 101, 79, 80]
    frame.loc[4, ["open", "high", "low", "close"]] = [80, 111, 79, 110]

    def forecast(history, future):
        return np.full((100, len(future)), 150.0)

    result = run_backtest(frame, forecast, lookback=3, horizon=3, fee_bps=0, slippage_bps=0)
    assert result["max_drawdown"] == pytest.approx((100 / 30 * 20) / 10_000)
    assert result["net_return"] > 0
    assert result["trade_log"][0]["exit_reason"] == "end_of_history"
    assert result["trade_log"][0]["holding_bars"] == 2


def test_next_open_rechecks_frozen_levels_and_rejects_unfavorable_gap():
    frame = candles(5)
    frame.loc[3, ["open", "high", "low", "close"]] = [109, 109.5, 108, 109]
    result = run_backtest(frame, bullish, lookback=3, horizon=2)
    assert result["trades"] == 0
    assert "Next open" in result["forecast_log"][0]["reason"]


def test_requested_dates_are_start_inclusive_end_exclusive_and_require_warmup():
    frame = candles()
    result = run_backtest(frame, bullish, start=frame.timestamps.iloc[3], end=frame.timestamps.iloc[5], lookback=3, horizon=2)
    assert result["coverage"]["evaluation_candles"] == 2
    assert result["forecasts"] == 1
    with pytest.raises(ValueError, match="warm-up"):
        run_backtest(frame, bullish, start=frame.timestamps.iloc[2], lookback=3, horizon=2)
    with pytest.raises(ValueError, match="overlapping"):
        run_backtest(frame, bullish, lookback=3, horizon=2, stride=1)


def test_backtest_rejects_incorrect_sample_shape_and_nonfinite_paths():
    with pytest.raises(ValueError, match="expected"):
        run_backtest(candles(), lambda history, future: np.full((1, 2), 110), lookback=3, horizon=2)
    with pytest.raises(ValueError, match="finite"):
        summarize_paths([[np.nan]], 100)


def test_monthly_forecast_calendar_uses_month_opens_not_an_average_timedelta():
    frame = candles(5)
    frame["timestamps"] = pd.date_range("2024-01-01", periods=5, freq="MS", tz="UTC")

    def forecast(history, future):
        assert list(future) == list(pd.to_datetime(["2024-04-01", "2024-05-01"], utc=True))
        return np.full((100, 2), 90)

    run_backtest(frame, forecast, lookback=3, horizon=2, interval="1M")


def test_costs_can_suppress_a_bullish_signal_and_sizing_respects_risk_and_cap():
    history = candles(3)
    tiny_upside = np.full((100, 2), 100.1)
    plan = make_trade_plan(summarize_paths(tiny_upside, 100), tiny_upside, history)
    assert plan["action"] == "WAIT"
    assert "fees" in plan["reason"]
    paths = np.full((100, 2), 110)
    capped = make_trade_plan(summarize_paths(paths, 100), paths, history, fee_bps=0, slippage_bps=0)
    risk_limited = make_trade_plan(summarize_paths(paths, 100), paths, history,
                                  fee_bps=0, slippage_bps=0, risk_fraction=0.002)
    assert capped["position_fraction"] == 0.25
    assert risk_limited["position_fraction"] == pytest.approx(0.002 * 100 / 3)
