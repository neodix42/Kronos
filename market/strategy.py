"""Frozen, explicit research rules for sampled Kronos close-price paths.

Sample frequencies describe this model's draws. They are not calibrated market
probabilities, and the path interval is not a measured confidence interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _paths(close_paths) -> np.ndarray:
    paths = np.asarray(close_paths, dtype=float)
    if paths.ndim != 2 or min(paths.shape) < 1:
        raise ValueError("close_paths must have shape [samples, forecast_horizon].")
    if not np.isfinite(paths).all() or (paths <= 0).any():
        raise ValueError("Forecast prices must be finite and positive.")
    return paths


def summarize_paths(close_paths, reference_price: float, *, consensus_threshold: float = 0.70) -> dict:
    """Classify terminal direction using a threshold fixed before the test.

    Flat terminal paths count toward neither direction. A regime is TRENDING
    only when at least ``consensus_threshold`` of *all* paths agree on a side.
    """
    paths = _paths(close_paths)
    reference = float(reference_price)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference_price must be finite and positive.")
    if not 0.5 < consensus_threshold <= 1:
        raise ValueError("consensus_threshold must be greater than 0.5 and at most 1.")
    terminal = paths[:, -1]
    up = float(np.mean(terminal > reference))
    down = float(np.mean(terminal < reference))
    agreement = max(up, down)
    trending = agreement >= consensus_threshold
    direction = ("BULLISH" if up > down else "BEARISH") if trending else "MIXED"
    low, median, high = np.quantile(terminal, [0.1, 0.5, 0.9])
    return {
        "regime": "TRENDING" if trending else "CHOPPY",
        "direction": direction,
        "reference_price": reference,
        "sample_count": int(paths.shape[0]),
        "horizon": int(paths.shape[1]),
        "consensus_threshold": float(consensus_threshold),
        "agreement": agreement,
        "up_probability": up,
        "down_probability": down,
        "flat_frequency": float(np.mean(terminal == reference)),
        "terminal_price_p10": float(low),
        "terminal_price_p50": float(median),
        "terminal_price_p90": float(high),
        "terminal_return_p10": float(low / reference - 1),
        "terminal_return_p50": float(median / reference - 1),
        "terminal_return_p90": float(high / reference - 1),
        "spread": float((high - low) / reference),
        "frequency_label": "Model sample frequencies; not calibrated probabilities",
        "spread_label": "Model terminal p10–p90 spread; not a calibrated risk bound",
    }


def _validate_costs(fee_bps: float, slippage_bps: float) -> tuple[float, float]:
    fee, slip = float(fee_bps) / 10_000, float(slippage_bps) / 10_000
    if not 0 <= fee < 1 or not 0 <= slip < 1:
        raise ValueError("Fees and slippage must each be between 0 and 10,000 bps.")
    return fee, slip


def _trade_economics(reference: float, stop: float, target: float, fee: float, slip: float) -> dict:
    entry_fill = reference * (1 + slip)
    entry_cost = entry_fill * (1 + fee)
    stop_proceeds = stop * (1 - slip) * (1 - fee)
    target_proceeds = target * (1 - slip) * (1 - fee)
    loss = entry_cost - stop_proceeds
    reward = target_proceeds - entry_cost
    return {
        "entry_fill": entry_fill,
        "entry_cost": entry_cost,
        "loss_per_unit": loss,
        "reward_per_unit": reward,
        "reward_risk": reward / loss if loss > 0 else 0.0,
    }


def make_trade_plan(
    summary: dict,
    close_paths,
    history_df: pd.DataFrame,
    *,
    fee_bps: float = 10,
    slippage_bps: float = 5,
    risk_fraction: float = 0.01,
    max_position_fraction: float = 0.25,
    minimum_reward_risk: float = 1.5,
) -> dict:
    """Return a spot long-only plan whose levels use only completed candles.

    Enter at the next candle's open only if frozen levels still pass the net
    reward/risk check. Stop at the wider of 1.5 ATR14 and terminal p10 below the
    reference. Take profit at the sampled median terminal close. Stops can gap;
    the allocation risk is an estimate, not a guaranteed maximum loss.
    """
    paths = _paths(close_paths)
    fee, slip = _validate_costs(fee_bps, slippage_bps)
    if not 0 < risk_fraction <= 1 or not 0 < max_position_fraction <= 1:
        raise ValueError("Risk and maximum position fractions must be in (0, 1].")
    if not np.isfinite(minimum_reward_risk) or minimum_reward_risk <= 0:
        raise ValueError("minimum_reward_risk must be finite and positive.")
    if history_df.empty or not {"high", "low", "close"}.issubset(history_df.columns):
        raise ValueError("history_df needs completed high, low, and close candles.")
    prices = history_df[["high", "low", "close"]].astype(float)
    if not np.isfinite(prices.to_numpy()).all() or (prices.to_numpy() <= 0).any():
        raise ValueError("Historical prices must be finite and positive.")
    reference = float(prices["close"].iloc[-1])
    if not np.isclose(float(summary["reference_price"]), reference, rtol=1e-10):
        raise ValueError("Summary reference_price must match the last completed close.")
    plan = {
        "action": "WAIT",
        "reason": "",
        "market_type": "spot_long_only",
        "regime": summary["regime"],
        "direction": summary["direction"],
        "reference_price": reference,
        "entry_estimate": float(reference * (1 + slip)),
        "stop_price": None,
        "take_profit_price": None,
        "reward_risk": None,
        "position_fraction": 0.0,
        "risk_fraction": float(risk_fraction),
        "max_position_fraction": float(max_position_fraction),
        "minimum_reward_risk": float(minimum_reward_risk),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "entry_rule": "Buy at the next candle open only when TRENDING/BULLISH and net reward/risk still meets the threshold.",
        "exit_rule": f"Sell at the frozen median terminal target, otherwise after {paths.shape[1]} candles.",
        "stop_rule": "Frozen stop: wider of 1.5 ATR14 or model terminal p10 below the reference. Gaps fill at the worse open; stop wins same-bar ambiguity.",
        "sizing_rule": "Allocate the smaller of the position cap and estimated stop-risk budget; no leverage. Gaps can exceed the risk budget.",
    }
    if summary["regime"] != "TRENDING":
        plan["reason"] = "CHOPPY: fewer than the required sampled paths agree on direction."
        return plan
    if summary["direction"] != "BULLISH":
        plan["reason"] = "BEARISH: this spot strategy holds cash instead of shorting."
        return plan

    previous_close = prices["close"].shift(1)
    true_range = pd.concat(
        [prices["high"] - prices["low"], (prices["high"] - previous_close).abs(), (prices["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.tail(14).mean())
    if not np.isfinite(atr) or atr <= 0:
        plan["reason"] = "No positive recent true range is available to set a stop."
        return plan
    p10, target = np.quantile(paths[:, -1], [0.1, 0.5])
    stop = min(reference - 1.5 * atr, float(p10) if p10 < reference else reference)
    if not 0 < stop < reference < target:
        plan["reason"] = "Forecast target and ATR stop do not define a valid long trade."
        return plan
    economics = _trade_economics(reference, stop, float(target), fee, slip)
    plan.update({
        "atr14": atr,
        "stop_price": float(stop),
        "take_profit_price": float(target),
        "reward_risk": float(economics["reward_risk"]),
        "estimated_net_reward_per_unit": float(economics["reward_per_unit"]),
        "estimated_stop_loss_per_unit": float(economics["loss_per_unit"]),
    })
    if economics["reward_per_unit"] <= 0:
        plan["reason"] = "Median upside does not cover both-side fees and adverse slippage."
        return plan
    if economics["reward_risk"] < minimum_reward_risk:
        plan["reason"] = "Net reward/risk is below the frozen minimum threshold."
        return plan
    allocation = min(max_position_fraction, risk_fraction * economics["entry_cost"] / economics["loss_per_unit"])
    plan.update({
        "action": "BUY",
        "position_fraction": float(allocation),
        "reason": "TRENDING/BULLISH with sufficient net reward/risk after estimated costs.",
    })
    return plan
