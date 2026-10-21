"""Forecast contracts tested without downloading or loading model weights."""

import numpy as np
import pandas as pd
import pytest
import torch

from market.forecast import FEATURES, ForecastEngine


def history_frame(count=512):
    times = pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC")
    return pd.DataFrame({
        "timestamps": times, "open": 100.0, "high": 102.0,
        "low": 98.0, "close": 101.0, "volume": 10.0, "amount": 1005.0,
        "close_time": times + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
    })


def future_times(history, count=24):
    return pd.Series(pd.date_range(history.timestamps.iloc[-1] + pd.Timedelta(hours=1), periods=count, freq="h"))


class DistinguishablePredictor:
    """Give each sampled path a different close, including across chunks."""

    def __init__(self, corrupt=None):
        self.calls = []
        self.produced = 0
        self.corrupt = corrupt

    def predict_batch(self, **kwargs):
        self.calls.append(kwargs)
        frames = []
        for _ in kwargs["df_list"]:
            base = 110.0 + self.produced
            frame = pd.DataFrame(np.tile([base, base + 2, base - 2, base + 1, 10, 1000], (kwargs["pred_len"], 1)), columns=FEATURES)
            self.produced += 1
            if self.corrupt == "nan":
                frame.loc[0, "close"] = np.nan
            elif self.corrupt == "negative":
                frame.loc[0, "close"] = -1
            elif self.corrupt == "short_horizon":
                frame = frame.iloc[:-1]
            frames.append(frame)
        if self.corrupt == "short_batch":
            frames = frames[:-1]
        # Exercise the forked RNG without requiring any model computation.
        torch.rand(1)
        return frames


def engine_with(predictor=None, batch_size=8):
    engine = ForecastEngine(device="cpu", batch_size=batch_size)
    engine.predictor = predictor or DistinguishablePredictor()
    return engine


def test_sample_preserves_each_future_and_batches_without_upstream_averaging():
    history = history_frame()
    times = future_times(history)
    engine = engine_with()
    progress = []
    rng_before = torch.random.get_rng_state().clone()
    result = engine.sample(history, times, sample_count=19, seed=123, progress=progress.append)

    assert result.shape == (19, 24, 6)
    np.testing.assert_array_equal(result[:, 0, 3], np.arange(111.0, 130.0))
    assert [len(call["df_list"]) for call in engine.predictor.calls] == [8, 8, 3]
    assert all(call["sample_count"] == 1 for call in engine.predictor.calls)
    assert all(call["pred_len"] == 24 for call in engine.predictor.calls)
    assert all(len(frame) == 512 for call in engine.predictor.calls for frame in call["df_list"])
    assert all(str(stamps.dtype) == "datetime64[ns, UTC]" for call in engine.predictor.calls for stamps in call["x_timestamp_list"])
    assert progress == pytest.approx([8 / 19, 16 / 19, 1.0])
    assert torch.equal(rng_before, torch.random.get_rng_state())


def test_forecast_keeps_paths_quantiles_and_original_closed_context():
    history = history_frame(520)
    engine = engine_with(batch_size=3)
    result = engine.forecast(history, symbol="BTCUSDT", interval="1h", sample_count=8)

    assert result["sample_count"] == 8
    assert result["context_length"] == 512
    assert result["horizon"] == 24
    assert len(result["history"]) == 512
    assert len(result["paths"]) == 8
    assert all(len(path) == 24 for path in result["paths"])
    np.testing.assert_array_equal([path[0]["value"] for path in result["paths"]], np.arange(111.0, 119.0))
    assert result["quantiles"][0]["p50"] == pytest.approx(114.5)
    assert result["quantiles"][0]["p10"] < result["quantiles"][0]["p90"]
    assert result["paths"][0][0]["time"] == int((history.timestamps.iloc[-1] + pd.Timedelta(hours=1)).timestamp())
    assert result["as_of"] == history.close_time.iloc[-1].isoformat()
    assert result["analysis"]["regime"] == "TRENDING"
    assert result["analysis"]["sample_count"] == 8
    assert "not calibrated" in result["interpretation"]


@pytest.mark.parametrize("count", [0, 501, True, 2.5])
def test_invalid_sample_counts_rejected_cleanly(count):
    history = history_frame()
    engine = engine_with()
    with pytest.raises(ValueError, match="sample_count"):
        engine.sample(history, future_times(history), sample_count=count)
    assert not engine.predictor.calls


@pytest.mark.parametrize("count", [511, 513])
def test_context_must_contain_exactly_512_completed_candles(count):
    history = history_frame(count)
    with pytest.raises(ValueError, match="512"):
        engine_with().sample(history, future_times(history))


@pytest.mark.parametrize("count", [0, 513])
def test_horizon_bounds_are_checked_before_model_inference(count):
    history = history_frame()
    with pytest.raises(ValueError, match="horizon"):
        engine_with().sample(history, future_times(history, count=count))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_is_rejected_without_calling_predictor(value):
    history = history_frame()
    history.loc[0, "open"] = value
    engine = engine_with()
    with pytest.raises(ValueError, match="non-finite"):
        engine.sample(history, future_times(history), sample_count=1)
    assert not engine.predictor.calls


@pytest.mark.parametrize("corrupt", ["nan", "negative"])
def test_invalid_sample_prices_never_reach_trade_planning(corrupt):
    history = history_frame()
    with pytest.raises(ValueError, match="invalid prices"):
        engine_with(DistinguishablePredictor(corrupt)).sample(history, future_times(history), sample_count=2)


@pytest.mark.parametrize("corrupt", ["short_batch", "short_horizon"])
def test_predictor_must_return_exactly_requested_sample_and_horizon_shape(corrupt):
    history = history_frame()
    with pytest.raises(ValueError, match="shape|batch|samples|horizon"):
        engine_with(DistinguishablePredictor(corrupt)).sample(history, future_times(history), sample_count=2)
