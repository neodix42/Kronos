"""Held-out comparison checks use tiny deterministic local checkpoint doubles."""

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from finetune_csv.evaluate import checkpoint_hash, evaluate_config, forecast_metrics, evaluation_windows
from finetune_csv.inference import FEATURES, LocalForecastEngine


def candles(count=120):
    values = 100 + np.arange(count) * 0.05
    return pd.DataFrame({
        "timestamps": pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC"),
        "open": values, "high": values + 1, "low": values - 1, "close": values + 0.1,
        "volume": np.ones(count), "amount": values,
    })


def configuration(tmp_path, frame=None):
    frame = candles() if frame is None else frame
    source = tmp_path / "candles.csv"
    frame.to_csv(source, index=False)
    original = tmp_path / "original"
    tokenizer = tmp_path / "tokenizer"
    trained = tmp_path / "run" / "basemodel" / "best_model"
    for directory in (original, tokenizer, trained):
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").write_bytes(f"checkpoint fixture {directory.name}".encode())
    config = {
        "data": {"data_path": str(source), "interval": "1h", "lookback_window": 8,
                 "predict_window": 4, "max_context": 12, "clip": 5.0,
                 "train_ratio": .6, "val_ratio": .2, "test_ratio": .2},
        "training": {"seed": 42},
        "model_paths": {"pretrained_predictor": str(original), "pretrained_tokenizer": str(tokenizer),
                        "finetuned_tokenizer": str(tokenizer), "base_save_path": str(tmp_path / "run")},
        "experiment": {"train_tokenizer": False}, "device": {"use_cuda": False},
        "evaluation": {"sample_count": 6, "batch_size": 2, "symbol": "BTCUSDT", "interval": "1h"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config


class FakeEngine:
    calls = []
    closed = []

    def __init__(self, model_path, tokenizer_path, **kwargs):
        self.model_path = str(model_path)
        self.kwargs = kwargs

    def sample(self, history, future, **kwargs):
        assert len(history) == 8
        assert history["timestamps"].iloc[-1] < future.iloc[0]
        assert list(history.columns) == ["timestamps", *FEATURES]
        self.calls.append((self.model_path, history.copy(), future.copy(), kwargs.copy()))
        # Prices depend only on the supplied historical context. Both models
        # get the same sampling seed/settings and differ by a fixed fixture bias.
        level = float(history["close"].iloc[-1]) + (10 if "best_model" in self.model_path else 6)
        result = np.full((kwargs["sample_count"], len(future), 6), level)
        result[:, :, 4:] = 1
        return result

    def close(self):
        self.closed.append(self.model_path)


@pytest.fixture(autouse=True)
def reset_engine():
    FakeEngine.calls, FakeEngine.closed = [], []


def test_paired_test_dates_sampling_costs_and_outputs(tmp_path):
    path, config = configuration(tmp_path)
    report = evaluate_config(path, predictor_factory=FakeEngine)
    assert report["coverage"]["evaluated_windows"] == 6
    assert report["coverage"]["test_start"] == candles().timestamps.iloc[96].isoformat()
    original, trained = report["models"]["original"], report["models"]["finetuned"]
    original_calls, trained_calls = FakeEngine.calls[:6], FakeEngine.calls[6:]
    for (_, history_a, future_a, args_a), (_, history_b, future_b, args_b) in zip(original_calls, trained_calls):
        pd.testing.assert_frame_equal(history_a, history_b)
        pd.testing.assert_series_equal(future_a, future_b)
        assert args_a == args_b == {"sample_count": 6, "seed": 42, "temperature": 1.0, "top_p": .9, "top_k": 0}
        assert future_a.iloc[0] >= candles().timestamps.iloc[96]
    assert original["forecast_metrics"]["close"]["mae"] < trained["forecast_metrics"]["close"]["mae"]
    assert original["forecast_metrics"]["terminal_directional_accuracy"] == 1
    for model in (original, trained):
        assert model["backtest"]["trades"] > 0
        assert model["backtest"]["total_fees"] > 0
        assert model["backtest"]["total_slippage"] > 0
        assert 0 <= model["backtest"]["max_drawdown"] < 1
        backtest = json.loads(Path(model["backtest_path"]).read_text())
        assert backtest["rules"]["fee_bps_per_side"] == 10
        assert backtest["rules"]["slippage_bps_per_fill"] == 5
        assert all(item["evaluated"] for item in backtest["forecast_log"])
    assert len(FakeEngine.closed) == 2
    assert Path(report["summary_csv_path"]).is_file()
    assert json.loads(Path(report["report_path"]).read_text()) == report
    charts = json.loads(Path(report["chart_data_path"]).read_text())
    assert len(charts) == 6
    assert set(charts[0]["models"]) == {"original", "finetuned"}
    assert len(charts[0]["actual_candles"]) == 4
    assert "historical context" in report["normalization"]
    assert report["coverage"]["pretraining_unseen_guaranteed"] is False


def test_cached_samples_resume_and_checkpoint_changes_invalidate_only_that_model(tmp_path):
    path, config = configuration(tmp_path)
    first = evaluate_config(path, predictor_factory=FakeEngine, max_windows=2)
    assert len(FakeEngine.calls) == 4
    FakeEngine.calls = []
    second = evaluate_config(path, predictor_factory=FakeEngine, max_windows=2)
    assert FakeEngine.calls == []
    assert all(model["cached_windows"] == 2 for model in second["models"].values())
    trained = Path(config["model_paths"]["base_save_path"]) / "basemodel" / "best_model" / "model.safetensors"
    trained.write_bytes(b"changed weights")
    third = evaluate_config(path, predictor_factory=FakeEngine, max_windows=2)
    assert len(FakeEngine.calls) == 2
    assert third["models"]["original"]["cached_windows"] == 2
    assert third["models"]["finetuned"]["checkpoint_hash"] != first["models"]["finetuned"]["checkpoint_hash"]
    assert second["coverage"]["skipped_limit_windows"] == 4
    assert second["coverage"]["partial_evaluation"] is True


def test_changed_source_csv_invalidates_cached_forecasts(tmp_path):
    path, config = configuration(tmp_path)
    first = evaluate_config(path, predictor_factory=FakeEngine, max_windows=1)
    source = Path(config["data"]["data_path"])
    frame = pd.read_csv(source)
    frame.loc[100, "volume"] += 1
    frame.to_csv(source, index=False)
    FakeEngine.calls = []
    second = evaluate_config(path, predictor_factory=FakeEngine, max_windows=1)
    assert len(FakeEngine.calls) == 2
    assert first["source_csv_sha256"] != second["source_csv_sha256"]


def test_gaps_and_incomplete_tail_are_shared_cash_only_exclusions(tmp_path):
    frame = candles(123)
    # Missing timestamp affects context of first and second scheduled windows;
    # prices remain valid and no missing candle is filled.
    frame.loc[93:, "timestamps"] += pd.Timedelta(hours=1)
    path, _ = configuration(tmp_path, frame)
    report = evaluate_config(path, predictor_factory=FakeEngine)
    coverage = report["coverage"]
    assert coverage["skipped_gap_windows"] > 0
    assert coverage["skipped_incomplete_windows"] == 1
    assert coverage["evaluated_windows"] < coverage["scheduled_windows"]
    for model in report["models"].values():
        result = json.loads(Path(model["backtest_path"]).read_text())
        excluded = [row for row in result["forecast_log"] if not row["evaluated"]]
        assert len(excluded) == coverage["skipped_gap_windows"] + 1
        assert all(row["action"] == "WAIT" and row["regime"] == "EXCLUDED" for row in excluded)
        excluded_origins = {row["origin_time"] for row in excluded}
        assert not any(trade["forecast_origin"] in excluded_origins for trade in result["trade_log"])


def test_metrics_use_median_ohlc_and_terminal_direction_with_flat_class():
    real = np.full((3, 2, 4), 100.0)
    point = real + 1
    reference = np.array([90, 100, 110])
    metrics = forecast_metrics(real, point, reference)
    assert metrics["ohlc"] == {"mae": 1.0, "rmse": 1.0, "mape": .01}
    assert metrics["terminal_directional_accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="matching"):
        forecast_metrics(real, point[:, :, :3], reference)


def test_checkpoint_hash_changes_for_content_but_not_mtime(tmp_path):
    path, config = configuration(tmp_path)
    original = Path(config["model_paths"]["pretrained_predictor"])
    before = checkpoint_hash(original)
    (original / "model.safetensors").touch()
    assert checkpoint_hash(original) == before
    (original / "config.json").write_text('{"changed":true}')
    assert checkpoint_hash(original) != before


@pytest.mark.parametrize("settings,match", [
    ({"stride": 3}, "overlapping"), ({"sample_count": 0}, "positive integer"),
    ({"fee_bps": -1}, "Fees"), ({"top_k": -1}, "nonnegative"),
])
def test_invalid_evaluation_settings_fail_before_inference(tmp_path, settings, match):
    path, config = configuration(tmp_path)
    config["evaluation"].update(settings)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=match):
        evaluate_config(path, predictor_factory=FakeEngine)
    assert FakeEngine.calls == []


def test_inference_passes_only_historical_features_and_preserves_individual_paths(monkeypatch):
    engine = LocalForecastEngine("fixture", "fixture", device="cpu", lookback=8, max_context=12, batch_size=2)
    calls = []

    class Predictor:
        def predict_batch(self, **kwargs):
            assert kwargs["sample_count"] == 1
            assert kwargs["T"] == 1 and kwargs["top_p"] == .9
            frames = kwargs["df_list"]
            assert all(list(frame.columns) == FEATURES and len(frame) == 8 for frame in frames)
            assert max(kwargs["x_timestamp_list"][0]) < min(kwargs["y_timestamp_list"][0])
            calls.append(len(frames))
            return [pd.DataFrame(np.full((4, 6), len(calls) * 100 + i), columns=FEATURES) for i in range(len(frames))]

    monkeypatch.setattr(engine, "load", lambda: Predictor())
    frame = candles(12)
    result = engine.sample(frame.iloc[:8], frame.timestamps.iloc[8:], sample_count=5, seed=42)
    assert result.shape == (5, 4, 6)
    assert calls == [2, 2, 1]
    assert len(set(result[:, 0, 0])) == 5
    with pytest.raises(ValueError, match="exactly 8"):
        engine.sample(frame.iloc[:7], frame.timestamps.iloc[8:])


def test_inferred_interval_without_metadata_is_supported(tmp_path):
    path, config = configuration(tmp_path)
    config["data"].pop("interval")
    config["evaluation"].pop("interval")
    path.write_text(yaml.safe_dump(config))
    report = evaluate_config(path, predictor_factory=FakeEngine, max_windows=1)
    assert report["coverage"]["evaluated_windows"] == 1


def test_training_manifest_prevents_evaluating_a_different_split(tmp_path):
    from finetune_csv.evaluate import file_hash

    path, config = configuration(tmp_path)
    manifest_path = Path(config["model_paths"]["base_save_path"]) / "training_manifest.json"
    manifest = {"data_sha256": file_hash(config["data"]["data_path"]), "rows": 120,
                "train_end": 72, "val_end": 96, "lookback_window": 8, "predict_window": 4,
                "tokenizer_path": config["model_paths"]["finetuned_tokenizer"]}
    manifest_path.write_text(json.dumps(manifest))
    report = evaluate_config(path, predictor_factory=FakeEngine, max_windows=1)
    assert report["training_manifest_verified"] is True
    manifest["val_end"] = 95
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="val_end"):
        evaluate_config(path, predictor_factory=FakeEngine, max_windows=1)
