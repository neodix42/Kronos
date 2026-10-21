"""Compare local checkpoints on the same frozen chronological test windows.

Run with ``python -m finetune_csv.evaluate --config path/to/config.yaml``.
Sample caches are content addressed and can be reused after interruption.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

from market.backtest import run_backtest
from market.binance import SUPPORTED_INTERVALS
from market.strategy import _validate_costs
from .config_loader import CustomFinetuneConfig
from .inference import FEATURES, LocalForecastEngine


ROOT = Path(__file__).resolve().parents[1]
CACHE_VERSION = 1


def _path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hash(path):
    """Hash actual local checkpoint bytes rather than its path or mtime."""
    directory = _path(path)
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise ValueError(f"Local checkpoint with config.json is required: {directory}")
    digest = hashlib.sha256()
    for item in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(item.relative_to(directory).as_posix().encode())
        digest.update(bytes.fromhex(file_hash(item)))
    return digest.hexdigest()


def trained_tokenizer_path(config):
    """Honor the exact tokenizer selected by predictor training."""
    value = getattr(config, "finetuned_tokenizer_path", None)
    if not value:
        raise ValueError("model_paths.finetuned_tokenizer must name the tokenizer used for predictor training.")
    path = _path(value)
    if not path.is_dir():
        raise ValueError(f"Trained predictor tokenizer is missing: {path}")
    return path


def _verify_training_manifest(config, frame, dataset_hash):
    from .data_utils import split_bounds

    path = Path(config.base_save_path) / "training_manifest.json"
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text())
    train_end, val_end = split_bounds(frame, config)
    expected = {"data_sha256": dataset_hash, "rows": len(frame), "train_end": train_end,
                "val_end": val_end, "lookback_window": config.lookback_window,
                "predict_window": config.predict_window}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Training manifest {key} differs from evaluation data/config; use the original frozen experiment.")
    if not manifest.get("tokenizer_path") or _path(manifest["tokenizer_path"]) != trained_tokenizer_path(config):
        raise ValueError("Evaluation tokenizer differs from the tokenizer paired with the trained checkpoint.")
    return True


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".evaluation-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def evaluation_windows(frame, config, interval, *, stride, max_windows=None):
    """Freeze origins before either model is loaded; exclude any gapped window."""
    from .data_utils import contiguous_window_starts, split_bounds

    _, test_start = split_bounds(frame, config)
    lookback, horizon = int(config.lookback_window), int(config.predict_window)
    if test_start < lookback or test_start >= len(frame):
        raise ValueError("The untouched test split needs history and at least one forecast window.")
    contiguous = set(int(i) + lookback for i in contiguous_window_starts(frame, lookback + horizon, interval))
    selected, skipped = [], {}
    scheduled = list(range(test_start, len(frame), stride))
    for first in scheduled:
        if first + horizon > len(frame):
            skipped[first] = "incomplete_final_window"
        elif first not in contiguous:
            skipped[first] = "noncontiguous_context_or_target"
        elif max_windows is not None and len(selected) >= max_windows:
            skipped[first] = "explicit_max_windows_limit"
        else:
            selected.append(first)
    if not selected:
        raise ValueError("No complete, contiguous forecast windows are available in the untouched test split.")
    test_end = frame["timestamps"].iloc[-1] + (pd.DateOffset(months=1) if interval == "1M" else pd.Timedelta(interval))
    coverage = {
        "test_start": frame["timestamps"].iloc[test_start].isoformat(),
        "test_end_exclusive": test_end.isoformat(),
        "test_candles": int(len(frame) - test_start),
        "scheduled_windows": len(scheduled),
        "evaluated_windows": len(selected),
        "skipped_gap_windows": sum(reason == "noncontiguous_context_or_target" for reason in skipped.values()),
        "skipped_incomplete_windows": sum(reason == "incomplete_final_window" for reason in skipped.values()),
        "skipped_limit_windows": sum(reason == "explicit_max_windows_limit" for reason in skipped.values()),
        "evaluated_candles": len(selected) * horizon,
        "window_coverage": len(selected) / len(scheduled),
        "partial_evaluation": len(selected) != len(scheduled),
        "untouched_test_for_this_experiment": True,
        "pretraining_unseen_guaranteed": False,
        "skip_policy": "Identical origins for both checkpoints; hold cash on excluded windows. No candles are fabricated.",
        "skipped_windows": [
            {"first_target_time": frame["timestamps"].iloc[first].isoformat(), "reason": reason}
            for first, reason in skipped.items()
        ],
    }
    return selected, skipped, coverage


def forecast_metrics(actual, predicted, references):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 3 or actual.shape[2] != 4:
        raise ValueError("Forecast metrics need matching [windows, horizon, OHLC] arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or (actual <= 0).any():
        raise ValueError("Forecast metrics require finite predictions and positive actual prices.")

    def error(real, point):
        delta = point - real
        return {"mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt((delta ** 2).mean())),
                "mape": float((np.abs(delta) / real).mean())}

    references = np.asarray(references, dtype=float)
    if references.shape != (len(actual),) or not np.isfinite(references).all():
        raise ValueError("Each forecast window needs its historical reference close.")
    actual_direction = np.sign(actual[:, -1, 3] - references)
    predicted_direction = np.sign(predicted[:, -1, 3] - references)
    return {
        "ohlc": error(actual, predicted),
        "close": error(actual[:, :, 3], predicted[:, :, 3]),
        "per_column": {name: error(actual[:, :, i], predicted[:, :, i]) for i, name in enumerate(FEATURES[:4])},
        "terminal_directional_accuracy": float(np.mean(actual_direction == predicted_direction)),
        "direction_definition": "Sign of horizon-terminal median close minus last historical close; exact flats form a third class.",
        "windows": len(actual), "forecast_candles": int(actual.shape[0] * actual.shape[1]),
        "point_forecast": "sample_median", "mape_unit": "fraction",
    }


def _records(frame):
    return [{"time": int(row.timestamps.timestamp()), **{name: float(getattr(row, name)) for name in FEATURES if hasattr(row, name)}}
            for row in frame.itertuples(index=False)]


def _chart(paths, timestamps):
    median = np.median(paths, axis=0)
    candles = pd.DataFrame(median, columns=FEATURES)
    candles["timestamps"] = pd.Series(timestamps).reset_index(drop=True)
    # Quantile aggregation need not preserve OHLC bounds; chart-only repair does
    # not alter raw predictions used for the error metrics.
    candles["high"] = candles[FEATURES[:4]].max(axis=1)
    candles["low"] = candles[FEATURES[:4]].min(axis=1)
    candles[["volume", "amount"]] = candles[["volume", "amount"]].clip(lower=0)
    quantiles = np.quantile(paths[:, :, 3], [0.1, 0.5, 0.9], axis=0)
    return {
        "median_candles": _records(candles),
        "quantiles": [{"time": int(pd.Timestamp(t).timestamp()), "p10": float(quantiles[0, i]),
                       "p50": float(quantiles[1, i]), "p90": float(quantiles[2, i])} for i, t in enumerate(timestamps)],
    }


def _cached_samples(cache_path, make_samples, *, shape):
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as saved:
                paths = saved["samples"]
            if paths.shape == shape and np.isfinite(paths).all() and (paths[:, :, :4] > 0).all():
                return paths, True
        except (ValueError, OSError, KeyError, EOFError, zipfile.BadZipFile):
            pass  # Interrupted or corrupted cache entries are regenerated.
    paths = np.asarray(make_samples(), dtype=np.float32)
    if paths.shape != shape or not np.isfinite(paths).all() or (paths[:, :, :4] <= 0).any():
        raise ValueError(f"Invalid model sample paths; expected {shape}, got {paths.shape}.")
    with tempfile.NamedTemporaryFile("wb", dir=cache_path.parent, prefix=".samples-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, samples=paths)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(cache_path)
    return paths, False


def evaluate_config(config_path, *, predictor_factory=None, progress=None, max_windows=None):
    """Run a reproducible, resumable paired comparison; return the report dict."""
    from .data_utils import get_interval, load_candles

    config = CustomFinetuneConfig(str(config_path))
    raw = config.loader.config
    settings = dict(raw.get("evaluation", {}))
    sampling = {"sample_count": 100, "seed": config.seed, "temperature": 1.0,
                "top_p": 0.9, "top_k": 0, "batch_size": 8}
    sampling.update({key: settings[key] for key in sampling if key in settings})
    sampling.update({key: value for key, value in settings.get("sampling", {}).items() if key in sampling})
    for key in ("sample_count", "batch_size"):
        _positive_integer(sampling[key], key)
    if not 1 <= sampling["sample_count"] <= 500:
        raise ValueError("sample_count must be between 1 and 500.")
    if isinstance(sampling["seed"], bool) or not isinstance(sampling["seed"], int):
        raise ValueError("seed must be an integer.")
    if not np.isfinite(sampling["temperature"]) or sampling["temperature"] <= 0 or not 0 < sampling["top_p"] <= 1:
        raise ValueError("temperature must be positive; top_p must be in (0, 1].")
    if isinstance(sampling["top_k"], bool) or not isinstance(sampling["top_k"], int) or sampling["top_k"] < 0:
        raise ValueError("top_k must be a nonnegative integer.")
    stride = _positive_integer(settings.get("stride", config.predict_window), "stride")
    if stride < config.predict_window:
        raise ValueError("Evaluation stride must be at least predict_window to avoid overlapping trades.")
    limit = max_windows if max_windows is not None else settings.get("max_windows")
    if limit is not None:
        _positive_integer(limit, "max_windows")
    frame = load_candles(config)
    interval = settings.get("interval") or get_interval(frame, config)
    if not interval:
        raise ValueError("Specify data.interval or evaluation.interval for calendar-correct test windows.")
    origins, skipped, coverage = evaluation_windows(frame, config, interval, stride=stride, max_windows=limit)
    output = _path(settings.get("output_dir", Path(config.base_save_path) / "evaluation"))
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    dataset_hash = file_hash(_path(config.data_path))
    training_verified = _verify_training_manifest(config, frame, dataset_hash)
    models = {
        "original": {"model_path": _path(config.pretrained_predictor_path), "tokenizer_path": _path(config.pretrained_tokenizer_path)},
        "finetuned": {"model_path": _path(config.basemodel_best_model_path), "tokenizer_path": trained_tokenizer_path(config)},
    }
    execution = {
        "fee_bps": settings.get("fee_bps", 10), "slippage_bps": settings.get("slippage_bps", 5),
        "initial_equity": settings.get("initial_equity", 10_000),
        "consensus_threshold": settings.get("consensus_threshold", 0.70),
        "risk_fraction": settings.get("risk_fraction", 0.01),
        "max_position_fraction": settings.get("max_position_fraction", 0.25),
        "minimum_reward_risk": settings.get("minimum_reward_risk", 1.5),
    }
    _validate_costs(execution["fee_bps"], execution["slippage_bps"])
    if not np.isfinite(execution["initial_equity"]) or execution["initial_equity"] <= 0:
        raise ValueError("initial_equity must be positive and finite.")
    if not 0 < execution["risk_fraction"] <= 1 or not 0 < execution["max_position_fraction"] <= 1:
        raise ValueError("risk_fraction and max_position_fraction must be in (0, 1].")
    if not np.isfinite(execution["minimum_reward_risk"]) or execution["minimum_reward_risk"] <= 0 or not 0.5 < execution["consensus_threshold"] <= 1:
        raise ValueError("minimum_reward_risk must be positive; consensus_threshold must be in (0.5, 1].")
    # Validate all model files before an expensive comparison starts.
    for model in models.values():
        model["checkpoint_hash"] = checkpoint_hash(model["model_path"])
        model["tokenizer_hash"] = checkpoint_hash(model["tokenizer_path"])
    actual = np.stack([frame.iloc[first:first + config.predict_window][FEATURES[:4]].to_numpy() for first in origins])
    references = [float(frame["close"].iloc[first - 1]) for first in origins]
    chart_records = [{
        "origin_time": int(frame["timestamps"].iloc[first - 1].timestamp()),
        "first_target_time": int(frame["timestamps"].iloc[first].timestamp()),
        "last_target_time": int(frame["timestamps"].iloc[first + config.predict_window - 1].timestamp()),
        "actual_candles": _records(frame.iloc[first:first + config.predict_window]), "models": {},
    } for first in origins]
    factory = predictor_factory or LocalForecastEngine
    device = f"cuda:{config.device_id}" if config.use_cuda else "cpu"
    # CPU-only hosts remain usable with the default use_cuda configuration.
    if config.use_cuda:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    for label, model in models.items():
        engine = factory(model["model_path"], model["tokenizer_path"], device=device,
                         max_context=config.max_context, clip=config.clip,
                         batch_size=sampling["batch_size"], lookback=config.lookback_window)
        predicted, samples_by_origin, cached_count = [], {}, 0
        cache_identity = {"version": CACHE_VERSION, "config": raw, "dataset_sha256": dataset_hash,
                          "model_sha256": model["checkpoint_hash"], "tokenizer_sha256": model["tokenizer_hash"],
                          "sampling": sampling, "interval": interval}
        identity = hashlib.sha256(json.dumps(cache_identity, sort_keys=True, default=str).encode()).hexdigest()
        try:
            for completed, first in enumerate(origins, start=1):
                history = frame.iloc[first - config.lookback_window:first].copy().reset_index(drop=True)
                future = frame["timestamps"].iloc[first:first + config.predict_window].reset_index(drop=True)
                origin_time = int(history["timestamps"].iloc[-1].timestamp())
                cache_path = cache / f"{identity}_{origin_time}.npz"
                sample_args = {key: value for key, value in sampling.items() if key != "batch_size"}
                paths, was_cached = _cached_samples(cache_path, lambda: engine.sample(history, future, **sample_args),
                                                  shape=(sampling["sample_count"], config.predict_window, len(FEATURES)))
                cached_count += int(was_cached)
                samples_by_origin[origin_time] = cache_path
                predicted.append(np.median(paths[:, :, :4], axis=0))
                chart_records[completed - 1]["models"][label] = _chart(paths, future)
                if progress:
                    progress({"model": label, "completed_windows": completed, "total_windows": len(origins),
                              "cached": was_cached, "origin": history["timestamps"].iloc[-1].isoformat()})

            def forecast(history, future):
                origin = int(history["timestamps"].iloc[-1].timestamp())
                if origin not in samples_by_origin:
                    # Excluded gap/tail/limited windows hold cash. These flat
                    # placeholders never enter the measured forecast errors.
                    return np.full((sampling["sample_count"], config.predict_window), float(history["close"].iloc[-1]))
                with np.load(samples_by_origin[origin], allow_pickle=False) as saved:
                    return saved["samples"][:, :, 3]

            backtest = run_backtest(
                frame, forecast, start=coverage["test_start"], end=coverage["test_end_exclusive"],
                lookback=config.lookback_window, horizon=config.predict_window,
                sample_count=sampling["sample_count"], stride=stride,
                interval=interval if isinstance(interval, str) and interval in SUPPORTED_INTERVALS else None, **execution,
            )
            for record in backtest["forecast_log"]:
                record["evaluated"] = record["origin_time"] in samples_by_origin
                if not record["evaluated"]:
                    first_index = int(frame["timestamps"].searchsorted(pd.Timestamp(record["origin_time"], unit="s", tz="UTC"))) + 1
                    record.update(action="WAIT", reason=skipped.get(first_index, "excluded_window"), regime="EXCLUDED", direction="EXCLUDED")
            backtest["coverage"]["paired_evaluation"] = coverage
            backtest["summary"]["evaluated_forecasts"] = len(origins)
            backtest["evaluated_forecasts"] = len(origins)
            backtest_path = output / f"{label}_backtest.json"
            _atomic_json(backtest_path, backtest)
            model.update(forecast_metrics=forecast_metrics(actual, np.stack(predicted), references),
                         backtest=backtest["summary"], backtest_path=str(backtest_path),
                         cached_windows=cached_count)
            model["model_path"], model["tokenizer_path"] = str(model["model_path"]), str(model["tokenizer_path"])
        finally:
            if hasattr(engine, "close"):
                engine.close()
    charts_path = output / "forecasts.json"
    _atomic_json(charts_path, chart_records)
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(_path(config_path)), "source_csv": str(_path(config.data_path)),
        "source_csv_sha256": dataset_hash, "interval": str(interval),
        "symbol": settings.get("symbol", raw.get("experiment", {}).get("symbol")),
        "coverage": coverage, "sampling": sampling, "models": models,
        "training_manifest_verified": training_verified,
        "execution_rules": backtest["rules"], "chart_data_path": str(charts_path),
        "normalization": "Per forecast, mean/std use only the historical context; target prices are never passed to KronosPredictor.",
        "interpretation": "The test split is untouched by this experiment's training and validation. Original pretraining data may overlap these dates. Costs are constant fee/slippage estimates, not measured fills.",
    }
    summary_path = output / "summary.csv"
    pd.DataFrame([{
        "model": label, "ohlc_mae": model["forecast_metrics"]["ohlc"]["mae"],
        "ohlc_rmse": model["forecast_metrics"]["ohlc"]["rmse"], "ohlc_mape": model["forecast_metrics"]["ohlc"]["mape"],
        "close_mae": model["forecast_metrics"]["close"]["mae"], "close_rmse": model["forecast_metrics"]["close"]["rmse"],
        "close_mape": model["forecast_metrics"]["close"]["mape"],
        "directional_accuracy": model["forecast_metrics"]["terminal_directional_accuracy"],
        **model["backtest"],
    } for label, model in models.items()]).to_csv(summary_path, index=False)
    report["summary_csv_path"] = str(summary_path)
    report["report_path"] = str(output / "report.json")
    _atomic_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-windows", type=int, help="Explicit partial smoke evaluation; reported as limited coverage.")
    args = parser.parse_args(argv)

    def progress(event):
        print(f"{event['model']}: {event['completed_windows']}/{event['total_windows']} test windows "
              f"({event['origin']}){' [cache]' if event['cached'] else ''}", flush=True)

    try:
        report = evaluate_config(args.config, progress=progress, max_windows=args.max_windows)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Paired test report: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
