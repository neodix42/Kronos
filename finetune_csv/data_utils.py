"""Shared chronological data contract for CSV training and held-out evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def raw_config(config: Any) -> dict:
    if isinstance(config, dict):
        return config
    if hasattr(config, "loader"):
        return config.loader.config
    if hasattr(config, "config") and isinstance(config.config, dict):
        return config.config
    return {"data": vars(config)}


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_candles(config: Any) -> pd.DataFrame:
    """Read candles without fitting transforms or filling gaps from later observations."""
    data = raw_config(config).get("data", {})
    path = resolve_path(data.get("data_path") or getattr(config, "data_path", ""))
    if not path.is_file():
        raise FileNotFoundError(f"Candle CSV not found: {path}")
    if data.get("sha256"):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != data["sha256"]:
            raise ValueError("Candle CSV changed after experiment creation (SHA256 mismatch); create a new experiment")
    frame = pd.read_csv(path)
    required = ["timestamps", *FEATURES]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Candle CSV is missing columns: {', '.join(sorted(missing))}")
    frame = frame[required].copy()
    frame["timestamps"] = pd.to_datetime(frame["timestamps"], utc=True, errors="raise")
    if frame["timestamps"].isna().any():
        raise ValueError("Candle timestamps must not be missing")
    frame = frame.sort_values("timestamps", kind="stable").reset_index(drop=True)
    if frame["timestamps"].duplicated().any():
        raise ValueError("Candle timestamps must be unique; remove duplicates before training")
    values = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError("Candle values must be finite; missing values are not filled")
    if (values[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Candle OHLC prices must be positive")
    if (values[["volume", "amount"]] < 0).any().any():
        raise ValueError("Candle volume and amount must be nonnegative")
    if ((values["high"] < values[["open", "close", "low"]].max(axis=1)) |
            (values["low"] > values[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Candle OHLC bounds are inconsistent")
    frame[FEATURES] = values
    frame.attrs["data_path"] = str(path)
    metadata_path = path.with_suffix(".metadata.json")
    if metadata_path.exists():
        frame.attrs["download_metadata"] = json.loads(metadata_path.read_text())
    frame.attrs["interval"] = get_interval(frame, config)
    return frame


def split_bounds(frame: pd.DataFrame, config: Any) -> tuple[int, int]:
    data = raw_config(config).get("data", {})
    train, val, test = (float(data.get(key, default)) for key, default in
                        (("train_ratio", .7), ("val_ratio", .1), ("test_ratio", .2)))
    if train <= 0 or val <= 0 or test < 0 or not np.isclose(train + val + test, 1):
        raise ValueError("Split ratios must sum to 1, with positive train/validation and nonnegative test")
    # Avoid moving an exact boundary back one row because .7 + .1 is represented
    # as .7999999999999999 (e.g. 5120 rows must leave 512 validation candles).
    return (int(np.floor(len(frame) * train + 1e-9)),
            int(np.floor(len(frame) * (train + val) + 1e-9)))


def get_interval(frame: pd.DataFrame, config: Any = None) -> str | pd.Timedelta:
    data = raw_config(config).get("data", {}) if config is not None else {}
    specified = data.get("interval") or data.get("period") or frame.attrs.get("interval")
    if specified:
        return specified
    metadata = frame.attrs.get("download_metadata", {})
    if metadata.get("interval"):
        return metadata["interval"]
    stamps = pd.DatetimeIndex(frame["timestamps"])
    if len(stamps) < 2:
        raise ValueError("Specify data.interval when fewer than two candles are present")
    # Monthly candles have a variable number of days and must not use a fixed delta.
    if (stamps.is_month_start.all() and len(stamps) > 1 and
            np.all(np.diff(stamps.year * 12 + stamps.month) >= 1)):
        return "1M"
    differences = pd.Series(stamps[1:] - stamps[:-1])
    return pd.Timedelta(differences.mode().iloc[0])


def contiguous_window_starts(frame: pd.DataFrame, window: int,
                             interval: str | pd.Timedelta | None = None) -> np.ndarray:
    """Return all starts whose complete window contains no missing candle slots."""
    if window < 1:
        raise ValueError("Window must be positive")
    if len(frame) < window:
        return np.empty(0, dtype=np.int64)
    interval = interval if interval is not None else get_interval(frame)
    stamps = pd.DatetimeIndex(frame["timestamps"])
    if interval == "1M":
        expected = stamps[:-1] + pd.DateOffset(months=1)
        bad = np.asarray(stamps[1:] != expected, dtype=np.int64)
    else:
        delta = pd.Timedelta(interval)
        if delta <= pd.Timedelta(0):
            raise ValueError("Candle interval must be positive")
        bad = np.asarray((stamps[1:] - stamps[:-1]) != delta, dtype=np.int64)
    prefix = np.concatenate(([0], np.cumsum(bad)))
    starts = np.arange(len(frame) - window + 1)
    return starts[(prefix[starts + window - 1] - prefix[starts]) == 0]


def split_manifest(frame: pd.DataFrame, config: Any) -> dict:
    train_end, val_end = split_bounds(frame, config)
    path = frame.attrs.get("data_path")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    result = {"rows": len(frame), "train_end": train_end, "val_end": val_end,
              "data_sha256": digest, "interval": str(get_interval(frame, config)),
              "normalization": "each forecast/training window's historical lookback only",
              "split_policy": "chronological raw rows; complete training/validation windows stay inside their split"}
    for name, start, end in (("train", 0, train_end), ("validation", train_end, val_end),
                             ("test", val_end, len(frame))):
        result[name] = {"start_index": start, "end_exclusive": end,
                        "first_timestamp": frame.iloc[start]["timestamps"].isoformat() if end > start else None,
                        "last_timestamp": frame.iloc[end - 1]["timestamps"].isoformat() if end > start else None}
    return result
