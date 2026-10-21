"""Preserve individual Kronos futures rather than averaging away uncertainty."""

from datetime import datetime, timezone
import threading
import uuid

import numpy as np
import pandas as pd
import torch

from market.binance import future_timestamps
from market.strategy import make_trade_plan, summarize_paths

MODEL_ID = "NeoQuasar/Kronos-small"
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def candle_records(frame):
    """Lightweight Charts uses candle open timestamps in UTC seconds."""
    return [
        {"time": int(pd.Timestamp(row.timestamps).timestamp()),
         **{name: float(getattr(row, name)) for name in FEATURES if hasattr(row, name)}}
        for row in frame.itertuples(index=False)
    ]


class ForecastEngine:
    def __init__(self, device=None, batch_size=8):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.predictor = None
        self.lock = threading.RLock()

    def load(self):
        with self.lock:
            if self.predictor is None:
                from model import Kronos, KronosPredictor, KronosTokenizer
                tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION).eval()
                model = Kronos.from_pretrained(MODEL_ID, revision=MODEL_REVISION).eval()
                self.predictor = KronosPredictor(model, tokenizer, device=self.device, max_context=512)
        return self.predictor

    def sample(self, history, timestamps, *, sample_count=100, seed=42, progress=None):
        """Return [samples, horizon, OHLCVA]; each path remains independent.

        Upstream sample_count averages all paths internally. Treat each sample
        as its own batch series with sample_count=1 to retain the distribution.
        Chunking bounds memory on laptops and smaller GPUs.
        """
        if len(history) != 512:
            raise ValueError("Kronos-small requires exactly 512 closed context candles here.")
        if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)) or not 1 <= sample_count <= 500:
            raise ValueError("sample_count must be an integer between 1 and 500.")
        if not 1 <= len(timestamps) <= 512:
            raise ValueError("Forecast horizon must be between 1 and 512 candles.")
        features = history[FEATURES].reset_index(drop=True)
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Input candles contain non-finite values.")
        x_time = pd.Series(pd.to_datetime(history["timestamps"], utc=True)).reset_index(drop=True)
        y_time = pd.Series(pd.to_datetime(timestamps, utc=True)).reset_index(drop=True)
        outputs = []
        with self.lock:
            predictor = self.load()
            devices = [torch.device(self.device).index or 0] if str(self.device).startswith("cuda") else []
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.manual_seed(seed)
                for offset in range(0, sample_count, self.batch_size):
                    count = min(self.batch_size, sample_count - offset)
                    batch = predictor.predict_batch(
                        df_list=[features] * count,
                        x_timestamp_list=[x_time] * count,
                        y_timestamp_list=[y_time] * count,
                        pred_len=len(y_time), T=1.0, top_k=0, top_p=0.9,
                        sample_count=1, verbose=False,
                    )
                    if len(batch) != count or any(
                        not isinstance(frame, pd.DataFrame) or len(frame) != len(y_time)
                        or not set(FEATURES).issubset(frame.columns) for frame in batch
                    ):
                        raise ValueError("Predictor returned an invalid number or shape of sample paths.")
                    outputs.extend(frame[FEATURES].to_numpy(dtype=float) for frame in batch)
                    if progress:
                        progress((offset + count) / sample_count)
        paths = np.stack(outputs)
        if not np.isfinite(paths).all() or (paths[:, :, :4] <= 0).any():
            raise ValueError("Model generated invalid prices; no trade plan was produced.")
        return paths

    def forecast(self, history, *, symbol, interval, sample_count=100,
                 fee_bps=10, slippage_bps=5, seed=42, progress=None):
        history = history.iloc[-512:].reset_index(drop=True)
        timestamps = future_timestamps(history["timestamps"].iloc[-1], interval, 24)
        samples = self.sample(history, timestamps, sample_count=sample_count, seed=seed, progress=progress)
        closes = samples[:, :, FEATURES.index("close")]
        reference = float(history["close"].iloc[-1])
        analysis = summarize_paths(closes, reference)
        plan = make_trade_plan(analysis, closes, history, fee_bps=fee_bps, slippage_bps=slippage_bps)
        # Aliases keep the chart's plain-language display independent of rule internals.
        plan.update(entry=plan.get("entry_estimate"), stop_loss=plan.get("stop_price"),
                    take_profit=plan.get("take_profit_price"), reasons=[plan.get("reason", "")])
        times = [int(pd.Timestamp(t).timestamp()) for t in timestamps]
        quantiles = np.quantile(closes, [.1, .25, .5, .75, .9], axis=0)
        median = np.median(samples, axis=0)
        median_frame = pd.DataFrame(median, columns=FEATURES)
        median_frame["timestamps"] = pd.Series(timestamps).reset_index(drop=True)
        median_frame["high"] = median_frame[["open", "high", "low", "close"]].max(axis=1)
        median_frame["low"] = median_frame[["open", "low", "close"]].min(axis=1)
        median_frame[["volume", "amount"]] = median_frame[["volume", "amount"]].clip(lower=0)
        return {
            "id": uuid.uuid4().hex, "symbol": symbol, "interval": interval,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "as_of": pd.Timestamp(history["close_time"].iloc[-1]).isoformat() if "close_time" in history else pd.Timestamp(history["timestamps"].iloc[-1]).isoformat(),
            "origin_time": int(pd.Timestamp(history["timestamps"].iloc[-1]).timestamp()),
            "model": MODEL_ID, "model_revision": MODEL_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION, "device": str(self.device),
            "context_length": 512, "horizon": 24, "sample_count": sample_count, "seed": seed,
            "history": candle_records(history), "median_candles": candle_records(median_frame),
            "paths": [[{"time": t, "value": float(value)} for t, value in zip(times, path)] for path in closes],
            "quantiles": [dict(time=t, **{key: float(quantiles[i, j]) for i, key in enumerate(["p10", "p25", "p50", "p75", "p90"])},
                               mean=float(closes[:, j].mean())) for j, t in enumerate(times)],
            "analysis": analysis, "plan": plan,
            "interpretation": "Fractions of generated paths and their spread are model summaries, not calibrated probabilities or measured market risk.",
        }
