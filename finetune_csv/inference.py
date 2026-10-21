"""Local checkpoint inference with normalization restricted to past candles."""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd


FEATURES = ["open", "high", "low", "close", "volume", "amount"]


class LocalForecastEngine:
    """Retain independent futures instead of the upstream sample average.

    ``KronosPredictor.predict_batch`` computes mean/std from each history alone.
    A future timestamp is a calendar feature, never a future price input.
    """

    def __init__(self, model_path, tokenizer_path, *, device=None, max_context=512,
                 clip=5.0, batch_size=8, lookback=488):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not 2 <= lookback <= max_context:
            raise ValueError("lookback must be between 2 and max_context.")
        self.model_path = str(model_path)
        self.tokenizer_path = str(tokenizer_path)
        self.device = device
        self.max_context = max_context
        self.clip = clip
        self.batch_size = batch_size
        self.lookback = lookback
        self.predictor = None
        self.lock = threading.RLock()

    def load(self):
        import torch
        from model import Kronos, KronosPredictor, KronosTokenizer

        with self.lock:
            if self.predictor is None:
                self.device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
                model = Kronos.from_pretrained(self.model_path, local_files_only=True).eval()
                tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_path, local_files_only=True).eval()
                self.predictor = KronosPredictor(
                    model, tokenizer, device=self.device, max_context=self.max_context, clip=self.clip,
                )
        return self.predictor

    def sample(self, history, timestamps, *, sample_count=100, seed=42,
               temperature=1.0, top_p=0.9, top_k=0, progress=None):
        import torch

        if len(history) != self.lookback:
            raise ValueError(f"Expected exactly {self.lookback} completed history candles.")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or not 1 <= sample_count <= 500:
            raise ValueError("sample_count must be an integer between 1 and 500.")
        if not 1 <= len(timestamps) <= self.max_context:
            raise ValueError("Forecast horizon must be between 1 and max_context.")
        if not np.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1].")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer.")
        features = history[FEATURES].astype(float).reset_index(drop=True)
        if not np.isfinite(features.to_numpy()).all() or (features[FEATURES[:4]] <= 0).any().any():
            raise ValueError("Historical candles contain invalid prices or nonfinite features.")
        x_time = pd.Series(pd.to_datetime(history["timestamps"], utc=True)).reset_index(drop=True)
        y_time = pd.Series(pd.to_datetime(timestamps, utc=True)).reset_index(drop=True)
        if not x_time.is_monotonic_increasing or x_time.duplicated().any() or x_time.isna().any():
            raise ValueError("History timestamps must be unique and chronological.")
        if not y_time.is_monotonic_increasing or y_time.duplicated().any() or y_time.isna().any() or y_time.iloc[0] <= x_time.iloc[-1]:
            raise ValueError("Forecast timestamps must follow the completed history.")
        outputs = []
        with self.lock:
            predictor = self.load()
            devices = [torch.device(self.device).index or 0] if str(self.device).startswith("cuda") else []
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.manual_seed(seed)
                for offset in range(0, sample_count, self.batch_size):
                    count = min(self.batch_size, sample_count - offset)
                    batch = predictor.predict_batch(
                        df_list=[features.copy() for _ in range(count)],
                        x_timestamp_list=[x_time.copy() for _ in range(count)],
                        y_timestamp_list=[y_time.copy() for _ in range(count)],
                        pred_len=len(y_time), T=temperature, top_p=top_p, top_k=top_k,
                        sample_count=1, verbose=False,
                    )
                    if len(batch) != count or any(
                        not isinstance(frame, pd.DataFrame) or len(frame) != len(y_time)
                        or not set(FEATURES).issubset(frame) for frame in batch
                    ):
                        raise ValueError("Predictor returned an invalid sample shape.")
                    outputs.extend(frame[FEATURES].to_numpy(dtype=float) for frame in batch)
                    if progress:
                        progress((offset + count) / sample_count)
        paths = np.stack(outputs)
        if not np.isfinite(paths).all() or (paths[:, :, :4] <= 0).any():
            raise ValueError("Model generated invalid prices or nonfinite features.")
        return paths

    def close(self):
        """Release one model before loading the other comparison checkpoint."""
        import gc
        import torch

        with self.lock:
            self.predictor = None
            gc.collect()
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
