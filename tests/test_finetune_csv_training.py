import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from finetune_csv.config_loader import CustomFinetuneConfig, ROOT
from finetune_csv.data_utils import (FEATURES, contiguous_window_starts, load_candles,
                                    split_bounds, split_manifest)
from finetune_csv.finetune_base_model import CustomKlineDataset, causal_validation, train_model
from finetune_csv.train_sequential import initialize_from_architecture, SequentialTrainer
from model import Kronos, KronosTokenizer


def candles(count=100):
    close = np.arange(count, dtype=float) + 100
    return pd.DataFrame({"timestamps": pd.date_range("2024-01-01", periods=count, freq="h", tz="UTC"),
                         "open": close, "high": close + 2, "low": close - 2, "close": close,
                         "volume": close * 2, "amount": close ** 2})


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "candles.csv"
    candles().to_csv(path, index=False)
    return path


def test_raw_split_and_exact_partition_windows(csv_path):
    frame = load_candles({"data": {"data_path": str(csv_path), "interval": "1h"}})
    assert split_bounds(frame, {}) == (70, 80)
    datasets = [CustomKlineDataset(csv_path, part, lookback_window=6, predict_window=4,
                                  interval="1h") for part in ("train", "val", "test")]
    assert [len(ds) for ds in datasets] == [61, 1, 11]
    assert datasets[0].timestamps.iloc[-1] < datasets[1].timestamps.iloc[0]
    assert datasets[1].timestamps.iloc[-1] < datasets[2].timestamps.iloc[0]
    assert datasets[0][0][0].shape == (10, 6)
    assert split_bounds(range(5120), {}) == (3584, 4096)


def test_future_changes_never_change_normalized_context(csv_path, tmp_path):
    first = CustomKlineDataset(csv_path, lookback_window=6, predict_window=4, clip=1e9)
    changed = pd.read_csv(csv_path)
    changed.loc[6:9, FEATURES] = changed.loc[6:9, FEATURES] * 1000
    other = tmp_path / "changed.csv"
    changed.to_csv(other, index=False)
    second = CustomKlineDataset(other, lookback_window=6, predict_window=4, clip=1e9)
    np.testing.assert_array_equal(first[0][0][:6], second[0][0][:6])
    assert not torch.equal(first[0][0][6:], second[0][0][6:])
    np.testing.assert_allclose(first[0][0][:6].mean(axis=0), 0, atol=1e-6)


def test_windows_crossing_missing_slots_are_removed(csv_path):
    frame = pd.read_csv(csv_path).drop(index=20)
    frame.to_csv(csv_path, index=False)
    ds = CustomKlineDataset(csv_path, lookback_window=6, predict_window=4, interval="1h")
    assert ds.skipped_gap_windows == 9
    for start in ds.starts:
        dates = ds.timestamps.iloc[start:start + 10]
        assert dates.diff().dropna().eq(pd.Timedelta(hours=1)).all()


def test_calendar_months_are_contiguous():
    frame = pd.DataFrame({"timestamps": pd.date_range("2023-11-01", periods=8, freq="MS", tz="UTC")})
    assert contiguous_window_starts(frame, 4, "1M").tolist() == [0, 1, 2, 3, 4]
    assert contiguous_window_starts(frame.drop(index=2), 4, "1M").tolist() == [2, 3]


def test_hash_change_rejected_and_manifest_has_split_dates(csv_path):
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    config = {"data": {"data_path": str(csv_path), "sha256": digest}}
    frame = load_candles(config)
    manifest = split_manifest(frame, config)
    assert manifest["data_sha256"] == digest
    assert manifest["test"]["start_index"] == 80
    csv_path.write_text(csv_path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_candles(config)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "invalid_price", "nan"])
def test_invalid_data_rejected(csv_path, problem):
    frame = pd.read_csv(csv_path)
    if problem == "duplicate":
        frame.loc[1, "timestamps"] = frame.loc[0, "timestamps"]
    elif problem == "missing":
        frame = frame.drop(columns="amount")
    elif problem == "invalid_price":
        frame.loc[0, "low"] = -1
    else:
        frame.loc[0, "volume"] = np.nan
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_candles({"data": {"data_path": str(csv_path)}})


def test_config_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "data": {"data_path": "data/example.csv", "lookback_window": 488, "predict_window": 24},
        "model_paths": {"pretrained_predictor": "pretrained/Kronos-small",
                        "pretrained_tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
                        "exp_name": "demo", "base_path": "finetune_csv/finetuned"},
        "experiment": {"train_tokenizer": False}, "model": {"from_scratch": True}}))
    monkeypatch.chdir(tmp_path)
    config = CustomFinetuneConfig(path)
    assert config.data_path == str(ROOT / "data/example.csv")
    assert config.pretrained_predictor_path == str(ROOT / "pretrained/Kronos-small")
    assert config.pretrained_tokenizer_path == "NeoQuasar/Kronos-Tokenizer-base"
    assert config.finetuned_tokenizer_path == config.pretrained_tokenizer_path
    assert not config.pre_trained_predictor
    assert not config.pre_trained_tokenizer


def test_small_split_fails_with_actionable_message(csv_path):
    with pytest.raises(ValueError, match="at least 5120 rows"):
        CustomKlineDataset(csv_path, "val", lookback_window=488, predict_window=24)


def tiny_architectures(tmp_path):
    tok = {"d_in": 6, "d_model": 16, "n_heads": 2, "ff_dim": 32,
           "n_enc_layers": 2, "n_dec_layers": 2, "ffn_dropout_p": 0.,
           "attn_dropout_p": 0., "resid_dropout_p": 0., "s1_bits": 2, "s2_bits": 2,
           "beta": .05, "gamma0": 1., "gamma": 1.1, "zeta": .05, "group_size": 2}
    pred = {"s1_bits": 2, "s2_bits": 2, "n_layers": 1, "d_model": 16,
            "n_heads": 2, "ff_dim": 32, "ffn_dropout_p": 0., "attn_dropout_p": 0.,
            "resid_dropout_p": 0., "token_dropout_p": 0., "learn_te": True}
    paths = []
    for name, config in (("tokenizer", tok), ("predictor", pred)):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config))
        paths.append(path)
    return paths


def test_random_architecture_initialization_never_reads_weight_files(tmp_path):
    tokenizer, predictor = tiny_architectures(tmp_path)
    torch.manual_seed(42)
    first = initialize_from_architecture(Kronos, predictor)
    torch.manual_seed(43)
    second = initialize_from_architecture(Kronos, predictor)
    assert not torch.equal(first.head.proj_s1.weight, second.head.proj_s1.weight)
    assert initialize_from_architecture(KronosTokenizer, tokenizer).d_in == 6


def test_teacher_forced_validation_never_reads_later_tokens(tmp_path):
    _, predictor = tiny_architectures(tmp_path)
    model = initialize_from_architecture(Kronos, predictor).eval()
    first, second = torch.zeros((1, 10), dtype=torch.long), torch.zeros((1, 10), dtype=torch.long)
    later_first, later_second = first.clone(), second.clone()
    later_first[:, 5:], later_second[:, 5:] = 3, 2
    with torch.no_grad(), causal_validation(model):
        before = model(first, second, use_teacher_forcing=True, s1_targets=first)
        after = model(later_first, later_second, use_teacher_forcing=True, s1_targets=later_first)
        for left, right in zip(before, after):
            torch.testing.assert_close(left[:, :5], right[:, :5], rtol=0, atol=0)
    assert not model.dep_layer.cross_attn.training


def test_real_tiny_predictor_training_freezes_tokenizer_and_saves(tmp_path, csv_path):
    import logging
    tok_path, model_path = tiny_architectures(tmp_path)
    tokenizer = initialize_from_architecture(KronosTokenizer, tok_path)
    model = initialize_from_architecture(Kronos, model_path)
    original = {name: param.detach().clone() for name, param in tokenizer.named_parameters()}
    config = SimpleNamespace(data_path=str(csv_path), lookback_window=6, predict_window=4,
                             clip=5., seed=42, train_ratio=.7, val_ratio=.1, test_ratio=.2,
                             interval="1h", batch_size=64, num_workers=0, basemodel_epochs=1,
                             predictor_learning_rate=1e-5, adam_beta1=.9, adam_beta2=.95,
                             adam_weight_decay=.1, log_interval=50)
    saved = tmp_path / "saved"
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        loss = train_model(model, tokenizer, torch.device("cpu"), config, saved,
                           logging.getLogger("tiny_training_test"))
    finally:
        torch.set_num_threads(old_threads)
    assert np.isfinite(loss)
    assert (saved / "best_model/config.json").exists()
    assert not tokenizer.training
    for name, param in tokenizer.named_parameters():
        assert not param.requires_grad
        torch.testing.assert_close(param, original[name], rtol=0, atol=0)


def test_scratch_predictor_reuses_learned_tokenizer(tmp_path, monkeypatch):
    from finetune_csv import train_sequential as module
    tokenizer_path, predictor_path = tiny_architectures(tmp_path)
    tokenizer = initialize_from_architecture(KronosTokenizer, tokenizer_path)
    saved_tok = tmp_path / "learned_tokenizer"
    tokenizer.save_pretrained(saved_tok)
    config = SimpleNamespace(skip_existing=False, basemodel_best_model_path=tmp_path / "predictor/best_model",
                             seed=42, finetuned_tokenizer_path=str(saved_tok),
                             pre_trained_predictor=False, pretrained_predictor_path=str(predictor_path),
                             basemodel_save_path=tmp_path / "predictor")
    trainer = SequentialTrainer.__new__(SequentialTrainer)
    trainer.config, trainer.device = config, torch.device("cpu")
    trainer._logger = lambda: None
    trainer._record_phase = lambda *args: None
    def fake_train(model, loaded, *args):
        for actual, expected in zip(loaded.parameters(), tokenizer.parameters()):
            torch.testing.assert_close(actual, expected)
        assert not loaded.training
        assert not any(param.requires_grad for param in loaded.parameters())
        return 1.
    monkeypatch.setattr(module, "train_model", fake_train)
    trainer.train_basemodel_phase()
