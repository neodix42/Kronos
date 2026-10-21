"""Orchestration tests avoid loading a neural network or contacting exchanges."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from finetune_csv import wizard


def test_requested_defaults_and_separate_scratch_stages(tmp_path):
    config = wizard.make_config()
    assert config["data"] == dict(data_path="data/bybit_spot_BTCUSDT_1h_2021-01-01.csv", interval="1h",
                                   lookback_window=488, predict_window=24, max_context=512, clip=5.0,
                                   train_ratio=.7, val_ratio=.1, test_ratio=.2)
    assert config["training"]["basemodel_epochs"] == 3
    assert config["training"]["predictor_learning_rate"] == .00001
    assert config["model_paths"]["finetuned_tokenizer"] == "pretrained/Kronos-Tokenizer-base"
    scratch = wizard.make_config(mode="scratch")
    files = wizard.config_files(scratch, tmp_path / "my_btc_small.yaml")
    assert len(files) == 3
    tokenizer, predictor = files[1][1], files[2][1]
    assert tokenizer["experiment"]["train_tokenizer"] and not tokenizer["experiment"]["train_basemodel"]
    assert not tokenizer["experiment"]["pre_trained_tokenizer"]
    assert not predictor["experiment"]["train_tokenizer"] and predictor["experiment"]["train_basemodel"]
    assert predictor["experiment"]["pre_trained_tokenizer"]
    assert not predictor["experiment"]["pre_trained_predictor"]
    assert tokenizer["model_paths"]["base_save_path"] == predictor["model_paths"]["base_save_path"]


@pytest.mark.parametrize("options", [dict(symbol="../BTCUSDT"), dict(exp_name="../escape"),
                                    dict(exchange="other"), dict(exchange="binance", category="inverse"),
                                    dict(interval="2m"), dict(start_date="2025-02-30"), dict(epochs=0),
                                    dict(fee_bps=float("nan")), dict(mode="invalid"), dict(sample_count=501)])
def test_invalid_inputs_rejected_before_any_mutation(options):
    with pytest.raises(ValueError):
        wizard.make_config(**options)


def setup_fake_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(wizard, "ROOT", tmp_path)
    config_path = tmp_path / "finetune_csv/configs/my_btc_small.yaml"
    csv = tmp_path / "data/bybit_spot_BTCUSDT_1h_2021-01-01.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("downloaded snapshot")

    def manifest(config):
        config["data"]["sha256"] = wizard.sha256(tmp_path / config["data"]["data_path"])
        return dict(partitions={name: dict(rows=1000, first_timestamp="2024-01-01", last_timestamp="2025-01-01")
                                for name in ("train", "validation", "test")})

    monkeypatch.setattr(wizard, "freeze_dataset", manifest)
    monkeypatch.setattr(wizard, "prepare_models", lambda: None)
    calls = []

    def run(command, log):
        calls.append(command)
        base = tmp_path / "finetune_csv/finetuned/my_btc_small"
        if "finetune_csv.train_sequential" in command:
            import yaml
            stage_config = yaml.safe_load(Path(command[-1]).read_text())
            component = "tokenizer" if stage_config["experiment"]["train_tokenizer"] else "basemodel"
            checkpoint = base / component / "best_model"
            checkpoint.mkdir(parents=True, exist_ok=True)
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model.safetensors").write_bytes(b"trained")
        elif "finetune_csv.evaluate" in command:
            report = base / "evaluation/report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}")

    monkeypatch.setattr(wizard, "run_logged", run)
    return config_path, csv, calls


def test_download_once_train_then_evaluate_and_resume_without_retraining(monkeypatch, tmp_path):
    config_path, csv, calls = setup_fake_pipeline(monkeypatch, tmp_path)
    args = ["--non-interactive", "--no-dashboard", "--config", str(config_path)]
    assert wizard.main(args) == 0
    assert [command[2] for command in calls] == ["finetune_csv.train_sequential", "finetune_csv.evaluate"]
    calls.clear()
    assert wizard.main(args) == 0
    assert calls == []
    original_yaml = config_path.read_text()
    csv.write_text("shared download updates must not change this frozen experiment")
    assert wizard.main(args) == 0
    assert calls == [] and config_path.read_text() == original_yaml
    frozen = tmp_path / "finetune_csv/finetuned/my_btc_small/dataset/candles.csv"
    assert frozen.read_text() == "downloaded snapshot"
    frozen.write_text("changed the frozen experiment itself")
    assert wizard.main(args) == 1
    assert calls == [] and config_path.read_text() == original_yaml


def test_refresh_tracked_run_rejected_before_overwriting_shared_source(monkeypatch, tmp_path):
    config_path, csv, calls = setup_fake_pipeline(monkeypatch, tmp_path)
    args = ["--non-interactive", "--prepare-only", "--config", str(config_path)]
    assert wizard.main(args) == 0
    calls.clear()
    assert wizard.main([*args, "--refresh-data"]) == 1
    assert calls == [] and csv.read_text() == "downloaded snapshot"


@pytest.mark.parametrize("options", [["--port", "1"], ["--port", "70000"], ["--sample-count", "501"]])
def test_invalid_settings_never_start_download_or_training(monkeypatch, tmp_path, options):
    config_path, _, calls = setup_fake_pipeline(monkeypatch, tmp_path)
    assert wizard.main(["--non-interactive", "--config", str(config_path), *options]) == 1
    assert calls == [] and not config_path.exists()


def test_modified_checkpoint_cannot_reuse_report(monkeypatch, tmp_path):
    config_path, _, calls = setup_fake_pipeline(monkeypatch, tmp_path)
    args = ["--non-interactive", "--no-dashboard", "--config", str(config_path)]
    assert wizard.main(args) == 0
    calls.clear()
    checkpoint = tmp_path / "finetune_csv/finetuned/my_btc_small/basemodel/best_model/model.safetensors"
    checkpoint.write_bytes(b"different")
    assert wizard.main(args) == 1
    assert not calls


def test_scratch_executes_tokenizer_then_predictor_configs(monkeypatch, tmp_path):
    config_path, _, calls = setup_fake_pipeline(monkeypatch, tmp_path)
    assert wizard.main(["--non-interactive", "--no-dashboard", "--mode", "scratch", "--config", str(config_path)]) == 0
    assert "scratch_tokenizer.yaml" in calls[0][-1]
    assert "scratch_predictor.yaml" in calls[1][-1]
    assert calls[2][2] == "finetune_csv.evaluate"


def test_prepare_only_never_starts_training_or_dashboard(monkeypatch, tmp_path):
    config_path, _, calls = setup_fake_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(wizard, "start_dashboard", lambda *args: pytest.fail("must not launch"))
    assert wizard.main(["--non-interactive", "--prepare-only", "--config", str(config_path)]) == 0
    assert calls == [] and config_path.is_file()


def test_missing_data_uses_download_script_before_preparation(monkeypatch, tmp_path):
    config_path, csv, calls = setup_fake_pipeline(monkeypatch, tmp_path)
    csv.unlink()
    old_run = wizard.run_logged

    def download(command, log):
        if command[0].endswith("download_data.sh"):
            calls.append(command)
            csv.write_text("downloaded snapshot")
        else:
            old_run(command, log)

    monkeypatch.setattr(wizard, "run_logged", download)
    assert wizard.main(["--non-interactive", "--prepare-only", "--config", str(config_path)]) == 0
    assert calls == [[str(tmp_path / "download_data.sh"), "bybit", "BTCUSDT", "1h", "2021-01-01", "--category", "spot"]]
