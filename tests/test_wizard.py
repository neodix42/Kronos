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
    assert config["evaluation"]["risk_fraction"] == .01
    assert config["evaluation"]["max_position_fraction"] == .25
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
                                    dict(fee_bps=float("nan")), dict(mode="invalid"), dict(sample_count=501),
                                    dict(risk_fraction=0), dict(risk_fraction=float("nan")),
                                    dict(risk_fraction=float("inf")), dict(risk_fraction=1.01),
                                    dict(max_position_fraction=-.1), dict(max_position_fraction=float("nan")),
                                    dict(max_position_fraction=float("inf")), dict(max_position_fraction=True)])
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


@pytest.mark.parametrize("options", [["--port", "1"], ["--port", "70000"], ["--sample-count", "501"],
    ["--risk-percent", "0"], ["--risk-percent", "-1"], ["--risk-percent", "nan"],
    ["--risk-percent", "inf"], ["--risk-percent", "101"], ["--max-position-percent", "0"],
    ["--max-position-percent", "nan"], ["--max-position-percent", "inf"], ["--max-position-percent", "101"]])
def test_invalid_settings_never_start_download_or_training(monkeypatch, tmp_path, options):
    config_path, _, calls = setup_fake_pipeline(monkeypatch, tmp_path)
    assert wizard.main(["--non-interactive", "--config", str(config_path), *options]) == 1
    assert calls == [] and not config_path.exists()


def test_cli_percentages_convert_to_fractions_in_both_scratch_configs(monkeypatch, tmp_path):
    import yaml
    config_path, _, calls = setup_fake_pipeline(monkeypatch, tmp_path)
    assert wizard.main(["--non-interactive", "--prepare-only", "--mode", "scratch", "--config", str(config_path),
                        "--risk-percent", "2.5", "--max-position-percent", "40"]) == 0
    for path in (config_path, config_path.with_name("my_btc_small_scratch_tokenizer.yaml"),
                 config_path.with_name("my_btc_small_scratch_predictor.yaml")):
        settings = yaml.safe_load(path.read_text())["evaluation"]
        assert settings["risk_fraction"] == .025
        assert settings["max_position_fraction"] == .4
    assert not calls


def test_percentages_allow_full_budget_and_allocation():
    assert wizard.percentage("100") == 100
    config = wizard.make_config(risk_fraction=1, max_position_fraction=1)
    assert config["evaluation"]["risk_fraction"] == 1
    assert config["evaluation"]["max_position_fraction"] == 1


def test_changed_risk_rejected_before_any_tracked_run_files_change(monkeypatch, tmp_path):
    config_path, csv, calls = setup_fake_pipeline(monkeypatch, tmp_path)
    args = ["--non-interactive", "--prepare-only", "--config", str(config_path)]
    assert wizard.main(args) == 0
    ledger = tmp_path / "finetune_csv/finetuned/my_btc_small/paper/account.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"cash": 10100, "enabled": true}')
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls.clear()
    # Even a missing frozen snapshot must not trigger a download for a changed budget.
    monkeypatch.setattr(wizard, "freeze_dataset", lambda *args: pytest.fail("must reject before data changes"))
    monkeypatch.setattr(wizard, "prepare_models", lambda: pytest.fail("must reject before preparing models"))
    assert wizard.main([*args, "--risk-percent", "2"]) == 1
    assert wizard.main([*args, "--max-position-percent", "30"]) == 1
    assert calls == []
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_default_risk_resumes_without_changing_config_bytes_or_fingerprint(monkeypatch, tmp_path):
    config_path, _, calls = setup_fake_pipeline(monkeypatch, tmp_path)
    args = ["--non-interactive", "--prepare-only", "--config", str(config_path)]
    assert wizard.main(args) == 0
    state_path = tmp_path / "finetune_csv/finetuned/my_btc_small/wizard-state.json"
    before = config_path.read_bytes(), state_path.read_bytes()
    assert wizard.main([*args, "--risk-percent", "1", "--max-position-percent", "25"]) == 0
    assert (config_path.read_bytes(), state_path.read_bytes()) == before


def test_prepared_setup_manifest_has_exact_resolved_descriptor(monkeypatch, tmp_path):
    config_path, _, _ = setup_fake_pipeline(monkeypatch, tmp_path)
    assert wizard.main(["--non-interactive", "--prepare-only", "--config", str(config_path), "--port", "7075"]) == 0
    run_dir = tmp_path / "finetune_csv/finetuned/my_btc_small"
    frozen_path = run_dir / "setup/config.yaml"
    expected = {"version": 1, "root": str(tmp_path), "config": str(frozen_path),
                "source_config": str(config_path.resolve()),
                "config_sha256": wizard.sha256(config_path), "port": 7075,
                "experiment_name": "my_btc_small", "base_save_path": str(run_dir), "prepared": True}
    assert json.loads((run_dir / "wizard-setup.json").read_text()) == expected
    assert json.loads((tmp_path / "outputs/wizard/last-setup.json").read_text()) == expected
    assert frozen_path.read_bytes() == config_path.read_bytes()
    config_path.write_text("editor configuration reused by another experiment")
    assert wizard.sha256(frozen_path) == expected["config_sha256"]


def test_existing_frozen_config_cannot_be_replaced_by_different_risk(monkeypatch, tmp_path):
    import yaml
    config_path, _, _ = setup_fake_pipeline(monkeypatch, tmp_path)
    assert wizard.main(["--non-interactive", "--prepare-only", "--config", str(config_path)]) == 0
    config = yaml.safe_load(config_path.read_text())
    frozen = tmp_path / "finetune_csv/finetuned/my_btc_small/setup/config.yaml"
    before = frozen.read_bytes()
    config["evaluation"]["risk_fraction"] = .02
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="different prepared configuration"):
        wizard.write_setup_manifest(config_path, 7071, config)
    assert frozen.read_bytes() == before


def test_risk_guard_reads_frozen_config_after_editor_file_is_reused(monkeypatch, tmp_path):
    import yaml
    config_path, _, _ = setup_fake_pipeline(monkeypatch, tmp_path)
    args = ["--non-interactive", "--prepare-only", "--config", str(config_path)]
    assert wizard.main(args) == 0
    config = yaml.safe_load(config_path.read_text())
    config["evaluation"]["risk_fraction"] = .02
    config_path.write_text(yaml.safe_dump(config))
    # The original 1% tracked run still resumes from its immutable setup.
    assert wizard.main(args) == 0
    assert yaml.safe_load(config_path.read_text())["evaluation"]["risk_fraction"] == .01


def test_model_preparation_failure_never_updates_last_setup(monkeypatch, tmp_path):
    config_path, _, _ = setup_fake_pipeline(monkeypatch, tmp_path)
    previous = tmp_path / "outputs/wizard/last-setup.json"
    previous.parent.mkdir(parents=True)
    previous.write_text('{"previous": true}')
    def fail():
        raise ValueError("model preparation failed")
    monkeypatch.setattr(wizard, "prepare_models", fail)
    assert wizard.main(["--non-interactive", "--prepare-only", "--config", str(config_path)]) == 1
    assert previous.read_text() == '{"previous": true}'
    assert not (tmp_path / "finetune_csv/finetuned/my_btc_small/wizard-setup.json").exists()


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


def test_finished_wizard_launches_the_immutable_setup_config(monkeypatch, tmp_path):
    config_path, _, calls = setup_fake_pipeline(monkeypatch, tmp_path)
    started = []
    monkeypatch.setattr(wizard, "dashboard_already_running", lambda *args: False)
    monkeypatch.setattr(wizard, "start_dashboard", lambda path, port: started.append((path, port)))
    assert wizard.main(["--non-interactive", "--config", str(config_path)]) == 0
    expected = tmp_path / "finetune_csv/finetuned/my_btc_small/setup/config.yaml"
    assert started == [(expected, 7071)]
    assert expected.read_bytes() == config_path.read_bytes()


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
