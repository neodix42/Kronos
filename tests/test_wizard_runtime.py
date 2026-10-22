"""Lifecycle checks use isolated fake checkouts and harmless Python servers."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest
import yaml

from finetune_csv import runtime, wizard


SERVER = '''
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
p=argparse.ArgumentParser()
p.add_argument('--config');p.add_argument('--port',type=int,required=True)
args,unknown=p.parse_known_args()
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'{"ok":true}')
 def log_message(self,*args):pass
HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError("Timed out waiting for fixture process")


@pytest.fixture
def checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(wizard, "ROOT", tmp_path)
    package = tmp_path / "finetune_csv"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "dashboard.py").write_text(SERVER)
    (tmp_path / "flask.py").write_text(SERVER)
    yield tmp_path
    # Only descendants of this isolated fake checkout are cleaned up.
    runtime.stop_setup(all_setups=True, timeout=.5)


def config_file(root, name="btc", filename=None):
    base = root / "finetune_csv/finetuned" / name
    source = root / (filename or f"finetune_csv/configs/{name}.yaml")
    source.parent.mkdir(parents=True, exist_ok=True)
    data = base / "dataset/candles.csv"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text("timestamps,open,high,low,close,volume,amount\n")
    predictor = base / "basemodel/best_model"
    original = root / "pretrained/Kronos-small"
    tokenizer = root / "pretrained/Kronos-Tokenizer-base"
    for folder in (predictor, original, tokenizer):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text("{}")
        (folder / "model.safetensors").write_bytes(b"fixture model")
    raw = {"data": {"data_path": str(data.relative_to(root)), "sha256": runtime.sha256(data)},
           "model_paths": {"exp_name": name, "base_save_path": str(base.relative_to(root)),
                           "pretrained_predictor": "pretrained/Kronos-small",
                           "pretrained_tokenizer": "pretrained/Kronos-Tokenizer-base",
                           "finetuned_tokenizer": "pretrained/Kronos-Tokenizer-base"},
           "experiment": {"train_tokenizer": False}}
    source.write_text("# Preserve these exact bytes for paper identity\n" + yaml.safe_dump(raw, sort_keys=False))
    wizard.write_json(base / "wizard-state.json", {"completed_stages": ["predictor", "evaluation"], "config": str(source),
                     "checkpoint_hashes": {"predictor": {"config.json": runtime.sha256(predictor / "config.json"),
                                                            "model.safetensors": runtime.sha256(predictor / "model.safetensors")}}})
    return source, raw


def record_setup(source, raw, port):
    return wizard.write_setup_manifest(source, port, raw)


def process_for(pid):
    wait_for(lambda: pid in runtime.process_snapshot())
    return runtime.process_snapshot()[pid]


def launch(root, module, config=None, port=None):
    command = [sys.executable, "-m", module]
    if config is not None:
        command += ["--config", str(config)]
    if port is not None:
        command += ["--port", str(port)]
    return subprocess.Popen(command, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_start_freezes_exact_bytes_without_training_and_resumes_existing_account(checkout, monkeypatch):
    source, raw = config_file(checkout)
    base = checkout / "finetune_csv/finetuned/btc"
    account = base / "paper/account.json"
    account.parent.mkdir()
    account.write_text('{"enabled":true,"position":{"id":1}}')
    saved = account.read_bytes()
    launches = []
    monkeypatch.setattr(wizard, "start_dashboard", lambda config, port: launches.append((config, port)))
    monkeypatch.setattr(wizard, "dashboard_already_running", lambda *args: False)
    descriptor = runtime.start_setup(config=source, port=17071)
    assert Path(descriptor["config"]).read_bytes() == source.read_bytes()
    assert launches == [(base / "setup/config.yaml", 17071)]
    assert account.read_bytes() == saved
    assert runtime.resolve_setup()["experiment_name"] == "btc"
    assert runtime.resolve_setup(name="btc")["config_sha256"] == runtime.sha256(source)


def test_real_fake_dashboard_start_is_idempotent_and_stop_leaves_unrelated_flask(checkout):
    source, raw = config_file(checkout)
    port, other_port = free_port(), free_port()
    other = launch(checkout, "flask", port=other_port)
    try:
        process_for(other.pid)
        descriptor = runtime.start_setup(config=source, port=port)
        first = runtime.read_json(checkout / f"outputs/wizard/dashboard-{port}.json")
        runtime.start_setup(name="btc")
        assert runtime.read_json(checkout / f"outputs/wizard/dashboard-{port}.json")["pid"] == first["pid"]
        stopped = runtime.stop_setup(name="btc", timeout=1)
        assert first["pid"] in stopped
        assert other.poll() is None
        assert runtime.stop_setup(name="btc", timeout=.1) == []
        assert Path(descriptor["config"]).is_file()
    finally:
        other.terminate()
        other.wait(timeout=5)


def test_legacy_latest_descriptor_migrates_without_yaml_rerender(checkout, monkeypatch):
    source, raw = config_file(checkout, name="old", filename="finetune_csv/configs/my_btc_small.yaml")
    path = checkout / "outputs/wizard/dashboard-17071.json"
    wizard.write_json(path, dict(pid=99999999, start_time="stale", config=str(source),
                                config_sha256=runtime.sha256(source), port=17071))
    monkeypatch.setattr(wizard, "start_dashboard", lambda *args: None)
    monkeypatch.setattr(wizard, "dashboard_already_running", lambda *args: False)
    descriptor = runtime.start_setup()
    assert descriptor["experiment_name"] == "old"
    assert Path(descriptor["config"]).read_bytes() == source.read_bytes()
    assert runtime.resolve_setup(name="old")["port"] == 17071


def test_name_never_reuses_overwritten_configuration_from_another_run(checkout):
    old, raw = config_file(checkout, name="old", filename="finetune_csv/configs/shared.yaml")
    config_file(checkout, name="new", filename="finetune_csv/configs/shared.yaml")
    with pytest.raises(ValueError, match="another experiment"):
        runtime.resolve_setup(name="old")


def test_stop_tracked_old_process_even_when_its_editable_filename_was_overwritten(checkout):
    source, raw = config_file(checkout, name="old", filename="finetune_csv/configs/shared.yaml")
    port = free_port()
    descriptor = record_setup(source, raw, port)
    process = launch(checkout, "finetune_csv.dashboard", config=source, port=port)
    identity = process_for(process.pid)
    wizard.write_json(checkout / f"outputs/wizard/dashboard-{port}.json", dict(pid=process.pid,
        start_time=identity.start_time, config=str(source), config_sha256=descriptor["config_sha256"], port=port))
    config_file(checkout, name="new", filename="finetune_csv/configs/shared.yaml")
    try:
        with pytest.raises(ValueError, match="already running"):
            runtime.start_setup(name="old", port=free_port())
        assert process.pid in runtime.stop_setup(name="old", timeout=1)
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_stop_pid_reuse_guard_and_explicit_port_preserve_other_setup(checkout):
    first_source, first_raw = config_file(checkout, name="first")
    second_source, second_raw = config_file(checkout, name="second")
    first_port, second_port = free_port(), free_port()
    first = record_setup(first_source, first_raw, first_port)
    record_setup(second_source, second_raw, second_port)
    other = launch(checkout, "finetune_csv.dashboard", config=second_source, port=second_port)
    identity = process_for(other.pid)
    # Stale metadata points at a live, different experiment: it is never trusted.
    wizard.write_json(checkout / f"outputs/wizard/dashboard-{first_port}.json", dict(pid=other.pid,
        start_time="wrong-reused-pid", config=first["config"], config_sha256=first["config_sha256"], port=first_port))
    try:
        assert runtime.stop_setup(name="first", timeout=.1) == []
        assert other.poll() is None
        assert runtime.stop_setup(name="second", port=first_port, timeout=.1) == []
        assert other.poll() is None
    finally:
        other.terminate()
        other.wait(timeout=5)


def test_start_rejects_same_account_on_another_port(checkout):
    source, raw = config_file(checkout)
    port = free_port()
    runtime.start_setup(config=source, port=port)
    with pytest.raises(ValueError, match="already running"):
        runtime.start_setup(name="btc", port=free_port())


def test_stop_captures_selected_training_descendants(checkout):
    source, raw = config_file(checkout)
    record_setup(source, raw, free_port())
    module = checkout / "finetune_csv/train_sequential.py"
    module.write_text("import subprocess,sys,time\nfrom pathlib import Path\n"
                      "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
                      "Path('child.pid').write_text(str(child.pid))\ntime.sleep(60)\n")
    process = launch(checkout, "finetune_csv.train_sequential", config=source)
    try:
        wait_for(lambda: (checkout / "child.pid").exists())
        child_pid = int((checkout / "child.pid").read_text())
        selected = runtime.stop_setup(name="btc", timeout=1)
        assert process.pid in selected and child_pid in selected
        process.wait(timeout=5)
        assert child_pid not in runtime.process_snapshot()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


@pytest.mark.parametrize("change,error", [
    ("incomplete", "not complete"), ("weights", "checkpoint changed"),
    ("dataset", "dataset changed"), ("missing", "checkpoint"),
])
def test_start_failures_do_not_launch_or_freeze_new_setup(checkout, monkeypatch, change, error):
    source, raw = config_file(checkout)
    base = checkout / "finetune_csv/finetuned/btc"
    if change == "incomplete":
        wizard.write_json(base / "wizard-state.json", {"completed_stages": []})
    elif change == "weights":
        (base / "basemodel/best_model/model.safetensors").write_bytes(b"changed")
    elif change == "dataset":
        (base / "dataset/candles.csv").write_text("changed")
    else:
        (checkout / "pretrained/Kronos-small/model.safetensors").unlink()
    monkeypatch.setattr(wizard, "start_dashboard", lambda *args: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match=error):
        runtime.start_setup(config=source)
    assert not (base / "setup/config.yaml").exists()


def test_all_stop_excludes_flask_and_cli_rejects_ambiguous_selectors(checkout):
    source, raw = config_file(checkout)
    record_setup(source, raw, free_port())
    assert runtime.main(["stop", "--all", "--name", "btc"]) == 1
    assert runtime.main(["start", "--all"]) == 1
    assert runtime.main(["start", "--name", "../outside"]) == 1
    assert runtime.main(["start", "--port", "80"]) == 1


def test_explicit_config_and_name_must_agree(checkout):
    source, raw = config_file(checkout)
    with pytest.raises(ValueError, match="different experiments"):
        runtime.resolve_setup(config=source, name="other")


def test_default_fallback_refuses_a_known_changed_legacy_filename(checkout):
    source, raw = config_file(checkout, filename="finetune_csv/configs/my_btc_small.yaml")
    wizard.write_json(checkout / "outputs/wizard/dashboard-17071.json", dict(config=str(source),
                     config_sha256=runtime.sha256(source), port=17071, pid=99999999, start_time="stale"))
    source.write_text(source.read_text() + "# Changed after recorded startup\n")
    with pytest.raises(ValueError, match="changed"):
        runtime.resolve_setup()


def test_interactive_wizard_name_default_cannot_retarget_selected_stop(checkout):
    source, raw = config_file(checkout, name="first")
    descriptor = record_setup(source, raw, 17071)
    process = runtime.Process(pid=9999, start_time="fixture", parent=1, cwd=checkout,
        args=(sys.executable, "-m", "finetune_csv.wizard", "--name", "first", "--config", str(source)),
        executable=sys.executable)
    assert runtime._matches_pipeline(process, descriptor) is False
