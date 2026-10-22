"""Start or stop a prepared wizard setup without training or changing its account."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_MODULES = {"finetune_csv.wizard", "finetune_csv.train_sequential", "finetune_csv.evaluate",
                    "finetune_csv.finetune_base_model", "finetune_csv.finetune_tokenizer"}


def local_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a setup object: {path}")
    return value


def valid_port(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1024 <= value <= 65535:
        raise ValueError("Choose a dashboard port between 1024 and 65535.")
    return value


def config_info(path):
    path = local_path(path)
    if not path.is_file():
        raise ValueError(f"Setup configuration is missing: {path}. Run ./wizard.sh to prepare it.")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("model_paths"), dict):
        raise ValueError(f"Configuration has no model_paths mapping: {path}")
    paths = raw["model_paths"]
    name = paths.get("exp_name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("The setup needs a valid model_paths.exp_name.")
    base = paths.get("base_save_path") or str(Path(paths.get("base_path", "finetune_csv/finetuned")) / name)
    base = local_path(str(base).format(exp_name=name))
    return {"config": path, "raw": raw, "name": name, "base": base, "sha256": sha256(path)}


def _candidate(path, port=7071, *, expected_name=None, expected_base=None, expected_hash=None):
    info = config_info(path)
    if expected_name is not None and info["name"] != expected_name:
        raise ValueError("The saved configuration now belongs to a different experiment; select its immutable setup with --name.")
    if expected_base is not None and info["base"] != local_path(expected_base):
        raise ValueError("The saved setup directory does not match its configuration.")
    if expected_hash is not None and info["sha256"] != expected_hash:
        raise ValueError("The saved configuration changed; restore the prepared snapshot or select another --name.")
    return {"version": 1, "root": str(ROOT), "config": str(info["config"]),
            "source_config": str(info["config"]), "config_sha256": info["sha256"],
            "port": valid_port(port), "experiment_name": info["name"],
            "base_save_path": str(info["base"]), "prepared": True}


def _descriptor(path):
    value = read_json(path)
    if value.get("version") != 1 or value.get("prepared") is not True or local_path(value.get("root", "")) != ROOT.resolve():
        raise ValueError(f"Setup descriptor does not belong to this checkout: {path}")
    required = ("config", "config_sha256", "port", "experiment_name", "base_save_path")
    if any(key not in value for key in required):
        raise ValueError(f"Incomplete setup descriptor: {path}")
    checked = _candidate(value["config"], value["port"], expected_name=value["experiment_name"],
                         expected_base=value["base_save_path"], expected_hash=value["config_sha256"])
    checked["source_config"] = value.get("source_config", checked["config"])
    return checked


def _legacy_states():
    return sorted((ROOT / "outputs/wizard").glob("dashboard-*.json"),
                  key=lambda path: path.stat().st_mtime_ns, reverse=True)


def _legacy_for_config(path, *, fallback_port=7071):
    path = local_path(path)
    for state_path in _legacy_states():
        state = None
        try:
            state = read_json(state_path)
            if local_path(state.get("config", "")) == path:
                return _candidate(path, state.get("port", fallback_port), expected_hash=state.get("config_sha256"))
        except (OSError, ValueError, TypeError, yaml.YAMLError):
            # A known filename with a changed startup hash must never be adopted.
            if state and local_path(state.get("config", "")) == path:
                raise ValueError("The legacy configuration changed after its recorded dashboard start; use a verified --name setup.")
    return _candidate(path, fallback_port)


def resolve_setup(*, config=None, name=None, port=None):
    """Select a setup without downloading, training, starting, or stopping it."""
    if port is not None:
        valid_port(port)
    if name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Experiment name must use letters, digits, underscores or dashes.")
    if config is not None:
        chosen = _legacy_for_config(config)
        if name is not None and chosen["experiment_name"] != name:
            raise ValueError("--config and --name select different experiments.")
    elif name is not None:
        base = ROOT / "finetune_csv/finetuned" / name
        descriptor = base / "wizard-setup.json"
        if descriptor.is_file():
            chosen = _descriptor(descriptor)
        elif (base / "setup/config.yaml").is_file():
            chosen = _candidate(base / "setup/config.yaml", expected_name=name, expected_base=base)
        elif (base / "wizard-state.json").is_file():
            source = read_json(base / "wizard-state.json").get("config")
            if not source:
                raise ValueError(f"No prepared configuration is recorded for {name}. Run ./wizard.sh --name {name}.")
            chosen = _legacy_for_config(source)
        else:
            raise ValueError(f"No prepared setup exists for {name}. Run ./wizard.sh --name {name}.")
        if chosen["experiment_name"] != name or local_path(chosen["base_save_path"]) != base.resolve():
            raise ValueError("The selected name points to another experiment; its mutable YAML will not be reused.")
    else:
        last = ROOT / "outputs/wizard/last-setup.json"
        if last.is_file():
            chosen = _descriptor(last)
        else:
            chosen = None
            for state_path in _legacy_states():
                try:
                    state = read_json(state_path)
                    chosen = _candidate(state["config"], state.get("port", 7071), expected_hash=state.get("config_sha256"))
                    break
                except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
                    continue
            if chosen is None:
                chosen = _candidate(ROOT / "finetune_csv/configs/my_btc_small.yaml")
    if port is not None:
        chosen["port"] = port
    return chosen


def require_ready(descriptor):
    info = config_info(descriptor["config"])
    paths, raw = info["raw"]["model_paths"], info["raw"]
    state_path = info["base"] / "wizard-state.json"
    if state_path.is_file():
        state = read_json(state_path)
        if "predictor" not in state.get("completed_stages", []):
            raise ValueError("Predictor training is not complete. Run ./wizard.sh with this experiment's original choices first.")
        for component, hashes in state.get("checkpoint_hashes", {}).items():
            folder = info["base"] / (paths.get("tokenizer_save_name", "tokenizer") if component == "tokenizer"
                                       else paths.get("basemodel_save_name", "basemodel")) / "best_model"
            if any(not (folder / filename).is_file() or sha256(folder / filename) != digest for filename, digest in hashes.items()):
                raise ValueError(f"The completed {component} checkpoint changed; restore it or prepare a new experiment.")
    tokenizer = paths.get("finetuned_tokenizer") or (str(info["base"] / paths.get("tokenizer_save_name", "tokenizer") / "best_model")
                    if raw.get("experiment", {}).get("train_tokenizer", True) else paths.get("pretrained_tokenizer"))
    required = {"trained predictor": info["base"] / paths.get("basemodel_save_name", "basemodel") / "best_model",
                "trained predictor tokenizer": tokenizer,
                "original local predictor": paths.get("pretrained_predictor"),
                "original local tokenizer": paths.get("pretrained_tokenizer")}
    for label, value in required.items():
        folder = local_path(value) if value else None
        if folder is None or not (folder / "config.json").is_file() or not any(
                (folder / filename).is_file() for filename in ("model.safetensors", "pytorch_model.bin")):
            raise ValueError(f"Missing {label} checkpoint. Run ./wizard.sh to finish preparing and training this setup.")
    data = raw.get("data", {})
    if not data.get("data_path") or not local_path(data["data_path"]).is_file():
        raise ValueError("The frozen candle dataset is missing. Restore this experiment's dataset or prepare a new setup with ./wizard.sh.")
    if data.get("sha256") and sha256(local_path(data["data_path"])) != data["sha256"]:
        raise ValueError("The experiment's frozen candle dataset changed; restore its original snapshot.")
    return info


@dataclass(frozen=True)
class Process:
    pid: int
    start_time: str
    parent: int
    cwd: Path
    args: tuple[str, ...]
    executable: str

    @property
    def module(self):
        if not Path(self.executable).name.startswith("python"):
            return None
        index = 1
        while index < len(self.args) and self.args[index] in {"-u", "-B", "-E", "-I", "-O", "-OO", "-s", "-S"}:
            index += 1
        return self.args[index + 1] if self.args[index:index + 1] == ("-m",) and index + 1 < len(self.args) else None

    def argument(self, option, default=None):
        for index, value in enumerate(self.args):
            if value.startswith(option + "="):
                return value.split("=", 1)[1]
            if value == option and index + 1 < len(self.args):
                return self.args[index + 1]
        return default


def process_snapshot():
    result = {}
    for proc in Path("/proc").iterdir():
        try:
            if not proc.name.isdecimal() or proc.stat().st_uid != os.getuid():
                continue
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] in {"Z", "X"}:
                continue
            args = tuple(os.fsdecode(arg) for arg in (proc / "cmdline").read_bytes().split(b"\0") if arg)
            result[int(proc.name)] = Process(int(proc.name), fields[19], int(fields[1]),
                                            (proc / "cwd").resolve(strict=True), args,
                                            str((proc / "exe").resolve(strict=True)))
        except (OSError, ValueError, IndexError):
            continue
    return result


def _process_config(proc):
    value = proc.argument("--config")
    return (proc.cwd / value).resolve() if value else None


def _matches_dashboard(proc, descriptor, *, any_port=False, stopping=False):
    if proc.cwd != ROOT.resolve() or proc.module != "finetune_csv.dashboard":
        return False
    try:
        port = int(proc.argument("--port", "7071"))
        if not any_port and port != descriptor["port"]:
            return False
        path = _process_config(proc)
        if path is None:
            return False
        state_path = ROOT / "outputs/wizard" / f"dashboard-{port}.json"
        state = read_json(state_path) if state_path.is_file() else None
        if state and state.get("pid") == proc.pid:
            if state.get("start_time") != proc.start_time or local_path(state.get("config", "")) != path:
                return False
            if state.get("config_sha256") != descriptor["config_sha256"]:
                return False
            if stopping:
                # The running process loaded the recorded startup bytes. Its
                # former editable filename may now contain another experiment.
                return True
        elif path != local_path(descriptor["config"]):
            return False
        info = config_info(path)
        return (info["sha256"] == descriptor["config_sha256"] and info["name"] == descriptor["experiment_name"]
                and info["base"] == local_path(descriptor["base_save_path"]))
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return False


def start_setup(*, config=None, name=None, port=None):
    descriptor = resolve_setup(config=config, name=name, port=port)
    info = require_ready(descriptor)
    for proc in process_snapshot().values():
        if _matches_dashboard(proc, descriptor, any_port=True):
            active_port = int(proc.argument("--port", "7071"))
            if active_port != descriptor["port"]:
                raise ValueError(f"This setup is already running at http://localhost:{active_port}/. Stop that instance before changing ports.")
    from .wizard import dashboard_already_running, freeze_config_bytes, start_dashboard, write_json

    # Freeze the exact YAML bytes. Reformatting would change paper identity.
    dashboard_already_running(info["config"], descriptor["port"])
    contents = info["config"].read_bytes()
    if hashlib.sha256(contents).hexdigest() != descriptor["config_sha256"]:
        raise ValueError("The setup configuration changed while starting; retry with its immutable --name setup.")
    frozen = freeze_config_bytes(info["base"] / "setup/config.yaml", contents)
    descriptor = {**descriptor, "config": str(frozen), "source_config": descriptor.get("source_config", str(info["config"]))}
    write_json(info["base"] / "wizard-setup.json", descriptor)
    write_json(ROOT / "outputs/wizard/last-setup.json", descriptor)
    start_dashboard(Path(descriptor["config"]), descriptor["port"])
    print("Stored paper positions and the paper entry setting are preserved.")
    return descriptor


def _matches_pipeline(proc, descriptor):
    if proc.cwd != ROOT.resolve() or proc.module not in PIPELINE_MODULES:
        return False
    if proc.module == "finetune_csv.wizard" and proc.argument("--name") == descriptor["experiment_name"]:
        return True
    path = _process_config(proc)
    if path is None:
        return False
    try:
        info = config_info(path)
        return info["name"] == descriptor["experiment_name"] and info["base"] == local_path(descriptor["base_save_path"])
    except (OSError, ValueError, yaml.YAMLError):
        return False


def _still_running(proc):
    try:
        fields = (Path("/proc") / str(proc.pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] not in {"Z", "X"} and fields[19] == proc.start_time
    except (OSError, IndexError):
        return False


def stop_setup(*, config=None, name=None, port=None, all_setups=False, timeout=10):
    if all_setups and any(value is not None for value in (config, name, port)):
        raise ValueError("Use --all by itself, or select one setup with --name/--config/--port.")
    descriptor = None if all_setups else resolve_setup(config=config, name=name, port=port)
    processes = process_snapshot()
    selected = {pid for pid, proc in processes.items() if proc.cwd == ROOT.resolve() and (
        (all_setups and proc.module in PIPELINE_MODULES | {"finetune_csv.dashboard"}) or
        (not all_setups and (_matches_dashboard(proc, descriptor, stopping=True) or _matches_pipeline(proc, descriptor))))}
    # A verified selected trainer may be the child of an interactive wizard
    # whose answers are absent from argv. Stop that parent before its next stage.
    for pid in list(selected):
        parent = processes.get(processes[pid].parent)
        if parent and parent.cwd == ROOT.resolve() and parent.module == "finetune_csv.wizard":
            selected.add(parent.pid)
    while True:
        children = {pid for pid, proc in processes.items() if proc.parent in selected}
        if children <= selected:
            break
        selected |= children
    targets = {pid: processes[pid] for pid in selected if pid != os.getpid()}
    for proc in targets.values():
        if _still_running(proc):
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout
    while targets and time.monotonic() < deadline:
        targets = {pid: proc for pid, proc in targets.items() if _still_running(proc)}
        if targets:
            time.sleep(.05)
    for proc in targets.values():
        if _still_running(proc):
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    while targets and time.monotonic() < deadline:
        targets = {pid: proc for pid, proc in targets.items() if _still_running(proc)}
        if targets:
            time.sleep(.05)
    if targets:
        raise RuntimeError("Unable to stop wizard processes: " + ", ".join(str(pid) for pid in targets))
    print("Wizard setup stopped." if selected else "Wizard setup is already stopped.")
    print("Checkpoints, data, reports and paper positions are preserved; the original Web UI is unchanged.")
    return sorted(selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--port", type=int)
    parser.add_argument("--all", action="store_true", dest="all_setups", help="Stop all wizard dashboards/pipelines in this checkout")
    args = parser.parse_args(argv)
    try:
        if args.action == "start":
            if args.all_setups:
                raise ValueError("--all is available only when stopping wizard setups.")
            start_setup(config=args.config, name=args.name, port=args.port)
        else:
            stop_setup(config=args.config, name=args.name, port=args.port, all_setups=args.all_setups)
        return 0
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
