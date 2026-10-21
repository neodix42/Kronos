"""Interactive, resumable local CSV training and matched test evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import yaml

from market.binance import SUPPORTED_INTERVALS
from market.bybit import SUPPORTED_INTERVALS as BYBIT_INTERVALS
from market.download import utc_date

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "finetune_csv/configs/my_btc_small.yaml"
PRETRAINED = (
    ("NeoQuasar/Kronos-small", "901c26c1332695a2a8f243eb2f37243a37bea320", "Kronos-small"),
    ("NeoQuasar/Kronos-Tokenizer-base", "0e0117387f39004a9016484a186a908917e22426", "Kronos-Tokenizer-base"),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ask(label, default, *, choices=None, non_interactive=False):
    if non_interactive:
        print(f"{label}: {default}")
        return str(default)
    while True:
        answer = input(f"{label} [{default}]: ").strip() or str(default)
        if choices is None or answer in choices:
            return answer
        print("Choose one of: " + ", ".join(choices))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise ValueError("Expected a positive integer.")
    return number


def prepare_models():
    """Copy pinned snapshots, retaining provenance and checking existing copies."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    for repo_id, revision, name in PRETRAINED:
        destination = ROOT / "pretrained" / name
        provenance = destination / "provenance.json"
        if destination.exists():
            if not provenance.is_file():
                raise ValueError(f"{destination} exists without wizard provenance. Move it aside before preparing a pinned copy.")
            info = json.loads(provenance.read_text())
            if info.get("repo_id") != repo_id or info.get("revision") != revision or not info.get("sha256"):
                raise ValueError(f"Pinned model provenance differs in {destination}.")
            if any(not (destination / name).is_file() or sha256(destination / name) != digest
                   for name, digest in info["sha256"].items()):
                raise ValueError(f"Local pretrained files changed in {destination}; restore them before continuing.")
            print(f"Reusing pinned local copy: {destination}", flush=True)
            continue
        options = dict(repo_id=repo_id, revision=revision,
                       allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin"])
        try:
            snapshot = Path(snapshot_download(**options, local_files_only=True))
        except LocalEntryNotFoundError:
            snapshot = Path(snapshot_download(**options))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".preparing-") as staging:
            staging = Path(staging)
            names = [name for name in ("config.json", "model.safetensors", "pytorch_model.bin")
                     if (snapshot / name).is_file()]
            if "config.json" not in names or not any(name in names for name in ("model.safetensors", "pytorch_model.bin")):
                raise ValueError(f"Incomplete pretrained snapshot for {repo_id}.")
            for filename in names:
                shutil.copy2(snapshot / filename, staging / filename)
            write_json(staging / "provenance.json", dict(repo_id=repo_id, revision=revision,
                       sha256={name: sha256(staging / name) for name in names}))
            staging.rename(destination)
        print(f"Saved pinned local copy: {destination}", flush=True)


def make_config(*, exchange="bybit", category="spot", symbol="BTCUSDT", interval="1h",
                start_date="2021-01-01", exp_name="my_btc_small", mode="finetune",
                epochs=3, batch_size=2, sample_count=100, fee_bps=10, slippage_bps=5):
    if exchange not in ("binance", "bybit") or category not in ("spot", "linear", "inverse"):
        raise ValueError("Unsupported exchange or market category.")
    if exchange == "binance" and category != "spot":
        raise ValueError("Binance downloads support spot here.")
    if interval not in (SUPPORTED_INTERVALS if exchange == "binance" else BYBIT_INTERVALS):
        raise ValueError(f"Unsupported {exchange} interval: {interval}")
    if not re.fullmatch(r"[A-Z0-9]{2,30}", symbol):
        raise ValueError("Use an exchange symbol such as BTCUSDT.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", exp_name):
        raise ValueError("Experiment name must use letters, digits, underscores or dashes.")
    try:
        utc_date(start_date)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc
    if mode not in ("finetune", "scratch"):
        raise ValueError("Training mode must be finetune or scratch.")
    for value in (epochs, batch_size, sample_count):
        positive_int(value)
    if not 1 <= sample_count <= 500:
        raise ValueError("Forecast paths must be between 1 and 500.")
    if not 0 <= fee_bps <= 100 or not 0 <= slippage_bps <= 100:
        raise ValueError("Fees and slippage must each be between 0 and 100 bps per side.")
    base = f"finetune_csv/finetuned/{exp_name}"
    scratch = mode == "scratch"
    return {
        "data": dict(data_path=f"data/{exchange}_{category}_{symbol}_{interval}_{start_date}.csv",
                     interval=interval, lookback_window=488, predict_window=24, max_context=512,
                     clip=5.0, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2),
        "training": dict(basemodel_epochs=int(epochs), tokenizer_epochs=int(epochs), batch_size=int(batch_size),
                         num_workers=2, log_interval=50, seed=42, predictor_learning_rate=0.00001,
                         tokenizer_learning_rate=0.0002, adam_beta1=0.9, adam_beta2=0.95,
                         adam_weight_decay=0.1, accumulation_steps=1),
        "model_paths": dict(pretrained_predictor="pretrained/Kronos-small",
                            pretrained_tokenizer="pretrained/Kronos-Tokenizer-base", exp_name=exp_name,
                            base_path="finetune_csv/finetuned", base_save_path=base,
                            finetuned_tokenizer=f"{base}/tokenizer/best_model" if scratch else "pretrained/Kronos-Tokenizer-base",
                            tokenizer_save_name="tokenizer", basemodel_save_name="basemodel"),
        "model": dict(from_scratch=scratch),
        "experiment": dict(name=exp_name, description=f"{symbol} {interval} {mode}; chronological held-out test",
                           use_comet=False, train_tokenizer=False, train_basemodel=True, skip_existing=False,
                           pre_trained_tokenizer=True, pre_trained_predictor=not scratch,
                           exchange=exchange, category=category, symbol=symbol, interval=interval,
                           start_date=start_date, mode=mode, dataset_manifest=f"{base}/dataset-manifest.json"),
        "device": dict(use_cuda=True, device_id=0),
        "evaluation": dict(exchange=exchange, category=category, symbol=symbol, interval=interval,
                           sample_count=int(sample_count), batch_size=8, seed=42, temperature=1.0,
                           top_k=0, top_p=0.9, stride=24, fee_bps=float(fee_bps),
                           slippage_bps=float(slippage_bps), initial_equity=10000,
                           consensus_threshold=0.70, risk_fraction=0.01,
                           max_position_fraction=0.25, minimum_reward_risk=1.5),
    }


def config_files(config, path):
    """A scratch run really executes two separate stage configurations."""
    path = Path(path)
    configs = [(path, config)]
    if config["model"]["from_scratch"]:
        tokenizer, predictor = deepcopy(config), deepcopy(config)
        tokenizer["experiment"].update(train_tokenizer=True, train_basemodel=False,
                                       pre_trained_tokenizer=False, pre_trained_predictor=False)
        predictor["experiment"].update(train_tokenizer=False, train_basemodel=True,
                                       pre_trained_tokenizer=True, pre_trained_predictor=False)
        configs += [(path.with_name(path.stem + "_scratch_tokenizer.yaml"), tokenizer),
                    (path.with_name(path.stem + "_scratch_predictor.yaml"), predictor)]
    return configs


def freeze_dataset(config):
    from finetune_csv.data_utils import load_candles, split_bounds, contiguous_window_starts

    path = ROOT / config["data"]["data_path"]
    config["data"]["sha256"] = sha256(path)
    frame = load_candles(config)
    train_end, val_end = split_bounds(frame, config)
    window = config["data"]["lookback_window"] + config["data"]["predict_window"]
    partitions = {"train": (0, train_end), "validation": (train_end, val_end), "test": (val_end, len(frame))}
    manifest = dict(csv=str(path), sha256=config["data"]["sha256"], rows=len(frame),
                    interval=config["data"]["interval"], normalization="Per window, using historical context only",
                    split_policy="Chronological raw rows, split before window creation", partitions={})
    for name, (start, end) in partitions.items():
        part = frame.iloc[start:end].reset_index(drop=True)
        if part.empty:
            raise ValueError(f"{name} partition is empty; download more history.")
        count = len(contiguous_window_starts(part, window, config["data"]["interval"]))
        manifest["partitions"][name] = dict(start_row=start, end_row_exclusive=end, rows=end-start,
                                            first_timestamp=part.timestamps.iloc[0].isoformat(),
                                            last_timestamp=part.timestamps.iloc[-1].isoformat(),
                                            contiguous_windows=count)
        if name != "test" and count < max(1, config["training"]["batch_size"] if name == "train" else 1):
            raise ValueError(f"Not enough contiguous {name} candles for {window}-bar windows. "
                             "Download more history; the default 10% validation split needs at least 5,120 rows.")
    if len(frame) - val_end < config["data"]["predict_window"]:
        raise ValueError("The held-out test partition needs at least one complete forecast horizon.")
    return manifest


def save_configs(config, path):
    files = config_files(config, path)
    for target, contents in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        rendered = yaml.safe_dump(contents, sort_keys=False)
        if target.exists() and target.read_text() != rendered:
            backup = ROOT / "outputs/wizard/config-backups" / (target.stem + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".yaml")
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        target.write_text(rendered, encoding="utf-8")
        print(f"Configuration: {target}", flush=True)
    fingerprint = hashlib.sha256(json.dumps([item for _, item in files], sort_keys=True).encode()).hexdigest()
    return files, fingerprint


def run_logged(command, log_path):
    """Forward cancellation to the process group, including DataLoader workers."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running: " + " ".join(str(part) for part in command), flush=True)
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True,
                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
    previous_handler = signal.getsignal(signal.SIGTERM)

    def cancel(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        if process.wait() != 0:
            raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}")
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        process.stdout.close()


def dashboard_already_running(config_path, port):
    """A reused filename must not disguise a different running experiment."""
    directory = ROOT / "outputs/wizard"
    state_path = directory / f"dashboard-{port}.json"
    with socket.socket() as probe:
        occupied = probe.connect_ex(("127.0.0.1", port)) == 0
    if occupied:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        proc = Path("/proc") / str(state.get("pid", "missing"))
        try:
            identity = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
            matches = (state.get("start_time") == identity and state.get("config") == str(config_path.resolve())
                       and state.get("config_sha256") == sha256(config_path))
        except FileNotFoundError:
            matches = False
        if matches:
            return True
        raise ValueError(f"Port {port} is already in use. Choose --port with another port, or run ./stop-webui.sh first.")
    return False


def start_dashboard(config_path, port):
    if not 1024 <= port <= 65535:
        raise ValueError("Choose a dashboard port between 1024 and 65535.")
    if dashboard_already_running(config_path, port):
        print(f"Dashboard already running: http://localhost:{port}/")
        return
    directory = ROOT / "outputs/wizard"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / f"dashboard-{port}.json"
    log_path = directory / f"dashboard-{port}.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "finetune_csv.dashboard", "--config", str(config_path),
                                    "--port", str(port)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    identity = (Path("/proc") / str(process.pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
    write_json(state_path, dict(pid=process.pid, start_time=identity, config=str(config_path.resolve()),
                               config_sha256=sha256(config_path),
                               port=port, log=str(log_path)))
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and process.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as response:
                if response.status == 200:
                    print(f"Trained-model dashboard: http://localhost:{port}/", flush=True)
                    print(f"Dashboard log: {log_path}\nStop all local Kronos dashboards: ./stop-webui.sh")
                    return
        except (OSError, TimeoutError):
            time.sleep(0.2)
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=10)
    raise RuntimeError(f"Dashboard did not start; see {log_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--non-interactive", action="store_true", help="Use supplied arguments or defaults without prompts")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare data, local models and configs without training")
    parser.add_argument("--refresh-data", action="store_true", help="Replace the existing data snapshot with a new download")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Generated YAML destination")
    for name, default in (("exchange", "bybit"), ("category", "spot"), ("symbol", "BTCUSDT"),
                          ("interval", "1h"), ("start-date", "2021-01-01"), ("name", "my_btc_small"),
                          ("mode", "finetune")):
        parser.add_argument("--" + name, default=default)
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--sample-count", type=positive_int, default=100)
    parser.add_argument("--fee-bps", type=float, default=10)
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--port", type=int, default=7071)
    args = parser.parse_args(argv)
    if not args.non_interactive and not sys.stdin.isatty():
        parser.error("Interactive input needs a terminal. Use --non-interactive to choose defaults explicitly.")
    try:
        print("Kronos training wizard: download → local checkpoints → training → held-out comparison → dashboard", flush=True)
        options = dict(non_interactive=args.non_interactive)
        exchange = ask("Exchange", args.exchange, choices=["bybit", "binance"], **options)
        category = ask("Market category", args.category, choices=["spot", "linear", "inverse"] if exchange == "bybit" else ["spot"], **options)
        symbol = ask("Exchange symbol", args.symbol, **options).upper().replace("/", "")
        interval = ask("Candle interval (1m=minute, 1M=month)", args.interval,
                       choices=list(BYBIT_INTERVALS if exchange == "bybit" else SUPPORTED_INTERVALS), **options)
        start_date = ask("History start, YYYY-MM-DD UTC", args.start_date, **options)
        mode = ask("Training mode (finetune=predictor only; scratch=tokenizer then predictor)", args.mode,
                   choices=["finetune", "scratch"], **options)
        name = ask("Experiment name (choose a new name for a different run)", args.name, **options)
        epochs = positive_int(ask("Training epochs", args.epochs, **options))
        batch_size = positive_int(ask("Training batch size", args.batch_size, **options))
        sample_count = positive_int(ask("Forecast paths per test origin, per model", args.sample_count, **options))
        fee_bps = float(ask("Fee per side, basis points", args.fee_bps, **options))
        slippage_bps = float(ask("Slippage per fill, basis points", args.slippage_bps, **options))
        port = positive_int(ask("Dashboard port", args.port, **options))
        if not 1024 <= port <= 65535:
            raise ValueError("Choose a dashboard port between 1024 and 65535.")
        config = make_config(exchange=exchange, category=category, symbol=symbol, interval=interval,
                             start_date=start_date, exp_name=name, mode=mode, epochs=epochs, batch_size=batch_size,
                             sample_count=sample_count, fee_bps=fee_bps, slippage_bps=slippage_bps)
        source_path = ROOT / config["data"]["data_path"]
        run_dir = ROOT / config["model_paths"]["base_save_path"]
        state_path = run_dir / "wizard-state.json"
        csv_path = run_dir / "dataset/candles.csv"
        if args.refresh_data and (state_path.exists() or csv_path.exists()):
            raise ValueError("This experiment already has a frozen dataset. Use a new --name with --refresh-data.")
        if not csv_path.exists() and (not source_path.exists() or args.refresh_data):
            run_logged([str(ROOT / "download_data.sh"), exchange, symbol, interval, start_date,
                        "--category", category], ROOT / "outputs/wizard/download.log")
        if not csv_path.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=csv_path.parent, prefix=".freezing-", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                shutil.copyfile(source_path, temporary_path)
                if sha256(source_path) != sha256(temporary_path):
                    raise ValueError("The source CSV changed while taking the snapshot; rerun after the download finishes.")
                temporary_path.replace(csv_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            if source_path.with_suffix(".metadata.json").is_file():
                shutil.copy2(source_path.with_suffix(".metadata.json"), csv_path.with_suffix(".metadata.json"))
            print(f"Frozen per-experiment copy: {csv_path}", flush=True)
        else:
            print(f"Reusing frozen experiment snapshot: {csv_path}", flush=True)
        config["experiment"]["source_data_path"] = str(source_path.relative_to(ROOT))
        config["data"]["data_path"] = str(csv_path.relative_to(ROOT))
        sidecar = csv_path.with_suffix(".metadata.json")
        if sidecar.is_file():
            metadata = json.loads(sidecar.read_text())
            if any(metadata.get(key) != value for key, value in
                   (("exchange", exchange), ("category", category), ("symbol", symbol), ("interval", interval))):
                raise ValueError("Downloaded file metadata does not match the requested market.")
            print(f"Snapshot: {metadata.get('first_candle')} through {metadata.get('last_candle')} UTC")
            for notice in metadata.get("notices", []):
                print("NOTICE: " + notice)
        manifest = freeze_dataset(config)
        files = config_files(config, args.config.resolve())
        fingerprint = hashlib.sha256(json.dumps([item for _, item in files], sort_keys=True).encode()).hexdigest()
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        checkpoints = list(run_dir.glob("*/best_model/config.json"))
        if (state and state.get("fingerprint") != fingerprint) or (checkpoints and not state):
            raise ValueError(f"{run_dir} already holds a different or untracked run. Choose a new --name to preserve its checkpoints and untouched test split.")
        if not state:
            state = dict(fingerprint=fingerprint, completed_stages=[], config=str(args.config.resolve()))
        for stage, hashes in state.get("checkpoint_hashes", {}).items():
            checkpoint = run_dir / ("tokenizer" if stage == "tokenizer" else "basemodel") / "best_model"
            if any(not (checkpoint / name).is_file() or sha256(checkpoint / name) != digest
                   for name, digest in hashes.items()):
                raise ValueError(f"The completed {stage} checkpoint changed. Choose a new experiment name.")
        save_configs(config, args.config.resolve())
        if not args.prepare_only and not args.no_dashboard:
            dashboard_already_running(args.config.resolve(), port)
        write_json(run_dir / "dataset-manifest.json", manifest)
        write_json(state_path, state)
        for split, values in manifest["partitions"].items():
            print(f"{split}: {values['rows']:,} rows, {values['first_timestamp']} → {values['last_timestamp']}", flush=True)
        prepare_models()
        if args.prepare_only:
            print("Preparation complete. Run ./wizard.sh again with the same choices to train, evaluate and launch the dashboard.")
            return 0
        stages = files[1:] if mode == "scratch" else files
        for index, (path, _) in enumerate(stages):
            stage = "tokenizer" if mode == "scratch" and index == 0 else "predictor"
            checkpoint = run_dir / ("tokenizer" if stage == "tokenizer" else "basemodel") / "best_model"
            if stage in state["completed_stages"] and (checkpoint / "config.json").is_file():
                print(f"Reusing completed {stage} training: {checkpoint}", flush=True)
                continue
            run_logged([sys.executable, "-m", "finetune_csv.train_sequential", "--config", str(path)],
                       run_dir / "logs" / f"wizard-{stage}.log")
            if not (checkpoint / "config.json").is_file():
                raise RuntimeError(f"Training finished without the expected checkpoint: {checkpoint}")
            state["completed_stages"].append(stage)
            state.setdefault("checkpoint_hashes", {})[stage] = {p.name: sha256(p) for p in checkpoint.iterdir() if p.is_file()}
            write_json(state_path, state)
        report = run_dir / "evaluation/report.json"
        if "evaluation" not in state["completed_stages"] or not report.exists():
            run_logged([sys.executable, "-m", "finetune_csv.evaluate", "--config", str(args.config.resolve())],
                       run_dir / "logs/wizard-evaluation.log")
            state["completed_stages"].append("evaluation")
            write_json(state_path, state)
        print(f"Matched held-out test comparison: {report}", flush=True)
        if not args.no_dashboard:
            start_dashboard(args.config.resolve(), port)
        return 0
    except KeyboardInterrupt:
        print("\nWizard stopped. Completed stages and cached test forecasts are preserved; rerun to continue.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
