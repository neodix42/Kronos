#!/usr/bin/env bash
set -euo pipefail

KRONOS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stopping the service also stops its entire control group and prevents restarts.
if command -v systemctl >/dev/null 2>&1; then
    KRONOS_SERVICE_ROOT="$(systemctl --user show kronos-webui.service \
        --property=WorkingDirectory --value 2>/dev/null || true)"
    if [[ "$KRONOS_SERVICE_ROOT" == "$KRONOS_ROOT" ]]; then
        systemctl --user stop kronos-webui.service
    fi
fi

KRONOS_PYTHON="$KRONOS_ROOT/.venv/bin/python"
if [[ ! -x "$KRONOS_PYTHON" ]]; then
    KRONOS_PYTHON=python3
fi

# Direct launches have no service. Match this checkout, never just a port/name.
"$KRONOS_PYTHON" - "$KRONOS_ROOT" <<'PY'
import os
from pathlib import Path
import signal
import sys
import time

root = Path(sys.argv[1])
legacy_scripts = {root / "webui" / name for name in ("app.py", "run.py")}
targets = {}
processes = {}
training_scripts = {root / "finetune_csv" / name for name in (
    "wizard.py", "dashboard.py", "evaluate.py", "train_sequential.py",
    "finetune_base_model.py", "finetune_tokenizer.py",
)}

def process_identity(proc):
    # Start time distinguishes a process from a later reuse of the same PID.
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    return None if fields[0] in ("Z", "X") else fields[19]

def signal_process(pid, identity, signum):
    try:
        if process_identity(Path("/proc") / pid) == identity:
            os.kill(int(pid), signum)
    except (FileNotFoundError, ProcessLookupError):
        pass

for proc in Path("/proc").iterdir():
    try:
        if not proc.name.isdigit() or proc.stat().st_uid != os.getuid():
            continue
        identity = process_identity(proc)
        if identity is None:
            continue
        parent = (proc / "stat").read_text().rsplit(")", 1)[1].split()[1]
        processes[proc.name] = (parent, identity)
        args = [os.fsdecode(arg) for arg in (proc / "cmdline").read_bytes().split(b"\0") if arg]
        if len(args) < 2:
            continue
        cwd = (proc / "cwd").resolve(strict=True)
        if cwd not in (root, root / "webui", root / "finetune_csv"):
            continue
        if not (proc / "exe").resolve(strict=True).name.startswith("python"):
            continue
        flask = cwd == root and args[1:6] == ["-m", "flask", "--app", "webui.app", "run"]
        legacy = (cwd / args[1]).resolve() in legacy_scripts
        training_modules = {
            "finetune_csv.wizard", "finetune_csv.dashboard", "finetune_csv.evaluate",
            "finetune_csv.train_sequential", "finetune_csv.finetune_base_model",
            "finetune_csv.finetune_tokenizer",
        }
        training = cwd == root and len(args) > 2 and args[1] == "-m" and args[2] in training_modules
        training = training or (cwd / args[1]).resolve() in training_scripts
        if not (flask or legacy or training):
            continue
        targets[proc.name] = identity
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue

# Discover descendants before signaling parents, including workers whose cwd or
# command line changed. Start-time checks still protect against PID reuse.
changed = True
while changed:
    changed = False
    for pid, (parent, identity) in processes.items():
        if parent in targets and pid not in targets:
            targets[pid] = identity
            changed = True
for pid, identity in targets.items():
    signal_process(pid, identity, signal.SIGTERM)

def wait_for_exit(seconds):
    deadline = time.monotonic() + seconds
    while targets and time.monotonic() < deadline:
        for pid, identity in list(targets.items()):
            try:
                running = process_identity(Path("/proc") / pid) == identity
            except (FileNotFoundError, ProcessLookupError):
                running = False
            if not running:
                targets.pop(pid)
                print(f"Stopped Kronos Web UI process {pid}.")
        if targets:
            time.sleep(0.1)

wait_for_exit(10)
for pid, identity in targets.items():
    signal_process(pid, identity, signal.SIGKILL)
wait_for_exit(5)
if targets:
    sys.exit("Unable to stop Kronos Web UI processes: " + ", ".join(targets))
print("Kronos dashboards and wizard training are stopped; their model memory is released.")
PY
