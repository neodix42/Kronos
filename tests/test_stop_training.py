"""The stop script must terminate this checkout's whole training process tree."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(not Path('/proc').exists(), reason='Linux process identity checks')
def test_stop_training_children_without_affecting_another_checkout(tmp_path):
    root = tmp_path / 'checkout'
    other = tmp_path / 'unrelated-checkout'
    script = Path(__file__).resolve().parents[1] / 'stop-webui.sh'
    processes = []
    for directory in (root, other):
        (directory / 'finetune_csv').mkdir(parents=True)
        (directory / '.venv/bin').mkdir(parents=True)
        (directory / '.venv/bin/python').symlink_to(sys.executable)
        # The worker moves outside the checkout and has no matching module name.
        (directory / 'finetune_csv/dashboard.py').write_text(
            "import pathlib, subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], cwd='/tmp')\n"
            "pathlib.Path('child.pid').write_text(str(child.pid))\n"
            "print('ready', flush=True)\n"
            "time.sleep(120)\n"
        )
        process = subprocess.Popen([sys.executable, '-m', 'finetune_csv.dashboard'], cwd=directory,
                                   stdout=subprocess.PIPE, text=True, start_new_session=True)
        processes.append(process)
        assert process.stdout.readline().strip() == 'ready'
    shutil.copy2(script, root / 'stop-webui.sh')
    try:
        result = subprocess.run(['bash', str(root / 'stop-webui.sh')], cwd=root,
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
        assert processes[0].wait(timeout=5) < 0
        child = Path('/proc') / (root / 'child.pid').read_text()
        assert not child.exists() or child.joinpath('stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z'
        assert processes[1].poll() is None
        assert (Path('/proc') / (other / 'child.pid').read_text()).exists()
    finally:
        for process in processes:
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            process.stdout.close()
