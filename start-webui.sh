#!/usr/bin/env bash
set -euo pipefail

KRONOS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$KRONOS_ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv; follow the local setup instructions in README.md." >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec .venv/bin/python -m flask --app webui.app run \
    --host 127.0.0.1 --port "${PORT:-7070}" --no-debugger --no-reload
