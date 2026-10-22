#!/usr/bin/env bash
set -euo pipefail

KRONOS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$KRONOS_ROOT/.venv/bin/python" ]]; then
    echo "Missing .venv; follow the setup instructions in README.md." >&2
    exit 1
fi
cd "$KRONOS_ROOT"
export PYTHONPATH="$KRONOS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "$KRONOS_ROOT/.venv/bin/python" -m finetune_csv.runtime stop "$@"
