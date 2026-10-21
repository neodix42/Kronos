#!/usr/bin/env bash
set -euo pipefail

KRONOS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KRONOS_PYTHON="$KRONOS_ROOT/.venv/bin/python"
if [[ ! -x "$KRONOS_PYTHON" ]]; then
    echo "Missing .venv; follow the local setup instructions in README.md." >&2
    exit 1
fi

# Resolve our module from any working directory; explicit output paths retain
# the caller's usual relative-path semantics.
export PYTHONPATH="$KRONOS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$KRONOS_PYTHON" -m market.download "$@"
