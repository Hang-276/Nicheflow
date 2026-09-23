#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${NICHEFLOW_PYTHON:-}" ]]; then
  python_bin="$NICHEFLOW_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
elif [[ -x /root/miniconda3/bin/python ]]; then
  python_bin=/root/miniconda3/bin/python
else
  python_bin=python3
fi
exec "$python_bin" -m nicheflow.cli "$@"
