#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
umask 077
mkdir -p runs/prescale_v073
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/run_v073.py "$@" >> runs/prescale_v073/console.log 2>&1
nicheflow_exit=$?
.venv/bin/python - "$nicheflow_exit" <<'PY'
from datetime import datetime,timezone
from pathlib import Path
import sys
from nicheflow.ledger import atomic_json
atomic_json(Path('runs/prescale_v073/exit.json'), {'exit_code':int(sys.argv[1]),'finished_at':datetime.now(timezone.utc).isoformat()})
PY
exit "$nicheflow_exit"
