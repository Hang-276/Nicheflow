#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
umask 077
mkdir -p runs/multidomain_v072_continuation
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/run_v072_continuation.py "$@" >> runs/multidomain_v072_continuation/console.log 2>&1
nicheflow_exit=$?
.venv/bin/python - "$nicheflow_exit" <<'PY'
from datetime import datetime,timezone
from pathlib import Path
import sys
from nicheflow.ledger import atomic_json
atomic_json(Path('runs/multidomain_v072_continuation/exit.json'),
            {'exit_code':int(sys.argv[1]),'finished_at':datetime.now(timezone.utc).isoformat()})
PY
exit "$nicheflow_exit"
