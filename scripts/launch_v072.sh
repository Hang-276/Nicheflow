#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/multidomain_v072
umask 077
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/run_v072.py "$@" >> runs/multidomain_v072/console.log 2>&1
nicheflow_exit=$?
.venv/bin/python - "$nicheflow_exit" <<'PY'
from datetime import datetime,timezone
import json,sys
from pathlib import Path
from nicheflow.ledger import atomic_json
record={'exit_code':int(sys.argv[1]),'finished_at':datetime.now(timezone.utc).isoformat(),
        'note':'Nonzero exit preserves state.sqlite3 and receipts; inspect progress.json before explicit resume.'}
atomic_json(Path('runs/multidomain_v072/exit.json'),record)
PY
exit "$nicheflow_exit"
