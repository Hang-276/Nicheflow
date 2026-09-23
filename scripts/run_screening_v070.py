#!/usr/bin/env python3
"""One-shot wrapper that records completion independently of SSH sessions."""
import json
from pathlib import Path
import time
import urllib.request

import model_selection_v070 as m

root = Path(__file__).resolve().parents[1]
pilot = json.loads((root / 'preflight/local_readiness/results.json').read_text())
m.require(pilot['completed'] == 35, 'local diagnostic pilot incomplete')
m.require((root / 'preflight/api_readiness/results.json').is_file(), 'API readiness incomplete')
with urllib.request.urlopen('http://127.0.0.1:8070/health', timeout=5) as response:
    m.require(response.status == 200, 'local service unavailable')
directory = root / 'runs/model_selection_v070'
m.require(not directory.exists(), 'existing run requires deliberate recovery, never fresh repeat')
code = 2
try:
    result = m.run(root / 'configs/model_selection_v070.plan.json', directory)
    code = 0
except Exception as exc:
    print(json.dumps({'status': 'stopped', 'error_type': type(exc).__name__}), flush=True)
finally:
    m.atomic_json(root / 'setup/main_exit.json', {'exit_code': code, 'finished_unix': time.time()})
raise SystemExit(code)
