#!/usr/bin/env python3
"""Advance once from the successful local pilot to the already authorized test."""
import json
from pathlib import Path
import subprocess
import sys
import time

import model_selection_v070 as m

root = Path(__file__).resolve().parents[1]
deadline = time.monotonic() + 1900
try:
    while True:
        state = json.loads((root / 'setup/deploy_progress.json').read_text())
        m.require(state['phase'] != 'failed', 'local deployment/pilot failed; main test not started')
        if state['phase'] == 'local_deployment_and_pilot_complete':
            break
        m.require(time.monotonic() < deadline, 'pilot wait timeout; main test not started')
        time.sleep(5)
    m.require(not (root / 'setup/main.pid').exists(), 'main test already launched')
    m.require(not (root / 'runs/model_selection_v070').exists(), 'main run already exists')
    with (root / 'setup/main.log').open('w') as log:
        proc = subprocess.Popen([sys.executable, str(root / 'scripts/run_screening_v070.py')],
                                cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (root / 'setup/main.pid').write_text(str(proc.pid) + '\n')
    m.atomic_json(root / 'setup/main_launch.json', {'status': 'started', 'pid': proc.pid,
                  'started_unix': time.time(), 'gate': 'local warmups passed and all 35 pilot receipts recorded without protocol/runtime errors'})
except Exception as exc:
    m.atomic_json(root / 'setup/main_launch.json', {'status': 'not_started', 'error_type': type(exc).__name__,
                                                  'detail': str(exc), 'updated_unix': time.time()})
    raise
