#!/usr/bin/env python3
"""One-shot deployment and diagnostic pilot; never starts paid screening."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import model_selection_v070 as m
from nicheflow.datasets import Task

ROOT = Path(__file__).resolve().parents[1]


def state(phase, **extra):
    m.atomic_json(ROOT / 'setup/deploy_progress.json', {'phase': phase, 'updated_unix': time.time(), **extra})
    print(json.dumps({'phase': phase, **extra}), flush=True)


def main():
    state('waiting_for_verified_model_download')
    deadline = time.monotonic() + 3600
    while not (ROOT / 'setup/model_path.txt').exists():
        m.require(time.monotonic() < deadline, 'model download timeout')
        time.sleep(5)
    state('starting_local_service')
    log = (ROOT / 'setup/serve.log').open('a')
    proc = subprocess.Popen([str(ROOT / '.serve-venv/bin/python'), str(ROOT / 'scripts/serve_qwen35_v070.py')],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (ROOT / 'setup/serve.pid').write_text(str(proc.pid) + '\n')
    deadline = time.monotonic() + 1200
    while True:
        m.require(proc.poll() is None, 'local service failed; inspect serve.log')
        m.require(time.monotonic() < deadline, 'local service readiness timeout')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8070/health', timeout=2) as response:
                if response.status == 200: break
        except Exception:
            time.sleep(5)
    state('warming_local_service')
    config = json.loads((ROOT / 'configs/model_selection_v070.plan.json').read_text())
    profile = config['models']['local']
    receipts = []
    for index in range(3):
        task = Task(f'warmup-{index}', 'math', 'train', 'evaluation',
                    f'Compute {20+index} + 3. Return only the boxed answer.', str(23+index), 'rational')
        payload, _ = m.request_payload(config, profile, task.model_input(), {'max_tokens': 256}, config['seed']+index)
        result = m.call_model(profile, payload, None)
        score = m.score_result(task, result)
        receipts.append({'payload': payload, 'response': result, 'score': score})
        m.atomic_json(ROOT / 'preflight/local_warmup.json', receipts)
        m.require(result['status'] == 'ok' and score['quality'] == 1., 'local warmup failed')
    state('running_35_question_local_pilot')
    with (ROOT / 'preflight/local_pilot.log').open('w') as log:
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/preflight_model_selection_v070.py'),
                                 '--stage', 'local'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    m.require(result.returncode == 0, 'local pilot interrupted; inspect receipts')
    result = json.loads((ROOT / 'preflight/local_readiness/results.json').read_text())
    state('local_deployment_and_pilot_complete', pilot=result)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc))
        raise
