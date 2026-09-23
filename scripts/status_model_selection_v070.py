#!/usr/bin/env python3
"""Read-only remote status. Reuses SSH; never asks for a password or calls a model."""
import json
import subprocess
import sys

PROGRAM = r'''
import json
from pathlib import Path
r=Path('/root/autodl-tmp/NicheFlow_Server_Handoff/NicheFlow_ModelSelection_v070')
def read(path):
    return json.loads(path.read_text()) if path.exists() else None
run=r/'runs/model_selection_v070'
result={'deployment':read(r/'setup/deploy_progress.json'),
        'pilot':read(r/'preflight/local_readiness/results.json'),
        'progress':read(run/'progress.json'),'stopped':read(run/'stopped.json'),
        'results':read(run/'results.json'),'audit':read(run/'audit.json'),
        'exit':read(r/'setup/main_exit.json')}
pidfile=r/'setup/main.pid'
result['main_process_alive']=False
if pidfile.exists():
    pid=pidfile.read_text().strip()
    cmd=Path('/proc')/pid/'cmdline'
    result['main_process_alive']=cmd.exists() and b'run_screening_v070.py' in cmd.read_bytes()
failures=[]
events=run/'events.jsonl'
if events.exists():
    for line in events.read_text().splitlines():
        try: event=json.loads(line)
        except ValueError: continue
        if event['kind']=='call_finished':
            response=event['payload']['response']
            if response['status']!='ok':
                failures.append({'id':event['id'],'status':response['status'],
                                 'http_status':response.get('http_status'),
                                 'error_code':response.get('error_code'),
                                 'quota_or_rate_limit':response.get('quota_or_rate_limit',False)})
result['failed_requests']=failures
print(json.dumps(result,ensure_ascii=False))
'''

command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
           '-o', 'ControlPath=/private/tmp/nicheflow-ssh-login/control-33474',
           '-p', '33474', 'root@connect.nmb1.seetacloud.com',
           '/root/autodl-tmp/NicheFlow_Server_Handoff/NicheFlow_Minimal_Probe/.venv/bin/python -']
try:
    result = subprocess.run(command, input=PROGRAM, capture_output=True, text=True, timeout=45)
    if result.returncode:
        print(json.dumps({'status': 'ssh_unavailable', 'returncode': result.returncode,
                          'detail': 'No password popup, no model requests, and no automatic restart.'}))
        sys.exit(2)
    print(json.dumps(json.loads(result.stdout), ensure_ascii=False, indent=2))
except Exception as exc:
    print(json.dumps({'status': 'status_unavailable', 'error_type': type(exc).__name__}))
    sys.exit(2)
