#!/usr/bin/env python3
"""Read-only live monitor for the separate v052 comparison run."""
import fcntl
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

ROOT=Path('/root/autodl-tmp/NicheFlow_Server_Handoff/NicheFlow_Evaluation_v052')
RUN=ROOT/'runs/comparison_v052_before_after'

def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def snapshot():
    code=ROOT/'runs/comparison_v052.exit_code.txt'
    if not (RUN/'events.jsonl').exists():
        return {'phase':'route_planning_or_model_setup','exit_code':code.read_text().strip() if code.is_file() else None,
            'console':str(ROOT/'runs/comparison_v052.console.log')}
    previous='0'*64;started={};finished={};counts=Counter();errors=[];partial=False;last=None;executions=[]
    with (RUN/'events.jsonl').open('rb') as f:
        for i,line in enumerate(f):
            if not line.endswith(b'\n'):partial=True;break
            e=json.loads(line);body={k:v for k,v in e.items() if k!='hash'}
            if e['seq']!=i or e['previous']!=previous or digest(body)!=e['hash']:
                raise ValueError('evaluation journal chain mismatch')
            previous=e['hash'];last=e;counts[e['kind']]+=1
            if e['kind']=='call_started':started[e['id']]=e
            elif e['kind']=='call_finished':finished[e['id']]=e['payload']
            elif e['kind']=='execution':
                executions.append(e['payload'])
                if e['payload']['status'] not in ('ok','truncated'):errors.append(e['id'])
    frozen=json.loads((RUN/'config.json').read_text())
    if frozen['fingerprint']!=digest({k:frozen[k] for k in ('config','max_calls','seconds')}):
        raise ValueError('evaluation frozen config hash mismatch')
    plan=json.loads((RUN/'plan.json').read_text())
    if digest(plan)!=frozen['config']['plan_digest']:raise ValueError('evaluation plan hash mismatch')
    live=False
    if (RUN/'.lock').exists():
        with (RUN/'.lock').open('r') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
            except BlockingIOError:live=True
    return {'phase':'finished' if counts['comparison_finished'] else 'evaluation',
        'unique_workflows_done':len(executions),'unique_workflows_total':plan['unique_executions'],
        'fraction_complete':len(executions)/plan['unique_executions'],'test_tasks_total':plan['task_count'],
        'logical_groups':len(plan['groups']),'calls_started':len(started),'calls_finished':len(finished),
        'calls_by_model':dict(Counter(started[k]['payload']['model'] for k in finished)),
        'api_usd':sum((v.get('api_charge') or {}).get('usd',0.) for v in finished.values()),
        'api_cap_usd':frozen['config']['limits']['max_api_usd'],
        'local_call_seconds':sum(v.get('elapsed_seconds') or 0 for k,v in finished.items() if started[k]['payload']['model'].startswith('/')),
        'execution_errors':errors,'truncated':sum(p['status']=='truncated' for p in executions),
        'unknown_cost_calls':[k for k,v in finished.items() if v.get('accounted_usd') is None],
        'in_flight':[{'id':k,'age_seconds':round(time.time()-v['time'],1)} for k,v in started.items() if k not in finished],
        'last_event_age_seconds':round(time.time()-last['time'],1) if last else None,
        'writer_active':live,'complete_event_chain_valid':True,'plan_valid':True,'partial_last_line_ignored':partial,
        'unexpected_learning_events':sum(counts[k] for k in ('router','feedback','generation','scheduler')),
        'exit_code':code.read_text().strip() if code.is_file() else None,
        'comparison_json_exists':(RUN/'comparison.json').exists(),
        'report_ready':(RUN/'COMPARISON_REPORT.md').exists(),
        'completion_audit_exists':(RUN/'completion_audit.json').exists(),
        'postprocess_failure':(RUN/'POSTPROCESS_FAILED.txt').read_text() if (RUN/'POSTPROCESS_FAILED.txt').exists() else None}

if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=False,indent=2))
