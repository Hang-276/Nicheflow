#!/usr/bin/env python3
"""Read-only compact progress, accounting and concurrency audit."""
from collections import Counter,defaultdict
import fcntl
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nicheflow.spec import digest

def monitor(directory):
    directory=Path(directory)
    if not (directory/'config.json').exists():return {'status':'not_started'}
    cfg=json.loads((directory/'config.json').read_text());settings=cfg['config']['settings']
    plan=json.loads((directory/'plan.json').read_text())
    running=False
    with (directory/'.lock').open('r') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:running=True
    raw=(directory/'events.jsonl').read_bytes();parts=raw.split(b'\n');partial=bool(parts[-1])
    events=[json.loads(x) for x in parts[:-1]]
    previous='0'*64
    for i,e in enumerate(events):
        assert e['seq']==i and e['previous']==previous
        assert digest({k:v for k,v in e.items() if k!='hash'})==e['hash']
        previous=e['hash']
    starts={};finished={};executions={};active=Counter();peaks=Counter();seconds=defaultdict(list);overlap=False
    profiles={p.get('path',p.get('model')):p['kind'] for p in settings['models'].values()}
    for e in events:
        id,p=e['id'],e['payload']
        if e['kind']=='call_started':
            starts[id]=p;kind=profiles[p['model']];active[kind]+=1;peaks[kind]=max(peaks[kind],active[kind])
            overlap|=active['api']>0 and active['local']>0
        elif e['kind']=='call_finished':
            finished[id]=p;kind=profiles[starts[id]['model']];active[kind]-=1
            if p.get('elapsed_seconds') is not None:seconds[kind].append(p['elapsed_seconds'])
        elif e['kind']=='execution':executions[id]=p
    result={'status':'running' if running else 'stopped','completed':len(executions),'total':plan['executions'],
        'by_group':dict(Counter(j['group'] for j in plan['jobs'] if j['execution_id'] in executions)),
        'calls_started':len(starts),'calls_finished':len(finished),'inflight_calls':len(starts)-len(finished),
        'unknown_cost_calls':sum(p.get('accounted_usd') is None for p in finished.values()),
        'call_errors':sum(p['status']!='ok' for p in finished.values()),
        'execution_errors':sum(p['status'] not in ('ok','truncated') for p in executions.values()),
        'truncated_executions':sum(p['status']=='truncated' for p in executions.values()),
        'actual_api_usd':sum((p.get('api_charge') or {}).get('usd',0) for p in finished.values()),
        'elapsed_minutes':(time.time()-cfg['started'])/60,'observed_peak_concurrency':dict(peaks),
        'observed_local_api_overlap':overlap,'call_timing':{k:{'n':len(v),'mean_seconds':statistics.mean(v)} for k,v in seconds.items()},
        'verified_complete_event_chain':True,'partial_tail_while_running':partial and running,
        'unexpected_partial_tail':partial and not running,'concurrency':settings['concurrency']}
    if (directory/'results.json').exists():
        final=json.loads((directory/'results.json').read_text());result['status']=final['run_status']
        result['elapsed_minutes']=final['elapsed_seconds']/60
    return result

if __name__=='__main__':
    print(json.dumps(monitor(sys.argv[1] if len(sys.argv)>1 else ROOT/'runs/fixed_validation_v060'),ensure_ascii=False,indent=2))
