#!/usr/bin/env python3
"""Run v072 technical checks and development baselines with durable receipts.

Stops for development analysis before workflow search or sealed final evaluation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.metadata
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nicheflow.datasets import load_tasks
from nicheflow.ledger import atomic_json
from nicheflow.spec import digest,file_hash
from nicheflow.v072.durable import Store,Paused
from scripts.model_selection_v070 import call_model,credentials,request_payload


def payload(config,profile,task,sample):
    value,bound=request_payload(config,profile,task.model_input(),{'max_tokens':config['max_tokens']},config['seed']+sample)
    if profile.get('reasoning_effort'):
        value['reasoning_effort']=profile['reasoning_effort'];value.pop('enable_thinking',None)
    visible=json.dumps(value,ensure_ascii=False)
    # Structural separation is provided by Task.model_input(); never serialize Task.record here.
    assert 'gold' not in task.model_input() and 'private' not in task.model_input()
    return value,bound


def run_score(task,response,config):
    if task.dataset=='mbpp':
        command=[sys.executable,str(ROOT/'scripts/v072_code_sandbox.py'),'--root',config['code_sandbox_root']]
    else:command=[sys.executable,str(ROOT/'scripts/score_v072_response.py')]
    result=subprocess.run(command,input=json.dumps({'task':task.record(),'response':response}),
        text=True,capture_output=True,timeout=180,env=None)
    if result.returncode:
        raise ValueError(f'{task.dataset} scoring subprocess failed (receipt retained), exit {result.returncode}')
    return json.loads(result.stdout)


def freeze(config):
    manifest=json.loads((ROOT/config['manifest']).read_text())
    assert set(config['models'])=={'L','M','H','D'}
    assert config['domains']==['math','mbpp','hotpotqa']
    assert config['samples_per_task']==2 and config['max_tokens']==8192
    assert manifest['counts']=={d:{'development':40,'screening':20,'calibration':40,'final':100} for d in config['domains']}
    for path,h in manifest['task_files'].items():
        if file_hash(ROOT/path)!=h:raise ValueError('dataset manifest mismatch: '+path)
    identity=json.loads(Path(config['local_identity']).read_text())
    if identity['revision']!=config['local_revision']:raise ValueError('local model revision mismatch')
    for name,h in identity['configuration_sha256'].items():
        if file_hash(Path(identity['snapshot_path'])/name)!=h:raise ValueError('local template/configuration changed')
    ready=json.loads((ROOT/config['readiness']).read_text())
    if not ready['passed']:raise ValueError('offline readiness checks not passed')
    if ready['manifest_sha256']!=file_hash(ROOT/config['manifest']):raise ValueError('readiness belongs to different data')
    if ready['sandbox_manifest_sha256']!=file_hash(Path(config['code_sandbox_root'])/'sandbox-manifest.json'):
        raise ValueError('code isolation image differs from checked runtime')
    if ready['runtime_lock_sha256']!=file_hash(ROOT/'requirements.lock'):
        raise ValueError('runtime dependencies differ from checked environment')
    source_paths=sorted(list((ROOT/'nicheflow').rglob('*.py'))+list((ROOT/'nicheflow_probe').rglob('*.py'))+
                        [ROOT/'scripts'/name for name in ['run_v072.py','score_v072_response.py','v072_code_sandbox.py','v072_code_worker.py','model_selection_v070.py']])
    code={str(p.relative_to(ROOT)):file_hash(p) for p in source_paths}
    return {'protocol':'v072-development-1','config':config,'limits':config['limits'],
            'manifest_sha256':file_hash(ROOT/config['manifest']),'local_identity':identity,
            'readiness':ready,'code':code,'packages':{n:importlib.metadata.version(n) for n in ['numpy','math-verify','latex2sympy2_extended','sympy','evalplus','ujson']}}


def summarize(store):
    from collections import defaultdict
    import numpy as np
    groups=defaultdict(list)
    rows=[]
    for key,value in store.db.execute("SELECT key,value FROM artifacts WHERE key LIKE 'answer/development/%'"):
        record=json.loads(value);rows.append(record);groups[(record['domain'],record['model'])].append(record)
    summary=[]
    for (domain,model),records in sorted(groups.items()):
        responses=[store.receipt(r['call_id']) for r in records]
        summary.append({'domain':domain,'model':model,'answers':len(records),
            'quality':float(np.mean([r['score']['quality'] for r in records])),
            'truncated':sum(r['score']['outcome']=='truncated' for r in records),
            'audit_required':sum(r['score'].get('audit_required',False) for r in records),
            'mean_seconds':float(np.mean([r['elapsed_seconds'] for r in responses])),
            'p95_seconds':float(np.percentile([r['elapsed_seconds'] for r in responses],95)),
            'input_tokens':sum(r['input_tokens'] for r in responses),
            'output_tokens':sum(r['output_tokens'] for r in responses),
            'currency':responses[0]['currency'],'reference_cost':sum(r['reference_cost'] for r in responses)})
    result={'status':'development_complete_pending_analysis' if len(rows)==960 else 'partial',
            'answers':len(rows),'expected':960,'groups':summary,'scores_provisional_until_audit':True,
            'final_evaluation_opened':False,'search_rounds_started':0}
    atomic_json(store.directory/'summary.json',result)
    atomic_json(store.directory/'audit_queue.json',[r for r in rows if r['score'].get('audit_required')])
    return result


def run(config,run_dir,resume):
    frozen=freeze(config);store=Store(run_dir,frozen,resume=resume)
    stopping=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopping.set())
    keys=credentials(config['credentials_file'])
    start=time.monotonic()
    tasks={d:load_tasks(ROOT/f'data/multidomain_v072/{d}/development.jsonl') for d in config['domains']}
    pending_scores=threading.BoundedSemaphore(config['scoring_concurrency'])
    def work(stage,domain,task,model,sample):
        call_id=f'{stage}/{domain}/{task.id}/{model}/{sample}'
        artifact='answer/'+call_id
        old=store.artifact(artifact)
        if old is not None:return old
        if stopping.is_set():store.pause('requested_stop')
        if time.monotonic()-start>config['max_session_seconds']:store.pause('session_time_limit')
        profile=config['models'][model]
        request,bound=payload(config,profile,task,sample)
        response=store.begin(call_id,model,request,profile,stage,bound,config['max_tokens'])
        if response is None:
            try:
                response=call_model(profile,request,keys.get(profile['key_environment']))
            except Exception as exc:
                response={'status':'unknown','error_type':type(exc).__name__,'reference_cost':None,'currency':profile['currency']}
            store.finish(call_id,response)  # Durable before parsing, evaluation, or aggregate updates.
        if response.get('status')!='ok':raise Paused('provider response requires investigation')
        try:
            with pending_scores:score=run_score(task,response,config)
        except Exception:
            store.pause('scoring_error_receipts_retained');raise
        record={'call_id':call_id,'domain':domain,'task_id':task.id,'model':model,'sample':sample,'score':score}
        store.artifact(artifact,record,write=True)
        store.progress(stage)
        return record
    try:
        if store.unresolved():raise Paused('unknown calls require reconciliation before resume')
        for stage in ['preflight','development']:
            if store.artifact('stage_complete/'+stage):continue
            store.progress(stage)
            jobs=[]
            for domain,rows in tasks.items():
                for task in rows[:1] if stage=='preflight' else rows:
                    for sample in range(1 if stage=='preflight' else config['samples_per_task']):
                        for model in config['models']:jobs.append((stage,domain,task,model,sample))
            jobs.sort(key=lambda x:digest([config['seed'],stage,x[1],x[2].id,x[3],x[4]]))
            errors=[];finished=0
            # Dedicated pools prevent local-model queueing from occupying API worker slots.
            with ThreadPoolExecutor(config['api_concurrency']) as api,ThreadPoolExecutor(config['local_concurrency']) as local:
                futures={}
                for job in jobs:
                    pool=local if config['models'][job[3]]['kind']=='local' else api
                    futures[pool.submit(work,*job)]=job
                for future in as_completed(futures):
                    if future.cancelled():continue
                    try:future.result();finished+=1
                    except Exception as exc:
                        errors.append(type(exc).__name__)
                        store.pause('worker_error:'+type(exc).__name__)
                        for other in futures:other.cancel()
                    if finished and finished%20==0:store.backup()
                    if finished%20==0 or errors:
                        print(json.dumps({'stage':stage,'finished_this_session':finished,'expected':len(jobs),'pause':store.get_meta('pause')}),flush=True)
            store.backup()
            if errors or store.get_meta('pause'):raise Paused(str(store.get_meta('pause')))
            store.artifact('stage_complete/'+stage,{'jobs':len(jobs)},write=True)
        report=summarize(store)
        if report['answers']!=960:raise ValueError('development matrix incomplete')
        store.set_meta('completion',report['status']);store.progress('development_complete_pending_analysis')
        print(json.dumps(report,ensure_ascii=False),flush=True)
        return 0
    except Exception as exc:
        store.pause('execution_error:'+type(exc).__name__)
        summarize(store);store.progress('paused');store.backup()
        print(json.dumps({'status':'paused','reason':store.get_meta('pause'),'completed_work_preserved':True}),flush=True)
        return 2
    finally:store.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/multidomain_v072.json');parser.add_argument('--run-dir',default='runs/multidomain_v072');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text());raise SystemExit(run(config,ROOT/args.run_dir,args.resume))
