#!/usr/bin/env python3
"""Bounded, independent Qwen3.8 pilot. Paid receipts precede offline grading."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]


def setup(base):
    sys.path.insert(0, str(base))
    sys.path.insert(0, str(base / 'scripts'))
    import model_selection_v070 as core
    return core


def prepare(base, m):
    import pyarrow.parquet as pq
    from nicheflow.math_protocol import SUBJECTS, learning_tasks, question_hash, distribution
    directory = ROOT / 'data/model_selection_v071'
    m.require(not directory.exists(), 'data already frozen')
    prior = sorted(p for p in (base/'data').rglob('*.jsonl') if p.name in {
        'tasks.jsonl','development.jsonl','calibration.jsonl','evaluation.jsonl','validation.jsonl'})
    old = [t.question for p in prior for t in m.load_tasks(p) if t.dataset == 'math']
    held = m.load_tasks(base/'data/main_math500_v050/evaluation.jsonl')
    sources = json.loads((base/'data/source_math_v041/downloads.json').read_text())
    for entry in sources:
        m.require(m.file_hash(base/entry['path'])==entry['sha256'],'source changed')
    rows = {s:pq.read_table(base/f'data/source_math_v041/{s}.parquet').to_pylist() for s in SUBJECTS}
    parts, exclusions = learning_tasks(rows,held,old,seed=20260923071,
                                       development_per_stratum=3,calibration_per_stratum=1)
    tasks = [replace(t,role='evaluation') for t in parts['development'] if t.private['level'] in (4,5)]
    forbidden={question_hash(q) for q in old}
    m.require(len(tasks)==42 and len({question_hash(t.question) for t in tasks})==42,'42 unique tasks required')
    m.require(not forbidden & {question_hash(t.question) for t in tasks},'historical overlap')
    directory.mkdir(parents=True)
    path=directory/'validation.jsonl'
    path.write_text(''.join(json.dumps(t.record(),ensure_ascii=False)+'\n' for t in tasks))
    m.atomic_json(directory/'manifest.json',{
        'seed':20260923071,'count':42,'tasks_sha256':m.file_hash(path),
        'distribution':distribution(tasks),'excluded_unique_questions':len(forbidden),
        'excluded_prior_files':{str(p.relative_to(base)):m.file_hash(p) for p in prior},
        'source_files':sources,'selection_before_model_calls':True,
        'selection_rule':'3 per subject and level 4/5, independent of any model scores',
        'pilot_not_final_test':True,'exclusions':exclusions})
    print(json.dumps({'tasks':42,'distribution':distribution(tasks),'overlap':0}),flush=True)


def payload(m, config, profile, public, tokens, seed):
    result,bound=m.request_payload(config,profile,public,{'max_tokens':tokens},seed)
    if profile.get('reasoning_effort'):
        result['reasoning_effort']=profile['reasoning_effort']
        result.pop('enable_thinking',None)
    return result,bound


def readiness(m,config,keys):
    from nicheflow.datasets import Task
    task=Task('v071-api-readiness','math','train','evaluation',
              'Calculate 17 times 19. Give only the final boxed answer.','323','rational')
    rows=[];eligible=[];blocked=False
    for name,profile in config['models'].items():
        request,bound=payload(m,config,profile,task.model_input(),256,config['seed'])
        frozen={'config':config,'model':name,'payload':request,
                'script_sha256':m.file_hash(Path(__file__)),
                'purpose':'availability, mode and usage only; not quality evaluation'}
        directory=ROOT/'preflight'/name
        ledger=m.Ledger(directory,frozen,{'CNY':.2,'USD':.02},1,600)
        try:
            identifier='readiness:'+name
            ledger.begin(identifier,{'model':name,'payload':request},profile['currency'],
                         m.reference_charge(profile,bound,256))
            response=m.call_model(profile,request,keys.get(profile['key_environment']))
            score=m.score_result(task,response)
            row={'model':name,'task_id':task.id,'response':response,'score':score}
            ledger.finish(identifier,row)
            m.atomic_json(directory/'results.json',row)
            rows.append(row)
            if response['status']=='ok' and score['quality']==1:eligible.append(name)
            if response.get('quota_or_rate_limit') or response['status']=='unknown':blocked=True
            print(json.dumps({'model':name,'status':response['status'],'returned_model':response.get('returned_model'),
                              'http_status':response.get('http_status'),'error_code':response.get('error_code'),
                              'quality':score['quality'],'quota_or_rate_limit':response.get('quota_or_rate_limit',False)}),flush=True)
        finally:ledger.close()
        if blocked:break
    m.atomic_json(ROOT/'preflight/results.json',{'status':'blocked' if blocked else 'complete',
                  'rows':rows,'eligible_models':eligible,
                  'note':'Per-model refusal is excluded, never retried or treated as wrong math.'})


def run(base,m,stage,resume):
    config_path=ROOT/'configs/qwen38_v071.plan.json'
    config=json.loads(config_path.read_text())
    keys=m.credentials(config['credentials_file'])
    if stage=='preflight':
        return readiness(m,config,keys)
    else:
        ready=json.loads((ROOT/'preflight/results.json').read_text())
        scorer_check=json.loads((ROOT/'scorer_selftest.json').read_text())
        m.require(scorer_check['passed'],'scoring regression must pass before paid comparison')
        m.require(ready['status']=='complete','quota/unknown request blocks main pilot')
        eligible=ready['eligible_models']
        m.require(all(n in eligible for n in ('qwen37_flash','deepseek')),'baseline readiness required')
        m.require(any(n.startswith('qwen38') for n in eligible),'at least one Qwen3.8 model required')
        config={**config,'models':{n:p for n,p in config['models'].items() if n in eligible}}
        manifest=json.loads((ROOT/config['manifest']).read_text())
        m.require(m.file_hash(ROOT/config['data'])==manifest['tasks_sha256'],'data changed')
        for path,h in manifest['excluded_prior_files'].items():
            m.require(m.file_hash(base/path)==h,'historical data changed')
        tasks=m.load_tasks(ROOT/config['data'])
        m.require(len(tasks)==42,'incorrect pilot sample size')
        tokens=config['max_tokens'];caps=config['caps']
    frozen={'config':config,'stage':stage,'tasks_sha256':m.digest([t.record() for t in tasks]),
            'script_sha256':m.file_hash(Path(__file__)),
            'receipt_library_sha256':m.file_hash(Path(m.__file__)),
            'scorer_sha256':m.file_hash(ROOT/'scripts/score_qwen38_v071.py'),
            'scorer_selftest_sha256':m.file_hash(ROOT/'scorer_selftest.json'),
            'grading':'deferred offline; all completed answers use the same semantic audit; truncations zero',
            'max_tokens':tokens,'caps':caps}
    directory=ROOT/('preflight' if stage=='preflight' else 'runs/qwen38_v071')
    ledger=m.Ledger(directory,frozen,caps,len(tasks)*len(config['models']),config['max_seconds'],resume)
    jobs=[]
    for task in tasks:
        for name in sorted(config['models'],key=lambda x:m.digest([config['seed'],task.id,x])):
            jobs.append((name,task))
    m.atomic_json(directory/'plan.json',[{'model':name,'task_id':t.id} for name,t in jobs])
    gates={'api':threading.BoundedSemaphore(config['api_concurrency']),
           'local':threading.BoundedSemaphore(config['local_concurrency'])}
    def worker(name,task):
        with gates[config['models'][name]['kind']]:
            return invoke(name,task)
    def invoke(name,task):
        profile=config['models'][name]
        identifier=f'{stage}:{name}:{task.id}'
        request,bound=payload(m,config,profile,task.model_input(),tokens,config['seed'])
        old=ledger.begin(identifier,{'model':name,'task_id':task.id,'payload':request},
                         profile['currency'],m.reference_charge(profile,bound,tokens))
        if old is not None:return old
        response=m.call_model(profile,request,keys.get(profile['key_environment']))
        if stage=='preflight':
            score=m.score_result(task,response)
        else:
            outcome=({'stop':'complete','length':'truncated'}.get(response.get('finish_reason'),'unexpected_finish')
                     if response['status']=='ok' else response['status'])
            score={'quality':None,'outcome':outcome,'assessment':'deferred_offline_semantic_audit'}
        result={'model':name,'task_id':task.id,'response':response,'score':score}
        ledger.finish(identifier,result)
        return result
    try:
        remaining=iter(jobs);pending=set();error=None
        with ThreadPoolExecutor(max_workers=config['concurrency']) as pool:
            while True:
                while len(pending)<config['concurrency'] and not ledger.stopped:
                    job=next(remaining,None)
                    if job is None:break
                    pending.add(pool.submit(worker,*job))
                if not pending:break
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    try: future.result()
                    except Exception as exc:
                        ledger.stopped=True;error=type(exc).__name__+': '+str(exc)
                progress={'completed':len(ledger.results),'total':len(jobs),'spent':ledger.spent,
                          'groups':dict(Counter(r['model'] for r in ledger.results.values())),
                          'elapsed_seconds':ledger.elapsed(),'stopped':ledger.stopped,'error':error}
                m.atomic_json(directory/'progress.json',progress)
                print(json.dumps(progress),flush=True)
        result={'status':'complete' if len(ledger.results)==len(jobs) and not ledger.stopped else 'stopped',
                'executions':len(ledger.results),'expected':len(jobs),'spend':ledger.spent,
                'elapsed_seconds':ledger.elapsed(),'rows':list(ledger.results.values()),'error':error}
        m.atomic_json(directory/'results.json',result)
        m.atomic_json(directory/'audit.json',{'events_sha256':m.file_hash(directory/'events.jsonl'),
                    'calls_started':len(ledger.starts),'calls_finished':len(ledger.results),
                    'unknown_calls':sorted(set(ledger.starts)-set(ledger.results)),
                    'learning_updates':0,'status':result['status']})
        m.require(result['status']=='complete','pilot stopped; inspect receipts, no automatic retry')
    finally:ledger.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','preflight','run'],required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();m=setup(args.base)
    if args.stage=='prepare':prepare(args.base,m)
    else:run(args.base,m,args.stage,args.resume)
