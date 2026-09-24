#!/usr/bin/env python3
"""Bounded pre-scale repair and optional paired search, with durable v072 receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import json
from pathlib import Path
import signal
import sqlite3
import sys
import threading
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nicheflow.ledger import atomic_json
from nicheflow.spec import digest,file_hash
from nicheflow.v072.durable import Store,Paused
from scripts.run_v072_continuation import Experiment as Previous,validate,seeds,rank,node
from scripts.model_selection_v070 import credentials
from scripts.v073_support import payload,score,apply_edit,VERSION

RUNS=ROOT/'runs/prescale_v073'
SOURCE_PARENT=ROOT/'runs/multidomain_v072/checkpoint.sqlite3'
SOURCE_CONTINUATION=ROOT/'runs/multidomain_v072_continuation/checkpoint.sqlite3'
SEARCH_SEEDS=[2026092401,2026092402]
POLICY={'version':'v073-prescale-1','repair_calls':140,'repair_answers':120,
    'search_total_upper':870,'shared_search_calls':150,'per_search_run_calls':180,
    'search_random_seeds':SEARCH_SEEDS,'search_slots':5,'proposal_attempts':2,
    'search_feedback_tasks':10,'search_screening_tasks':20,
    'mixed_schedule':['elite','restart','elite','nonelite','elite'],
    'search_note':'shared 90-call initialization plus 60-call single-model screening controls; four equal 180-attempt runs, within earlier 880-call cap; budget may prevent filling all slots',
    'decision':'engineering sample checks only; no independent final-set claim; no new final tasks opened'}


def read(db,key):
    row=db.execute('SELECT value FROM artifacts WHERE key=?',(key,)).fetchone()
    if row is None:raise ValueError('missing source artifact: '+key)
    return json.loads(row[0])


def seed_record(db,key):
    r=read(db,key);calls=[r['call_id']] if 'call_id' in r else r['calls']
    receipts=[json.loads(db.execute("SELECT response FROM attempts WHERE call_id=? AND status='complete'",(c,)).fetchone()[0]) for c in calls]
    return r,receipts


class CappedStore(Store):
    def begin(self,identifier,*args,**kwargs):
        with self.mutex:
            if self.receipt(identifier) is None and self.db.execute('SELECT count(*) FROM attempts').fetchone()[0]>=self.frozen['total_attempt_cap']:
                self.pause('total_attempt_cap');raise Paused('total_attempt_cap')
            return super().begin(identifier,*args,**kwargs)


class Experiment(Previous):
    def __init__(self,mode,strategy,seed,resume=False):
        self.mode,self.strategy,self.seed=mode,strategy,seed
        self.config=json.loads((ROOT/'configs/multidomain_v072.json').read_text())
        self.parent=sqlite3.connect(f'file:{SOURCE_PARENT}?mode=ro',uri=True,check_same_thread=False)
        self.prior=sqlite3.connect(f'file:{SOURCE_CONTINUATION}?mode=ro',uri=True,check_same_thread=False)
        assert read(self.prior,'complete')['status']=='all_planned_generations_complete'
        self.manifest=json.loads((ROOT/self.config['manifest']).read_text());self.parent_lock=threading.RLock()
        cap=140 if mode=='repair' else 150 if mode=='search-seeds' else 180
        models={'L':100,'M':40,'H':0,'D':0} if mode=='repair' else {m:cap for m in self.config['models']}
        limits={'models':{m:{'calls':n,'input_tokens':12000000,'output_tokens':2000000} for m,n in models.items()},
                'currency':{'CNY':1. if mode=='repair' else 3. if mode=='search-seeds' else 5.,'USD':.25 if mode=='search' else 0.}}
        name=mode if mode!='search' else f'search_{strategy}_{seed}'
        paths=[Path(__file__),ROOT/'scripts/v073_support.py',ROOT/'scripts/v073_offline.py',
            *[ROOT/'scripts'/n for n in ['run_v072.py','run_v072_continuation.py','model_selection_v070.py','score_v072_response.py','v072_code_worker.py','v072_code_sandbox.py']],
            *sorted((ROOT/'nicheflow').rglob('*.py')),*sorted((ROOT/'nicheflow_probe').rglob('*.py'))]
        identity=json.loads(Path(self.config['local_identity']).read_text())
        assert identity['revision']==self.config['local_revision']
        for p,h in identity['configuration_sha256'].items():assert file_hash(Path(identity['snapshot_path'])/p)==h
        migration=json.loads((ROOT/'setup/v072/scoring_memory_migration_20260924.json').read_text())
        assert file_hash(Path(self.config['code_sandbox_root'])/'sandbox-manifest.json')==migration['sandbox_manifest']['new_sha256']
        assert file_hash(Path(self.config['code_sandbox_root'])/'worker.py')==migration['source_changes']['scripts/v072_code_worker.py']['new_sha256']
        frozen={'policy':POLICY,'mode':mode,'strategy':strategy,'seed':seed,'config':self.config,'limits':limits,'total_attempt_cap':cap,
            'source_dbs':{str(p.relative_to(ROOT)):file_hash(p) for p in [SOURCE_PARENT,SOURCE_CONTINUATION]},
            'code':{str(p.relative_to(ROOT)):file_hash(p) for p in paths},
            'task_files':self.manifest['task_files'],'identity':identity,
            'lock':file_hash(ROOT/'requirements.lock'),'migration':file_hash(ROOT/'setup/v072/scoring_memory_migration_20260924.json'),
            'sandbox':file_hash(Path(self.config['code_sandbox_root'])/'sandbox-manifest.json')}
        if mode!='repair':
            frozen['repair_summary_sha256']=file_hash(RUNS/'repair/summary.json')
            frozen['repair_review_sha256']=file_hash(RUNS/'repair/review.approved.json')
        if mode=='search':frozen['shared_seed_checkpoint_sha256']=file_hash(RUNS/'search-seeds/checkpoint.sqlite3')
        for p,h in self.manifest['task_files'].items():assert file_hash(ROOT/p)==h
        self.cap=cap;self.store=CappedStore(RUNS/name,frozen,resume=resume);self.load_audits()
        self.keys=credentials(self.config['credentials_file']);self.start=time.monotonic();self.completed=0
        self.local=threading.BoundedSemaphore(1);self.api=threading.BoundedSemaphore(4);self.scorers=threading.BoundedSemaphore(2)
        self.stop=threading.Event()
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:self.stop.set())

    def execute(self,stage,task,candidate,graph,sample,stopping=False):
        key=f'answer/{stage}/{task.dataset}/{task.id}/{candidate}/{sample}'
        old=self.store.artifact(key)
        if old is not None:return self.effective(key,old)
        outputs={};receipts=[];calls=[];result=None
        for n in graph['nodes']:
            request,bound=payload(self.config,task,n,outputs,sample,n['id']==graph['output'],stopping)
            cid=key[len('answer/'):]+'/'+n['id'];r=self.response(cid,n['model'],request,bound,stage)
            receipts.append(r);calls.append(cid);outputs[n['id']]=r['text']
            if r['finish_reason']=='length':
                result={'quality':0.,'outcome':'truncated' if n['id']==graph['output'] else 'upstream_truncated','audit_required':False};break
            if r['finish_reason']!='stop':raise ValueError('unexpected finish')
        if result is None:
            with self.scorers:result=score(task,receipts[-1],self.config)
        record={'domain':task.dataset,'task_id':task.id,'candidate':candidate,'sample':sample,'score':result,'calls':calls,
            'currency':'CNY','cost_cny':sum(r['reference_cost'] for r in receipts),'reference_cost':sum(r['reference_cost'] for r in receipts),
            'input_tokens':sum(r['input_tokens'] for r in receipts),'output_tokens':sum(r['output_tokens'] for r in receipts),
            'elapsed_seconds':sum(r['elapsed_seconds'] for r in receipts),'scoring_version':VERSION}
        self.store.artifact(key,record,write=True)
        with self.store.mutex:
            self.completed+=1;self.store.progress(stage)
            if self.completed%10==0:
                self.store.backup();print(json.dumps({'mode':self.mode,'stage':stage,'session_answers':self.completed}),flush=True)
        return self.effective(key,record)

    def batch(self,stage,tasks,graphs,samples=1,stopping=False):
        jobs=sorted([(t,n,g,s) for t in tasks for n,g in graphs.items() for s in range(samples)],key=lambda x:digest([stage,x[0].id,x[1],x[3]]))
        records=[];errors=[]
        with ThreadPoolExecutor(6) as pool:
            futures=[pool.submit(self.execute,stage,*job,stopping) for job in jobs]
            for f in as_completed(futures):
                if f.cancelled():continue
                try:records.append(f.result())
                except Exception as exc:
                    errors.append(exc);self.store.pause(type(exc).__name__+':'+str(exc)[:150])
                    for other in futures:other.cancel()
        self.store.backup()
        if errors:raise errors[0]
        return sorted(records,key=lambda r:(r['task_id'],r['candidate'],r['sample']))

    def feedback(self,domain,n):
        ids=set(self.manifest['candidate_feedback_ids'][domain])
        tasks=[t for t in self.tasks(domain,'development') if t.id in ids]
        return sorted(tasks,key=lambda t:digest(['v073-feedback',t.id]))[:n]

    def repair(self):
        graph=read(self.prior,'deployment/hotpotqa')['graphs'][read(self.prior,'deployment/hotpotqa')['searched']]
        self.batch('repair',self.tasks('math','development'),{'L_stop':seeds()['L']},2,True)
        self.batch('repair',self.feedback('hotpotqa',20),{'M_contract':seeds()['M'],'search_contract':graph})
        self.gate('repair')
        groups={};comparisons=[]
        for key,raw in self.store.db.execute("SELECT key,value FROM artifacts WHERE key LIKE 'answer/repair/%' ORDER BY key"):
            new=self.effective(key,json.loads(raw));t=next(t for t in self.tasks(new['domain'],'development') if t.id==new['task_id'])
            with self.parent_lock:
                if new['candidate']=='search_contract':
                    name=read(self.prior,'deployment/hotpotqa')['searched'];old,receipts=seed_record(self.prior,f'answer/search/hotpotqa/{t.id}/{name}/0')
                else:
                    model='L' if new['domain']=='math' else 'M';old,receipts=seed_record(self.parent,f"answer/development/{t.dataset}/{t.id}/{model}/{new['sample']}")
            old_score=old['score']
            if t.dataset=='hotpotqa' and 'truncated' not in old_score['outcome']:old_score=score(t,receipts[-1],self.config)
            old_compatible={'score':old_score,'output_tokens':sum(r['output_tokens'] for r in receipts),'elapsed_seconds':sum(r['elapsed_seconds'] for r in receipts),'cost_cny':sum(r['reference_cost'] for r in receipts)}
            comparisons.append({'task_id':t.id,'sample':new['sample'],'candidate':new['candidate'],'old':old_compatible,'new':new})
        for name in ['L_stop','M_contract','search_contract']:
            rr=[r for r in comparisons if r['candidate']==name];groups[name]={}
            for side in ['old','new']:
                rs=[r[side] for r in rr];groups[name][side]={'answers':len(rs),'quality':float(np.mean([r['score']['quality'] for r in rs])),
                    'truncated':sum('truncated' in r['score']['outcome'] for r in rs),'malformed':sum(r['score']['outcome']=='malformed' for r in rs),
                    'mean_output_tokens':float(np.mean([r['output_tokens'] for r in rs])),'mean_seconds':float(np.mean([r['elapsed_seconds'] for r in rs]))}
        old,new=groups['L_stop']['old'],groups['L_stop']['new']
        math_accept=new['truncated']<old['truncated'] and new['quality']>=old['quality']-1e-12 and new['mean_output_tokens']<=old['mean_output_tokens']
        qa=['M_contract','search_contract'];before=sum(groups[n]['old']['malformed'] for n in qa);after=sum(groups[n]['new']['malformed'] for n in qa)
        qa_accept=after<before and all(groups[n]['new']['malformed']<=groups[n]['old']['malformed'] for n in qa) and sum(groups[n]['new']['quality'] for n in qa)>=sum(groups[n]['old']['quality'] for n in qa)-1e-12
        summary={'groups':groups,'math_stopping_accepted':bool(math_accept),'qa_contract_numeric_acceptance':bool(qa_accept),
            'review_required':'Review format regressions and semantic failures before approving the bounded search; numerical rules are engineering gates, not significance tests.',
            'comparison_rows':comparisons,'new_calls_limit':140,'new_answers':len(comparisons)}
        assert len(comparisons)==120
        self.store.artifact('repair_summary',summary,write=True);atomic_json(self.store.directory/'summary.json',summary)
        self.store.artifact('complete',{'status':'repair_complete_pending_review'},write=True)

    def require_repair_review(self):
        directory=RUNS/'repair';summary=json.loads((directory/'summary.json').read_text());approval=json.loads((directory/'review.approved.json').read_text())
        assert approval['summary_sha256']==digest(summary) and approval['qa_regressions_reviewed'] and approval['reason']
        assert summary['qa_contract_numeric_acceptance'] and approval['start_bounded_search'] is True

    def initialize_search(self):
        self.require_repair_review();tasks=self.feedback('mbpp',10)
        rows=self.batch('seed',tasks,seeds())
        # These three controls use the same v073 prompt and the same 20 screening tasks as winners.
        self.batch('screen_controls',self.tasks('mbpp','screening'),{n:seeds()[n] for n in ['L','M','H']})
        self.store.artifact('seed_statistics',[self.stats(n,g,rows) for n,g in seeds().items()],write=True)
        self.store.artifact('complete',{'status':'shared_search_initialization_complete'},write=True)

    def choose_parent(self,graphs,stats,slot):
        niches={}
        for s in stats:niches.setdefault(s['niche'],[]).append(s)
        elite=[rank(v,0.)[0] for _,v in sorted(niches.items())];elite_ids={s['id'] for s in elite}
        mode=POLICY['mixed_schedule'][slot] if self.strategy=='mixed' else 'elite'
        if mode=='restart':return None,'restart',False
        pool=[s for s in stats if s['id'] not in elite_ids] if mode=='nonelite' else elite
        fallback=not pool
        if fallback:pool=elite
        pool=sorted(pool,key=lambda s:s['id']);i=int(digest([self.seed,slot,mode])[:12],16)%len(pool)
        return pool[i]['id'],mode,fallback

    def proposal(self,graphs,stats,slot):
        key=f'proposal/{slot}';old=self.store.artifact(key)
        if old:return old
        parent,branch,fallback=self.choose_parent(graphs,stats,slot)
        selected_type=['model_swap','prompt_edit','add','delete_merge','rewire'][slot]
        if parent is not None:
            if selected_type=='delete_merge' and len(graphs[parent]['nodes'])==1:selected_type='prompt_edit'
            if selected_type=='add' and len(graphs[parent]['nodes'])==3:selected_type='prompt_edit'
            if selected_type=='rewire' and len(graphs[parent]['nodes'])<3:selected_type='prompt_edit'
        error=None
        for attempt in range(2):
            request={'model':self.config['models']['D']['model'],'max_tokens':8192,'temperature':.7,'thinking':{'type':'disabled'},
                'messages':[{'role':'system','content':'Design a task-independent MBPP workflow. Return one JSON object only. Never include benchmark answers or tests.'},
                 {'role':'user','content':json.dumps({'branch':branch,'parent':graphs[parent] if parent else None,'statistics':stats,
                    'requested_operation':selected_type if parent else 'restart','previous_error':error,
                    'rules':['All graphs: 1-3 nodes, L/M/H, at most one H; node fields id/model/role/prompt/inputs; roles plan/solve/review; IDs topologically ordered; final node is output and not plan; no dead nodes; prompts <=2400 chars.',
                        'Restart: generate a fresh JSON graph with nodes and output without a parent. Choose any permitted composition. No task instance is provided.',
                        'Otherwise return one operation with type exactly requested_operation. Do not return a rewritten graph.',
                        'model_swap: {type,node:existing ID,model:L/M/H}; prompt_edit: {type,node:existing ID,prompt:new text}; delete_merge: {type,node:existing ID}.',
                        'rewire: {type,node:existing ID,inputs:list of earlier IDs}; add: {type,node:full new node object with a fresh ID,consumer:existing ID}; the new node is inserted immediately before consumer and added to its inputs.',
                        'Seek useful quality-cost tradeoffs or complementary behavior. Source and final output contracts are fixed by the runtime.'],
                    'example_graph':seeds()['M']},ensure_ascii=False)}]}
            bound=sum(len(m['content'].encode())+64 for m in request['messages'])+1024
            response=self.response(f'proposal/{slot}/{attempt}','D',request,bound,'proposal')
            try:
                if response['finish_reason']!='stop':raise ValueError('truncated proposal')
                obj=json.loads(response['text']);change=None
                if parent is None:graph=validate(obj);change={'operation':{'type':'restart'},'before':None,'after':graph}
                else:
                    if obj['type']!=selected_type:raise ValueError('wrong operation type')
                    graph,change=apply_edit(graphs[parent],obj)
                if digest(graph) in {digest(g) for g in graphs.values()}:raise ValueError('duplicate graph')
                result={'valid':True,'graph':graph,'change':change,'parent':parent,'branch':branch,'fallback_to_elite':fallback,'id':f'candidate_{slot}','attempts':attempt+1}
                break
            except (ValueError,TypeError,KeyError,IndexError) as exc:
                error=str(exc)[:180];self.store.artifact(key+f'/invalid/{attempt}',{'error':error},write=True)
        else:result={'valid':False,'parent':parent,'branch':branch,'fallback_to_elite':fallback,'error':error,'attempts':2}
        self.store.artifact(key,result,write=True);return result

    def search(self):
        self.require_repair_review();source=sqlite3.connect(f'file:{RUNS}/search-seeds/checkpoint.sqlite3?mode=ro',uri=True)
        read(source,'complete');stats=read(source,'seed_statistics');source.close();graphs=seeds();tasks=self.feedback('mbpp',10)
        for slot in range(5):
            existing=self.store.artifact(f'slot/{slot}')
            if existing:
                if existing.get('evaluated'):graphs[existing['id']]=existing['graph'];stats.append(existing['statistics'])
                continue
            with self.store.mutex:
                used=self.store.db.execute('SELECT count(*) FROM attempts').fetchone()[0]
                begun=self.store.db.execute('SELECT 1 FROM calls WHERE id LIKE ? LIMIT 1',(f'proposal/{slot}/%',)).fetchone()
            if not begun and self.cap-used<72:
                self.store.artifact(f'slot/{slot}',{'evaluated':False,'reason':'reserve 60 screening calls plus smallest next candidate/proposals'},write=True);continue
            proposal=self.proposal(graphs,stats,slot)
            if not proposal['valid']:
                self.store.artifact(f'slot/{slot}',{'evaluated':False,'reason':'invalid_proposal'},write=True);continue
            name,graph=proposal['id'],proposal['graph']
            with self.store.mutex:used=self.store.db.execute('SELECT count(*) FROM attempts').fetchone()[0]
            # Existing candidate receipts are reused on resume and do not need reservation again.
            existing_calls=self.store.db.execute("SELECT count(*) FROM calls WHERE stage='search' AND id LIKE ?",('%/'+name+'/%',)).fetchone()[0]
            if self.cap-used<10*len(graph['nodes'])-existing_calls+60:
                self.store.artifact(f'slot/{slot}',{'evaluated':False,'reason':'candidate exceeds remaining budget with screening reserve'},write=True);continue
            rows=self.batch('search',tasks,{name:graph});stat=self.stats(name,graph,rows);graphs[name]=graph;stats.append(stat)
            self.store.artifact(f'slot/{slot}',{'evaluated':True,'id':name,'graph':graph,'statistics':stat},write=True)
        candidates=[s for s in stats if s['id'].startswith('candidate_')]
        if not candidates:
            self.store.artifact('complete',{'status':'no_valid_evaluated_search_candidate'},write=True);return
        winner=rank(candidates)[0]['id'];self.store.artifact('selected',{'candidate':winner,'graph':graphs[winner]},write=True)
        rows=self.batch('screening',self.tasks('mbpp','screening'),{winner:graphs[winner]})
        self.store.artifact('summary',{'strategy':self.strategy,'seed':self.seed,'candidate_count':len(candidates),
            'winner':winner,'screening':self.stats(winner,graphs[winner],rows),'statistics':stats},write=True)
        self.store.artifact('complete',{'status':'bounded_search_complete'},write=True)

    def run(self):
        try:
            if self.store.unresolved():raise Paused('unknown outcomes require reconciliation')
            if not self.store.artifact('complete'):
                if self.mode=='repair':self.repair()
                elif self.mode=='search-seeds':self.initialize_search()
                else:self.search()
            self.store.progress('complete');self.store.backup();return 0
        except Exception as exc:
            self.store.pause(type(exc).__name__+':'+str(exc)[:180]);self.store.progress('paused');self.store.backup()
            print(json.dumps({'status':'paused','reason':str(exc),'receipts_preserved':True}),flush=True);return 2
        finally:self.store.close();self.parent.close();self.prior.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['repair','search-seeds','search'],default='repair');p.add_argument('--strategy',choices=['elite','mixed'],default='elite');p.add_argument('--seed',type=int,choices=SEARCH_SEEDS,default=SEARCH_SEEDS[0]);p.add_argument('--resume',action='store_true')
    a=p.parse_args();raise SystemExit(Experiment(a.mode,a.strategy,a.seed,a.resume).run())
