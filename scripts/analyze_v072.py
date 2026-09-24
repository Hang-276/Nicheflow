#!/usr/bin/env python3
"""Offline v072 summaries and paired, task-clustered uncertainty. No API calls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nicheflow.datasets import load_tasks
from nicheflow.ledger import atomic_json
from nicheflow.spec import digest,file_hash
from scripts.run_v072_continuation import features


def connect(path):
    db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    return db


def artifacts(db,prefix):
    result=[]
    for key,v in db.execute('SELECT key,value FROM artifacts WHERE key LIKE ?',(prefix+'%',)).fetchall():
        r=json.loads(v)
        if key.startswith('answer/'):
            decision=get(db,'adjudication/'+key)
            if decision:
                r['score']['raw_quality']=r['score']['quality']
                r['score']['quality']=decision['quality'];r['score']['audit_required']=False
        result.append(r)
    return result


def get(db,key):
    row=db.execute('SELECT value FROM artifacts WHERE key=?',(key,)).fetchone()
    return json.loads(row[0]) if row else None


def paired(a,b,seed=20260923):
    def values(rows):
        grouped=defaultdict(list)
        for r in rows:grouped[r['task_id']].append(r['score']['quality'])
        return {k:float(np.mean(v)) for k,v in grouped.items()}
    av,bv=values(a),values(b);assert av.keys()==bv.keys()
    ids=sorted(av);delta=np.array([av[k]-bv[k] for k in ids]);rng=np.random.default_rng(seed)
    boots=delta[rng.integers(len(ids),size=(20000,len(ids)))].mean(1)
    return {'mean_difference':float(delta.mean()),'paired_task_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),
            'independent_tasks':len(ids),'first_better_tasks':int(sum(delta>1e-12)),
            'equal_tasks':int(sum(abs(delta)<=1e-12)),'second_better_tasks':int(sum(delta<-1e-12)),
            'interpretation':'exploratory, no multiplicity adjustment; two answers remain clustered per task'}


def summary(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['domain'],r['candidate']].append(r)
    groups=[]
    for (d,name),rs in sorted(grouped.items()):
        q=[r['score']['quality'] for r in rs];lat=[r['elapsed_seconds'] for r in rs]
        metrics=sorted(set(k for r in rs for k in r['score'].get('metrics',{}) if k in {'em','f1','sp_em','sp_f1','joint_em','joint_f1','base_passed','plus_passed','source_hidden_passed'}))
        groups.append({'domain':d,'candidate':name,'answers':len(rs),'tasks':len(set(r['task_id'] for r in rs)),
            'quality':float(np.mean(q)),'truncated':sum('truncated' in r['score']['outcome'] for r in rs),
            'missing_or_malformed':sum(r['score']['outcome'] in ('missing_final','malformed') for r in rs),
            'mean_seconds':float(np.mean(lat)),'median_seconds':float(np.median(lat)),
            'p95_seconds':float(np.percentile(lat,95)),'max_seconds':float(max(lat)),
            'currency':rs[0]['currency'],'reference_cost':sum(r['reference_cost'] for r in rs),
            'mean_reference_cost':float(np.mean([r['reference_cost'] for r in rs])),
            'metrics':{m:float(np.mean([r['score'].get('metrics',{}).get(m,0.) for r in rs])) for m in metrics}})
    contrasts=[]
    for d in ['math','mbpp','hotpotqa']:
        arms={name:rs for (domain,name),rs in grouped.items() if domain==d}
        if not arms:continue
        comparisons=[('H','M'),('M','L'),('D','M')]
        comparisons += [(name,'M') for name in arms if name.startswith(('fixed_','search_'))]
        fixed=[n for n in arms if n.startswith('fixed_')];searched=[n for n in arms if n.startswith('search_')]
        if len(fixed)==len(searched)==1:comparisons.append((searched[0],fixed[0]))
        for a,b in comparisons:
            if a in arms and b in arms:contrasts.append({'domain':d,'first':a,'second':b,**paired(arms[a],arms[b])})
    return {'groups':groups,'paired_contrasts':contrasts}


def baseline_rows(db):
    out=[]
    for r in artifacts(db,'answer/development/'):
        a=json.loads(db.execute("SELECT response FROM attempts WHERE call_id=? AND status='complete'",(r['call_id'],)).fetchone()[0])
        out.append({**r,'candidate':r['model'],'currency':a['currency'],'reference_cost':a['reference_cost'],
                    'cost_cny':a['reference_cost'] if a['currency']=='CNY' else None,'elapsed_seconds':a['elapsed_seconds']})
    return out


def accounting(db):
    groups=defaultdict(lambda:{'attempts':0,'complete':0,'input_tokens':0,'output_tokens':0,'reference_cost':0.,'unknown':0,'rejected':0})
    for stage,model,state,raw in db.execute('SELECT c.stage,c.model,a.status,a.response FROM attempts a JOIN calls c ON c.id=a.call_id'):
        response=json.loads(raw) if raw else {};currency=response.get('currency','unknown')
        s=groups[stage,model,currency];s['attempts']+=1
        if state=='complete':
            s['complete']+=1;s['input_tokens']+=response['input_tokens'];s['output_tokens']+=response['output_tokens'];s['reference_cost']+=response['reference_cost']
        elif state=='rejected':s['rejected']+=1
        else:s['unknown']+=1
    return [{'stage':stage,'model':model,'currency':currency,**s} for (stage,model,currency),s in sorted(groups.items())]


def replay(db,final_rows):
    frozen=get(db,'final_protocol_frozen');weights=get(db,'frozen_routers')
    assert frozen and digest(weights)==frozen['router_sha256']
    results=[]
    for d in ['math','mbpp','hotpotqa']:
        deployment=frozen['deployments'][d];w=weights[d]
        tasks=load_tasks(ROOT/f'data/multidomain_v072/{d}/final.jsonl')
        idx={(r['task_id'],r['candidate'],r['sample']):r for r in final_rows if r['domain']==d}
        pools={'L_H':['L','H'],'M_H':['M','H'],'L_M_H':['L','M','H'],
               'with_fixed':['L','M','H',deployment['fixed']],
               'with_searched':['L','M','H',deployment['searched']],
               'five_arms':['L','M','H',deployment['fixed'],deployment['searched']]}
        for name,arms in pools.items():
            choices=[]
            for t in tasks:
                x=np.asarray(features(t));pred={a:float(np.clip(x@w[a]['quality'],0,1)) for a in arms}
                costs={a:max(0.,float(x@w[a]['cost'])) for a in arms};best=max(pred.values())
                choices.append(min([a for a in arms if pred[a]>=best-.01-1e-12],key=lambda a:(costs[a],a)))
            meanbest=max(w[a]['mean_quality'] for a in arms)
            constant=min([a for a in arms if w[a]['mean_quality']>=meanbest-.01-1e-12],key=lambda a:(w[a]['mean_cost'],a))
            random_choices=np.asarray(choices)[np.random.default_rng(int(digest([d,name,'matched_random'])[:12],16)).permutation(len(choices))].tolist()
            for policy,selected in [('ridge',choices),('constant',[constant]*len(tasks)),('matched_use_random',random_choices)]:
                chosen=[idx[t.id,a,s] for t,a in zip(tasks,selected) for s in range(2)]
                flash=[idx[t.id,'M',s] for t in tasks for s in range(2)]
                results.append({'domain':d,'pool':name,'policy':policy,
                    'quality':float(np.mean([r['score']['quality'] for r in chosen])),
                    'mean_cost_cny':float(np.mean([r['cost_cny'] for r in chosen])),
                    'mean_serial_call_seconds':float(np.mean([r['elapsed_seconds'] for r in chosen])),
                    'choices_by_task':{a:selected.count(a) for a in arms},'difference_from_flash':paired(chosen,flash),
                    'mode':'stored-response offline replay, not live conditional execution'})
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--continuation');p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    parent=connect(args.parent);result={'development':summary(baseline_rows(parent)),'accounting':accounting(parent),
        'source_parent_sha256':file_hash(args.parent),'final_complete':False}
    if args.continuation:
        db=connect(args.continuation);result['source_continuation_sha256']=file_hash(args.continuation)
        for phase in ['seeds','search','screening','calibration','final']:
            rows=artifacts(db,'answer/'+phase+'/')
            result[phase]=summary(rows) if rows else None
        complete=get(db,'complete');result['final_complete']=bool(complete)
        result['rounds_completed']=len(artifacts(db,'round/'))
        result['deployment']={d:get(db,'deployment/'+d) for d in ['math','mbpp','hotpotqa']}
        result['proposal_slots']=len([k for k, in db.execute("SELECT key FROM artifacts WHERE key LIKE 'proposal/%'") if k.count('/')==3])
        result['accounting']+=accounting(db)
        result['proposals']=[{'key':key,**json.loads(value)} for key,value in db.execute("SELECT key,value FROM artifacts WHERE key LIKE 'proposal/%'") if key.count('/')==3]
        result['search_statistics']=get(db,'search_summary')
        result['adjudication_count']=len(artifacts(db,'adjudication/'))
        if complete:result['routing_replay']=replay(db,artifacts(db,'answer/final/'))
        db.close()
    parent.close();atomic_json(out/'analysis.json',result)
    print(json.dumps({'final_complete':result['final_complete'],'output':str(out/'analysis.json')}))


if __name__=='__main__':main()
