#!/usr/bin/env python3
"""One fixed grouped-CV router diagnostic and stored-output parser replay. No API."""
import argparse
import json
from pathlib import Path
import sys
import sqlite3
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nicheflow.datasets import load_tasks
from nicheflow.spec import digest,file_hash
from nicheflow.ledger import atomic_json
from scripts.analyze_v072 import artifacts,get,paired
from scripts.run_v072_continuation import features
from scripts.v073_support import score


def fit_predict(x,y,c,train,test,standardized):
    a=x[train,1:];b=x[test,1:]
    if standardized:
        center=a.mean(0);scale=a.std(0);scale[scale<1e-10]=1.
        a=(a-center)/scale;b=(b-center)/scale
    a=np.column_stack([np.ones(len(a)),a]);b=np.column_stack([np.ones(len(b)),b])
    penalty=np.diag([0.,10.,10.,10.])
    return np.clip(b@np.linalg.solve(a.T@a+penalty,a.T@y[train]),0,1),np.maximum(0,b@np.linalg.solve(a.T@a+penalty,a.T@c[train]))


def choose(q,c):
    return [min([j for j in range(q.shape[1]) if q[i,j]>=q[i].max()-.01-1e-12],key=lambda j:(c[i,j],j)) for i in range(len(q))]


def run(parent,continuation,out):
    db=sqlite3.connect(f'file:{continuation}?mode=ro',uri=True);pd=sqlite3.connect(f'file:{parent}?mode=ro',uri=True)
    result={'version':'v073-offline-1','selection':'fixed 5 task-grouped folds; ridge=10 unchanged; only train-fold feature standardization differs; no final data',
            'source_parent_sha256':file_hash(parent),'source_continuation_sha256':file_hash(continuation),'router':[]}
    rows=artifacts(db,'answer/calibration/')
    for d in ['math','mbpp','hotpotqa']:
        tasks=sorted(load_tasks(ROOT/f'data/multidomain_v072/{d}/calibration.jsonl'),key=lambda t:digest(['v073-cv',t.id]))
        deployment=get(db,'deployment/'+d);arms=['L','M','H',deployment['fixed'],deployment['searched']]
        idx={(r['task_id'],r['candidate'],r['sample']):r for r in rows if r['domain']==d}
        x=np.asarray([features(t) for t in tasks]);q=np.asarray([[np.mean([idx[t.id,a,s]['score']['quality'] for s in range(2)]) for a in arms] for t in tasks])
        c=np.asarray([[np.mean([idx[t.id,a,s]['cost_cny'] for s in range(2)]) for a in arms] for t in tasks])
        selections={k:[None]*40 for k in ['raw_ridge10','standardized_ridge10','constant','matched_random','flash']}
        for f in range(5):
            train=np.array([i for i in range(40) if i%5!=f]);test=np.array([i for i in range(40) if i%5==f])
            qp,cp=fit_predict(x,q,c,train,test,False);raw=choose(qp,cp)
            qp,cp=fit_predict(x,q,c,train,test,True);new=choose(qp,cp)
            const=choose(q[train].mean(0,keepdims=True),c[train].mean(0,keepdims=True))[0]
            random=np.random.default_rng(20260924073+f).permutation(new).tolist()
            for j,i in enumerate(test):
                for name,choice in [('raw_ridge10',raw[j]),('standardized_ridge10',new[j]),('constant',const),('matched_random',random[j]),('flash',1)]:selections[name][i]=choice
        selected={name:[idx[t.id,arms[j],s] for t,j in zip(tasks,choices) for s in range(2)] for name,choices in selections.items()}
        for name,rs in selected.items():
            result['router'].append({'domain':d,'policy':name,'tasks':40,'quality':float(np.mean([r['score']['quality'] for r in rs])),
                'mean_cost_cny':float(np.mean([r['cost_cny'] for r in rs])), 'choices':{a:selections[name].count(i) for i,a in enumerate(arms)},
                'difference_from_constant':paired(rs,selected['constant']),'difference_from_flash':paired(rs,selected['flash'])})
    cfg=json.loads((ROOT/'configs/multidomain_v072.json').read_text());tasks={t.id:t for t in load_tasks(ROOT/'data/multidomain_v072/hotpotqa/development.jsonl')}
    replay=[]
    for source,connection,prefix in [('parent',pd,'answer/development/hotpotqa/'),('continuation',db,'answer/search/hotpotqa/')]:
        for key,raw in connection.execute('SELECT key,value FROM artifacts WHERE key LIKE ?',(prefix+'%',)):
            r=json.loads(raw);cid=r.get('call_id') or r['calls'][-1]
            response=json.loads(connection.execute("SELECT response FROM attempts WHERE call_id=? AND status='complete'",(cid,)).fetchone()[0])
            new=score(tasks[r['task_id']],response,cfg)
            if 'truncated' in r['score']['outcome']:new=r['score']
            replay.append({'key':key,'source':source,'old_quality':r['score']['quality'],'new_quality':new['quality'],'old_outcome':r['score']['outcome'],'new_outcome':new['outcome']})
    result['parser_replay']={'answers':len(replay),'changed':[r for r in replay if r['old_quality']!=r['new_quality'] or r['old_outcome']!=r['new_outcome']],
        'principle':'complete JSON or one outer fence only; strict field types and unique keys; no content invention'}
    out=Path(out);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'offline.json',result)
    print(json.dumps({'output':str(out/'offline.json'),'paid_calls':0,'router_rows':len(result['router']),'parser_changes':len(result['parser_replay']['changed'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--continuation',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.parent,a.continuation,a.out)
