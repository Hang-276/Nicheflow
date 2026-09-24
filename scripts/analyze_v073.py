#!/usr/bin/env python3
"""Summarize bounded v073 work without starting or retrying inference."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nicheflow.ledger import atomic_json
from nicheflow.spec import file_hash
from scripts.analyze_v072 import artifacts,get,paired,accounting


def run(directory,out):
    directory=Path(directory);result={'analysis_source_sha256':file_hash(__file__),'scope':'v073 engineering/development comparisons, not fresh independent confirmation','runs':[],'search_contrasts':[]}
    records={}
    for path in sorted(directory.glob('*/checkpoint.sqlite3')):
        db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        rows=artifacts(db,'answer/');name=path.parent.name
        status=get(db,'complete');meta={k:json.loads(v) for k,v in db.execute("SELECT key,value FROM meta WHERE key IN ('pause','pending_audit')")}
        entry={'name':name,'source_sha256':file_hash(path),'complete':status,'state':meta,'answers':len(rows),'accounting':accounting(db)}
        if name=='repair':
            summary=get(db,'repair_summary')
            if summary:entry['repair_summary']={k:v for k,v in summary.items() if k!='comparison_rows'}
        if name.startswith('search_'):
            entry['summary']=get(db,'summary');entry['slots']=[{'key':k,**json.loads(v)} for k,v in db.execute("SELECT key,value FROM artifacts WHERE key LIKE 'slot/%'")]
            records[name]=[r for r in rows if '/screening/' in 'answer/'+r['calls'][0]]
        if name=='search-seeds':records[name]=[r for r in rows if r['calls'][0].startswith('screen_controls/')]
        result['runs'].append(entry);db.close()
    controls=records.get('search-seeds',[])
    for name,rows in records.items():
        if name=='search-seeds' or len(rows)!=20:continue
        for model in ['L','M','H']:
            baseline=[r for r in controls if r['candidate']==model]
            if len(baseline)==20:result['search_contrasts'].append({'first':name,'second':model,**paired(rows,baseline)})
    for seed in [2026092401,2026092402]:
        left,right=f'search_mixed_{seed}',f'search_elite_{seed}'
        if len(records.get(left,[]))==len(records.get(right,[]))==20:result['search_contrasts'].append({'first':left,'second':right,**paired(records[left],records[right])})
    out=Path(out);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'analysis.json',result)
    print(json.dumps({'output':str(out/'analysis.json'),'runs':len(result['runs']),'paid_calls':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='runs/prescale_v073');p.add_argument('--out',default='reports/prescale_v073_20260924');a=p.parse_args();run(a.run_dir,a.out)
