#!/usr/bin/env python3
"""Freeze three disjoint domains before any v072 model calls."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gzip
import json
from pathlib import Path
import random
import shutil
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nicheflow.datasets import Task,normalize,load_tasks
from nicheflow.math_protocol import SUBJECTS,question_hash
from nicheflow.spec import file_hash,digest
from nicheflow.ledger import atomic_json

SOURCES={
 'math':('EleutherAI/hendrycks_math','21a5633873b6a120296cce3e2df9d5550074f4a3'),
 'mbpp':('google-research-datasets/mbpp','4bb6404fdc6cacfda99d4ac4205087b89d32030c'),
 'hotpotqa':('hotpotqa/hotpot_qa','1908d6afbbead072334abe2965f91bd2709910ab')}
SEED=20260923072
SPLITS={'development':40,'screening':20,'calibration':40,'final':100}


def files():
    for subject in SUBJECTS:
        for split in ['train','test']:
            yield 'math',f'{subject}/{split}-00000-of-00001.parquet'
    for split in ['train','test']:
        yield 'mbpp',f'sanitized/{split}-00000-of-00001.parquet'
    for name in ['train-00000-of-00002','train-00001-of-00002','validation-00000-of-00001']:
        yield 'hotpotqa',f'distractor/{name}.parquet'


def fetch_one(item):
    name,relative=item
    repo,revision=SOURCES[name]
    url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{relative}'
    path=ROOT/'data/source_v072'/name/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        req=urllib.request.Request(url,headers={'User-Agent':'NicheFlow-v072-data-preparation'})
        with urllib.request.urlopen(req,timeout=120) as r,path.with_suffix('.partial').open('wb') as f:
            shutil.copyfileobj(r,f)
        path.with_suffix('.partial').replace(path)
    return {'path':str(path.relative_to(ROOT)),'url':url,'sha256':file_hash(path),'bytes':path.stat().st_size}


def plus_source():
    path=ROOT/'data/source_v072/MbppPlus-v0.2.0.jsonl.gz'
    url='https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz'
    if not path.exists():
        with urllib.request.urlopen(url,timeout=120) as r,path.with_suffix('.partial').open('wb') as f:
            shutil.copyfileobj(r,f)
        path.with_suffix('.partial').replace(path)
    with gzip.open(path,'rt') as f:rows={int(r['task_id'].split('/')[-1]):r for r in map(json.loads,f)}
    return rows,{'path':str(path.relative_to(ROOT)),'url':url,'sha256':file_hash(path),'bytes':path.stat().st_size}


def old_tasks():
    paths=[ROOT/'data/tasks.jsonl']
    for pattern in ['main_*/*.jsonl','fixed_validation_*/*.jsonl','model_selection_*/*.jsonl','benchmarks/*/tasks.jsonl']:
        paths.extend((ROOT/'data').glob(pattern))
    rows=[]
    for p in sorted(set(paths)):rows.extend(load_tasks(p))
    return rows,{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(set(paths))}


def stratified(items,count,key,tag):
    pools=defaultdict(list)
    for item in items:pools[key(item)].append(item)
    rng=random.Random(int(digest([SEED,tag])[:16],16))
    for pool in pools.values():rng.shuffle(pool)
    labels=sorted(pools);rng.shuffle(labels);selected=[]
    while len(selected)<count:
        changed=False
        for label in labels:
            if pools[label] and len(selected)<count:
                selected.append(pools[label].pop());changed=True
        if not changed:raise ValueError(f'insufficient eligible data: {tag} ({len(selected)}/{count})')
    return selected


def prepare():
    import pyarrow.parquet as pq
    out=ROOT/'data/multidomain_v072'
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text())
        for f,h in manifest['task_files'].items():assert file_hash(ROOT/f)==h
        print(json.dumps({'status':'already_frozen','manifest':str(out/'manifest.json')}));return
    with ThreadPoolExecutor(6) as pool:sources=list(pool.map(fetch_one,files()))
    plus,source=plus_source();sources.append(source)
    exclusions_snapshot=json.loads((ROOT/'data/v072_exclusions.json').read_text())
    history=exclusions_snapshot['history_files']
    forbidden=set(exclusions_snapshot['question_hashes'])
    prior_mbpp=set(exclusions_snapshot['mbpp_task_ids'])
    domains={name:{'learning':[],'final':[]} for name in ['math','mbpp','hotpotqa']}
    exclusions=[]
    for name,relative in files():
        rows=pq.read_table(ROOT/'data/source_v072'/name/relative).to_pylist()
        split='train' if '/train-' in relative else 'test' if '/test-' in relative else 'validation'
        for idx,row in enumerate(rows):
            if name=='math':
                subject=relative.split('/')[0]
                try:
                    t=normalize('math',row,idx,split=split,role='development' if split=='train' else 'evaluation')
                except ValueError as exc:
                    if str(exc)!='MATH reference missing boxed answer':raise
                    exclusions.append({'id':f'math/{split}/{subject}/{idx}','reason':'reference_missing_boxed_answer'})
                    continue
                t=replace(t,id=f'math/{split}/{subject}/{idx}',private={**t.private,'subject':subject})
            elif name=='mbpp':
                ident=int(row['task_id'])
                if ident not in plus or ident in prior_mbpp:continue
                # Match sanitized task text/code to the EvalPlus task identity.
                prompt=row.get('prompt',row.get('text'))
                tests=row.get('test_list') or row.get('tests')
                if not prompt or not tests:raise ValueError('MBPP sanitized schema unsupported')
                canonical=plus[ident]['prompt']+plus[ident]['canonical_solution']
                t=Task(f'mbpp/{split}/{ident}','mbpp',split,'development' if split=='train' else 'evaluation',
                    prompt,canonical,'python_tests',public_tests=tuple(tests[:1]),
                    private={'tests':list(tests[1:]),'setup':'\n'.join(row.get('test_imports',[]))+ '\n'+row.get('test_setup_code',''),
                             'evalplus':plus[ident],'source_task_id':ident,'base_test_count':len(tests)-1})
            else:
                t=normalize(name,row,idx,split=split,role='development' if split=='train' else 'evaluation')
                # Use explicit title and zero-based sentence identifiers in visible context.
                context=t.context
                if isinstance(context,dict):context=list(zip(context['title'],context['sentences']))
                t=replace(t,context=[{'title':title,'sentences':[{'id':i,'text':text} for i,text in enumerate(sentences)]}
                                    for title,sentences in context],private={**t.private,'level':row.get('level'),'type':row.get('type')})
            if question_hash(t.question) in forbidden:continue
            domains[name]['learning' if split=='train' else 'final'].append(t)
    all_tasks=[];selection={};seen=set(forbidden)
    for name,sets in domains.items():
        if name=='math':key=lambda t:(t.private['subject'],t.private['level'])
        elif name=='hotpotqa':key=lambda t:(str(t.private['type']),str(t.private['level']))
        else:key=lambda t:('all',)
        selection[name]={}
        for part,n in SPLITS.items():
            source=sets['final' if part=='final' else 'learning']
            chosen=stratified([t for t in source if question_hash(t.question) not in seen],n,key,[name,part])
            for t in chosen:seen.add(question_hash(t.question))
            chosen=[replace(t,role='evaluation' if part in ('screening','final') else part) for t in chosen]
            selection[name][part]=[t.id for t in chosen]
            target=out/name/(part+'.jsonl');target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(''.join(json.dumps(t.record(),ensure_ascii=False)+'\n' for t in chosen))
            all_tasks.extend(chosen)
    assert len(all_tasks)==600 and len({t.id for t in all_tasks})==600
    assert len({question_hash(t.question) for t in all_tasks})==600
    manifest={'version':'multidomain_v072','seed':SEED,'created_before_v072_calls':True,
        'sources':sources,'history_files':history,'history_unique_questions':len(forbidden),
        'source_revisions':SOURCES,'counts':{name:{p:len(ids) for p,ids in parts.items()} for name,parts in selection.items()},
        'selected_ids':selection,'selection_rule':'fixed-seed balanced source strata, no model scores; MBPP sanitized intersection with EvalPlus v0.2.0',
        'task_files':{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(out.glob('*/*.jsonl'))},
        'heldout_policy':'final tasks sealed until model settings, candidate choices and routing are frozen',
        'candidate_feedback_ids':{name:ids['development'][:20] for name,ids in selection.items()},
        'mbpp_protocol':'first source example public; remaining source examples private; EvalPlus base+plus evaluated separately; primary private-source and enhanced pass',
        'overlap_previous':0,'overlap_between_splits':0,'source_eligibility_exclusions':exclusions}
    atomic_json(out/'manifest.json',manifest)
    print(json.dumps({'status':'prepared','counts':manifest['counts'],'eligible_counts':{name:{s:len(r) for s,r in sets.items()} for name,sets in domains.items()},'manifest_sha256':file_hash(out/'manifest.json')},ensure_ascii=False),flush=True)


if __name__=='__main__':prepare()
