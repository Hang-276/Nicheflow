#!/usr/bin/env python3
"""Freeze fresh MATH validation and four workflows before any new model calls."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nicheflow.datasets import load_tasks
from nicheflow.graph import Node,WorkflowGraph
from nicheflow.ledger import atomic_json
from nicheflow.math_protocol import SUBJECTS,learning_tasks,question_hash,distribution
from nicheflow.spec import IntegrityError,digest,file_hash

BEST='6e8752093f669ff4ab416e3dc3e58a439ec5316be984de17206dcd4f96eecbc9'
SEED=2026092206

def select_fresh(rows,heldout,old):
    # Existing trusted normalizer handles reference validity and duplicate source questions.
    parts,exclusions=learning_tasks(rows,heldout,old,seed=SEED,development_per_stratum=4,calibration_per_stratum=1)
    tasks=[replace(t,role='evaluation') for t in parts['development']]
    forbidden={question_hash(t.question) for t in heldout}|{question_hash(q) for q in old}
    assert len(tasks)==140 and len({question_hash(t.question) for t in tasks})==140
    assert all(question_hash(t.question) not in forbidden for t in tasks)
    assert Counter((t.private['subject'],t.private['level']) for t in tasks)==Counter({(s,l):4 for s in SUBJECTS for l in range(1,6)})
    return tasks,exclusions

def make_groups(settings,best):
    a=WorkflowGraph((Node('a','Generate',settings['seed_prompts']['solve'],model='strong'),),'a')
    b=WorkflowGraph((Node('a','Generate',settings['seed_prompts']['solve']+' Use concise but sufficient reasoning. Avoid restating the problem, lengthy introductions, repeated calculations, and unnecessary alternative solutions. Prioritize reaching and checking the exact final answer within the output budget. Put the final answer in the required format.',model='strong'),),'a')
    assert [n.id for n in best.nodes]==['a','review','final']
    assert best.nodes[1].model=='local' and best.nodes[2].inputs==('review',)
    d=replace(best,nodes=(best.nodes[0],replace(best.nodes[2],inputs=('a',))),parents=(best.version,))
    names=['A_flash_original','B_flash_concise','C_best_with_qwen','D_without_qwen']
    return {name:g.validate(settings['models']).definition() for name,g in zip(names,(a,b,best,d))}

def prepare(source):
    import pyarrow.parquet as pq
    source=Path(source); target=ROOT/'data/fixed_validation_v060'
    config_path=ROOT/'configs/fixed_validation_v060.json'
    if target.exists() or config_path.exists():raise IntegrityError('immutable validation pack already exists')
    config=json.loads((source/'config.json').read_text())['config']['settings']
    state=json.loads((source/'checkpoints/round_0030.json').read_text())['state']
    assert state['records'][BEST]['quality']==max(r['quality'] for r in state['records'].values())
    best=WorkflowGraph.from_dict(state['graphs'][BEST]);assert best.version==BEST
    downloads=json.loads((ROOT/'data/source_math_v041/downloads.json').read_text())
    for entry in downloads:
        if file_hash(ROOT/entry['path'])!=entry['sha256']:raise IntegrityError('source data changed')
    old_paths=sorted(p for p in (ROOT/'data').rglob('*.jsonl') if p.name in ('tasks.jsonl','development.jsonl','calibration.jsonl','evaluation.jsonl'))
    old=[t.question for p in old_paths for t in load_tasks(p) if t.dataset=='math']
    heldout=load_tasks(ROOT/'data/main_math500_v050/evaluation.jsonl')
    rows={s:pq.read_table(ROOT/f'data/source_math_v041/{s}.parquet').to_pylist() for s in SUBJECTS}
    tasks,exclusions=select_fresh(rows,heldout,old)
    groups=make_groups(config,best)
    target.mkdir(parents=True)
    task_path=target/'validation.jsonl'
    task_path.write_text(''.join(json.dumps(t.record(),ensure_ascii=False)+'\n' for t in tasks))
    manifest={'schema':'fixed_validation_v060_data','sampling_seed':SEED,'source_split':'train',
       'runtime_role':'evaluation: prevents learning within this fixed-workflow validation',
       'count':140,'per_subject_level':4,'distribution':distribution(tasks),'tasks_sha256':file_hash(task_path),
       'source_files':downloads,'excluded_prior_files':{str(p.relative_to(ROOT)):file_hash(p) for p in old_paths},
       'excluded_unique_questions':len({question_hash(q) for q in old}), 'exclusions':exclusions,
       'selection_before_model_calls':True,'no_overlap_with_prior_data':True}
    atomic_json(target/'manifest.json',manifest)
    frozen={k:config[k] for k in ('models','credentials_file','decode','allowed_operators','workflow_output_policy','length_limit_outcome','cost_objective')}
    frozen.update(schema='fixed_validation_v060',experiment_id='four_fixed_workflows_140x2',synthetic=False,
       data='data/fixed_validation_v060/validation.jsonl',data_sha256=file_hash(task_path),
       manifest='data/fixed_validation_v060/manifest.json',manifest_sha256=file_hash(target/'manifest.json'),
       groups=groups,samples_per_task=2,planning_seed=SEED,max_api_usd=10.,max_seconds=8*3600,
       concurrency={'workflow_workers':8,'api_calls':4,'local_calls':1},
       primary_contrasts=[['B_flash_concise','A_flash_original'],['C_best_with_qwen','A_flash_original'],
                          ['C_best_with_qwen','B_flash_concise'],['C_best_with_qwen','D_without_qwen']],
       source_training={'directory':str(source),'hashes':{n:file_hash(source/n) for n in ('config.json','events.jsonl','checkpoints/round_0030.json','environment.json')},'best_workflow':BEST},
       decode_note='Original common defaults retained. C keeps its discovered node temperatures 0.7/0.3/0.5; D retains the same planner/final temperatures. A/B both use 0.7. No claim that C-vs-A isolates topology alone.',
       sampling_note='Two independent draws per task/group, average accuracy; no best-of-two selection. Groups do not share generated planner outputs. Flash does not support fixed seed.',
       concurrency_note='Fixed dispatch order with at most 8 independent workflows, 4 concurrent API requests and 1 local inference. DAG dependencies remain sequential. Stop new calls on unknown billing; drain in-flight requests without retry. Report summed call service time separately from run wall time.',
       reporting_cost='Frozen peak uncached reference cost plus actual returned-usage tariff cost; local API spend=0, local seconds separate.')
    atomic_json(config_path,frozen)
    print(json.dumps({'config':str(config_path),'config_sha256':file_hash(config_path),'tasks':len(tasks),
                      'workflow_executions':len(tasks)*4*2,'max_model_calls':len(tasks)*2*sum(WorkflowGraph.from_dict(g).model_calls for g in groups.values()),
                      'new_model_calls':0,'overlap':False},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    prepare(p.parse_args().source)
