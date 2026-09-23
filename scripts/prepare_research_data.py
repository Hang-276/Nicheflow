#!/usr/bin/env python3
"""Prepare 70 development / 280 calibration / 500 held-out tasks; no model calls."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nicheflow.datasets import load_tasks
from nicheflow.math_protocol import SUBJECTS, TEXT_PROTOCOL, MATH500_SHA256, learning_tasks, distribution
from nicheflow.spec import file_hash, IntegrityError


def main():
    import pyarrow.parquet as pq
    source, target = ROOT/'data/source_math_v041', ROOT/'data/main_math500_v050'
    if target.exists():
        raise IntegrityError('immutable prepared pack already exists')
    downloads = json.loads((source/'downloads.json').read_text())
    for record in downloads:
        if file_hash(ROOT/record['path']) != record['sha256']:
            raise IntegrityError('source download changed')
    if file_hash(source/'math500.jsonl') != MATH500_SHA256:
        raise IntegrityError('MATH-500 source changed')
    evaluation = load_tasks(ROOT/'data/main_math500_v1/evaluation.jsonl')
    old = [t.question for path in [ROOT/'data/tasks.jsonl', *[ROOT/f'data/{pack}/{role}.jsonl'
            for pack in ('main_math_learning_v1','main_math500_v1') for role in ('development','calibration')]] for t in load_tasks(path)]
    training = {s:pq.read_table(source/f'{s}.parquet').to_pylist() for s in SUBJECTS}
    parts, exclusions = learning_tasks(training,evaluation,old,seed=20260921,development_per_stratum=2,calibration_per_stratum=8)
    parts['evaluation']=evaluation
    target.mkdir()
    entries={}
    for role,tasks in parts.items():
        path=target/f'{role}.jsonl'
        path.write_text(''.join(json.dumps(t.record(),ensure_ascii=False)+'\n' for t in tasks))
        entries[role]={'path':path.name,'count':len(tasks),'sha256':file_hash(path),'distribution':distribution(tasks)}
    manifest={'schema':'nicheflow_data_pack_v1','protocol':'main_math500_v1','input_protocol':TEXT_PROTOCOL,
              'sampling_seed':20260921,'development_per_subject_level':2,'calibration_per_subject_level':8,
              'selection_before_model_calls':True,'evaluation_exclusions':[],
              'source_files':downloads,'exclusions_from_learning_pool':exclusions,'partitions':entries}
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'manifest':str(target/'manifest.json'),'sha256':file_hash(target/'manifest.json'),
                      'counts':{r:len(ts) for r,ts in parts.items()},'model_calls':0}))


if __name__=='__main__':main()
