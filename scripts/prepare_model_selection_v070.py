#!/usr/bin/env python3
"""Freeze a fresh, balanced screening set without inspecting any model answers."""
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nicheflow.datasets import load_tasks
from nicheflow.ledger import atomic_json
from nicheflow.math_protocol import SUBJECTS, learning_tasks, question_hash, distribution
from nicheflow.spec import file_hash

SEED = 2026092307


def prepare():
    import pyarrow.parquet as pq
    target = ROOT / 'data/model_selection_v070'
    if target.exists():
        raise ValueError('immutable data pack already exists')
    names = {'tasks.jsonl', 'development.jsonl', 'calibration.jsonl',
             'evaluation.jsonl', 'validation.jsonl'}
    previous = sorted(p for p in (ROOT / 'data').rglob('*.jsonl') if p.name in names)
    old = [t.question for p in previous for t in load_tasks(p) if t.dataset == 'math']
    heldout = load_tasks(ROOT / 'data/main_math500_v050/evaluation.jsonl')
    sources = json.loads((ROOT / 'data/source_math_v041/downloads.json').read_text())
    for entry in sources:
        assert file_hash(ROOT / entry['path']) == entry['sha256']
    rows = {s: pq.read_table(ROOT / f'data/source_math_v041/{s}.parquet').to_pylist() for s in SUBJECTS}
    parts, exclusions = learning_tasks(rows, heldout, old, seed=SEED,
                                      development_per_stratum=4, calibration_per_stratum=1)
    tasks = [replace(t, role='evaluation') for t in parts['development']]
    forbidden = {question_hash(q) for q in old} | {question_hash(t.question) for t in heldout}
    assert len(tasks) == len({question_hash(t.question) for t in tasks}) == 140
    assert not {question_hash(t.question) for t in tasks} & forbidden
    assert Counter((t.private['subject'], t.private['level']) for t in tasks) == Counter(
        {(s, level): 4 for s in SUBJECTS for level in range(1, 6)})
    target.mkdir()
    path = target / 'validation.jsonl'
    path.write_text(''.join(json.dumps(t.record(), ensure_ascii=False) + '\n' for t in tasks))
    manifest = {'schema': 'model_selection_v070_data', 'seed': SEED, 'count': len(tasks),
                'distribution': distribution(tasks), 'tasks_sha256': file_hash(path),
                'excluded_prior_files': {str(p.relative_to(ROOT)): file_hash(p) for p in previous},
                'excluded_unique_questions': len(forbidden), 'source_files': sources,
                'exclusions': exclusions, 'selection_before_model_calls': True,
                'validation_not_final_test': True}
    atomic_json(target / 'manifest.json', manifest)
    print(json.dumps({'count': len(tasks), 'overlap': 0, 'sha256': file_hash(path),
                      'excluded_unique_questions': len(forbidden)}))


if __name__ == '__main__':
    prepare()
