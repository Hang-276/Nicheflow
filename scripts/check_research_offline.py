#!/usr/bin/env python3
"""Real frozen CPU encoder + synthetic model receipts; no model API/GPU inference.

This diagnostic does not establish middle-model access, accuracy or live routing.
It verifies the current feature implementation across a saved/restored stage.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_research_v050 import Pool, revised_fixture, state
from nicheflow.main_entry import run_main
from nicheflow.ledger import Journal, atomic_json
from nicheflow.spec import digest, file_hash


class RealEncoderSyntheticModels(Pool):
    semantic_encoder_factory = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='runs/revision_v050_offline_final')
    args = parser.parse_args()
    directory = Path(args.out_dir).resolve()
    if directory.exists():
        parser.error('output must be new; no overwrite')
    directory.mkdir(parents=True)
    runargs, config = revised_fixture(directory, rounds=2, deferred=True)
    runargs.run_dir = str(directory / 'uninterrupted')
    uninterrupted, c0 = run_main(runargs, ROOT, RealEncoderSyntheticModels)
    first, second = directory / 'stage1', directory / 'stage2'
    runargs.run_dir, runargs.stop_after_round = str(first), 1
    one, c1 = run_main(runargs, ROOT, RealEncoderSyntheticModels)
    source_hash = file_hash(first / 'events.jsonl')
    runargs.run_dir, runargs.stop_after_round, runargs.resume_from = str(second), 2, str(first)
    two, c2 = run_main(runargs, ROOT, RealEncoderSyntheticModels)
    events = Journal.read(first / 'events.jsonl') + Journal.read(second / 'events.jsonl')
    call_ids = [e['id'] for e in events if e['kind'] == 'call_started']
    checks = {
        'all_stages_complete': c0 == c1 == c2 == 0,
        'real_encoder_restored_state_matches': digest(state(directory / 'uninterrupted', 2)) == digest(state(second, 2)),
        'source_unchanged': source_hash == file_hash(first / 'events.jsonl'),
        'no_duplicate_calls': len(call_ids) == len(set(call_ids)),
        'all_model_backends_synthetic': all(r['synthetic_backend'] for r in (uninterrupted, one, two)),
        'healthy': all(r['numerical_health']['all_finite_spd'] for r in (uninterrupted, one, two)),
        'no_execution_errors': all(not r['execution_errors'] for r in (uninterrupted, one, two)),
        'parent_credit_exercised': any(e['kind'] == 'parent_credit' for e in events),
        'block_rescheduling': sum(e['kind'] == 'scheduler' for e in events) == 4,
        'no_heldout_evaluation': not any(e['kind'] == 'evaluation_started' for e in events),
    }
    result = {'passed': all(checks.values()), 'checks': checks, 'real_model_calls': 0,
              'model_backends': 'synthetic fixtures', 'semantic_encoder': 'real frozen MiniLM CPU',
              'middle_model_selected': False, 'expanded_experiment_started': False,
              'performance_or_three_tier_readiness_established': False}
    atomic_json(directory / 'check.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 6


if __name__ == '__main__':
    raise SystemExit(main())
