#!/usr/bin/env python3
"""Explicit, bounded real integration check; never starts the six-round experiment."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nicheflow.main_config import load_protocol
from nicheflow.main_entry import run_main
from nicheflow.spec import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Explicitly authorize this bounded real check')
    parser.add_argument('--run-dir', default='runs/candidate_chain_v044')
    args = parser.parse_args()
    if not args.execute:
        parser.error('--execute required; this check consumes real model calls')
    run = Path(args.run_dir).resolve()
    if run.exists():
        parser.error('Run directory exists; no overwrite, resume, retry or extra generation')
    config, parts, _ = load_protocol(ROOT / 'configs/main_math.json', ROOT, 1)
    config = copy.deepcopy(config)
    config['queries_per_deploy_block'] = 2
    config['evaluation'] = {'status': 'deferred', 'selection': 'frozen_oful_chebyshev',
                            'samples_per_task': 1, 'pass_k': [1],
                            'note': 'Targeted integration check only; no held-out evaluation.'}
    config['limits'].update(fixed_seconds=900, seconds_per_round=900,
                            api_usd_fixed=.5, api_usd_per_round=.5)
    config['notes'].append('DIAGNOSTIC ONLY: first two development and calibration tasks; one production-driver cycle, no formal experiment.')
    inputs = run / 'inputs'
    inputs.mkdir(parents=True)
    manifest = {'synthetic': False, 'protocol': 'candidate_chain_check_v044', 'partitions': {}}
    for role in ('development', 'calibration'):
        path = inputs / (role + '.jsonl')
        selected = parts[role][:2]
        path.write_text(''.join(json.dumps(t.record(), ensure_ascii=False) + '\n' for t in selected))
        manifest['partitions'][role] = {'path': path.name, 'count': len(selected), 'sha256': file_hash(path)}
    pack = inputs / 'manifest.json'; pack.write_text(json.dumps(manifest, indent=2) + '\n')
    config['data_manifest'], config['data_manifest_sha256'] = str(pack), file_hash(pack)
    cfg = inputs / 'check_config.json'; cfg.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    summary, code = run_main(SimpleNamespace(config=str(cfg), rounds=1, run_dir=str(run)), ROOT)
    from nicheflow.ledger import Journal
    events = Journal.read(run / 'events.jsonl')
    generated = {e['payload']['version'] for e in events if e['kind'] == 'generation' and e['payload']['status'] == 'valid'}
    evaluated = {e['payload']['workflow'] for e in events if e['kind'] == 'execution'
                 and e['id'].startswith('round:') and ':search:' in e['id']}
    archive_decisions = [e for e in events if e['kind'] == 'archive' and e['id'].startswith('round:')]
    checks = {'valid_generation': bool(generated), 'generated_candidate_executed': bool(generated & evaluated),
              'candidate_archive_decision': bool(archive_decisions),
              'one_cycle_complete': summary['rounds_completed'] == 1,
              'numerically_healthy': summary['numerical_health']['all_finite_spd'] is True,
              'no_unknown_calls_or_cost': not summary['unknown_calls'] and not summary['unknown_cost_calls'],
              'system_errors_absent': not summary['execution_errors'], 'driver_exit_zero': code == 0}
    result = {'diagnostic_only': True, 'six_round_experiment_started': False, 'checks': checks,
              'passed': all(checks.values()), 'real_calls': summary['calls_finished'],
              'api_tariff_usd': summary['api_tariff_usd'], 'elapsed_seconds': summary['elapsed_seconds'],
              'limits': summary['limits'], 'new_elite_route_feedback': summary['coverage']['new_elite_route_feedback']}
    (run / 'chain_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 6


if __name__ == '__main__':
    raise SystemExit(main())
