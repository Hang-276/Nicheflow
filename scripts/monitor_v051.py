#!/usr/bin/env python3
"""Read-only live snapshot. Does not load models, call APIs or rewrite run files."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nicheflow.spec import digest
from nicheflow.checkpoint_compat import verify_checkpoint


def snapshot(directory):
    directory = Path(directory)
    if not (directory/'events.jsonl').exists():
        return {'state':'not_started_or_initializing', 'directory':str(directory)}
    config = json.loads((directory/'config.json').read_text())
    settings = config['config']['settings']
    source = config['config'].get('stage', {}).get('source')
    start_round = source['round'] if source else 0
    events, previous, partial = [], '0'*64, False
    raw = (directory/'events.jsonl').read_bytes()
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if index == len(lines)-1 and not line.endswith(b'\n'):
            partial = True
            break
        event = json.loads(line)
        # Same ledger hash contract, checked without opening its writer.
        if event['seq'] != len(events) or event['previous'] != previous:
            raise ValueError('event sequence/hash chain mismatch')
        body = {k:v for k,v in event.items() if k != 'hash'}
        if digest(body) != event['hash']:
            raise ValueError('event hash mismatch')
        previous = event['hash']
        events.append(event)
    kinds = Counter(e['kind'] for e in events)
    started = {e['id']:e for e in events if e['kind']=='call_started'}
    finished = {e['id']:e['payload'] for e in events if e['kind']=='call_finished'}
    local_ids = {p['path'] for p in settings['models'].values() if p['kind']=='local'}
    by_model = Counter(started[k]['payload']['model'] for k in finished)
    completion = next((e['payload'] for e in reversed(events) if e['kind']=='run_finished'), None)
    operations = Counter(e['payload']['kind'] for e in events if e['kind']=='operation_finished')
    generations = Counter(e['payload']['status'] for e in events if e['kind']=='generation')
    checkpoints = sorted((directory/'checkpoints').glob('round_*.json'))
    last = json.loads(checkpoints[-1].read_text()) if checkpoints else None
    state = last['state'] if last else None
    checkpoint_validation = None
    if last:
        round_index = int(checkpoints[-1].stem.split('_')[1])
        expected = next((e['payload'] for e in events if e['kind']=='state_snapshot' and e['id']==f'round:{round_index}'), None)
        checkpoint_validation = verify_checkpoint(last, expected, allow_legacy=True)
    active = None
    if (directory/'.lock').exists():
        with (directory/'.lock').open('r') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
                active = False
            except BlockingIOError:
                active = True
    health = [e['payload']['numerically_healthy'] for e in events if e['kind']=='parameter_health']
    in_flight = [{'id':k,'age_seconds':round(time.time()-e['time'],1)} for k,e in started.items() if k not in finished]
    errors = [e['id'] for e in events if e['kind']=='execution'
              and e['payload']['status'] not in {'ok','truncated'}]
    return {
        'directory':str(directory), 'state':completion['status'] if completion else ('running' if active else 'no_active_writer'),
        'rounds_completed':completion['rounds_completed'] if completion else start_round+kinds['cycle_finished'],
        'stage_rounds_completed':kinds['cycle_finished'], 'rounds_requested':settings['rounds'],
        'phase':'learning' if kinds['scheduler'] or kinds['stage_restored'] else 'bootstrap',
        'bootstrap_workflows_done':sum(e['kind']=='feedback' and e['id'].startswith('bootstrap:') for e in events),
        'bootstrap_calls_reserved':config['config']['limits']['bootstrap_calls'],
        'calls_started':len(started), 'calls_finished':len(finished), 'calls_by_requested_model':dict(by_model),
        'api_tariff_usd':sum((c.get('api_charge') or {}).get('usd',0.) for c in finished.values()),
        'api_usd_cap':config['config']['limits']['max_api_usd'],
        'cost_objective':settings.get('cost_objective'),
        'local_inference_accounting_usd':sum(c.get('local_accounting_usd') or 0 for c in finished.values()),
        'local_inference_seconds':sum(c.get('elapsed_seconds') or 0 for k,c in finished.items() if started[k]['payload']['model'] in local_ids),
        'operations_completed':dict(operations), 'generation_statuses':dict(generations),
        'parent_credit_updates':kinds['parent_credit'], 'router_updates':kinds['router'],
        'latest_checkpoint':str(checkpoints[-1]) if checkpoints else None,
        'checkpoint_validation':checkpoint_validation,
        'archive_elites':len(state['archive']['elites']) if state else None,
        'evaluated_workflows':len(state['records']) if state else None,
        'numerically_healthy':all(health) if health else None,
        'execution_errors':errors, 'truncated_workflows':sum(e['kind']=='execution' and e['payload']['status']=='truncated' for e in events),
        'unknown_cost_calls':[k for k,c in finished.items() if c.get('accounted_usd') is None],
        'calls_in_flight':in_flight, 'writer_active':active, 'complete_event_hash_chain_valid':True,
        'partial_last_event_ignored':partial,
        'last_event_age_seconds':round(time.time()-events[-1]['time'],1) if events else None,
        'evaluation_started':bool(kinds['evaluation_started']),
        'completion':completion,
        'exit_code':(directory/'exit_code.txt').read_text().strip() if (directory/'exit_code.txt').exists() else None,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', nargs='?', default='runs/main_v051_local_flash_r30')
    args = parser.parse_args()
    print(json.dumps(snapshot(args.run_dir), ensure_ascii=False, indent=2))
