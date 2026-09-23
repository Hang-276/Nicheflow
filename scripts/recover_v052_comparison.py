#!/usr/bin/env python3
"""Explicit recovery into a NEW directory; original interrupted evidence is immutable.

Only a final, unfinished, zero-dollar local call may be reissued. No API retry.
The derived journal preserves the exact completed prefix and records the excluded
reservation/start as an abandoned attempt, also counted against the call cap.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nicheflow.cli import code_identity
from nicheflow.comparison_eval import summarize
from nicheflow.graph import WorkflowGraph
from nicheflow.ledger import Journal, atomic_json
from nicheflow.main_budget import ResourceJournal
from nicheflow.main_config import load_protocol
from nicheflow.main_models import ModelPool, model_readiness
from nicheflow.runtime import GraphExecutor
from nicheflow.spec import IntegrityError, digest, file_hash


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def recovery_prefix(events, cfg):
    starts = {e['id']: e for e in events if e['kind'] == 'call_started'}
    ends = {e['id']: e for e in events if e['kind'] == 'call_finished'}
    require(len(starts) == sum(e['kind'] == 'call_started' for e in events), 'duplicate call start')
    require(len(ends) == sum(e['kind'] == 'call_finished' for e in events), 'duplicate call end')
    require(set(ends) <= set(starts), 'call completion without start')
    unknown = set(starts) - set(ends)
    require(len(unknown) == 1, 'recovery requires exactly one unfinished local call')
    key = next(iter(unknown))
    tail = events[-2:]
    require(len(tail) == 2 and [e['kind'] for e in tail] == ['resource_reservation', 'call_started']
            and all(e['id'] == key for e in tail), 'unfinished call must be the final reservation/start')
    profiles = [p for p in cfg['models'].values() if p.get('path', p.get('model')) == starts[key]['payload']['model']]
    require(len(profiles) == 1 and profiles[0]['kind'] == 'local'
            and profiles[0]['accounting_usd_per_gpu_hour'] == 0, 'cannot retry API or unknown-cost model')
    require(all(e['payload'].get('accounted_usd') is not None for e in ends.values()), 'unknown completed-call cost')
    require(not any(e['kind'] == 'comparison_finished' for e in events), 'run already finalized')
    require(not any(e['kind'] in ('router', 'feedback', 'generation', 'scheduler') for e in events), 'unexpected learning')
    return events[:-2], tail


def verify_training(plan):
    source = Path(plan['source']['directory'])
    for name, key in [('config.json', 'config_sha256'), ('events.jsonl', 'events_sha256')]:
        require(file_hash(source / name) == plan['source'][key], 'training evidence changed: ' + name)
    graphs = {}
    for c in plan['checkpoint_checks'].values():
        p = source / 'checkpoints' / f"round_{c['round']:04d}.json"
        require(file_hash(p) == c['file_sha256'], 'training checkpoint changed')
        graphs.update({k: WorkflowGraph.from_dict(v) for k, v in json.loads(p.read_text())['state']['graphs'].items()})
    return graphs


class RecoveryJournal(ResourceJournal):
    def __init__(self, *args, abandoned, **kwargs):
        self.abandoned = abandoned
        super().__init__(*args, **kwargs)

    def reserve(self, count):
        # One extra physical local attempt occurred before the interruption.
        return super().reserve(count + 1)

    def call(self, id, backend, messages, params):
        if id == self.abandoned['id']:
            require(digest({'messages': messages, 'params': params, 'model': backend.model_id})
                    == digest(self.abandoned['payload']), 'reissued local request differs from original')
        return super().call(id, backend, messages, params)


def recover(root, source, output, config_path, execute=False, backend_factory=ModelPool):
    root, source, output = map(Path, (root, source, output))
    require(source.resolve() != output.resolve() and not output.exists(), 'recovery requires a new output directory')
    # Do not race another writer, even while copying evidence.
    with (source / '.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        frozen = json.loads((source / 'config.json').read_text())
        require(frozen['fingerprint'] == digest({k: frozen[k] for k in ('config', 'max_calls', 'seconds')}), 'config hash mismatch')
        contract = frozen['config']; cfg = contract['settings']
        require(contract['code'] == code_identity(root), 'frozen execution code changed')
        require(not contract['synthetic'] or getattr(backend_factory, 'is_synthetic', False), 'synthetic/real mismatch')
        require(contract['synthetic'] or not getattr(backend_factory, 'is_synthetic', False), 'synthetic/real mismatch')
        loaded, tasks, _ = load_protocol(config_path, root)
        require(loaded == cfg, 'recovery settings differ')
        plan = json.loads((source / 'plan.json').read_text())
        require(digest(plan) == contract['plan_digest'], 'plan changed')
        graphs = verify_training(plan)
        raw = (source / 'events.jsonl').read_bytes()
        events = Journal.read(source / 'events.jsonl')
        prefix, tail = recovery_prefix(events, cfg)
        done = {e['id'] for e in prefix if e['kind'] == 'execution'}
        require(done <= {j['execution_id'] for j in plan['jobs']}, 'unplanned execution')
        require(time.time() - frozen['started'] < frozen['seconds'], 'original wall-clock budget exhausted')
        manifest = {'schema': 'nicheflow_local_interruption_recovery_v1', 'source_directory': str(source.resolve()),
                    'source_hashes': {n: file_hash(source/n) for n in ('events.jsonl', 'config.json', 'plan.json', 'environment.json')},
                    'completed_executions_reused': len(done), 'remaining_executions': len(plan['jobs'])-len(done),
                    'abandoned_local_events': tail, 'abandoned_local_attempts': 1,
                    'abandoned_local_elapsed_seconds': None, 'abandoned_api_usd': 0,
                    'prefix_event_count': len(prefix), 'original_deadline': frozen['started'] + frozen['seconds'],
                    'budget_policy': 'unchanged original time/API/call caps; abandoned local attempt counts toward call cap',
                    'recovery_script_sha256': file_hash(Path(__file__)), 'created_at': time.time()}
        print(json.dumps({k: manifest[k] for k in ('completed_executions_reused', 'remaining_executions', 'original_deadline')}) , flush=True)
        if not execute:
            return manifest
        if not contract['synthetic']:
            require(not model_readiness(cfg), 'model or credential unavailable')
        output.mkdir(parents=True)
        for name in ('config.json', 'plan.json', 'environment.json'):
            shutil.copy2(source/name, output/name)
        # Byte-for-byte prefix; original source retains the complete interrupted tail.
        (output/'events.jsonl').write_bytes(b''.join(raw.splitlines(keepends=True)[:len(prefix)]))
        require(Journal.read(output/'events.jsonl') == prefix, 'copied prefix differs')
        atomic_json(output/'recovery_manifest.json', manifest)
    task_map = {t.id: t for t in tasks['evaluation']}
    limits = contract['limits']
    with RecoveryJournal(output, contract, frozen['max_calls'], frozen['seconds'], limits=limits, abandoned=tail[-1]) as journal:
        journal.append('interruption_recovery', 'local_retry:1', manifest)
        journal.reserve(0)
        pool = backend_factory(cfg)
        try:
            identity = {k: v.environment for k, v in pool.backends.items()}
            old = json.loads((source/'environment.json').read_text())
            require(set(identity) == set(old), 'model set changed')
            for k in identity:
                require(identity[k]['model'] == old[k]['model'], 'model identity changed')
                if cfg['models'][k]['kind'] == 'local' and not contract['synthetic']:
                    for field in ('software', 'awq_kernels', 'backend'):
                        require(identity[k].get(field) == old[k].get(field), 'local backend changed: '+field)
                    for field in ('name', 'compute_capability', 'cuda_runtime', 'cudnn'):
                        require(identity[k]['gpu'].get(field) == old[k]['gpu'].get(field), 'GPU environment changed: '+field)
            atomic_json(output/'recovery_environment.json', identity)
            executor = GraphExecutor(journal, pool.backends, cfg['decode'], failure_limit=3,
                                     allowed_operators=cfg['allowed_operators'], workflow_output_policy=cfg['workflow_output_policy'])
            executor.length_limit_as_zero = cfg['length_limit_outcome'] == 'zero_quality_no_retry'
            with journal.scope('evaluation', frozen['max_calls'], limits['max_accounting_usd']):
                for i, job in enumerate(plan['jobs'], 1):
                    executor.execute(graphs[job['workflow']], task_map[job['task_id']], job['execution_id'])
                    executor.check_failures(); journal.reserve(0)
                    if job['execution_id'] not in done:
                        progress = {'completed': i, 'total': len(plan['jobs']), 'api_usd': journal.spending()['api_usd'],
                                    'recovery_new_executions': i-len(done)}
                        atomic_json(output/'progress.json', progress)
                        print(json.dumps(progress), flush=True)
        finally:
            pool.close()
        verify_training(plan)
        require(all(file_hash(source/n) == h for n,h in manifest['source_hashes'].items()), 'interrupted source changed')
        require(code_identity(root) == contract['code'], 'code changed during recovery')
        result = summarize(plan, journal.events, cfg)
        require(result['status'] == 'complete' and not result['unknown_call_ids'] and not result['unknown_cost_ids'], 'incomplete recovery')
        result.update(run_status='main_completed_with_execution_errors' if result['all_execution_errors'] else 'main_run_complete',
                      error=None, elapsed_seconds=time.time()-frozen['started'], planning_seconds=None,
                      learning_state_unchanged=True, recovery=manifest,
                      physical_attempts_including_abandoned=result['physical_calls']+1,
                      time_notes=result['time_notes']+' Interrupted local compute is unknown and excluded; elapsed wall time includes downtime.')
        journal.append('comparison_finished', 'main', result)
        atomic_json(output/'comparison.json', result)
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--out-dir', required=True)
    p.add_argument('--config', required=True); p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    result = recover(ROOT, args.source, args.out_dir, args.config, args.execute)
    print(json.dumps({k: result[k] for k in ('status', 'run_status', 'physical_api_usd') if k in result}), flush=True)
    if args.execute:
        raise SystemExit(0 if result['run_status'] == 'main_run_complete' else 2)
