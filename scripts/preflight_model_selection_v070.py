#!/usr/bin/env python3
"""Small readiness checks kept separate from fresh 140-question screening."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import model_selection_v070 as m
from nicheflow.datasets import Task


def main(stage):
    config = json.loads((m.ROOT / 'configs/model_selection_v070.plan.json').read_text())
    keys = m.credentials(config['credentials_file'])
    if stage == 'api':
        task = Task('arithmetic-readiness', 'math', 'train', 'evaluation',
                    'Calculate 17 times 19.', '323', 'rational')
        jobs = [(name, task, 128) for name in ('qwen_flash', 'qwen_plus', 'deepseek')]
        extra = {'purpose': 'endpoint, snapshot, thinking and usage contract only; not model quality'}
    else:
        source = m.ROOT / 'data/fixed_validation_v060/validation.jsonl'
        strata = defaultdict(list)
        for task in m.load_tasks(source):
            strata[task.private['subject'], task.private['level']].append(task)
        tasks = [min(strata[k], key=lambda t: m.digest([config['seed'], 'pilot', t.id])) for k in sorted(strata)]
        m.require(len(tasks) == 35, 'pilot requires one problem per subject/level')
        jobs = [('local', t, 4096) for t in tasks]
        extra = {'source_sha256': m.file_hash(source),
                 'local_identity': json.loads((m.ROOT / 'setup/local_model_identity.json').read_text()),
                 'purpose': 'diagnostic pilot on previous validation questions; excluded from fresh screening'}
    frozen = {'stage': stage, 'config_sha256': m.file_hash(m.ROOT / 'configs/model_selection_v070.plan.json'),
              'runner_sha256': m.file_hash(Path(m.__file__)), 'pilot_runner_sha256': m.file_hash(Path(__file__)),
              'extra': extra, 'jobs': [{'model': name, 'task': task.record(), 'max_tokens': tokens}
                                     for name, task, tokens in jobs]}
    directory = m.ROOT / 'preflight' / (stage + '_readiness')
    ledger = m.Ledger(directory, frozen, {'CNY': .1, 'USD': .01}, len(jobs), 1800)
    try:
        for index, (name, task, limit) in enumerate(jobs):
            profile = config['models'][name]
            payload, bound = m.request_payload(config, profile, task.model_input(), {'max_tokens': limit},
                                               config['seed'] + index)
            identifier = f'{stage}:{index}:{name}:{task.id}'
            ledger.begin(identifier, {'model': name, 'payload': payload}, profile['currency'],
                         m.reference_charge(profile, bound, limit))
            result = m.call_model(profile, payload, keys.get(profile.get('key_environment')))
            score = m.score_result(task, result)
            ledger.finish(identifier, {'model': name, 'task_id': task.id, 'response': result, 'score': score})
            print(json.dumps({'completed': index+1, 'total': len(jobs), 'model': name,
                              'status': result['status'], 'outcome': score['outcome'],
                              'quality': score['quality'], 'elapsed_seconds': result['elapsed_seconds'],
                              'reference_spend': ledger.spent,
                              'quota_or_rate_limit': result.get('quota_or_rate_limit', False)}), flush=True)
            m.require(not ledger.stopped, 'readiness failure; inspect receipt without retrying')
        rows = list(ledger.results.values())
        result = {'stage': stage, 'completed': len(rows), 'accuracy': statistics.mean(r['score']['quality'] for r in rows),
                  'mean_call_seconds': statistics.mean(r['response']['elapsed_seconds'] for r in rows),
                  'mean_output_tokens': statistics.mean(r['response']['output_tokens'] for r in rows),
                  'truncated': sum(r['score']['outcome'] == 'truncated' for r in rows),
                  'reference_spend': ledger.spent, 'elapsed_seconds': ledger.elapsed(), 'purpose': extra['purpose']}
        m.atomic_json(directory / 'results.json', result)
        print(json.dumps(result), flush=True)
    finally:
        ledger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['api', 'local'], required=True)
    main(parser.parse_args().stage)
