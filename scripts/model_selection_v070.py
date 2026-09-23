#!/usr/bin/env python3
"""Fixed model screening with native-currency caps and resumable receipts.

No optimizer, hidden-label routing, retries, currency conversion, or pass@k.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nicheflow.cli import code_identity
from nicheflow.datasets import load_tasks
from nicheflow.ledger import atomic_json
from nicheflow.math_protocol import question_hash
from nicheflow.scoring import evaluate
from nicheflow.spec import digest, file_hash


def require(value, message):
    if not value:
        raise ValueError(message)


def credentials(path):
    p = Path(path)
    require(p.is_file() and p.stat().st_mode & 0o077 == 0, 'private credentials file required')
    out = {}
    for line in p.read_text().splitlines():
        key, sep, value = line.partition('=')
        if sep and key in {'DASHSCOPE_API_KEY', 'DEEPSEEK_API_KEY'}:
            out[key] = value.strip()
    return out


def request_payload(config, profile, public, group, seed):
    payload = {'model': profile['model'], 'messages': [
        {'role': 'system', 'content': config['system']},
        {'role': 'user', 'content': config['prompt'] + '\n' + json.dumps(
            {'task': public, 'upstream_outputs': {}}, ensure_ascii=False)}],
        'max_tokens': group['max_tokens'], 'temperature': profile['temperature']}
    if profile['provider'] == 'deepseek':
        payload['thinking'] = {'type': 'disabled'}
    else:
        payload.update(top_p=profile['top_p'], top_k=profile['top_k'],
                       presence_penalty=profile['presence_penalty'])
        if profile['provider'] == 'vllm':
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        else:
            payload['enable_thinking'] = False
    if profile.get('supports_seed'):
        payload['seed'] = seed
    input_bound = sum(len(m['content'].encode('utf-8')) + 64 for m in payload['messages']) + 1024
    require(input_bound <= profile['max_input_bound'], 'input exceeds frozen conservative bound')
    return payload, input_bound


def reference_charge(profile, inputs, outputs):
    for n in (inputs, outputs):
        require(type(n) is int and n >= 0, 'invalid or unknown token usage')
    return (inputs * profile['input_per_million'] + outputs * profile['output_per_million']) / 1e6


def call_model(profile, payload, key, timeout=None):
    endpoint = profile['endpoint']
    require(endpoint.startswith('https://') or
            (profile['kind'] == 'local' and endpoint.startswith('http://127.0.0.1:')),
            'model endpoint must be HTTPS or localhost')
    headers = {'Content-Type': 'application/json'}
    if profile['kind'] == 'api':
        require(bool(key), 'configured API credential missing')
        headers['Authorization'] = 'Bearer ' + key
    start = time.monotonic()
    try:
        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout or profile['timeout']) as response:
            data = json.load(response)
        choice = data['choices'][0]
        message, usage = choice['message'], data.get('usage', {})
        inputs, outputs = usage.get('prompt_tokens'), usage.get('completion_tokens')
        amount = reference_charge(profile, inputs, outputs)
        reason = message.get('reasoning_content') or message.get('reasoning') or ''
        details = usage.get('completion_tokens_details') or {}
        valid = (data.get('model') in profile['accepted_response_models'] and
                 isinstance(message.get('content'), str) and not reason and
                 (details.get('reasoning_tokens') in (None, 0)))
        return {'status': 'ok' if valid else 'protocol_error',
                'text': message.get('content') or '', 'reasoning_content': reason,
                'finish_reason': choice.get('finish_reason'), 'usage': usage,
                'input_tokens': inputs, 'output_tokens': outputs,
                'reference_cost': amount, 'currency': profile['currency'],
                'returned_model': data.get('model'), 'response_id': data.get('id'),
                'provider_created': data.get('created'), 'system_fingerprint': data.get('system_fingerprint'),
                'elapsed_seconds': time.monotonic() - start,
                'cost_basis': 'frozen_standard_uncached_tariff_not_invoice',
                'error': None if valid else 'model identity, content, or non-thinking contract mismatch'}
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
            code = str(body.get('error', {}).get('code', ''))
            code = code if len(code) < 100 and all(c.isalnum() or c in '._-' for c in code) else None
        except Exception:
            code = None
        # An HTTP error is recorded without response body or Authorization headers.
        return {'status': 'http_error', 'http_status': exc.code, 'error_code': code,
                'reference_cost': None if profile['kind'] == 'api' else 0.,
                'currency': profile['currency'], 'elapsed_seconds': time.monotonic() - start,
                'quota_or_rate_limit': exc.code in (402, 429) or any(
                    x in (code or '').lower() for x in ('quota', 'balance', 'arrear', 'limit'))}
    except Exception as exc:
        return {'status': 'unknown', 'error_type': type(exc).__name__,
                'reference_cost': None if profile['kind'] == 'api' else 0.,
                'currency': profile['currency'], 'elapsed_seconds': time.monotonic() - start}


def score_result(task, result):
    if result['status'] != 'ok':
        return {'quality': 0., 'outcome': result['status'], 'assessment': None}
    if result['finish_reason'] == 'length':
        return {'quality': 0., 'outcome': 'truncated', 'assessment': None}
    if result['finish_reason'] != 'stop':
        return {'quality': 0., 'outcome': 'unexpected_finish', 'assessment': None}
    try:
        assessment = evaluate(task, result['text'])
    except Exception as exc:
        # Preserve the paid provider receipt even when the local scorer fails.
        return {'quality': None, 'outcome': 'scoring_error', 'assessment': None,
                'error_type': type(exc).__name__}
    return {'quality': assessment['quality'], 'outcome': 'complete', 'assessment': assessment}


class Ledger:
    """One process/one writer, native-currency in-flight reservations, hash chain."""
    def __init__(self, directory, frozen, caps, max_calls, max_seconds, resume=False):
        self.directory = Path(directory)
        existed = self.directory.exists()
        require(not existed or resume, 'existing run requires explicit --resume')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock_file = (self.directory / 'run.lock').open('a+')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.mutex = threading.RLock()
        self.caps = caps
        self.max_calls, self.max_seconds = max_calls, max_seconds
        self.started = time.monotonic()
        self.previous_elapsed = 0.
        self.pending, self.starts, self.results = {}, {}, {}
        self.spent = {k: 0. for k in caps}
        self.stopped = False
        self.seq, self.previous_hash = 0, '0' * 64
        frozen_path = self.directory / 'frozen.json'
        if existed:
            require(json.loads(frozen_path.read_text()) == frozen, 'frozen protocol/model/data changed')
            for line in (self.directory / 'events.jsonl').read_text().splitlines():
                event = json.loads(line)
                core = {k: v for k, v in event.items() if k != 'hash'}
                require(event['seq'] == self.seq and event['previous_hash'] == self.previous_hash
                        and digest(core) == event['hash'], 'invalid event hash chain')
                self.seq += 1
                self.previous_hash = event['hash']
                self.previous_elapsed = max(self.previous_elapsed, event['elapsed_seconds'])
                if event['kind'] == 'call_started':
                    require(event['id'] not in self.starts, 'duplicate call start')
                    self.starts[event['id']] = event['payload']
                elif event['kind'] == 'call_finished':
                    require(event['id'] not in self.results, 'duplicate call receipt')
                    self.results[event['id']] = event['payload']
                    amount = event['payload']['response'].get('reference_cost')
                    require(amount is not None, 'unknown previous cost; manual reconciliation required')
                    self.spent[event['payload']['response']['currency']] += amount
            require(set(self.starts) == set(self.results), 'unfinished call; automatic repeat refused')
            require(all(r['response']['status'] == 'ok' and r['score']['outcome'] in ('complete', 'truncated')
                        for r in self.results.values()), 'prior failed request requires explicit investigation')
        else:
            atomic_json(frozen_path, frozen)
        self.stream = (self.directory / 'events.jsonl').open('a')

    def elapsed(self):
        return self.previous_elapsed + time.monotonic() - self.started

    def append(self, kind, identifier, payload):
        with self.mutex:
            core = {'seq': self.seq, 'previous_hash': self.previous_hash,
                    'elapsed_seconds': self.elapsed(), 'kind': kind, 'id': identifier, 'payload': payload}
            event = {**core, 'hash': digest(core)}
            self.stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.previous_hash = event['hash']
            self.seq += 1

    def begin(self, identifier, record, currency, bound):
        with self.mutex:
            require(not self.stopped, 'run stopped')
            if identifier in self.results:
                require(self.starts[identifier] == record, 'receipt reused with changed input')
                return self.results[identifier]
            require(identifier not in self.starts, 'unfinished request cannot be repeated')
            require(self.elapsed() < self.max_seconds, 'wall-time ceiling reached')
            require(len(self.starts) < self.max_calls, 'call ceiling reached')
            reserved = sum(value[1] for value in self.pending.values() if value[0] == currency)
            require(self.spent[currency] + reserved + bound <= self.caps[currency] + 1e-12,
                    'native-currency budget ceiling reached')
            self.append('call_started', identifier, record)
            self.starts[identifier] = record
            self.pending[identifier] = (currency, bound)
            return None

    def finish(self, identifier, record):
        with self.mutex:
            require(identifier in self.pending, 'unreserved receipt')
            self.append('call_finished', identifier, record)
            self.results[identifier] = record
            currency, bound = self.pending.pop(identifier)
            cost = record['response'].get('reference_cost')
            if cost is None or not math.isfinite(cost) or cost < 0:
                self.stopped = True
            else:
                self.spent[currency] += cost
                if cost > bound + 1e-12 or self.spent[currency] > self.caps[currency] + 1e-12:
                    self.stopped = True
            if record['response']['status'] != 'ok' or record['score']['outcome'] not in ('complete', 'truncated'):
                self.stopped = True

    def close(self):
        self.stream.close()
        fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        self.lock_file.close()


def freeze(plan_path):
    config = json.loads(Path(plan_path).read_text())
    require(all(p['thinking'] is False for p in config['models'].values()), 'non-thinking phase only')
    manifest = json.loads((ROOT / config['manifest']).read_text())
    require(file_hash(ROOT / config['data']) == manifest['tasks_sha256'], 'data changed')
    tasks = load_tasks(ROOT / config['data'])
    require(len(tasks) == manifest['count'] == 140, 'incorrect screening sample size')
    forbidden = set()
    for path, h in manifest['excluded_prior_files'].items():
        require(file_hash(ROOT / path) == h, 'previous data changed')
        forbidden.update(question_hash(t.question) for t in load_tasks(ROOT / path) if t.dataset == 'math')
    require(not forbidden & {question_hash(t.question) for t in tasks}, 'previous question overlap')
    require(all(t.role == 'evaluation' and t.split == 'train' for t in tasks), 'invalid screening role')
    for entry in manifest['source_files']:
        require(file_hash(ROOT / entry['path']) == entry['sha256'], 'source data changed')
    identity_path = ROOT / 'setup/local_model_identity.json'
    require(identity_path.is_file(), 'local model identity not frozen')
    identity = json.loads(identity_path.read_text())
    # Hashes were produced before readiness testing; recheck identity metadata here.
    for name, h in identity['configuration_sha256'].items():
        require(file_hash(Path(identity['snapshot_path']) / name) == h, 'local model configuration changed')
    frozen = {'config': config, 'manifest_sha256': file_hash(ROOT / config['manifest']),
              'data_sha256': file_hash(ROOT / config['data']), 'local_identity': identity,
              'code': code_identity(ROOT), 'runner_sha256': file_hash(Path(__file__)),
              'config_sha256': file_hash(Path(plan_path))}
    jobs = []
    for task in tasks:
        for sample in range(config['samples_per_task']):
            groups = sorted(config['groups'], key=lambda group: digest([config['seed'], task.id, sample, group]))
            for group in groups:
                jobs.append({'id': f'v070:{task.id}:{sample}:{group}', 'task_id': task.id,
                             'sample': sample, 'group': group})
    require(len(jobs) == config['max_calls'], 'call count differs from declared plan')
    return config, tasks, jobs, frozen


def summarize(config, tasks, ledger):
    import numpy as np
    groups, values = [], {}
    for group in config['groups']:
        rows = [r for r in ledger.results.values() if r['job']['group'] == group]
        require(len(rows) == len(tasks) * config['samples_per_task'], 'incomplete group')
        lookup = {(r['job']['task_id'], r['job']['sample']): r['score']['quality'] for r in rows}
        values[group] = np.array([[lookup[t.id, s] for s in range(config['samples_per_task'])] for t in tasks])
        profile = config['models'][config['groups'][group]['model']]
        groups.append({'id': group, 'correct': float(values[group].sum()), 'executions': len(rows),
                       'accuracy': float(values[group].mean()),
                       'sample_accuracies': values[group].mean(axis=0).tolist(),
                       'currency': profile['currency'],
                       'reference_cost_per_1000_answers': sum(r['response']['reference_cost'] for r in rows) / len(rows) * 1000,
                       'mean_call_seconds': float(np.mean([r['response']['elapsed_seconds'] for r in rows])),
                       'p50_call_seconds': float(np.median([r['response']['elapsed_seconds'] for r in rows])),
                       'p95_call_seconds': float(np.percentile([r['response']['elapsed_seconds'] for r in rows], 95)),
                       'mean_input_tokens': float(np.mean([r['response']['input_tokens'] for r in rows])),
                       'mean_output_tokens': float(np.mean([r['response']['output_tokens'] for r in rows])),
                       'truncated': sum(r['score']['outcome'] == 'truncated' for r in rows),
                       'sample_correctness_flips': int((values[group][:, 0] != values[group][:, 1]).sum())})
    contrasts = []
    ncomparisons = len(config['primary_contrasts'])
    for index, (a, b) in enumerate(config['primary_contrasts']):
        delta = (values[a] - values[b]).mean(axis=1)
        rng = np.random.default_rng(config['seed'] + index)
        boot = delta[rng.integers(len(tasks), size=(10000, len(tasks)))].mean(axis=1) * 100
        alpha = 2.5 / ncomparisons
        contrasts.append({'a': a, 'b': b, 'delta_pp': float(delta.mean() * 100),
                          'paired_question_bootstrap_95pct_pp': np.percentile(boot, [2.5, 97.5]).tolist(),
                          'multiplicity_sensitivity_interval_pp': np.percentile(boot, [alpha, 100-alpha]).tolist()})
    complementary = []
    names = list(config['groups'])[:4]
    for index, a in enumerate(names):
        for b in names[index+1:]:
            x, y = values[a], values[b]
            complementary.append({'a': a, 'b': b, 'both_correct': int(((x == 1) & (y == 1)).sum()),
                                  'a_only': int(((x == 1) & (y == 0)).sum()),
                                  'b_only': int(((x == 0) & (y == 1)).sum()),
                                  'both_wrong': int(((x == 0) & (y == 0)).sum()),
                                  'note': 'paired independent draws; descriptive, not an executable router'})
    return {'status': 'complete', 'tasks': len(tasks), 'executions': len(ledger.results),
            'groups': groups, 'contrasts': contrasts, 'complementarity': complementary,
            'reference_spend_by_currency': ledger.spent, 'elapsed_seconds': ledger.elapsed(),
            'validation_not_final_test': True, 'learning_updates': 0,
            'statistical_note': 'Average of two draws, clustered by question; not best-of-two or evidence of 1pp non-inferiority.'}


def write_report(directory, result):
    lines = ['# 三档模型与 DeepSeek 基线选型验证', '',
             '140道新题，每组每题两次独立采样，全部平均。四个模型使用4096输出上限，另有DeepSeek 2048对照。', '',
             '| 方案 | 准确率 | 标准费用/千次作答 | 平均调用秒 | 截断 |',
             '|---|---:|---:|---:|---:|']
    for g in result['groups']:
        lines.append(f"| {g['id']} | {g['accuracy']:.2%} | {g['reference_cost_per_1000_answers']:.4f} {g['currency']} | {g['mean_call_seconds']:.2f} | {g['truncated']}/{g['executions']} |")
    lines += ['', '| 比较 | 差值百分点 | 按题配对95%区间 |', '|---|---:|---|']
    for c in result['contrasts']:
        lo, hi = c['paired_question_bootstrap_95pct_pp']
        lines.append(f"| {c['a']} − {c['b']} | {c['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', '费用为冻结未缓存标准/高峰费率的参照值，不是发票；人民币与美元分列，未暗设汇率。本地API支出为0，耗时单列。',
              '时间不含客户端队列等待；并发API调用时间不等于独占服务延迟。截断计0，无自动重试。',
              '这是模型选型验证，不是最终测试。两次响应不能当成两道独立题；五项主要比较的多重比较敏感性区间见results.json。',
              '同采样编号的模型回答独立生成。互补计数不代表可以根据未知正确答案部署路由。']
    (Path(directory) / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def run(plan_path, directory, resume=False):
    config, tasks, jobs, frozen = freeze(plan_path)
    secret = credentials(config['credentials_file'])
    ledger = Ledger(directory, frozen, config['caps'], config['max_calls'], config['max_seconds'], resume)
    taskmap = {t.id: t for t in tasks}
    gates = {'api': threading.BoundedSemaphore(config['api_concurrency']),
             'local': threading.BoundedSemaphore(config['local_concurrency'])}
    atomic_json(Path(directory) / 'plan.json', jobs)

    def progress():
        with ledger.mutex:
            result = {'completed': len(ledger.results), 'total': len(jobs),
                      'groups_completed': dict(Counter(r['job']['group'] for r in ledger.results.values())),
                      'reference_spend': dict(ledger.spent), 'elapsed_seconds': ledger.elapsed(),
                      'pending': len(ledger.pending), 'stopped': ledger.stopped}
            atomic_json(Path(directory) / 'progress.json', result)
            print(json.dumps(result), flush=True)

    def execute(job):
        group = config['groups'][job['group']]
        profile = config['models'][group['model']]
        task = taskmap[job['task_id']]
        seed = int(digest([config['seed'], job['id']])[:8], 16) % (2**31)
        payload, input_bound = request_payload(config, profile, task.model_input(), group, seed)
        request = {'job': job, 'payload': payload, 'profile': group['model']}
        bound = reference_charge(profile, input_bound, group['max_tokens'])
        with gates[profile['kind']]:
            prior = ledger.begin(job['id'], request, profile['currency'], bound)
            if prior is not None:
                return
            response = call_model(profile, payload, secret.get(profile.get('key_environment')),
                                  timeout=min(profile['timeout'], max(1, config['max_seconds']-ledger.elapsed())))
            score = score_result(task, response)
            ledger.finish(job['id'], {'job': job, 'response': response, 'score': score})
            require(not ledger.stopped, 'request failed, quota exhausted, unknown cost, or protocol mismatch; see receipt')

    pending = iter(jobs)
    try:
        with ThreadPoolExecutor(max_workers=config['workers']) as pool:
            active = set()
            for _ in range(config['workers']):
                job = next(pending, None)
                if job:
                    active.add(pool.submit(execute, job))
            try:
                while active:
                    done, active = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                    if len(ledger.results) % 20 < len(done) or not active:
                        progress()
                    for _ in done:
                        job = next(pending, None)
                        if job:
                            active.add(pool.submit(execute, job))
            except BaseException:
                with ledger.mutex:
                    ledger.stopped = True
                for future in active:
                    future.cancel()
                raise
        require(freeze(plan_path)[-1] == frozen, 'frozen inputs changed during execution')
        result = summarize(config, tasks, ledger)
        ledger.append('run_finished', 'main', result)
        atomic_json(Path(directory) / 'results.json', result)
        write_report(directory, result)
        atomic_json(Path(directory) / 'audit.json', {'events_sha256': file_hash(Path(directory) / 'events.jsonl'),
                    'frozen_digest': digest(frozen), 'calls_started': len(ledger.starts),
                    'calls_finished': len(ledger.results), 'unknown_calls': sorted(set(ledger.starts)-set(ledger.results)),
                    'learning_updates': 0, 'completed': True})
        progress()
        return result
    except BaseException as exc:
        # Do not include exception messages from third-party libraries in status.
        atomic_json(Path(directory) / 'stopped.json', {'error_type': type(exc).__name__,
                    'completed': len(ledger.results), 'unknown_calls': sorted(set(ledger.starts)-set(ledger.results)),
                    'last_failure_ids': [k for k, r in ledger.results.items() if r['response']['status'] != 'ok'],
                    'reference_spend': ledger.spent})
        progress()
        raise
    finally:
        ledger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/model_selection_v070.plan.json'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.execute:
        try:
            result = run(args.config, args.out_dir, args.resume)
            print(json.dumps({'status': result['status'], 'executions': result['executions']}))
        except Exception as exc:
            print(json.dumps({'status': 'stopped', 'error_type': type(exc).__name__,
                              'detail': 'inspect stopped.json and recorded receipt; no automatic retry'}), flush=True)
            raise SystemExit(2)
    else:
        cfg, tasks, jobs, frozen = freeze(args.config)
        print(json.dumps({'tasks': len(tasks), 'calls': len(jobs), 'caps': cfg['caps'],
                          'frozen_digest': digest(frozen), 'new_model_calls': 0}))
