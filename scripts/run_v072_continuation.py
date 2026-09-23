#!/usr/bin/env python3
"""Bounded v072 search, held-out selection and frozen evaluation.

Phase A is read-only. This separate journal links to its audited receipts.
No generated Python is executed here: code answers use the existing sandbox.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import re
import signal
import sqlite3
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from nicheflow.datasets import load_tasks
from nicheflow.ledger import atomic_json
from nicheflow.spec import digest, file_hash
from nicheflow.v072.durable import Store, Paused
from scripts.model_selection_v070 import call_model, credentials
from scripts.run_v072 import payload as baseline_payload, run_score

DOMAINS = ['math', 'mbpp', 'hotpotqa']
EDITS = [['model_swap', 'prompt_edit'], ['delete_merge', 'rewire'],
         ['model_swap', 'add'], ['delete_merge', 'prompt_edit'], ['rewire', 'prompt_edit']]
POLICY = {
    'version': 'v072-continuation-1', 'rounds': 5, 'slots': 2, 'proposal_attempts': 2,
    'max_nodes': 3, 'max_high_calls': 1, 'feedback_tasks': 20, 'samples': 2,
    'ranking': 'empirical quality within 0.01 of best, then mean CNY API cost, node count, ID',
    'selection_tolerance': .01, 'ridge': 10., 'router': 'four public features, per-arm ridge means; no UCB',
    'cost': 'frozen CNY API tariffs for L/M/H; GPU time separate; D excluded from selection',
    'parent': 'deterministic round-robin occupied structure/model niches, elite quality then cost',
    'edits': EDITS, 'upstream_failure': 'stop on truncation; do not silently clip task or upstream text',
    'math_audit': 'retain frozen strict Math-Verify scores; inspect nonliteral labels at phase gates',
    'gates': ['seeds', 'search', 'screening', 'calibration', 'final'],
    'final_is_sealed_until': 'candidate identities and calibration-only router parameters committed',
}


def canonical(graph):
    """Normalize node identifiers so renaming is not a new workflow."""
    mapping = {n['id']: f'n{i}' for i, n in enumerate(graph['nodes'])}
    return {'nodes': [{**n, 'id': mapping[n['id']], 'inputs': sorted(mapping[x] for x in n['inputs'])}
                      for n in graph['nodes']], 'output': mapping[graph['output']]}


def validate(graph):
    if not isinstance(graph, dict) or set(graph) != {'nodes', 'output'}:
        raise ValueError('graph must contain only nodes and output')
    nodes = graph['nodes']
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 3:
        raise ValueError('one to three nodes required')
    seen = set()
    for n in nodes:
        if set(n) != {'id', 'model', 'role', 'prompt', 'inputs'}:
            raise ValueError('invalid node fields')
        if not isinstance(n['id'], str) or not re.fullmatch('[a-z][a-z0-9_]{0,30}', n['id']) or n['id'] in seen:
            raise ValueError('invalid node ID')
        if n['model'] not in ('L', 'M', 'H') or n['role'] not in ('plan', 'solve', 'review'):
            raise ValueError('invalid model or role')
        if not isinstance(n['prompt'], str) or len(n['prompt']) > 2400:
            raise ValueError('invalid prompt')
        if not isinstance(n['inputs'], list) or len(set(n['inputs'])) != len(n['inputs']) or not set(n['inputs']) <= seen:
            raise ValueError('nodes must be topologically ordered')
        if n['role'] == 'review' and not n['inputs']:
            raise ValueError('review requires upstream material')
        seen.add(n['id'])
    if sum(n['model'] == 'H' for n in nodes) > 1:
        raise ValueError('at most one H call')
    if graph['output'] != nodes[-1]['id'] or nodes[-1]['role'] == 'plan':
        raise ValueError('last node must produce the final answer')
    needed = {graph['output']}
    for n in reversed(nodes):
        if n['id'] in needed: needed.update(n['inputs'])
    if needed != seen:
        raise ValueError('dead nodes forbidden')
    return canonical(graph)


def node(model, role, prompt='', inputs=(), name='n0'):
    return {'id': name, 'model': model, 'role': role, 'prompt': prompt, 'inputs': list(inputs)}


def seeds():
    singles = {m: {'nodes': [node(m, 'solve')], 'output': 'n0'} for m in ('L', 'M', 'H')}
    mixed = {
        'fixed_LM': {'nodes': [node('L', 'solve'), node('M', 'review', inputs=['n0'], name='n1')], 'output': 'n1'},
        'fixed_MH': {'nodes': [node('M', 'solve'), node('H', 'review', inputs=['n0'], name='n1')], 'output': 'n1'},
        'fixed_HL': {'nodes': [node('H', 'plan'), node('L', 'solve', inputs=['n0'], name='n1')], 'output': 'n1'},
    }
    return {k: validate(v) for k, v in {**singles, **mixed}.items()}


def workflow_payload(config, task, n, outputs, sample, final):
    request, _ = baseline_payload(config, config['models'][n['model']], task, sample)
    public = task.model_input()
    if not final:
        public = {**public, 'output_contract': ('Return a concise solution plan, not a final answer.'
                  if n['role'] == 'plan' else public['output_contract'])}
    role = {
        'plan': 'Produce a concise, actionable plan using the original task. Do not give the final answer.',
        'solve': 'Solve the original task independently. Use upstream material only after checking it.',
        'review': 'Independently verify the upstream draft against the original task. Preserve correct content; change it only when you identify a concrete error. Return the complete final answer, not a critique.',
    }[n['role']]
    instruction = role + ' ' + n['prompt'] + ' Keep reasoning concise. Stop after the requested output.'
    request['messages'][1]['content'] = instruction + '\n' + json.dumps({
        'task': public, 'upstream_outputs': {k: outputs[k] for k in n['inputs']}}, ensure_ascii=False)
    bound = sum(len(m['content'].encode()) + 64 for m in request['messages']) + 1024
    return request, bound


def features(task):
    q = task.question
    context = json.dumps(task.context, ensure_ascii=False) if task.context else ''
    return [1., np.log1p(len(q)) / 10., sum(c.isdigit() for c in q) / max(1, len(q)),
            np.log1p(len(context)) / 12.]


def rank(stats, tolerance=.01):
    best = max(s['quality'] for s in stats)
    return sorted(stats, key=lambda s: (s['quality'] < best - tolerance - 1e-12,
                  s['cost'] if s['quality'] >= best - tolerance - 1e-12 else -s['quality'],
                  s['nodes'], s['id']))


def fit_router(tasks, arms, records, ridge=10.):
    x = np.asarray([features(t) for t in tasks]); penalty = np.diag([0., ridge, ridge, ridge])
    weights = {}
    for arm in arms:
        rows = {(r['task_id'], r['sample']): r for r in records if r['candidate'] == arm}
        q = np.array([np.mean([rows[(t.id, s)]['score']['quality'] for s in range(2)]) for t in tasks])
        cost = np.array([np.mean([rows[(t.id, s)]['cost_cny'] for s in range(2)]) for t in tasks])
        weights[arm] = {'quality': np.linalg.solve(x.T @ x + penalty, x.T @ q).tolist(),
                        'cost': np.linalg.solve(x.T @ x + penalty, x.T @ cost).tolist(),
                        'mean_quality': float(q.mean()), 'mean_cost': float(cost.mean())}
    return weights


class Experiment:
    def __init__(self, config, directory, parent, audit, resume=False):
        self.config = config
        self.parent = sqlite3.connect(f'file:{parent}?mode=ro', uri=True, check_same_thread=False)
        self.parent_lock = threading.RLock()
        assert self.parent.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        parent_frozen = json.loads(self.parent.execute("SELECT value FROM meta WHERE key='frozen'").fetchone()[0])
        assert parent_frozen['config'] == config, 'Phase A model/request configuration changed'
        for path, h in parent_frozen['code'].items():
            assert file_hash(ROOT/path) == h, 'Phase A scoring or execution code changed: '+path
        assert file_hash(ROOT/'requirements.lock') == parent_frozen['readiness']['runtime_lock_sha256']
        assert self.parent.execute("SELECT count(*) FROM attempts WHERE status='complete'").fetchone()[0] == 972
        assert self.parent.execute("SELECT count(*) FROM attempts WHERE status!='complete'").fetchone()[0] == 0
        adjudications = json.loads(Path(audit).read_text())
        assert adjudications['source_database_sha256'] == file_hash(parent)
        assert adjudications['all_31_reviewed'] and adjudications['label_changes'] == 0
        self.manifest = json.loads((ROOT / config['manifest']).read_text())
        for path, h in self.manifest['task_files'].items():
            assert file_hash(ROOT / path) == h
        spent = {'CNY': 0., 'USD': 0.}; used = {m: [0, 0, 0] for m in config['models']}
        for model, raw in self.parent.execute('SELECT c.model,a.response FROM calls c JOIN attempts a ON a.call_id=c.id'):
            r = json.loads(raw); spent[r['currency']] += r['reference_cost']
            used[model][0] += 1; used[model][1] += r['input_tokens']; used[model][2] += r['output_tokens']
        limits = {'currency': {k: config['limits']['currency'][k] - v for k, v in spent.items()},
                  'models': {m: {'calls': (4050 if m == 'H' else 15000) - u[0],
                                    'input_tokens': 20000000 - u[1], 'output_tokens': 10000000 - u[2]}
                             for m, u in used.items()}}
        sources = [Path(__file__), ROOT/'scripts/run_v072.py', ROOT/'scripts/model_selection_v070.py',
                   ROOT/'scripts/v072_code_worker.py', ROOT/'scripts/v072_code_sandbox.py',
                   ROOT/'scripts/score_v072_response.py', *sorted((ROOT/'nicheflow').rglob('*.py'))]
        frozen = {'policy': POLICY, 'config': config, 'limits': limits, 'parent_sha256': file_hash(parent),
                  'audit_sha256': file_hash(audit), 'parent_spend': spent,
                  'manifest_sha256': file_hash(ROOT/config['manifest']),
                  'source_hashes': {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                  'runtime_lock_sha256': file_hash(ROOT/'requirements.lock')}
        self.store = Store(directory, frozen, resume=resume)
        self.load_audits()
        self.keys = credentials(config['credentials_file'])
        self.local = threading.BoundedSemaphore(1); self.api = threading.BoundedSemaphore(4)
        self.scorers = threading.BoundedSemaphore(2); self.stop = threading.Event()
        self.start = time.monotonic(); self.completed = 0
        for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, lambda *_: self.stop.set())

    def load_audits(self):
        """Materialize explicit case decisions while retaining immutable raw labels."""
        for path in sorted(self.store.directory.glob('audit_*.approved.json')):
            phase = path.name[len('audit_'):-len('.approved.json')]
            queue_path = self.store.directory / f'audit_{phase}.json'
            if not queue_path.exists(): continue
            queue = json.loads(queue_path.read_text()); approval = json.loads(path.read_text())
            if approval.get('queue_sha256') != digest(queue): continue
            decisions = approval.get('reviewed_rows', {})
            if set(decisions) != {r['artifact_key'] for r in queue}:
                raise ValueError('audit must cover every queued answer')
            for r in queue:
                key = r['artifact_key']; decision = decisions[key]
                assert decision['quality'] in (0., 1.) and decision.get('reason')
                raw = self.store.artifact(key)
                assert raw and raw == {k: v for k, v in r.items() if k != 'artifact_key'}
                old = self.store.artifact('adjudication/' + key)
                if old:
                    assert old['quality'] == decision['quality'], 'previously used audit label changed'
                else:
                    self.store.artifact('adjudication/' + key, decision, write=True)

    def effective(self, key, record):
        decision = self.store.artifact('adjudication/' + key)
        if decision:
            record = copy.deepcopy(record)
            record['score']['raw_quality'] = record['score']['quality']
            record['score']['quality'] = decision['quality']
            record['score']['audit_reason'] = decision['reason']
            record['score']['audit_required'] = False
        return record

    def tasks(self, domain, split):
        return load_tasks(ROOT / f'data/multidomain_v072/{domain}/{split}.jsonl')

    def response(self, ident, model, request, bound, stage):
        profile = self.config['models'][model]
        if bound > profile['max_input_bound']: raise ValueError('input exceeds frozen bound')
        with self.local if model == 'L' else self.api:
            if self.stop.is_set(): self.store.pause('requested_stop')
            if time.monotonic() - self.start > self.config['max_session_seconds']:
                self.store.pause('session_time_limit')
            # SQLite connection reads must share the writer mutex as well.
            with self.store.mutex:
                old = self.store.receipt(ident)
            if old is not None: return old
            if model == 'L':
                token_request = {'model': profile['model'], 'messages': request['messages'],
                                 'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}
                req = urllib.request.Request('http://127.0.0.1:8070/tokenize', data=json.dumps(token_request).encode(),
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=20) as r: count = json.load(r)['count']
                if count + request['max_tokens'] > 24576:
                    raise ValueError('local_context_overflow_no_silent_truncation')
            value = self.store.begin(ident, model, request, profile, stage, bound, request['max_tokens'])
            if value is None:
                value = call_model(profile, request, self.keys.get(profile['key_environment']))
                self.store.finish(ident, value)
            if value['status'] != 'ok': raise Paused('provider failure; receipts preserved')
            return value

    def baseline(self, stage, task, model, sample):
        if stage == 'development':
            key = f'answer/development/{task.dataset}/{task.id}/{model}/{sample}'
            with self.parent_lock:
                r = json.loads(self.parent.execute('SELECT value FROM artifacts WHERE key=?', (key,)).fetchone()[0])
                response = json.loads(self.parent.execute('SELECT response FROM attempts WHERE call_id=?', (r['call_id'],)).fetchone()[0])
            return {'candidate': model, 'domain': task.dataset, 'task_id': task.id, 'sample': sample,
                    'score': r['score'], 'calls': [r['call_id']], 'parent_receipt': True,
                    'cost_cny': response['reference_cost'] if model != 'D' else None,
                    'currency': response['currency'], 'reference_cost': response['reference_cost'],
                    'elapsed_seconds': response['elapsed_seconds']}
        return self.execute(stage, task, model, None, sample)

    def execute(self, stage, task, candidate, graph, sample):
        ident = f'{stage}/{task.dataset}/{task.id}/{candidate}/{sample}'
        key = 'answer/' + ident
        old = self.store.artifact(key)
        if old is not None: return self.effective(key, old)
        responses = []; calls = []; outputs = {}; score = None
        if graph is None:
            request, bound = baseline_payload(self.config, self.config['models'][candidate], task, sample)
            cid = ident + '/single'
            response = self.response(cid, candidate, request, bound, stage)
            responses.append(response); calls.append(cid)
        else:
            for n in graph['nodes']:
                request, bound = workflow_payload(self.config, task, n, outputs, sample, n['id'] == graph['output'])
                cid = ident + '/' + n['id']
                response = self.response(cid, n['model'], request, bound, stage)
                responses.append(response); calls.append(cid); outputs[n['id']] = response['text']
                if response['finish_reason'] == 'length':
                    score = {'quality': 0., 'outcome': 'upstream_truncated' if n['id'] != graph['output'] else 'truncated', 'audit_required': False}
                    break
                if response['finish_reason'] != 'stop': raise ValueError('unexpected completion reason')
        if score is None:
            with self.scorers: score = run_score(task, responses[-1], self.config)
        currencies = set(r['currency'] for r in responses)
        assert len(currencies) == 1
        r = {'candidate': candidate, 'domain': task.dataset, 'task_id': task.id, 'sample': sample,
             'score': score, 'calls': calls, 'parent_receipt': False,
             'cost_cny': sum(r['reference_cost'] for r in responses) if currencies == {'CNY'} else None,
             'currency': responses[0]['currency'], 'reference_cost': sum(r['reference_cost'] for r in responses),
             'elapsed_seconds': sum(r['elapsed_seconds'] for r in responses)}
        self.store.artifact(key, r, write=True)
        with self.store.mutex:
            self.completed += 1
            if self.completed % 20 == 0:
                self.store.backup()
                print(json.dumps({'stage': stage, 'session_answers': self.completed}), flush=True)
            self.store.progress(stage)
        return self.effective(key, r)

    def matrix(self, stage, domain, tasks, candidates, samples):
        jobs = [(t, name, graph, s) for t in tasks for name, graph in candidates.items() for s in range(samples)]
        jobs.sort(key=lambda x: digest([self.config['seed'], stage, x[0].id, x[1], x[3]]))
        records = []; errors = []
        with ThreadPoolExecutor(10) as pool:
            futures = [pool.submit(self.baseline, stage, t, n, s) if g is None
                       else pool.submit(self.execute, stage, t, n, g, s) for t, n, g, s in jobs]
            for f in as_completed(futures):
                if f.cancelled(): continue
                try: records.append(f.result())
                except Exception as exc:
                    errors.append(type(exc).__name__ + ':' + str(exc)[:160])
                    self.store.pause('worker_error:' + errors[-1])
                    for pending in futures: pending.cancel()
        self.store.backup()
        if errors: raise Paused(errors[0])
        assert len(records) == len(jobs)
        return records

    def stats(self, name, graph, records):
        rows = sorted((r for r in records if r['candidate'] == name), key=lambda r: (r['task_id'], r['sample']))
        return {'id': name, 'quality': float(np.mean([r['score']['quality'] for r in rows])),
                'cost': float(np.mean([r['cost_cny'] for r in rows])), 'nodes': len(graph['nodes']),
                'truncated': sum('truncated' in r['score']['outcome'] for r in rows),
                'n': len(rows), 'niche': self.niche(graph)}

    @staticmethod
    def niche(graph):
        return str(len(graph['nodes'])) + ':' + ''.join(sorted(set(n['model'] for n in graph['nodes']))) + ':' + (
            'branch' if sum(not n['inputs'] for n in graph['nodes']) > 1 else 'chain')

    def propose(self, domain, round_id, slot, graphs, stats):
        ident = f'proposal/{domain}/{round_id}/{slot}'
        old = self.store.artifact(ident)
        if old is not None: return old
        edit = EDITS[round_id][slot]
        niches = {}
        for s in stats: niches.setdefault(s['niche'], []).append(s)
        eligible = sorted(niches)
        if edit == 'delete_merge': eligible = [n for n in eligible if not n.startswith('1:')] or eligible
        selected = eligible[(round_id * 2 + slot) % len(eligible)]
        parent = rank(niches[selected], 0.)[0]['id']
        schema = {'nodes': [node('M', 'solve')], 'output': 'n0'}
        last_error = None
        for attempt in range(2):
            request, _ = baseline_payload(self.config, self.config['models']['D'], self.tasks(domain, 'development')[0], attempt)
            request['messages'] = [
                {'role': 'system', 'content': 'Design a reusable workflow, not a solution to a particular task. Return only one JSON object with nodes and output. No code execution or tools are available.'},
                {'role': 'user', 'content': json.dumps({
                    'domain': domain, 'requested_edit': edit, 'parent': graphs[parent],
                    'parent_statistics': next(s for s in stats if s['id'] == parent),
                    'archive_statistics': stats, 'schema': schema,
                    'rules': ['1 to 3 LLM nodes; at most one H node; L/M/H models only.',
                              'Node fields exactly id/model/role/prompt/inputs. IDs in topological order; last node is output; no dead nodes.',
                              'role is plan, solve or review; review requires upstream input; final role cannot be plan.',
                              'Keep prompts task-independent and under 2400 characters. No task-specific answers, hidden tests, labels or scoring code.',
                              'Improve a useful quality-cost tradeoff. Deletion, cheaper model replacement and concise prompts are valid.',
                              'Make a meaningful change, not just renamed IDs. Keep unchanged node prompts exactly.'],
                    'previous_validation_error': last_error}, ensure_ascii=False)}]
            bound = sum(len(m['content'].encode()) + 64 for m in request['messages']) + 1024
            response = self.response(ident + f'/{attempt}', 'D', request, bound, 'proposal')
            try:
                if response['finish_reason'] != 'stop': raise ValueError('proposal truncated')
                text = response['text'].strip()
                if text.startswith('```'): text = text.split('\n', 1)[1].rsplit('```', 1)[0]
                graph = validate(json.loads(text))
                if digest(graph) in {digest(g) for g in graphs.values()}: raise ValueError('duplicate graph')
                result = {'valid': True, 'graph': graph, 'parent': parent, 'requested_edit': edit,
                          'attempts': attempt + 1, 'id': f'search_r{round_id+1}_s{slot+1}'}
                break
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                last_error = str(exc)[:180]
                self.store.artifact(ident + f'/invalid/{attempt}', {'error': last_error}, write=True)
        else:
            result = {'valid': False, 'parent': parent, 'requested_edit': edit, 'attempts': 2, 'error': last_error}
        self.store.artifact(ident, result, write=True)
        return result

    def reference_checks(self, split):
        key = 'reference_checks/' + split
        if self.store.artifact(key): return
        # This only checks scoring infrastructure; scores never feed graph or router selection.
        for task in self.tasks('mbpp', split):
            result = run_score(task, {'status': 'ok', 'finish_reason': 'stop', 'text': '```python\n'+task.gold+'\n```'}, self.config)
            if result['quality'] != 1: raise ValueError('MBPP canonical reference failed: ' + task.id)
        self.store.artifact(key, {'passed': True}, write=True)

    def gate(self, phase):
        rows = []
        for key, raw in self.store.db.execute('SELECT key,value FROM artifacts WHERE key LIKE ?', ('answer/'+phase+'/%',)):
            r = json.loads(raw)
            if r['domain'] == 'math' and r['score'].get('audit_required'): rows.append({'artifact_key': key, **r})
        atomic_json(self.store.directory / f'audit_{phase}.json', rows)
        self.store.backup()
        approval = self.store.directory / f'audit_{phase}.approved.json'
        signature = digest(rows)
        if rows and (not approval.exists() or json.loads(approval.read_text()).get('queue_sha256') != signature):
            self.store.set_meta('pending_audit', {'phase': phase, 'queue_sha256': signature, 'count': len(rows)})
            raise Paused('audit gate: '+phase)
        if rows:
            self.load_audits()
            assert all(self.store.artifact('adjudication/' + r['artifact_key']) for r in rows)
        self.store.set_meta('pending_audit', None)
        self.store.artifact('gate/' + phase + '/' + signature, {'queue_sha256': signature, 'count': len(rows)}, write=True)

    def run(self):
        try:
            if self.store.unresolved(): raise Paused('unknown call outcomes require reconciliation')
            all_graphs = {}; all_stats = {}; feedback = {}
            for d in DOMAINS:
                ids = set(self.manifest['candidate_feedback_ids'][d])
                tasks = [t for t in self.tasks(d, 'development') if t.id in ids]
                assert len(tasks) == 20
                feedback[d] = tasks; graphs = seeds(); all_graphs[d] = graphs
                records = self.matrix('development', d, tasks, {m: None for m in ['L', 'M', 'H']}, 1)
                records += self.matrix('seeds', d, tasks, {n: g for n, g in graphs.items() if n.startswith('fixed_')}, 1)
                all_stats[d] = [self.stats(n, g, records) for n, g in graphs.items()]
            self.gate('seeds')
            for r in range(5):
                for d in DOMAINS:
                    parent_stats = list(all_stats[d])
                    for slot in range(2):
                        proposal = self.propose(d, r, slot, all_graphs[d], parent_stats)
                        if not proposal['valid']: continue
                        n, g = proposal['id'], proposal['graph']
                        rows = self.matrix('search', d, feedback[d], {n: g}, 1)
                        all_graphs[d][n] = g
                        all_stats[d].append(self.stats(n, g, rows))
                    if not self.store.artifact(f'round/{d}/{r}'):
                        self.store.artifact(f'round/{d}/{r}', {'statistics_before_audit': all_stats[d]}, write=True)
                self.store.progress('search', rounds_completed=r+1)
                self.gate('search')
            self.gate('search')
            self.store.artifact('search_summary', all_stats, write=True)
            for d in DOMAINS:
                fixed = rank([s for s in all_stats[d] if s['id'].startswith('fixed_')])[0]['id']
                pool = [s for s in all_stats[d] if s['id'].startswith('search_')]
                searched = []
                while pool and len(searched) < 2:
                    winner = rank(pool)[0]; searched.append(winner['id']); pool.remove(winner)
                if not searched: raise ValueError('no valid search candidate in '+d)
                self.store.artifact('screen_selection/'+d, {'fixed': fixed, 'searched': searched}, write=True)
            self.reference_checks('screening')
            for d in DOMAINS:
                selected = self.store.artifact('screen_selection/'+d)
                names = [selected['fixed'], *selected['searched']]
                candidates = {**{m: None for m in ['L', 'M', 'H', 'D']}, **{n: all_graphs[d][n] for n in names}}
                rows = self.matrix('screening', d, self.tasks(d, 'screening'), candidates, 2)
                stats = [self.stats(n, all_graphs[d][n], rows) for n in names]
                if not self.store.artifact('screen_statistics_before_audit/'+d):
                    self.store.artifact('screen_statistics_before_audit/'+d, stats, write=True)
            self.gate('screening')
            for d in DOMAINS:
                selected = self.store.artifact('screen_selection/'+d)
                names = [selected['fixed'], *selected['searched']]
                rows = self.matrix('screening', d, self.tasks(d, 'screening'), {n: all_graphs[d][n] for n in names}, 2)
                stats = [self.stats(n, all_graphs[d][n], rows) for n in names]
                self.store.artifact('screen_statistics/'+d, stats, write=True)
                searched = rank([s for s in stats if s['id'] in selected['searched']])[0]['id']
                self.store.artifact('deployment/'+d, {'fixed': selected['fixed'], 'searched': searched,
                    'graphs': {n: all_graphs[d][n] for n in [selected['fixed'], searched]}}, write=True)
            self.reference_checks('calibration')
            calibration_rows = {}
            for d in DOMAINS:
                deploy = self.store.artifact('deployment/'+d)
                candidates = {**{m: None for m in ['L', 'M', 'H', 'D']}, **deploy['graphs']}
                calibration_rows[d] = self.matrix('calibration', d, self.tasks(d, 'calibration'), candidates, 2)
            self.gate('calibration')
            routers = {}
            for d in DOMAINS:
                deploy = self.store.artifact('deployment/'+d)
                arms = ['L', 'M', 'H', deploy['fixed'], deploy['searched']]
                routers[d] = fit_router(self.tasks(d, 'calibration'), arms, calibration_rows[d])
            self.store.artifact('frozen_routers', routers, write=True)
            self.store.artifact('final_protocol_frozen', {'router_sha256': digest(routers),
                'deployments': {d: self.store.artifact('deployment/'+d) for d in DOMAINS}, 'policy': POLICY}, write=True)
            # No final tasks or answers have been loaded before the preceding durable commit.
            self.reference_checks('final')
            for d in DOMAINS:
                deploy = self.store.artifact('deployment/'+d)
                candidates = {**{m: None for m in ['L', 'M', 'H', 'D']}, **deploy['graphs']}
                self.matrix('final', d, self.tasks(d, 'final'), candidates, 2)
            self.gate('final')
            self.store.artifact('complete', {'status': 'all_planned_generations_complete'}, write=True)
            self.store.progress('complete'); self.store.backup()
            return 0
        except Exception as exc:
            self.store.pause(type(exc).__name__+':'+str(exc)[:220])
            self.store.progress('paused'); self.store.backup()
            print(json.dumps({'status': 'paused', 'reason': str(exc), 'receipts_preserved': True}), flush=True)
            return 2
        finally:
            self.store.close(); self.parent.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/multidomain_v072.json')
    parser.add_argument('--run-dir', default='runs/multidomain_v072_continuation')
    parser.add_argument('--parent', default='runs/multidomain_v072/checkpoint.sqlite3')
    parser.add_argument('--audit', default='setup/v072/development_audit.json')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text())
    experiment = Experiment(config, ROOT/args.run_dir, ROOT/args.parent, ROOT/args.audit, args.resume)
    raise SystemExit(experiment.run())
