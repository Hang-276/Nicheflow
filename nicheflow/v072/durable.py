"""Crash-safe call receipts, stage state, and concurrent token reservations."""
from __future__ import annotations
import fcntl
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from nicheflow.spec import digest


class Paused(RuntimeError):
    pass


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class Store:
    def __init__(self, directory, frozen, *, resume=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / 'writer.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.mutex = threading.RLock()
        self.db = sqlite3.connect(self.directory / 'state.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY, model TEXT NOT NULL,
                contract TEXT NOT NULL, stage TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts(call_id TEXT NOT NULL, number INTEGER NOT NULL,
                status TEXT NOT NULL, input_bound INTEGER NOT NULL, output_bound INTEGER NOT NULL,
                currency TEXT NOT NULL, cost_bound REAL NOT NULL, response TEXT,
                started REAL NOT NULL, ended REAL, PRIMARY KEY(call_id,number));
            CREATE TABLE IF NOT EXISTS artifacts(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                created REAL NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
        ''')
        old = self.get_meta('frozen')
        if old is not None:
            if not resume:
                self.close()
                raise ValueError('existing run requires --resume')
            if old != frozen:
                self.close()
                raise ValueError('frozen code/configuration/data changed; use an audited migration')
            # A process died while a request might have reached the provider.
            with self.db:
                self.db.execute("UPDATE attempts SET status='unknown' WHERE status='started'")
        else:
            self.set_meta('frozen', frozen)
            self.set_meta('created', time.time())
        self.frozen = frozen
        self.limits = frozen['limits']
        self.set_meta('session_started', time.time())
        if self.unresolved():
            self.pause('unknown_call_outcome_requires_reconciliation')
        elif resume:
            # Only explicit invocation resumes rejected requests; never an automatic loop.
            self.set_meta('pause', None)
            self.event('explicit_resume', {})

    def get_meta(self, key):
        row = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, key, value):
        with self.mutex, self.db:
            self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', (key, encoded(value)))

    def event(self, kind, payload):
        with self.mutex, self.db:
            self.db.execute('INSERT INTO events(created,kind,payload) VALUES(?,?,?)',
                            (time.time(), kind, encoded(payload)))

    def artifact(self, key, value=None, *, write=False):
        with self.mutex:
            row = self.db.execute('SELECT value FROM artifacts WHERE key=?', (key,)).fetchone()
            old = json.loads(row[0]) if row else None
            if write:
                if row and old != value:
                    raise ValueError('immutable artifact changed: ' + key)
                with self.db:
                    self.db.execute('INSERT OR IGNORE INTO artifacts VALUES(?,?)', (key, encoded(value)))
                return value
            return old

    def pause(self, reason):
        with self.mutex:
            if not self.get_meta('pause'):
                self.set_meta('pause', {'reason': reason, 'time': time.time()})
                self.event('paused', {'reason': reason})

    def unresolved(self):
        return [r[0] for r in self.db.execute("SELECT call_id FROM attempts WHERE status='unknown'")]

    def receipt(self, identifier):
        row = self.db.execute("SELECT response FROM attempts WHERE call_id=? AND status='complete' ORDER BY number DESC LIMIT 1", (identifier,)).fetchone()
        return json.loads(row[0]) if row else None

    def totals(self):
        models, currencies = {}, {}
        for model, state, ib, ob, currency, cb, raw in self.db.execute('''
                SELECT c.model,a.status,a.input_bound,a.output_bound,a.currency,a.cost_bound,a.response
                FROM attempts a JOIN calls c ON c.id=a.call_id'''):
            m = models.setdefault(model, {'attempts': 0, 'complete': 0, 'rejected': 0,
                'input_tokens': 0, 'output_tokens': 0, 'reserved_input': 0, 'reserved_output': 0})
            m['attempts'] += 1
            v = currencies.setdefault(currency, {'spent': 0., 'reserved': 0.})
            if state == 'complete':
                response = json.loads(raw)
                m['complete'] += 1
                m['input_tokens'] += response['input_tokens']
                m['output_tokens'] += response['output_tokens']
                v['spent'] += response['reference_cost']
            elif state == 'rejected':
                m['rejected'] += 1
            else:
                m['reserved_input'] += ib
                m['reserved_output'] += ob
                v['reserved'] += cb
        return {'models': models, 'currencies': currencies}

    def begin(self, identifier, model, payload, profile, stage, input_bound, output_bound):
        contract = encoded({'payload': payload, 'profile': profile})
        with self.mutex:
            row = self.db.execute('SELECT contract,model,stage FROM calls WHERE id=?', (identifier,)).fetchone()
            if row and row != (contract, model, stage):
                raise ValueError('call identity reused with different input')
            old = self.receipt(identifier)
            if old is not None:
                return old
            if self.get_meta('pause'):
                raise Paused(self.get_meta('pause')['reason'])
            pending = self.db.execute("SELECT status FROM attempts WHERE call_id=? ORDER BY number DESC LIMIT 1", (identifier,)).fetchone()
            if pending and pending[0] != 'rejected':
                raise Paused('unfinished call cannot be repeated')
            total = self.totals()
            stats = total['models'].get(model, {})
            limit = self.limits['models'][model]
            cost = (input_bound * profile['input_per_million'] + output_bound * profile['output_per_million']) / 1e6
            money = total['currencies'].get(profile['currency'], {})
            reason = None
            if stats.get('attempts', 0) >= limit['calls']:
                reason = 'call_budget:' + model
            for side, bound in [('input', input_bound), ('output', output_bound)]:
                if stats.get(side+'_tokens', 0) + stats.get('reserved_'+side, 0) + bound > limit[side+'_tokens']:
                    reason = 'token_budget:' + model + ':' + side
            if money.get('spent', 0.) + money.get('reserved', 0.) + cost > self.limits['currency'][profile['currency']]:
                reason = 'currency_budget:' + profile['currency']
            if reason:
                self.pause(reason)
                raise Paused(reason)
            number = self.db.execute('SELECT COALESCE(MAX(number),0)+1 FROM attempts WHERE call_id=?', (identifier,)).fetchone()[0]
            with self.db:
                self.db.execute('INSERT OR IGNORE INTO calls VALUES(?,?,?,?)', (identifier, model, contract, stage))
                self.db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (identifier, number, 'started', input_bound, output_bound, profile['currency'], cost, None, time.time(), None))
            return None

    def finish(self, identifier, response):
        with self.mutex:
            row = self.db.execute('SELECT number,input_bound,output_bound,cost_bound FROM attempts WHERE call_id=? AND status=?', (identifier,'started')).fetchone()
            if row is None:
                raise ValueError('no in-flight reservation')
            n, ib, ob, cb = row
            known = (all(type(response.get(k)) is int and response[k] >= 0 for k in ('input_tokens','output_tokens'))
                     and isinstance(response.get('reference_cost'), (int,float))
                     and math.isfinite(response['reference_cost']) and response['reference_cost'] >= 0)
            # Explicit provider refusal: retain the error and retry only after an explicit resume.
            # Unknown transmission/5xx/timeout remains reserved and is never automatically reissued.
            rejected = response.get('status') == 'http_error' and response.get('http_status') in (400,401,402,403,404,422,429)
            state = 'complete' if known else 'rejected' if rejected else 'unknown'
            with self.db:
                self.db.execute('UPDATE attempts SET status=?,response=?,ended=? WHERE call_id=? AND number=?',
                    (state, encoded(response), time.time(), identifier, n))
            if state != 'complete':
                self.pause('provider_rejected:' + str(response.get('error_code') or response.get('http_status')) if rejected else 'unknown_call_outcome_requires_reconciliation')
            elif response.get('status') != 'ok':
                self.pause('provider_protocol_error')
            elif response['input_tokens'] > ib or response['output_tokens'] > ob or response['reference_cost'] > cb + 1e-9:
                self.pause('provider_usage_exceeded_reservation')
            self.progress()
            return response

    def progress(self, stage=None, **extra):
        from nicheflow.ledger import atomic_json
        with self.mutex:
            if stage:
                self.set_meta('stage', stage)
            report = {'stage': self.get_meta('stage'), 'pause': self.get_meta('pause'),
                'totals': self.totals(), 'unresolved_calls': self.unresolved(),
                'updated_at': time.time(), **extra}
            atomic_json(self.directory / 'progress.json', report)
            return report

    def backup(self):
        with self.mutex:
            target = self.directory / 'checkpoint.sqlite3.tmp'
            out = sqlite3.connect(target)
            self.db.backup(out)
            out.close()
            with target.open('rb') as f:
                os.fsync(f.fileno())
            os.replace(target, self.directory / 'checkpoint.sqlite3')

    def close(self):
        if getattr(self, 'db', None):
            self.db.close()
            self.db = None
        if getattr(self, 'lock', None):
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None
