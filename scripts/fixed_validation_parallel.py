"""Bounded parallel calls with one durable writer and conservative in-flight budgets."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import copy
from threading import RLock, BoundedSemaphore

from nicheflow.ledger import Journal
from nicheflow.main_budget import ResourceJournal
from nicheflow.main_config import max_call_usd
from nicheflow.spec import BudgetStop, IntegrityError, digest


class ParallelResourceJournal(ResourceJournal):
    def __init__(self, *args, api_concurrency=4, **kwargs):
        self.mutex = RLock()
        self.pending = {}
        self.stopped = False
        self.api_gate = BoundedSemaphore(api_concurrency)
        self.local_gate = BoundedSemaphore(1)
        super().__init__(*args, **kwargs)

    def append(self, *args, **kwargs):
        with self.mutex:
            return super().append(*args, **kwargs)

    def lookup(self, *args, **kwargs):
        with self.mutex:
            return super().lookup(*args, **kwargs)

    def spending(self, scope=None):
        with self.mutex:
            return super().spending(scope)

    def stop(self):
        with self.mutex:
            self.stopped = True

    def reserve(self, count):
        with self.mutex:
            if self.stopped:
                raise IntegrityError('parallel run stopped; no further model calls')
            Journal.reserve(self, count)
            spent = self.spending()
            if set(spent['unknown_cost_calls']) - set(self.pending):
                self.stopped = True
                raise IntegrityError('unknown billed outcome; no automatic continuation')
            pending_cost = sum(v['accounting'] for v in self.pending.values())
            reserve_cost = count * self.limits['max_call_accounting_usd']
            if spent['accounted_usd'] + pending_cost + reserve_cost > self.limits['max_accounting_usd'] + 1e-9:
                raise BudgetStop('total accounting USD reservation')
            if self.active_scope:
                envelope = self.lookup('resource_envelope', self.active_scope)
                stage = self.spending(self.active_scope)
                stage_pending = sum(v['accounting'] for v in self.pending.values() if v['scope'] == self.active_scope)
                if stage['calls'] + count > envelope['call_cap']:
                    raise BudgetStop('stage call cap')
                if stage['accounted_usd'] + stage_pending + reserve_cost > envelope['accounting_usd_cap'] + 1e-9:
                    raise BudgetStop('stage accounting USD reservation')

    def call(self, id, backend, messages, params):
        is_api = backend.profile['kind'] == 'api'
        # Hold a backend slot before recording a start. Waiting work is not a paid attempt.
        with self.api_gate if is_api else self.local_gate:
            with self.mutex:
                request = {'messages': messages, 'params': params, 'model': backend.model_id}
                start = self.lookup('call_started', id)
                if start is not None:
                    if digest(start) != digest(request):
                        raise IntegrityError('call id reused with changed input')
                    result = self.lookup('call_finished', id)
                    if result is None:
                        raise IntegrityError('previous call outcome unknown; duplicate refused')
                    return result
                if self.active_scope is None:
                    raise IntegrityError('model call outside resource envelope')
                self.reserve(1)
                bound = max_call_usd(backend.profile, params['max_new_tokens'])
                api_bound = bound if is_api else 0.
                phase = 'evaluation' if self.active_scope == 'evaluation' else 'learning'
                pending_api = sum(v['api'] for v in self.pending.values())
                phase_api = sum(v['api'] for v in self.pending.values() if v['phase'] == phase)
                if self.spending()['api_usd'] + pending_api + api_bound > self.limits['max_api_usd'] + 1e-9:
                    raise BudgetStop('external API dollar cap including in-flight calls')
                if self.spending(f'__{phase}__')['api_usd'] + phase_api + api_bound > self.limits[f'{phase}_api_usd'] + 1e-9:
                    raise BudgetStop('stage API cap including in-flight calls')
                self.append('resource_reservation', id, {'scope': self.active_scope,
                    'maximum_accounting_usd': self.limits['max_call_accounting_usd'],
                    'maximum_api_usd': api_bound})
                self.pending[id] = {'scope': self.active_scope, 'phase': phase,
                                    'accounting': self.limits['max_call_accounting_usd'], 'api': api_bound}
                self.append('call_started', id, request)
                started = self.clock()
                # API requests are stateless. A per-call copy isolates timeout adjustment.
                target = copy(backend) if is_api else backend
                if hasattr(target, 'call_timeout'):
                    target.call_timeout = min(target.profile['call_timeout_seconds'], max(.01, self.remaining_seconds()))
            try:
                result = target.generate(messages, **params)
            except Exception as exc:
                # Do not serialize exception text: third-party exceptions can include credentials.
                result = {'status': 'error', 'finish_reason': 'error', 'text': '',
                          'error': type(exc).__name__, 'input_tokens': None, 'output_tokens': None,
                          'elapsed_seconds': self.clock() - started, 'accounted_usd': None}
            with self.mutex:
                result = self.append('call_finished', id, {**result, 'call_id': id})
                del self.pending[id]
                # Flush accounting now, before any waiting call can acquire a new reservation.
                self.spending()
                if result.get('accounted_usd') is None:
                    self.stopped = True
                return result


def execute_jobs(executor, journal, groups, task_map, plan, workers, on_progress):
    """Keep only a bounded window submitted; drain in-flight calls on any error."""
    jobs = iter(plan['jobs'])
    completed = 0

    def execute(job):
        result = executor.execute(groups[job['group']], task_map[job['task_id']], job['execution_id'])
        with journal.mutex:
            executor.check_failures()
            journal.reserve(0)
        return result

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='fixed-workflow') as pool:
        active = set()
        try:
            for _ in range(workers):
                job = next(jobs, None)
                if job is not None:
                    active.add(pool.submit(execute, job))
            while active:
                done, active = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    completed += 1
                on_progress(completed)
                for _ in done:
                    job = next(jobs, None)
                    if job is not None:
                        active.add(pool.submit(execute, job))
        except BaseException:
            journal.stop()
            for future in active:
                future.cancel()
            raise
