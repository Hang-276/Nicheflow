"""Block-feedback scheduling with explicit bounded stages; terminal rule unchanged."""
from .main_driver import MainDriver
from .ledger import atomic_json
from .spec import BudgetStop, EnvironmentBlocked, ExecutionStop, IntegrityError, SpecGap, digest


class ResearchDriver(MainDriver):
    def run(self, source=None, stop_after=None):
        existing = self.journal.lookup('run_finished', 'main:done')
        if existing:
            return existing
        start = source['round'] if source else 0
        end = stop_after or self.config['rounds']
        self.journal.append('main_contract', 'main', {'profile': self.config['profile'], 'rounds': self.config['rounds'],
            'stage_start': start, 'stage_end': end, 'partitions': {k: [t.record() for t in ts] for k, ts in self.tasks.items()},
            'seeds': [g.version for g in self.policy.seeds], 'provenance': self.provenance,
            'limits': self.limits, 'synthetic': self.policy.synthetic})
        try:
            if source:
                from .research_state import restore
                restore(self.policy, source['state'])
                self.journal.append('stage_restored', 'source', source['evidence'])
                previous = self.policy.export_state()
                self.snapshot(start, previous)
            else:
                initial = self.policy.export_state()
                with self.scope('bootstrap', self.limits['bootstrap_calls']):
                    for i, graph in enumerate(self.policy.seeds):
                        key = f'bootstrap:{i}'
                        result = self.policy.search.evaluate(graph, self.tasks['development'], key)
                        if result['status'] != 'evaluated':
                            raise ExecutionStop('bootstrap evaluation incomplete')
                        self.feedback(key, 'bootstrap', [result])
                self.policy.finish_bootstrap()
                previous = self.snapshot(0, initial)
            for round_index in range(start + 1, end + 1):
                for step in range(self.config['policy']['blocks_per_cycle']):
                    key = f'round:{round_index}:step:{step}'
                    context = self.policy.context()
                    self.journal.append('block_context', key, context)
                    decision = self.policy.scheduler.decide(self.policy.completed_blocks + 1, context, context['remaining_research_budget'])
                    decision.update(theoretical_confidence_established=False,
                                    utility_scope='snapshot archive HV; adaptive heterogeneous observations, diagnostic epsilon',
                                    rewards={'search': 'delta Vendi / 100', 'deploy': 'delta deployment HV'},
                                    allocation_rule_status='explicit engineering heuristic; no transferred approximation guarantee')
                    self.journal.append('scheduler', key, decision)
                    kind = self.policy.next_operation(decision)
                    op = f'round:{round_index}:{step}:{kind}'
                    self.journal.append('operation_plan', key, {'operations': [kind], 'feedback_boundary': 'every_block',
                        'accounting_usd_per_block': self.limits['block_accounting_usd']})
                    with self.scope(op, self.limits['block_calls']):
                        if kind == 'search':
                            selection, parents = self.policy.selection()
                            results = self.policy.search.run(op, selection, parents, self.tasks['development'], max_nodes=self.config['generation_max_nodes'])
                        else:
                            results = []
                            for q in range(self.config['queries_per_deploy_block']):
                                task = self.tasks['calibration'][self.policy.calibration_cursor % len(self.tasks['calibration'])]
                                self.policy.calibration_cursor += 1
                                results.append(self.policy.deployment.run(f'{op}:query:{q}', task, self.policy.graphs, self.policy.calibration_cursor))
                                self.executor.check_failures()
                        self.feedback(op, kind, results)
                        self.executor.check_failures()
                        self.journal.append('operation_finished', op, {'kind': kind, 'resource_usage': self.journal.spending(op)})
                previous = self.snapshot(round_index, previous)
                self.journal.append('cycle_finished', f'round:{round_index}', {'state_digest': digest(previous), 'calibration_queries_seen': self.policy.calibration_cursor})
            state_digest = digest(previous)
            if end < self.config['rounds']:
                status = 'stage_paused'
            else:
                self.journal.append('learning_finished', 'main:learning', {'rounds': end, 'state_digest': state_digest})
                atomic_json(self.journal.directory / 'trained_state.json', {'digest': state_digest, 'state': previous})
                status = 'learning_complete_evaluation_pending'
                if self.config['evaluation']['status'] == 'configured':
                    self.evaluate_frozen(previous)
                    status = 'main_run_complete'
            failures = [e['id'] for e in self.journal.events if e['kind'] == 'execution' and not self.executor.assessment_complete(e['payload'])]
            if failures:
                status = 'main_completed_with_execution_errors'
            return self.journal.append('run_finished', 'main:done', {'status': status, 'synthetic': self.policy.synthetic,
                'rounds_completed': end, 'stage_rounds_completed': end-start, 'state_digest': state_digest,
                'evaluation_complete': status == 'main_run_complete', 'failed_executions': failures,
                'formal_training_automatically_started': False})
        except (BudgetStop, EnvironmentBlocked, ExecutionStop, SpecGap, ValueError) as exc:
            return self.journal.append('run_finished', 'main:done', {'status': {BudgetStop:'budget_stopped', EnvironmentBlocked:'environment_blocked',
                ExecutionStop:'execution_stopped', SpecGap:'spec_blocked', ValueError:'implementation_failure'}[type(exc)],
                'synthetic': self.policy.synthetic, 'rounds_completed': start + sum(e['kind']=='cycle_finished' for e in self.journal.events),
                'evaluation_complete': False, 'error': f'{type(exc).__name__}: {exc}'})
