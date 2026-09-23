"""v0.5 engineering alignment: source-parent feedback, semantics, block scheduling.

No claim of a theorem for adaptive proxy-filtered feedback or snapshot curvature.
"""
from dataclasses import asdict
import math
import numpy as np
from .archive import cell_id, cost_boundaries, occupancy_density, pareto_admission
from .graph import Node, WorkflowGraph
from .main_policy import MainPolicy, main_seeds
from .policy import neighbors
from .search import QualityEstimate, rbf_ucb_kernel, select_niches
from .semantic import SemanticFeatures
from .spec import IntegrityError, digest

REVISION = 'feedback_semantic_v2'


def research_seeds(config):
    config = {**config, 'models': {m: config['models'][m] for m in ('local', 'middle', 'strong') if m in config['models']}}
    # Comparable single-model controls are available as genuine initial arms.
    singles = [WorkflowGraph((Node('a', 'Generate', config['seed_prompts']['solve'], model=m),), 'a')
               for m in config['models']]
    return [g.validate(config['models']) for g in [*singles, *[g for g in main_seeds(config) if len(g.nodes) > 1]]]


class ParentEstimate(QualityEstimate):
    def ucb(self, t, exploration):
        if self.count == 0:
            # Explicit cold prior: quality upper bound 1, denominator 1 for bonus.
            # This is NOT a fabricated observation and count remains zero.
            return 1. + exploration * math.sqrt(math.log(max(2, t)))
        return super().ucb(max(2, t), exploration)


class ResearchPolicy(MainPolicy):
    def __init__(self, executor, config, tasks, *, synthetic=False, encoder=None):
        self.semantic_features = SemanticFeatures(config, encoder=encoder)
        self.search_rounds, self.search_visits, self.credited = 0, {}, {}
        self.calibration_cursor = 0
        self.fixed_boundaries = False
        super().__init__(executor, config, tasks, synthetic=synthetic)
        self.search.parent_feedback = self.parent_feedback

    def bootstrap(self, config):
        return research_seeds(config)

    def router_dimension(self):
        return self.semantic_features.router_dimension

    def workflow_dimension(self):
        return self.semantic_features.workflow_dimension

    def workflow_features(self, graph):
        return self.semantic_features.workflow(graph)

    def features(self, public, graph):
        return self.semantic_features.route(public, graph)

    def _rebin(self):
        if not self.fixed_boundaries:
            self.boundaries = cost_boundaries([r['usd_per_query'] for r in self.records.values()])
        changes = self.archive.rebin(lambda e: self.describe(self.graphs[e.workflow], e.usd_cost), pareto_admission)
        for i, e in self.archive.elites.items():
            self.estimates.setdefault(i, ParentEstimate())
            if self.router.add_arm(e.workflow):
                self.journal.append('router_arm', e.workflow, {'epoch': self.router.epoch, 'source': 'bootstrap_quantiles' if not self.fixed_boundaries else 'fixed_bins',
                                                             'receipts': list(e.receipt_ids)})
        self.journal.append('archive_rebin', f'feedback:{self.feedback_index}',
                            {'boundaries': self.boundaries, 'moves': changes, 'frozen_after_bootstrap': self.fixed_boundaries})

    def finish_bootstrap(self):
        self.fixed_boundaries = True
        # No search actions occurred during bootstrap; seed scores are not visits.
        self.estimates = {i: ParentEstimate() for i in self.archive.elites}
        self.journal.append('search_credit_protocol', 'fixed', {
            'quality_credit': 'evaluated_child_to_selected_parent_niche',
            'visit_count': 'all_selected_parents', 'ucb_count': 'real_new_candidate_evaluations_only',
            'proxy_rejected_invalid_duplicate': 'no_quality_observation',
            'cost_boundaries': self.boundaries, 'rebin_policy': 'freeze_bootstrap_quantiles_before_first_search',
            'cold_prior': 'quality_upper_bound_1_bonus_denominator_1_no_fake_sample',
            'source_exact_theorem_claim': False})

    def selection(self):
        if not self.fixed_boundaries:
            raise IntegrityError('freeze descriptor before search')
        self.search_rounds += 1
        ids = sorted(self.archive.elites)
        for i in ids:
            self.estimates.setdefault(i, ParentEstimate())
        occupied = {e.cell for e in self.archive.elites.values()}
        features = [[c / n for c, n in zip(self.archive.elites[i].cell, (4, 4, 3))]
                    + self.workflow_features(self.graphs[self.archive.elites[i].workflow]) for i in ids]
        # Offset only avoids the log(1)=0 cold-start bonus, clock counts searches.
        t = self.search_rounds + 1
        kernel = rbf_ucb_kernel(features, [self.estimates[i].ucb(t, self.p['search_exploration']) for i in ids], self.p['kernel_bandwidth'])
        result = select_niches(ids, {i: len(self.archive.population[i]) for i in ids},
            {i: occupancy_density(self.archive.elites[i].cell, occupied, neighbors) for i in ids}, self.estimates,
            kernel, t=t, k=self.p['k'], population_threshold=self.p['population_threshold'], rho0=self.p['rho0'],
            alpha=self.p['alpha'], diversity_weight=self.p['diversity_weight'], exploration=self.p['search_exploration'])
        for i in result['selected']:
            self.search_visits[i] = self.search_visits.get(i, 0) + 1
        result.update(clock='1 + actual search blocks', search_round=self.search_rounds, kernel=kernel.tolist(),
                      features=features, ids=ids, descriptor_version=digest(self.boundaries))
        return result, {i: self.graphs[self.archive.elites[i].workflow] for i in ids}

    def parent_feedback(self, graph):
        r = self.records[graph.version]
        return {'workflow': graph.version, 'role': 'development', 'quality': r['quality'],
                'usd_per_query': r['usd_per_query'], 'sample_count': len(r['execution_ids']),
                'failure_counts': r.get('failure_counts', {}), 'has_test_labels': False}

    def observe(self, stage, results):
        for r in results:
            if r['status'] == 'evaluated':
                executions = [self.journal.lookup('execution', i) for i in r['execution_ids']]
                if any(x is None or x['role'] != 'development' for x in executions):
                    raise IntegrityError('candidate feedback must have development executions')
                r['failure_counts'] = {
                    'length': sum(x['status'] == 'truncated' for x in executions),
                    'parse': sum(x['status'] == 'ok' and x['evaluation']['metrics'].get('parsed') is False for x in executions),
                    'incorrect': sum(self.executor.assessment_quality(x) == 0 for x in executions)}
                if stage == 'search':
                    i = r['parent_niche']
                    if self.search_visits.get(i, 0) < 1:
                        raise IntegrityError('feedback parent was never selected')
                    if r['workflow'] not in self.credited:
                        self.estimates.setdefault(i, ParentEstimate()).observe(r['quality'])
                        self.credited[r['workflow']] = {'parent_niche': i, 'destination_niche': cell_id(tuple(r['cell'])),
                                                        'quality': r['quality'], 'receipts': r['receipts']}
                        self.journal.append('parent_credit', r['workflow'], self.credited[r['workflow']])
        value = super().observe(stage, results)
        value.update(search_selected_counts=dict(self.search_visits),
                     search_real_feedback_counts={i: e.count for i, e in self.estimates.items()}, search_clock=self.search_rounds)
        return value

    def allocate(self, context, estimate, remaining_budget):
        result = super().allocate(context, estimate, remaining_budget)
        return result

    def next_operation(self, decision):
        self.rounding_carry += decision['allocation']['search']
        if self.rounding_carry >= 1.:
            self.rounding_carry -= 1.
            return 'search'
        return 'deploy'

    def export_state(self):
        state = super().export_state()
        state['schema'] = 'nicheflow_state_v2'
        state['research'] = {'search_rounds': self.search_rounds, 'search_visits': {str(k): v for k, v in self.search_visits.items()},
                             'credited': self.credited, 'fixed_boundaries': self.fixed_boundaries,
                             'calibration_cursor': self.calibration_cursor,
                             'router_receipt_signatures': self.router.receipts,
                             'semantic_contract': digest(self.config['semantic']),
                             'curvature_interpretation': 'snapshot_HV_diagnostic_not_certified_confidence_bound'}
        return state
