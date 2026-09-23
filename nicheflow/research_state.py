"""Exact v2 learning restoration and explicit immutable stage ancestry."""
from collections import deque
import json
from pathlib import Path
import numpy as np
from .archive import Evaluation, cell_id
from .graph import WorkflowGraph
from .ledger import Journal
from .main_driver import frozen_router
from .research_policy import ParentEstimate
from .router import RidgeHead
from .scheduler import MarginalObservation
from .spec import IntegrityError, digest, file_hash
from .checkpoint_compat import verify_code_transition, verify_checkpoint


def read_stage_source(directory, config, code, compatibility_manifest=None):
    directory = Path(directory).resolve()
    frozen = json.loads((directory / 'config.json').read_text())
    if frozen['fingerprint'] != digest({k: frozen[k] for k in ('config', 'max_calls', 'seconds')}):
        raise IntegrityError('source frozen contract hash mismatch')
    old = frozen['config']
    transition = verify_code_transition(old['code'], code, compatibility_manifest)
    if {k: v for k, v in old['settings'].items() if k != 'rounds'} != {k: v for k, v in config.items() if k != 'rounds'}:
        raise IntegrityError('stage extension may change horizon only; code/models/features/data must match')
    events = Journal.read(directory / 'events.jsonl')
    ends = [e['payload'] for e in events if e['kind'] == 'run_finished']
    if not ends or ends[-1]['status'] not in {'stage_paused', 'learning_complete_evaluation_pending'}:
        raise IntegrityError('source must be a clean completed learning stage')
    if any(e['kind'] == 'evaluation_started' for e in events):
        raise IntegrityError('cannot extend a stage after held-out evaluation')
    calls = {e['id'] for e in events if e['kind'] == 'call_started'}
    finished = {e['id']: e['payload'] for e in events if e['kind'] == 'call_finished'}
    if calls != finished.keys() or any(x.get('accounted_usd') is None for x in finished.values()):
        raise IntegrityError('source contains unknown call/billing outcomes')
    round_index = ends[-1]['rounds_completed']
    if not 0 < round_index < config['rounds']:
        raise IntegrityError('extended horizon must exceed completed source round')
    path = directory / 'checkpoints' / f'round_{round_index:04d}.json'
    checkpoint = json.loads(path.read_text())
    snapshot = next((e['payload'] for e in events if e['kind'] == 'state_snapshot' and e['id'] == f'round:{round_index}'), None)
    validation = verify_checkpoint(checkpoint, snapshot, allow_legacy=True)
    if checkpoint['state']['completed_blocks'] != round_index * config['policy']['blocks_per_cycle']:
        raise IntegrityError('checkpoint is not at a complete round boundary')
    from .main_policy import stability
    if not stability(checkpoint['state'])['numerically_healthy']:
        raise IntegrityError('source checkpoint numerical state is unhealthy')
    return {'directory': str(directory), 'round': round_index, 'state': checkpoint['state'],
            'evidence': {'directory': str(directory), 'round': round_index, 'checkpoint_digest': checkpoint['digest'],
                         'events_sha256': file_hash(directory / 'events.jsonl'), 'config_sha256': file_hash(directory / 'config.json'),
                         'checkpoint_validation': validation, 'code_transition': transition,
                         'old_horizon': old['settings']['rounds'], 'new_horizon': config['rounds'],
                         'budget_context': 'remaining/total uses new explicitly extended horizon; not equivalent to original longer-horizon run'},
            'environment': json.loads((directory / 'environment.json').read_text())}


def restore(policy, state):
    if state['schema'] != 'nicheflow_state_v2' or state['policy_config_digest'] != digest({k: v for k, v in policy.config.items() if k != 'rounds'}):
        raise IntegrityError('checkpoint schema/protocol mismatch')
    policy.graphs = {v: WorkflowGraph.from_dict(g).validate(policy.config['models']) for v, g in state['graphs'].items()}
    if any(g.version != v for v, g in policy.graphs.items()):
        raise IntegrityError('checkpoint workflow identity mismatch')
    policy.seeds = [policy.graphs[v] for v in state['seed_versions']]
    policy.router = frozen_router(state)
    research = state['research']
    policy.router.receipts = {k: (v[0], tuple(v[1]), v[2], v[3]) for k, v in research['router_receipt_signatures'].items()}
    if set(policy.router.receipts) != set(state['router']['receipt_ids']):
        raise IntegrityError('checkpoint receipt signatures incomplete')
    evaluations = {v: Evaluation(e['workflow'], tuple(e['cell']), e['quality'], e['usd_cost'], tuple(e['receipt_ids']),
                                e['split_role'], e['descriptor_version']) for v, e in state['archive']['evaluations'].items()}
    policy.archive.evaluations = evaluations
    policy.archive.elites = {int(i): evaluations[v] for i, v in state['archive']['elites'].items()}
    policy.archive.population = {i: {} for i in range(100)}
    for v, e in evaluations.items():
        policy.archive.population[cell_id(e.cell)][v] = e
    policy.boundaries = state['archive']['boundaries']
    for name in ('w1', 'b1', 'w2'):
        setattr(policy.proxy, name, np.asarray(state['proxy'][name], float))
    policy.proxy.b2 = state['proxy']['b2']
    policy.proxy.training_receipts = tuple(state['proxy']['training_receipts'])
    policy.estimates = {int(i): ParentEstimate(e['count'], e['total']) for i, e in state['search_estimates'].items()}
    for name, e in state['meta'].items():
        head = RidgeHead(len(e['b']), e['regularization'])
        head.v, head.b, head.count = np.asarray(e['v']), np.asarray(e['b']), e['count']
        policy.meta[name] = head
    def observation(e):
        return MarginalObservation(e['id'], e['element'], tuple(e['conditioning_set']), e['marginal'], e['empty_marginal'], e['utility_definition'], tuple(e['receipt_ids']))
    policy.curvature.window = deque(map(observation, state['curvature']['window']), maxlen=policy.curvature.window.maxlen)
    policy.curvature.seen = {k: observation(e) for k, e in state['curvature']['seen'].items()}
    for name in ('completed_blocks', 'feedback_index', 'rounding_carry', 'search_vendi', 'deploy_hv', 'deploy_observations', 'records'):
        setattr(policy, name, state[name])
    policy.search_rounds, policy.fixed_boundaries = research['search_rounds'], research['fixed_boundaries']
    policy.search_visits = {int(k): v for k, v in research['search_visits'].items()}
    policy.credited, policy.calibration_cursor = research['credited'], research['calibration_cursor']
    policy.search.archive, policy.search.router, policy.deployment.router = policy.archive, policy.router, policy.router
    if digest(policy.export_state()) != digest(state):
        raise IntegrityError('restored learning state differs from checkpoint')


def stage_limits(config, original, start, end):
    limit = dict(original)
    bootstrap = 0 if start else original['bootstrap_calls']
    evaluation = original['evaluation_calls'] if end == config['rounds'] else 0
    blocks = (end - start) * config['policy']['blocks_per_cycle']
    learning = bootstrap + blocks * original['block_calls']
    settings = config['limits']
    limit.update(bootstrap_calls=bootstrap, learning_calls=learning, evaluation_calls=evaluation,
                 evaluation_workflows=original['evaluation_workflows'] if evaluation else 0,
                 learning_seconds=settings['fixed_seconds'] + (end-start) * settings['seconds_per_round'],
                 evaluation_seconds=original['evaluation_seconds'] if evaluation else 0,
                 learning_api_usd=settings['api_usd_fixed'] + (end-start) * settings['api_usd_per_round'],
                 evaluation_api_usd=original['evaluation_api_usd'] if evaluation else 0,
                 max_calls=learning + evaluation,
                 max_accounting_usd=(learning+evaluation)*original['max_call_accounting_usd'])
    limit['max_seconds'] = limit['learning_seconds'] + limit['evaluation_seconds']
    limit['max_api_usd'] = limit['learning_api_usd'] + limit['evaluation_api_usd']
    return limit
