"""Frozen before/after evaluation with a shared, single-sample response bank.

Routes are fixed using public questions before any test responses are generated.
Identical (task, workflow, sample) requests share the same draw across policies;
logical per-policy costs and physical experiment spending are reported separately.
"""
import copy
import json
import math
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .checkpoint_compat import verify_checkpoint
from .cli import code_identity
from .graph import WorkflowGraph
from .ledger import Journal, atomic_json
from .main_budget import ResourceJournal
from .main_config import load_protocol, derive_limits
from .main_models import ModelPool
from .research_policy import ResearchPolicy, research_seeds
from .research_state import read_stage_source, restore
from .runtime import GraphExecutor
from .spec import BudgetStop, EnvironmentBlocked, ExecutionStop, IntegrityError, digest, file_hash


def prepare_comparison(root, source_dir, config_path, compatibility_manifest=None, encoder_factory=None):
    root, source_dir = Path(root), Path(source_dir)
    frozen = json.loads((source_dir/'config.json').read_text())['config']
    learning = frozen['settings']
    check = copy.deepcopy(learning); check['rounds'] += 1
    source = read_stage_source(source_dir, check, code_identity(root), compatibility_manifest)
    config, tasks, provenance = load_protocol(config_path, root, learning['rounds'])
    def core(c):
        c = copy.deepcopy(c); c.pop('evaluation')
        for k in ('evaluation_seconds', 'evaluation_api_usd'): c['limits'].pop(k, None)
        return c
    if core(config) != core(learning) or config['evaluation']['status'] != 'configured':
        raise IntegrityError('comparison must retain the frozen learning protocol')
    if config['evaluation']['samples_per_task'] != 1 or config['evaluation']['pass_k'] != [1]:
        raise IntegrityError('comparison protocol requires one shared sample per task/workflow')
    events = Journal.read(source_dir/'events.jsonl')
    snapshots = {e['id']:e['payload'] for e in events if e['kind']=='state_snapshot'}
    states, checks, policies = {}, {}, {}
    for label, r in [('before',0),('after',source['round'])]:
        path = source_dir/'checkpoints'/f'round_{r:04d}.json'
        c = json.loads(path.read_text())
        checks[label] = {**verify_checkpoint(c, snapshots.get(f'round:{r}'), allow_legacy=True),
                         'round':r, 'file_sha256':file_hash(path)}
        states[label] = c['state']
        policy = ResearchPolicy(SimpleNamespace(journal=None), learning, tasks['development'],
            synthetic=frozen['synthetic'], encoder=encoder_factory(learning['semantic']) if encoder_factory else None)
        restore(policy, c['state']); policies[label] = policy
    groups = []
    for wi, weight in enumerate(config['policy']['weight_grid']):
        for label in ('before','after'):
            groups.append({'id':f'{label}_w{wi}','checkpoint':label,'weight_index':wi,'weights':weight})
    for role in ('local','strong'):
        wid = next(v for v in states['before']['seed_versions']
                   if len(states['before']['graphs'][v]['nodes'])==1
                   and states['before']['graphs'][v]['nodes'][0]['model']==role)
        groups.append({'id':f'baseline_{role}','fixed_workflow':wid})
    graphs = {v:WorkflowGraph.from_dict(g) for s in states.values() for v,g in s['graphs'].items()}
    indices = provenance.get('evaluation_source_indices',list(range(len(tasks['evaluation']))))
    routes, jobs = [], {}
    plan_started = time.monotonic()
    for ti, task in zip(indices, tasks['evaluation']):
        task_features, confidence = {}, {}
        for label, policy in policies.items():
            task_features[label] = {a:policy.features(task.model_input(),policy.graphs[a]) for a in policy.router.arms}
            confidence[label] = policy.router_confidence(policy.router, ti+1)
        task_routes = []
        for group in groups:
            label = group.get('checkpoint')
            if label:
                policy = policies[label]
                bq, bc = confidence[label]
                decision = policy.router.choose(task_features[label],weights=group['weights'],
                    beta_quality=bq,beta_cost=bc,cost_reward_definition='frozen main quality/cost reward',
                    fallback_threshold=0.,safe_arm=states[label]['seed_versions'][0])
                wid = decision['arm']
            else:
                wid = group['fixed_workflow']; decision = {'arm':wid,'fixed_baseline':True}
            eid = f'comparison:sample:0:task:{ti}:workflow:{wid}'
            task_routes.append({'id':f"{group['id']}:task:{ti}",'group':group['id'],
                'task_id':task.id,'task_index':ti,'workflow':wid,'execution_id':eid,'decision':decision})
        # Fixed question/graph ordering interleaves groups without consulting labels.
        for route in sorted(task_routes,key=lambda r:digest([config['decode']['seed'],r['task_id'],r['workflow']])):
            jobs.setdefault(route['execution_id'],{k:route[k] for k in ('task_id','task_index','workflow','execution_id')})
        routes.extend(task_routes)
    for label, policy in policies.items():
        if digest(policy.export_state()) != digest(states[label]):
            raise IntegrityError('route planning changed checkpoint learning state')
    plan = {'schema':'nicheflow_frozen_comparison_v1','groups':groups,'routes':routes,'jobs':list(jobs.values()),
        'task_count':len(tasks['evaluation']),'logical_executions':len(routes),'unique_executions':len(jobs),
        'checkpoint_checks':checks,'source':source['evidence'],
        'state_digests':{k:digest(s) for k,s in states.items()},
        'route_counts':{g['id']:dict(Counter(r['workflow'] for r in routes if r['group']==g['id'])) for g in groups},
        'sampling':'one response per task/workflow, reused across groups; correlated paired observations',
        'execution_order':'public deterministic hash within each task; no outcome-dependent order',
        'max_model_calls':sum(graphs[j['workflow']].model_calls for j in jobs.values())}
    return {'plan':plan,'planning_seconds':time.monotonic()-plan_started,'config':config,'tasks':tasks,
            'provenance':provenance,'source':source,'graphs':graphs,'synthetic':frozen['synthetic']}


def summarize(plan, events, config):
    calls = {e['id']:e['payload'] for e in events if e['kind']=='call_finished'}
    starts = {e['id']:e['payload'] for e in events if e['kind']=='call_started'}
    executions = {e['id']:e['payload'] for e in events if e['kind']=='execution'}
    model_roles = {}
    for role, profile in config['models'].items():
        model_roles[role] = role
        model_roles[profile.get('path',profile.get('model'))] = role
    groups, values = [], {}
    for group in plan['groups']:
        rows = [r for r in plan['routes'] if r['group']==group['id']]
        if any(r['execution_id'] not in executions for r in rows):
            continue
        scores, costs, reference, times, local_times = [], [], [], [], []
        truncated = errors = parse_failures = 0
        for route in rows:
            ex = executions[route['execution_id']]
            assert ex['task_id']==route['task_id'] and ex['workflow']==route['workflow'] and ex['role']=='evaluation'
            scores.append(ex['evaluation']['quality'] if ex['status']=='ok' else 0.)
            truncated += ex['status']=='truncated'
            errors += ex['status'] not in ('ok','truncated')
            parse_failures += ex['status']=='ok' and ex['evaluation'].get('metrics',{}).get('parsed') is False
            cs = [calls[k] for k in ex['calls']]
            costs.append(sum((c.get('api_charge') or {}).get('usd',0.) for c in cs))
            times.append(sum(c.get('elapsed_seconds') or 0 for c in cs))
            local_times.append(sum(calls[k].get('elapsed_seconds') or 0 for k in ex['calls']
                                   if config['models'][model_roles[starts[k]['model']]]['kind']=='local'))
            normalized = 0.
            for k in ex['calls']:
                role = model_roles[starts[k]['model']]; profile = config['models'][role]; c=calls[k]
                if profile['kind']=='api':
                    rate = profile['pricing'].get('off_peak',profile['pricing'])
                    normalized += ((c.get('input_tokens') or 0)*rate['input_per_million_usd']
                                   +(c.get('output_tokens') or 0)*rate['output_per_million_usd'])/1e6
            reference.append(normalized)
        values[group['id']] = scores
        groups.append({**group,'tasks':len(rows),'correct':sum(scores),'accuracy':float(np.mean(scores)),
            'api_usd_per_query':float(np.mean(costs)),'logical_api_usd':sum(costs),
            'reference_api_usd_per_query':float(np.mean(reference)),
            'mean_workflow_call_seconds':float(np.mean(times)), 'p95_workflow_call_seconds':float(np.percentile(times,95)),
            'mean_local_call_seconds':float(np.mean(local_times)),
            'truncated':truncated,'execution_errors':errors,'parse_failures':parse_failures})
    paired=[]
    for wi in range(len(config['policy']['weight_grid'])):
        a,b=f'before_w{wi}',f'after_w{wi}'
        if a not in values or b not in values: continue
        va,vb=np.array(values[a]),np.array(values[b]); delta=vb-va
        rng=np.random.default_rng(20260922+wi)
        boot=np.mean(delta[rng.integers(0,len(delta),size=(5000,len(delta)))],axis=1)
        wins=int(sum((va==0)&(vb==1)));losses=int(sum((va==1)&(vb==0))); n=wins+losses
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n)) if n else 1.
        paired.append({'weight_index':wi,'after_minus_before_pp':float(np.mean(delta)*100),
            'after_only_correct':wins,'before_only_correct':losses,
            'paired_bootstrap_95pct_pp':(np.percentile(boot,[2.5,97.5])*100).tolist(),
            'mcnemar_exact_two_sided_p_unadjusted':p})
    complete = len(executions)==plan['unique_executions'] and len(groups)==len(plan['groups'])
    return {'status':'complete' if complete else 'incomplete','groups':groups,'paired_comparisons':paired,
        'task_count':plan['task_count'],'unique_executions_completed':len(executions),
        'unique_executions_planned':plan['unique_executions'],'physical_calls':len(calls),
        'physical_api_usd':sum((c.get('api_charge') or {}).get('usd',0.) for c in calls.values()),
        'unknown_call_ids':sorted(starts.keys()-calls.keys()),
        'unknown_cost_ids':[k for k,v in calls.items() if v.get('accounted_usd') is None],
        'all_execution_errors':[k for k,v in executions.items() if v['status'] not in ('ok','truncated')],
        'cost_notes':'Physical spend deduplicates shared executions. Group costs charge each selected workflow fully. Reference fees use frozen off-peak uncached tariffs, local=0, to remove clock/cache tariff differences.',
        'time_notes':'Sum of serial workflow node call times; excludes route precomputation and model loading.',
        'statistics_notes':'Paired task bootstrap, one shared draw per workflow/task, no repeated-seed evidence; three exploratory comparisons, p values unadjusted.',
        'source':plan['source'],'checkpoint_checks':plan['checkpoint_checks'],'state_digests':plan['state_digests']}


def execute_comparison(prepared, out_dir, root, backend_factory=ModelPool):
    out_dir, root = Path(out_dir), Path(root)
    if out_dir.exists():
        raise IntegrityError('comparison output must be new; no implicit retry or overwrite')
    cfg, plan = prepared['config'],prepared['plan']
    if prepared['synthetic'] != bool(getattr(backend_factory,'is_synthetic',False)):
        raise IntegrityError('cannot mix synthetic and real checkpoint evidence')
    limits=derive_limits(cfg,prepared['tasks'],research_seeds(cfg)); count=plan['max_model_calls']
    limits.update(max_calls=count,evaluation_calls=count,learning_calls=0,
        max_seconds=cfg['limits']['evaluation_seconds'],evaluation_seconds=cfg['limits']['evaluation_seconds'],
        learning_seconds=cfg['limits']['evaluation_seconds'],max_api_usd=cfg['limits']['evaluation_api_usd'],
        evaluation_api_usd=cfg['limits']['evaluation_api_usd'],learning_api_usd=0.,
        max_accounting_usd=count*limits['max_call_accounting_usd'])
    frozen={'mode':'frozen_before_after_comparison','settings':cfg,'code':code_identity(root),
        'plan_digest':digest(plan),'source':plan['source'],'synthetic':prepared['synthetic'],'limits':limits}
    task_map={t.id:t for t in prepared['tasks']['evaluation']}
    start=time.time(); status='main_run_complete'; error=None
    with ResourceJournal(out_dir,frozen,count,limits['max_seconds'],limits=limits) as journal:
        atomic_json(out_dir/'plan.json',plan)
        journal.append('comparison_started','main',{'plan_digest':digest(plan),'groups':plan['groups'],
            'task_count':plan['task_count'],'state_digests':plan['state_digests'],'learning_updates_allowed':False})
        pool=backend_factory(cfg)
        try:
            identity={k:v.environment for k,v in pool.backends.items()}
            if any(identity[k]['model']!=prepared['source']['environment'][k]['model'] for k in identity):
                raise IntegrityError('comparison model environment changed')
            atomic_json(out_dir/'environment.json',identity)
            executor=GraphExecutor(journal,pool.backends,cfg['decode'],failure_limit=3,
                allowed_operators=cfg['allowed_operators'],workflow_output_policy=cfg['workflow_output_policy'])
            executor.length_limit_as_zero=cfg['length_limit_outcome']=='zero_quality_no_retry'
            with journal.scope('evaluation',count,limits['max_accounting_usd']):
                for i, job in enumerate(plan['jobs'],1):
                    executor.execute(prepared['graphs'][job['workflow']],task_map[job['task_id']],job['execution_id'])
                    executor.check_failures();journal.reserve(0)
                    if i%25==0 or i==len(plan['jobs']):
                        progress={'completed':i,'total':len(plan['jobs']),'elapsed_seconds':time.time()-start,
                            'api_usd':journal.spending()['api_usd'],'last_task_index':job['task_index']}
                        atomic_json(out_dir/'progress.json',progress)
                        print(json.dumps(progress),flush=True)
        except (BudgetStop,EnvironmentBlocked,ExecutionStop) as exc:
            status={BudgetStop:'budget_stopped',EnvironmentBlocked:'environment_blocked',ExecutionStop:'execution_stopped'}[type(exc)]
            error=f'{type(exc).__name__}: {exc}'
        finally:
            pool.close()
        for label, check in plan['checkpoint_checks'].items():
            path=Path(prepared['source']['directory'])/'checkpoints'/f"round_{check['round']:04d}.json"
            if file_hash(path)!=check['file_sha256']:
                raise IntegrityError('source checkpoint changed during comparison')
        source_dir=Path(prepared['source']['directory'])
        if file_hash(source_dir/'events.jsonl')!=plan['source']['events_sha256']:
            raise IntegrityError('source journal changed during comparison')
        result=summarize(plan,journal.events,cfg)
        if result['all_execution_errors'] and status=='main_run_complete':status='main_completed_with_execution_errors'
        result.update(run_status=status,error=error,elapsed_seconds=time.time()-start,
            learning_state_unchanged=True,planning_seconds=prepared['planning_seconds'])
        journal.append('comparison_finished','main',result)
        atomic_json(out_dir/'comparison.json',result)
    return result
