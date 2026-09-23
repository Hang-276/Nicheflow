#!/usr/bin/env python3
"""Bounded real two-round check, paused after one round and explicitly restored.

No held-out evaluation, no expanded experiment, no automatic retries.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nicheflow.main_config import load_protocol, model_selection_problems
from nicheflow.main_entry import run_main
from nicheflow.ledger import Journal,atomic_json
from nicheflow.spec import file_hash,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--run-dir',default='runs/revision_v050_short')
    p.add_argument('--config',default='configs/main_math_v050.json')
    args=p.parse_args()
    if not args.execute:p.error('--execute required for bounded real calls')
    directory=Path(args.run_dir).resolve()
    if directory.exists():p.error('existing check directory; no overwrite or automatic retry')
    config,parts,_=load_protocol(ROOT/args.config,ROOT)
    if model_selection_problems(config):
        p.error('; '.join(model_selection_problems(config)))
    config=copy.deepcopy(config)
    config.update(rounds=2,queries_per_deploy_block=4)
    config['limits'].update(fixed_seconds=900,seconds_per_round=900,api_usd_fixed=.5,api_usd_per_round=.5)
    config['notes'].append('DIAGNOSTIC ONLY: 2 development, 4 calibration tasks; 2 rounds in two explicit stages; no terminal test or expanded run.')
    inputs=directory/'inputs';inputs.mkdir(parents=True)
    manifest={'synthetic':False,'protocol':'revision_v050_short','partitions':{}}
    for role,count in [('development',2),('calibration',4)]:
        path=inputs/f'{role}.jsonl';selected=parts[role][:count]
        path.write_text(''.join(json.dumps(t.record(),ensure_ascii=False)+'\n' for t in selected))
        manifest['partitions'][role]={'path':path.name,'count':count,'sha256':file_hash(path)}
    pack=inputs/'manifest.json';atomic_json(pack,manifest)
    config['data_manifest'],config['data_manifest_sha256']=str(pack),file_hash(pack)
    cfg=inputs/'config.json';atomic_json(cfg,config)
    first=directory/'stage1';second=directory/'stage2'
    runargs=SimpleNamespace(config=str(cfg),rounds=None,run_dir=str(first),stop_after_round=1,resume_from=None)
    one,c1=run_main(runargs,ROOT)
    if c1 or one['status']!='stage_paused':
        atomic_json(directory/'check.json',{'passed':False,'stage1':one,'expanded_experiment_started':False})
        return 6
    source_hash=file_hash(first/'events.jsonl')
    runargs.run_dir=str(second);runargs.stop_after_round=2;runargs.resume_from=str(first)
    two,c2=run_main(runargs,ROOT)
    ev1,ev2=Journal.read(first/'events.jsonl'),Journal.read(second/'events.jsonl')
    events=ev1+ev2
    calls=[e['payload'] for e in events if e['kind']=='call_started']
    gens=[e for e in events if e['kind']=='generation' and e['payload']['status']=='valid']
    credits=[e for e in events if e['kind']=='parent_credit']
    checks={'both_stages_complete':c1==c2==0 and two['rounds_completed']==2,
            'source_unchanged':file_hash(first/'events.jsonl')==source_hash,
            'no_bootstrap_repeat':not any(e['kind']=='call_started' and e['id'].startswith('bootstrap') for e in ev2),
            'valid_mutation':bool(gens),'real_parent_credit':bool(credits),
            'all_configured_models_called':{v['model'] for v in calls}=={m['path'] if m['kind']=='local' else m['model'] for m in config['models'].values()},
            'router_updated':any(e['kind']=='router' for e in events),
            'block_rescheduling':sum(e['kind']=='scheduler' for e in events)==4,
            'healthy':all(s['numerical_health']['all_finite_spd'] for s in (one,two)),
            'no_unknown_calls_or_cost':all(not s['unknown_calls'] and not s['unknown_cost_calls'] for s in (one,two)),
            'no_system_errors':all(not s['execution_errors'] for s in (one,two)),
            'no_test_evaluation':not any(e['kind']=='evaluation_started' for e in events)}
    snapshot=json.loads((first/'checkpoints/round_0000.json').read_text())['state']
    singles=[]
    for version in snapshot['seed_versions'][:len(config['models'])]:
        graph=snapshot['graphs'][version];r=snapshot['records'][version]
        singles.append({'role':graph['nodes'][0]['model'],'quality':r['quality'],'usd_per_query':r['usd_per_query'],
                        'failures':r['failure_counts'],'sample_count':len(r['execution_ids'])})
    result={'passed':all(checks.values()),'checks':checks,'diagnostic_only':True,'expanded_experiment_started':False,
            'real_calls':one['calls_finished']+two['calls_finished'],'api_tariff_usd':one['api_tariff_usd']+two['api_tariff_usd'],
            'local_accounting_usd':one['local_inference_accounting_usd']+two['local_inference_accounting_usd'],
            'single_model_diagnostic':singles,'quality_ranking_established':False,
            'length_limited_workflows':one['length_limited_workflows']+two['length_limited_workflows'],
            'stage1':one,'stage2':two}
    atomic_json(directory/'check.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'stage1','stage2'}},ensure_ascii=False,indent=2))
    return 0 if result['passed'] else 6


if __name__=='__main__':raise SystemExit(main())
