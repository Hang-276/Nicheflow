#!/usr/bin/env python3
"""Plan or execute explicitly authorized frozen before/after comparisons."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nicheflow.comparison_eval import prepare_comparison,execute_comparison
from nicheflow.ledger import atomic_json
from nicheflow.main_entry import completion_exit_code

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--compatibility-manifest')
    p.add_argument('--plan-out')
    p.add_argument('--out-dir')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.execute and not a.out_dir:p.error('--execute requires --out-dir')
    prepared=prepare_comparison(ROOT,a.source,a.config,a.compatibility_manifest)
    if a.plan_out:atomic_json(a.plan_out,prepared['plan'])
    plan=prepared['plan']
    print(json.dumps({k:plan[k] for k in ['task_count','logical_executions','unique_executions','max_model_calls','route_counts']},ensure_ascii=False),flush=True)
    if a.execute:
        result=execute_comparison(prepared,a.out_dir,ROOT)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        raise SystemExit(completion_exit_code(result['run_status']))
