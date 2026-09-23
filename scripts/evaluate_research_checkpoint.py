#!/usr/bin/env python3
"""Explicit independent frozen v2 evaluation; never resumes learning."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nicheflow.research_eval import evaluate
from nicheflow.main_entry import completion_exit_code

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--source',required=True)
    p.add_argument('--config',default='configs/main_math_v050_full_eval.json')
    p.add_argument('--out-dir',required=True)
    p.add_argument('--checkpoint-round',type=int)
    p.add_argument('--compatibility-manifest')
    a=p.parse_args()
    if not a.execute:p.error('--execute required; real held-out model calls consume a separately bounded budget')
    result=evaluate(ROOT,a.source,a.config,a.out_dir,a.checkpoint_round,compatibility_manifest=a.compatibility_manifest)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(completion_exit_code(result['status']))
