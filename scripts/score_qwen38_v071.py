#!/usr/bin/env python3
"""Offline pilot grading; exact answers and Math-Verify, with review receipts."""
import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'score_packages'))
from math_verify import parse,verify


def pair(gold,answer):
    if answer is None:return {'quality':0.,'decision':'missing_box'}
    if gold.strip()==answer.strip():return {'quality':1.,'decision':'literal_match'}
    g=parse('\\boxed{'+gold+'}',fallback_mode='no_fallback',parsing_timeout=5)
    a=parse('\\boxed{'+answer+'}',fallback_mode='no_fallback',parsing_timeout=5)
    result=bool(g and a and verify(g,a,float_rounding=12,numeric_precision=30,strict=True,timeout_seconds=5))
    return {'quality':float(result),'decision':'symbolic_match' if result else 'needs_review',
            'parsed_gold':str(g),'parsed_answer':str(a)}


def selftest():
    cases=[(r'35,\!280','35280',True),
           (r'-\frac1{\sqrt2}',r'-\frac{\sqrt2}{2}',True),
           (r'\text{(C)}','C',True),
           (r'AB=12+12\sqrt3',r'12(1+\sqrt3)',True),
           ('-1,2,-2','-2,-1,2',True),
           ('7.5',r'\frac{15}{2}',True),
           ('(1,2)','(2,1)',False),('1/3','.333',False),
           ('32','1.6',False),('41','44',False),
           ('[0,1]','(0,1)',False),(r'\frac1{52}',r'\frac5{884}',False)]
    results=[]
    for g,a,expected in cases:
        result=pair(g,a);ok=bool(result['quality'])==expected
        results.append({'gold':g,'answer':a,'expected':expected,'passed':ok,**result})
    out={'passed':all(r['passed'] for r in results),'cases':results,
         'packages':{n:importlib.metadata.version(n) for n in ['math-verify','latex2sympy2_extended','sympy','mpmath','antlr4-python3-runtime']}}
    (ROOT/'scorer_selftest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))
    assert out['passed'],'semantic scorer regression failed'


def audit(base):
    sys.path.insert(0,str(base))
    from nicheflow_probe.evaluation import extract_boxed
    from nicheflow.datasets import load_tasks
    from nicheflow.spec import digest,file_hash
    tasks={t.id:t for t in load_tasks(ROOT/'data/model_selection_v071/validation.jsonl')}
    run=ROOT/'runs/qwen38_v071'
    result=json.loads((run/'results.json').read_text())
    rows=[]
    for r in result['rows']:
        task=tasks[r['task_id']];response=r['response'];answer=extract_boxed(response.get('text',''))
        score=pair(task.gold,answer) if r['score']['outcome']=='complete' else {'quality':0.,'decision':r['score']['outcome']}
        rows.append({'model':r['model'],'task_id':task.id,'gold':task.gold,'answer':answer,
                     'question':task.question,'finish_reason':response.get('finish_reason'),**score})
    groups=[]
    for name in sorted({r['model'] for r in rows}):
        scores=[r for r in rows if r['model']==name];calls=[r['response'] for r in result['rows'] if r['model']==name]
        groups.append({'model':name,'correct':sum(r['quality'] for r in scores),'n':len(scores),
                       'accuracy':statistics.mean(r['quality'] for r in scores),
                       'currency':calls[0]['currency'],'cost_per_1000':sum(r['reference_cost'] for r in calls)/len(calls)*1000,
                       'mean_seconds':statistics.mean(r['elapsed_seconds'] for r in calls),
                       'mean_output_tokens':statistics.mean(r['output_tokens'] for r in calls),
                       'truncated':sum(r['finish_reason']=='length' for r in scores),
                       'needs_review':sum(r['decision']=='needs_review' for r in scores)})
    report={'status':result['status'],'events_sha256':file_hash(run/'events.jsonl'),
            'scorer_sha256':file_hash(Path(__file__)),'math_verify_version':importlib.metadata.version('math-verify'),
            'automated_scores_provisional_until_all_nonliteral_cases_reviewed':True,'new_model_calls':0,
            'groups':groups,'rows':rows}
    (run/'semantic_auto.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'groups':groups},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path);p.add_argument('--selftest',action='store_true');a=p.parse_args()
    selftest() if a.selftest else audit(a.base)
