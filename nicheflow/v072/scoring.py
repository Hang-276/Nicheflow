"""Versioned v072 scoring, separate from historical quality labels."""
import json
import re
from nicheflow.datasets import Task
from nicheflow.scoring import evaluate
from nicheflow_probe.evaluation import extract_boxed

VERSION='v072-semantic-math-hotpot-f1-mbpp-hidden-plus-1'


def math_score(task,text):
    from math_verify import parse,verify
    answer=extract_boxed(text)
    if answer is None:return {'quality':0.,'outcome':'missing_final','answer':None,'audit_required':True}
    gold=str(task.gold)
    if answer.strip()==gold.strip():return {'quality':1.,'outcome':'complete','answer':answer,'decision':'literal_match','audit_required':False}
    # Keep answer structure intact; never erase brackets, reorder tuples, or strip units globally.
    g=parse('\\boxed{'+gold+'}',fallback_mode='no_fallback',parsing_timeout=5)
    a=parse('\\boxed{'+answer+'}',fallback_mode='no_fallback',parsing_timeout=5)
    matched=bool(g and a and verify(g,a,float_rounding=12,numeric_precision=30,strict=True,timeout_seconds=5))
    # Symbolic results are provisional until consistent all-model case audit.
    return {'quality':float(matched),'outcome':'complete','answer':answer,
            'decision':'symbolic_match' if matched else 'nonliteral_mismatch',
            'parsed_gold':str(g),'parsed_answer':str(a),'audit_required':True}


def score(task,response,code_scorer=None):
    if response.get('status')!='ok':raise ValueError('provider failure cannot become a task-quality label')
    if response.get('finish_reason')=='length':
        return {'quality':0.,'outcome':'truncated','audit_required':False}
    if response.get('finish_reason')!='stop':raise ValueError('unexpected provider finish reason')
    text=response['text']
    if task.dataset=='math':return math_score(task,text)
    if task.dataset=='hotpotqa':
        # Only unwrap a single outer JSON fence; no LLM formatting or answer repair.
        match=re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*',text,re.S)
        result=evaluate(task,match.group(1) if match else text)
        metrics=result['metrics']
        return {'quality':float(metrics.get('f1',0.)), 'outcome':'complete' if metrics.get('parsed') else 'malformed',
                'metrics':metrics,'quality_metric':'answer_f1','audit_required':False}
    if task.dataset=='mbpp':
        if code_scorer is None:raise ValueError('isolated code scorer required before inference')
        return code_scorer(task,text)
    raise ValueError('unregistered v072 domain')


def selftest():
    cases=[('35,\\!280','35280',True),('-\\frac1{\\sqrt2}','-\\frac{\\sqrt2}{2}',True),
        ('(1,2)','(2,1)',False),('[0,1]','(0,1)',False),('1/3','.333',False),('1971','100',False),
        ('32','1.6',False),('41','44',False),('7.5','\\frac{15}{2}',True)]
    rows=[]
    for g,a,expected in cases:
        t=Task('test','math','train','development','Find the stated value.',g,'math_official')
        value=math_score(t,'\\boxed{'+a+'}')
        assert bool(value['quality'])==expected,(g,a,value)
        rows.append({'gold':g,'answer':a,'passed':True})
    t=Task('qa','hotpotqa','train','development','Who?', 'Ada Lovelace','hotpot_official',
           private={'supporting_facts':[['Biography',0]]})
    r=score(t,{'status':'ok','finish_reason':'stop','text':json.dumps({'answer':'Ada','sp':[['Biography',0]]})})
    assert 0 < r['quality'] < 1 and r['metrics']['em']==0 and r['metrics']['sp_em']==1
    assert score(t,{'status':'ok','finish_reason':'length','text':'correct'})['quality']==0
    assert score(t,{'status':'ok','finish_reason':'stop','text':'not JSON'})['quality']==0
    return {'version':VERSION,'math_cases':rows,'hotpot_f1_and_support_checks':True,'passed':True}
