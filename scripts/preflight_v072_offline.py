#!/usr/bin/env python3
"""Data, scoring, isolation, model-file and tokenizer checks; no generation calls."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nicheflow.datasets import load_tasks
from nicheflow.spec import file_hash
from nicheflow.ledger import atomic_json
from nicheflow.v072.scoring import selftest,score
from scripts.model_selection_v070 import credentials
from scripts.run_v072 import payload,run_score


def main():
    config=json.loads((ROOT/'configs/multidomain_v072.json').read_text())
    out=ROOT/'setup/v072';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/config['manifest']).read_text())
    for path,h in manifest['task_files'].items():assert file_hash(ROOT/path)==h,path
    math_qa=selftest()
    code=json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/v072_code_sandbox.py'),
                                             '--root',config['code_sandbox_root'],'--selftest'],text=True))
    assert code['passed']
    keys=credentials(config['credentials_file'])
    assert all(keys.get(p['key_environment']) for p in config['models'].values() if p['kind']=='api')
    identity=json.loads(Path(config['local_identity']).read_text());assert identity['revision']==config['local_revision']
    for path,h in {**identity['configuration_sha256'],**identity['weight_sha256']}.items():
        assert file_hash(Path(identity['snapshot_path'])/path)==h,path
    with urllib.request.urlopen('http://127.0.0.1:8070/v1/models',timeout=10) as r:service=json.load(r)
    entry=next(x for x in service['data'] if x['id']==config['models']['L']['model'])
    assert entry['root']==identity['snapshot_path'] and entry['max_model_len']==24576
    checks=[];max_prompt_tokens=0
    for domain in config['domains']:
        tasks=load_tasks(ROOT/f'data/multidomain_v072/{domain}/development.jsonl')
        for i,task in enumerate(tasks):
            fake=replace(task,gold='PRIVATE_GOLD_SENTINEL_V072',private={'secret':'PRIVATE_REFERENCE_SENTINEL_V072'})
            visible,_=payload(config,config['models']['L'],fake,0)
            assert 'PRIVATE_GOLD_SENTINEL_V072' not in json.dumps(visible)
            assert 'PRIVATE_REFERENCE_SENTINEL_V072' not in json.dumps(visible)
            request,_=payload(config,config['models']['L'],task,0)
            token_request={'model':config['models']['L']['model'],'messages':request['messages'],
                           'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}}
            req=urllib.request.Request('http://127.0.0.1:8070/tokenize',data=json.dumps(token_request).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=20) as r:tokens=json.load(r)['count']
            max_prompt_tokens=max(max_prompt_tokens,tokens)
            assert tokens+config['max_tokens']<=entry['max_model_len'],(task.id,tokens)
            if domain=='mbpp':
                response={'status':'ok','finish_reason':'stop','text':'```python\n'+task.gold+'\n```'}
                result=run_score(task,response,config)
                if result['quality']!=1:raise ValueError('MBPP reference fails frozen tests: '+task.id+' '+json.dumps(result))
                checks.append({'task_id':task.id,'reference_passed':True,'metrics':result['metrics']})
            elif domain=='hotpotqa':
                response={'status':'ok','finish_reason':'stop','text':json.dumps({'answer':task.gold,'sp':task.private['supporting_facts']})}
                result=score(task,response);assert result['quality']==1 and result['metrics']['joint_f1']==1
                context={x['title']:x['sentences'] for x in task.context}
                for title,number in task.private['supporting_facts']:assert 0<=number<len(context[title])
            if i%10==0:print(json.dumps({'domain':domain,'checked':i+1,'model_generation_calls':0}),flush=True)
    result={'passed':True,'model_generation_calls':0,'math_and_qa':math_qa,'code_isolation':code,
            'mbpp_reference_checks':checks,'max_local_prompt_tokens':max_prompt_tokens,
            'manifest_sha256':file_hash(ROOT/config['manifest']),
            'sandbox_manifest_sha256':file_hash(Path(config['code_sandbox_root'])/'sandbox-manifest.json'),
            'local_weight_hashes_verified':True,'visible_input_leakage_checks':True,
            'runtime_lock_sha256':file_hash(ROOT/'requirements.lock')}
    atomic_json(ROOT/config['readiness'],result)
    print(json.dumps({'passed':True,'code_reference_tasks':len(checks),'max_local_prompt_tokens':max_prompt_tokens,'model_generation_calls':0}),flush=True)


if __name__=='__main__':main()
