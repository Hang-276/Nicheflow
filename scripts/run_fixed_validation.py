#!/usr/bin/env python3
"""Fixed-workflow validation, paired by question with two retained samples."""
import argparse
from collections import Counter
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from nicheflow.cli import code_identity
from nicheflow.datasets import load_tasks
from nicheflow.graph import WorkflowGraph
from nicheflow.ledger import Journal,atomic_json
from nicheflow.main_budget import ResourceJournal
from nicheflow.main_config import max_call_usd
from nicheflow.main_models import ModelPool,model_readiness
from nicheflow.runtime import GraphExecutor
from nicheflow.spec import IntegrityError,digest,file_hash
from fixed_validation_parallel import ParallelResourceJournal,execute_jobs

def require(value,message):
    if not value:raise IntegrityError(message)

def prepare(config_path,root=ROOT):
    cfg=json.loads(Path(config_path).read_text())
    require(cfg['concurrency']=={'workflow_workers':8,'api_calls':4,'local_calls':1},'unexpected concurrency protocol')
    require(file_hash(root/cfg['data'])==cfg['data_sha256'],'validation data changed')
    require(file_hash(root/cfg['manifest'])==cfg['manifest_sha256'],'validation manifest changed')
    manifest=json.loads((root/cfg['manifest']).read_text())
    if not cfg.get('synthetic'):
        for name,h in manifest['excluded_prior_files'].items():require(file_hash(root/name)==h,'prior data changed')
        for item in manifest['source_files']:require(file_hash(root/item['path'])==item['sha256'],'raw source changed')
        source=Path(cfg['source_training']['directory'])
        for name,h in cfg['source_training']['hashes'].items():require(file_hash(source/name)==h,'source training changed')
    tasks=load_tasks(root/cfg['data'])
    require(len(tasks)==manifest['count'] and all(t.role=='evaluation' and t.split=='train' for t in tasks),'invalid validation roles')
    groups={k:WorkflowGraph.from_dict(v).validate(cfg['models']) for k,v in cfg['groups'].items()}
    jobs=[]
    for task in tasks:
        for sample in range(cfg['samples_per_task']):
            for group in sorted(groups,key=lambda k:digest([cfg['planning_seed'],task.id,sample,k])):
                jobs.append({'task_id':task.id,'sample':sample,'group':group,'workflow':groups[group].version,
                             'execution_id':f"fixed:v060:task:{task.id}:sample:{sample}:group:{group}"})
    count=sum(groups[j['group']].model_calls for j in jobs)
    bound=max(max_call_usd(p,cfg['decode']['max_new_tokens']) for p in cfg['models'].values())
    limits={'max_calls':count,'max_seconds':cfg['max_seconds'],'evaluation_seconds':cfg['max_seconds'],
            'learning_seconds':cfg['max_seconds'],'max_api_usd':cfg['max_api_usd'],
            'evaluation_api_usd':cfg['max_api_usd'],'learning_api_usd':0.,
            'max_accounting_usd':count*bound,'max_call_accounting_usd':bound}
    plan={'jobs':jobs,'groups':list(groups),'tasks':len(tasks),'samples':cfg['samples_per_task'],
          'executions':len(jobs),'max_calls':count,'contrasts':cfg['primary_contrasts'],
          'concurrency':cfg['concurrency'],
          'ordering':'deterministic dispatch, interleaved within each question/sample; parallel completion order may vary'}
    frozen={'settings':cfg,'code':code_identity(root),'runner_sha256':file_hash(Path(__file__)),
            'parallel_runner_sha256':file_hash(Path(__file__).with_name('fixed_validation_parallel.py')),
            'plan_digest':digest(plan),'limits':limits,'synthetic':cfg.get('synthetic',False)}
    return cfg,tasks,groups,plan,limits,frozen

def summarize(cfg,tasks,plan,events):
    starts={e['id']:e['payload'] for e in events if e['kind']=='call_started'}
    calls={e['id']:e['payload'] for e in events if e['kind']=='call_finished'}
    ex={e['id']:e['payload'] for e in events if e['kind']=='execution'}
    groups=[];values={};per_question={};profiles={p.get('path',p.get('model')):p for p in cfg['models'].values()}
    def cost(keys,peak=False,local_time=False):
        total=0.
        for k in keys:
            c=calls[k];p=profiles[starts[k]['model']]
            if local_time:total+=(c.get('elapsed_seconds') or 0) if p['kind']=='local' else 0
            elif peak and p['kind']=='api':
                r=p['pricing']['peak'];total+=(c['input_tokens']*r['input_per_million_usd']+c['output_tokens']*r['output_per_million_usd'])/1e6
            elif not peak:total+=(c.get('api_charge') or {}).get('usd',0.)
        return total
    for group in plan['groups']:
        jobs=[j for j in plan['jobs'] if j['group']==group]
        require(all(j['execution_id'] in ex for j in jobs),'incomplete validation')
        rows=[ex[j['execution_id']] for j in jobs]
        scores=[e['evaluation']['quality'] if e['status']=='ok' else 0. for e in rows]
        values[group]={(j['task_id'],j['sample']):v for j,v in zip(jobs,scores)}
        per_question[group]=[np.mean([values[group][t.id,s] for s in range(plan['samples'])]) for t in tasks]
        keys=[k for e in rows for k in e['calls']]
        groups.append({'id':group,'executions':len(rows),'correct':sum(scores),'accuracy':float(np.mean(scores)),
            'sample_accuracies':[float(np.mean([values[group][t.id,s] for t in tasks])) for s in range(plan['samples'])],
            'peak_api_usd_per_query':cost(keys,peak=True)/len(rows),'actual_api_usd_per_query':cost(keys)/len(rows),
            'mean_call_seconds':sum(calls[k].get('elapsed_seconds') or 0 for k in keys)/len(rows),
            'mean_local_seconds':cost(keys,local_time=True)/len(rows),
            'truncated':sum(e['status']=='truncated' for e in rows),
            'execution_errors':sum(e['status'] not in ('ok','truncated') for e in rows),
            'model_calls':len(keys)})
    contrasts=[]
    for i,(a,b) in enumerate(plan['contrasts']):
        delta=np.array(per_question[a])-np.array(per_question[b]);rng=np.random.default_rng(2026092206+i)
        boots=delta[rng.integers(len(tasks),size=(10000,len(tasks)))].mean(axis=1)*100
        contrasts.append({'a':a,'b':b,'delta_pp':float(delta.mean()*100),'question_cluster_bootstrap_95pct_pp':np.percentile(boots,[2.5,97.5]).tolist()})
    return {'status':'complete','groups':groups,'paired_contrasts':contrasts,'task_count':len(tasks),
        'executions':len(ex),'calls_started':len(starts),'calls_finished':len(calls),
        'actual_api_usd':cost(list(calls)),'peak_api_usd':cost(list(calls),peak=True),
        'unknown_calls':sorted(set(starts)-set(calls)),
        'unknown_cost_calls':[k for k,v in calls.items() if v.get('accounted_usd') is None],
        'statistical_note':'Average of both draws, not pass@2; paired intervals resample questions and keep both draws together. Four unadjusted exploratory contrasts, no across-run stability claim.',
        'validation_not_final_test':True}

def report(directory,result,plan):
    lines=['# 四个固定流程验证结果','',f"生成时间：{datetime.now(timezone(timedelta(hours=8))).isoformat()}",'',
       f"{result['task_count']}道新验证题，每题每方案{plan['samples']}次采样；全部取平均，不选最好一次。无搜索或路由更新。",'',
       '| 方案 | 平均正确率 | 两次采样正确率 | 高峰美元/千题 | 实际美元/千题 | 调用秒/题 | 截断 |',
       '|---|---:|---|---:|---:|---:|---:|']
    for g in result['groups']:
        samples=' / '.join(f'{x:.2%}' for x in g['sample_accuracies'])
        lines.append(f"| {g['id']} | {g['accuracy']:.2%} | {samples} | {g['peak_api_usd_per_query']*1000:.4f} | {g['actual_api_usd_per_query']*1000:.4f} | {g['mean_call_seconds']:.2f} | {g['truncated']}/{g['executions']} |")
    lines+=['','| 预先指定的比较 | 差值（百分点） | 按题配对95%区间 |','|---|---:|---|']
    for c in result['paired_contrasts']:
        lo,hi=c['question_cluster_bootstrap_95pct_pp'];lines.append(f"| {c['a']} − {c['b']} | {c['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines+=['',f"实际回执核算API支出 ${result['actual_api_usd']:.6f}；统一高峰未缓存参照支出 ${result['peak_api_usd']:.6f}。本地货币成本按约定为0，调用耗时单列。",'',
       '高峰价为冻结的Flash输入$0.30/百万tokens、输出$1.20/百万tokens。所有组独立执行，未跨组共享规划输出。',
       '这是分层验证集诊断，不是新的最终测试成绩；学科和难度均衡分布，不直接与MATH-500总分比较。区间按题重采样并保留同题两次响应，不把280次响应当成280道独立题。四组区间未做多重比较校正。',
       'C保留原最佳流程，包括节点温度0.7/0.3/0.5；D仅删除Qwen节点并重连，保留规划及最终解题参数与提示。A/B均使用原默认温度0.7。C与A差异不能只归因于拓扑。',
       'Flash不支持固定seed，双采样反映有限采样波动；没有失败重试、测试集调参或自动扩展实验。',
       f"并发设置：{plan['concurrency']}；表内为各节点调用时间之和，不含排队，不等于并行运行的总墙钟时间，也不是独占资源的延迟基准。总墙钟 {result['elapsed_seconds']/60:.1f} 分钟。"]
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')

def run(config_path,directory,root=ROOT,backend_factory=ModelPool,resume=False):
    cfg,tasks,groups,plan,limits,frozen=prepare(config_path,root)
    directory=Path(directory)
    require(bool(cfg.get('synthetic'))==bool(getattr(backend_factory,'is_synthetic',False)),'synthetic/real mismatch')
    require(not directory.exists() or resume,'existing output requires explicit --resume')
    if not cfg.get('synthetic'):require(not model_readiness(cfg),'model configuration/credentials unavailable')
    with ParallelResourceJournal(directory,frozen,limits['max_calls'],limits['max_seconds'],limits=limits,
                                 api_concurrency=cfg['concurrency']['api_calls']) as journal:
        existing=journal.lookup('fixed_validation_finished','main')
        if existing:return existing
        require(not journal.audit()['unknown_calls'],'unfinished call: manual recovery required, no automatic retry')
        journal.reserve(0)
        atomic_json(directory/'plan.json',plan)
        journal.append('fixed_validation_started','main',{'plan_digest':digest(plan),'updates_allowed':False})
        pool=backend_factory(cfg)
        try:
            identity={k:v.environment for k,v in pool.backends.items()}
            if not cfg.get('synthetic'):
                old=json.loads((Path(cfg['source_training']['directory'])/'environment.json').read_text())
                require(set(identity)==set(old),'model profiles changed')
                for k,v in identity.items():require(v['model']==old[k]['model'],'model snapshot/profile changed')
            environment=directory/'environment.json'
            if environment.exists():
                old=json.loads(environment.read_text())
                require(all(identity[k]['model']==old[k]['model'] for k in identity),'resume model changed')
            else:atomic_json(environment,identity)
            executor=GraphExecutor(journal,pool.backends,cfg['decode'],failure_limit=3,
                allowed_operators=cfg['allowed_operators'],workflow_output_policy=cfg['workflow_output_policy'])
            executor.length_limit_as_zero=True
            task_map={t.id:t for t in tasks}
            with journal.scope('evaluation',limits['max_calls'],limits['max_accounting_usd']):
                def progress_update(i):
                    progress={'completed':i,'total':len(plan['jobs']),'fraction':i/len(plan['jobs']),
                              'api_usd':journal.spending()['api_usd'],'elapsed_seconds':time.time()-journal.started}
                    atomic_json(directory/'progress.json',progress)
                    if i%8==0 or i==len(plan['jobs']):print(json.dumps(progress),flush=True)
                execute_jobs(executor,journal,groups,task_map,plan,cfg['concurrency']['workflow_workers'],progress_update)
        finally:pool.close()
        require(prepare(config_path,root)[-1]==frozen,'frozen inputs changed during run')
        result=summarize(cfg,tasks,plan,journal.events)
        require(not result['unknown_calls'] and not result['unknown_cost_calls'],'unknown call/cost')
        result['run_status']='complete' if not sum(g['execution_errors'] for g in result['groups']) else 'complete_with_execution_errors'
        result['elapsed_seconds']=time.time()-journal.started
        journal.append('fixed_validation_finished','main',result)
        atomic_json(directory/'results.json',result)
        verified=Journal.read(directory/'events.jsonl')
        require(len(verified)==len(journal.events),'journal readback differs')
        require(not any(e['kind'] in ('router','generation','feedback','scheduler') for e in verified),'unexpected learning')
        report(directory,result,plan)
        atomic_json(directory/'completion_audit.json',{'verified':True,'events_sha256':file_hash(directory/'events.jsonl'),
                    'config_fingerprint':digest({'config':frozen,'max_calls':limits['max_calls'],'seconds':limits['max_seconds']}),
                    'learning_updates':0,'known_costs':True,'source_unchanged':True,'synthetic':cfg.get('synthetic',False)})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out-dir')
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.execute:
        if not a.out_dir:p.error('--execute requires --out-dir')
        result=run(a.config,a.out_dir,resume=a.resume);print(json.dumps(result),flush=True)
        raise SystemExit(0 if result['run_status']=='complete' else 2)
    else:
        cfg,tasks,groups,plan,limits,frozen=prepare(a.config)
        print(json.dumps({'tasks':len(tasks),'groups':list(groups),'samples':plan['samples'],'executions':plan['executions'],
                          'max_model_calls':limits['max_calls'],'max_api_usd':limits['max_api_usd'],'max_hours':limits['max_seconds']/3600,
                          'concurrency':plan['concurrency'],
                          'frozen_digest':digest(frozen),'model_calls':0},ensure_ascii=False))
