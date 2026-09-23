#!/usr/bin/env python3
"""Standalone, read-only verification followed by a derived comparison report.

Runs after the evaluator exits. Never invokes a model or modifies source evidence.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def render(run, exit_code):
    run=Path(run)
    if exit_code != 0 or not (run/'comparison.json').is_file():
        run.mkdir(exist_ok=True,parents=True)
        message=f'评测未正常完成，退出码为 {exit_code}。请检查 comparison_v052.console.log；没有自动重试、重启或追加模型调用。\n'
        (run/'COMPARISON_REPORT.md').write_text(message)
        return False
    result=json.loads((run/'comparison.json').read_text())
    frozen=json.loads((run/'config.json').read_text())
    plan=json.loads((run/'plan.json').read_text())
    assert frozen['fingerprint']==digest({k:frozen[k] for k in ('config','max_calls','seconds')})
    assert frozen['config']['plan_digest']==digest(plan)
    previous='0'*64; calls_started=set();start_payloads={};calls_finished={};executions={};end=None
    for i,line in enumerate((run/'events.jsonl').open()):
        event=json.loads(line);body={k:v for k,v in event.items() if k!='hash'}
        assert event['seq']==i and event['previous']==previous and digest(body)==event['hash']
        previous=event['hash']
        if event['kind']=='call_started':
            calls_started.add(event['id']);start_payloads[event['id']]=event['payload']
        elif event['kind']=='call_finished':calls_finished[event['id']]=event['payload']
        elif event['kind']=='execution':executions[event['id']]=event['payload']
        elif event['kind']=='comparison_finished':end=event['payload']
        assert event['kind'] not in ('router','feedback','generation','scheduler'), 'learning occurred during evaluation'
    assert end==result
    assert result['status']=='complete' and result['run_status']=='main_run_complete'
    assert calls_started==calls_finished.keys() and all(c.get('accounted_usd') is not None for c in calls_finished.values())
    assert set(executions)=={j['execution_id'] for j in plan['jobs']}
    assert len(result['groups'])==8 and all(g['tasks']==plan['task_count'] for g in result['groups'])
    assert not result['all_execution_errors']
    source=Path(plan['source']['directory'])
    assert sha(source/'events.jsonl')==plan['source']['events_sha256']
    assert sha(source/'config.json')==plan['source']['config_sha256']
    for c in plan['checkpoint_checks'].values():
        assert sha(source/'checkpoints'/f"round_{c['round']:04d}.json")==c['file_sha256']
    recovery=result.get('recovery')
    if recovery:
        assert json.loads((run/'recovery_manifest.json').read_text())==recovery
        interrupted=Path(recovery['source_directory'])
        assert all(sha(interrupted/n)==h for n,h in recovery['source_hashes'].items())
        old_lines=(interrupted/'events.jsonl').read_bytes().splitlines(keepends=True)
        new_lines=(run/'events.jsonl').read_bytes().splitlines(keepends=True)
        n=recovery['prefix_event_count']
        assert old_lines[:n]==new_lines[:n]
        assert [json.loads(x) for x in old_lines[n:]]==recovery['abandoned_local_events']
        previous='0'*64
        for i,line in enumerate(old_lines):
            e=json.loads(line);body={k:v for k,v in e.items() if k!='hash'}
            assert e['seq']==i and e['previous']==previous and digest(body)==e['hash']
            previous=e['hash']
        abandoned=recovery['abandoned_local_events'][-1]
        assert abandoned['kind']=='call_started'
        assert any(p['kind']=='local' and p.get('path')==abandoned['payload']['model']
                   and p['accounting_usd_per_gpu_hour']==0 for p in frozen['config']['settings']['models'].values())
        restarted=[json.loads(x) for x in new_lines[n:] if json.loads(x)['kind']=='call_started' and json.loads(x)['id']==abandoned['id']]
        assert len(restarted)==1 and restarted[0]['payload']==abandoned['payload']
        assert result['physical_attempts_including_abandoned']==len(calls_finished)+1<=frozen['max_calls']
    # Derived report pricing only: preserve original result and immutable receipts.
    profiles={p.get('path',p.get('model')):p for p in frozen['config']['settings']['models'].values()}
    peak_calls={}
    for key,call in calls_finished.items():
        profile=profiles[start_payloads[key]['model']]
        if profile['kind']=='api':
            rates=profile['pricing']['peak']
            assert type(call.get('input_tokens')) is int and type(call.get('output_tokens')) is int
            peak_calls[key]=(call['input_tokens']*rates['input_per_million_usd']
                             +call['output_tokens']*rates['output_per_million_usd'])/1e6
        else:
            peak_calls[key]=0.
    peak_per_execution={k:sum(peak_calls[c] for c in e['calls']) for k,e in executions.items()}
    peak_groups={g['id']:sum(peak_per_execution[r['execution_id']] for r in plan['routes']
                           if r['group']==g['id'])/g['tasks'] for g in result['groups']}
    peak_pricing={'basis':'frozen_peak_uncached_input_all_output_local_zero',
                  'rates':{k:p['pricing']['peak'] for k,p in profiles.items() if p['kind']=='api'},
                  'api_usd_per_query_by_group':peak_groups,
                  'unique_execution_peak_api_usd':sum(peak_calls.values()),
                  'actual_recorded_api_usd':result['physical_api_usd'],
                  'original_comparison_unchanged':True}
    (run/'peak_pricing.json').write_text(json.dumps(peak_pricing,ensure_ascii=False,indent=2)+'\n')
    groups={g['id']:g for g in result['groups']}
    text=['# 500 题训练前后对比结果','',
        f"核验时间：{datetime.now(timezone(timedelta(hours=8))).isoformat()}。完整事件链、调用账本、冻结计划与原检查点均验证通过。",'',
        f"每组 {plan['task_count']} 道相同测试题；第 0 轮与第 30 轮冻结状态，共 8 组。一次采样，2048-token 上限，本地货币成本计 0。",'',
        '## 前后准确率','',
        '| 质量/费用权重 | 学习前 | 学习后 | 提升百分点 | 配对 95% 区间（百分点） |',
        '|---|---:|---:|---:|---:|']
    for p in result['paired_comparisons']:
        wi=p['weight_index'];a=groups[f'before_w{wi}'];b=groups[f'after_w{wi}'];lo,hi=p['paired_bootstrap_95pct_pp']
        text.append(f"| {a['weights']} | {a['correct']:.0f}/{a['tasks']} ({a['accuracy']:.2%}) | {b['correct']:.0f}/{b['tasks']} ({b['accuracy']:.2%}) | {p['after_minus_before_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    text+=['','## 全部结果与成本','',
        '| 组别 | 正确率 | 高峰参照 API 美元/千题 | 实际费率 API 美元/千题 | 平均工作流调用秒/题 | 截断/题数 |',
        '|---|---:|---:|---:|---:|---:|']
    names={'baseline_local':'Qwen 单独作答','baseline_strong':'Flash 单独作答'}
    for g in result['groups']:
        name=names.get(g['id'],('学习前' if g['id'].startswith('before') else '学习后')+f" {g.get('weights')}")
        text.append(f"| {name} | {g['accuracy']:.2%} | {peak_groups[g['id']]*1000:.4f} | {g['api_usd_per_query']*1000:.4f} | {g['mean_workflow_call_seconds']:.2f} | {g['truncated']}/{g['tasks']} |")
    text+=['',
        f"本次实际评测 API 支出：**${result['physical_api_usd']:.6f}**；{result['physical_calls']} 次模型调用，{result['unique_executions_completed']} 次唯一工作流执行，评测执行用时 {result['elapsed_seconds']/3600:.2f} 小时。",'',
        '按用户要求，参照费用统一使用实验冻结的 DeepSeek 高峰单价：输入 $0.30/百万 tokens、输出 $1.20/百万 tokens。全部输入按未缓存计价，本地货币成本仍为0；这是统一高峰参照口径，非供应商账单。实际费用列保留原始回执的峰谷时段与缓存计价。组别费用完整计入该策略每次选中工作流的全部调用，不按共享组数分摊。', '',
        f"去重后的整次评测若统一按高峰、未缓存计价，参照支出为 **${sum(peak_calls.values()):.6f}**；实际回执核算支出仍为 **${result['physical_api_usd']:.6f}**。高峰重算明细见 peak_pricing.json；原 comparison.json 的非高峰参照字段保留作为历史结果。", '',
        '同一题上相同工作流共享一次响应，因此本次实际支出小于八组分别独立执行的总额。相同执行的前后差异为零；配对比较利用同题结果。单次采样的 bootstrap 区间不代表跨随机种子的稳定性。', '',
        '工作流调用耗时包含顺序执行节点的等待时间，不包含路由预计算与模型加载；不能直接当作完整部署服务的端到端延迟。', '',
        '这是工作流搜索与路由学习的前后对比，Qwen 和 Flash 模型权重没有经过微调。初始状态来自当时保存的第 0 轮，不是事后重新构造。', '',
        '所有路由在测试响应产生前固定；没有任何学习更新、自动续训、参数调整或重复失败调用。三个权重的配对统计见 comparison.json；p 值未作多重比较校正。', '',
        '原训练日志、配置、首末检查点保持不变。旧检查点摘要通过原始整数键规则兼容验证，没有覆写旧摘要。']
    if recovery:
        text += ['', '## 服务器中断与恢复', '',
            f"服务器中断后，复用 {recovery['completed_executions_reused']} 次已完成执行，在独立目录补完剩余 {recovery['remaining_executions']} 次执行。原始中断目录、完整哈希日志和训练证据保持不变。",
            '仅重试了最后一条没有保存结果的本地零美元调用；没有重试未知付费 API 调用或重采样已完成回答。上文“没有重复失败调用”指已完成的失败回执；此处明确记录未完成本地调用的一次恢复重试。',
            f"含中断的本地尝试，实际发起调用数为 {result['physical_attempts_including_abandoned']}。中断调用的计算时间未知，未计入完成调用耗时；上文评测执行用时为从原启动至恢复完成的墙钟时间，包含停机间隔。",
            '保持原始评测计划、执行代码、模型与费用／调用／24小时时限。恢复前核对模型文件、推理软件和GPU环境；同一采样种子不构成断电前后逐位相同输出的保证。']
    (run/'COMPARISON_REPORT.md').write_text('\n'.join(text)+'\n')
    audit={'verified':True,'events_sha256':sha(run/'events.jsonl'),'plan_digest':digest(plan),
           'physical_calls':len(calls_finished),'source_unchanged':True,'learning_updates':0,
           'report':str(run/'COMPARISON_REPORT.md'), 'recovery_verified':bool(recovery),
           'report_pricing_basis':peak_pricing['basis'], 'peak_pricing_sha256':sha(run/'peak_pricing.json')}
    (run/'completion_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--exit-code-file',required=True)
    a=p.parse_args()
    try:
        ok=render(a.run,int(Path(a.exit_code_file).read_text().strip()))
    except Exception as e:
        directory=Path(a.run);directory.mkdir(exist_ok=True,parents=True)
        (directory/'POSTPROCESS_FAILED.txt').write_text(f'{type(e).__name__}: {e}\n')
        raise
    raise SystemExit(0 if ok else 1)
