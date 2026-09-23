import copy
from dataclasses import replace
import json
from pathlib import Path
import runpy
import tempfile
import unittest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from nicheflow.datasets import Task
from nicheflow.graph import WorkflowGraph
from nicheflow.ledger import Journal
from nicheflow.math_protocol import SUBJECTS,question_hash
from nicheflow.spec import IntegrityError,BudgetStop,file_hash
from nicheflow.main_config import max_call_usd
from test_main import ROOT,SyntheticMainBackend

prep=runpy.run_path(str(ROOT/'scripts/prepare_fixed_validation.py'))
runner=runpy.run_path(str(ROOT/'scripts/run_fixed_validation.py'))

class FakePool:
    is_synthetic=True
    made=0
    def __init__(self,cfg):
        FakePool.made+=1;self.backends={}
        for role,p in cfg['models'].items():
            model=p.get('path',p.get('model'));self.backends[role]=SyntheticMainBackend(model,p)
    def close(self):pass

def config_fixture(d):
    cfg=json.loads((ROOT/'configs/main_math_v051_local_flash.json').read_text())
    best=WorkflowGraph.from_dict(json.loads((ROOT/'tests/fixtures/v051_best_graph.json').read_text()))
    groups=prep['make_groups'](cfg,best)
    tasks=[Task(f'fixed/{i}','math','train','evaluation',f'Compute 2-1. Case {i}.','1','rational',private={'secret':'PRIVATE_SENTINEL'}) for i in range(3)]
    path=d/'tasks.jsonl';path.write_text(''.join(json.dumps(t.record())+'\n' for t in tasks))
    manifest=d/'manifest.json';manifest.write_text(json.dumps({'count':len(tasks)}))
    cfg.update(synthetic=True,groups=groups,data=str(path),data_sha256=file_hash(path),manifest=str(manifest),manifest_sha256=file_hash(manifest),
       samples_per_task=2,planning_seed=2026092206,max_api_usd=10.,max_seconds=3600,
       concurrency={'workflow_workers':8,'api_calls':4,'local_calls':1},
       primary_contrasts=[['B_flash_concise','A_flash_original'],['C_best_with_qwen','D_without_qwen']])
    p=d/'fixed.json';p.write_text(json.dumps(cfg));return p,cfg

class FixedValidationTests(unittest.TestCase):
    def test_bounded_overlap_and_serial_local_backend(self):
        counts={'api':0,'local':0,'api_peak':0,'local_peak':0,'overlap':False}
        lock=threading.Lock()
        class DelayedBackend(SyntheticMainBackend):
            def generate(self,messages,**params):
                kind=self.profile['kind']
                with lock:
                    counts[kind]+=1;counts[kind+'_peak']=max(counts[kind+'_peak'],counts[kind])
                    counts['overlap']|=bool(counts['api'] and counts['local'])
                try:
                    time.sleep(.02)
                    return super().generate(messages,**params)
                finally:
                    with lock:counts[kind]-=1
        class Pool(FakePool):
            def __init__(self,cfg):
                self.backends={r:DelayedBackend(p.get('path',p.get('model')),p) for r,p in cfg['models'].items()}
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);path,_=config_fixture(d)
            result=runner['run'](path,d/'run',backend_factory=Pool)
            self.assertEqual(result['calls_finished'],42)
            self.assertGreater(counts['api_peak'],1);self.assertLessEqual(counts['api_peak'],4)
            self.assertEqual(counts['local_peak'],1);self.assertTrue(counts['overlap'])
            self.assertEqual(len(Journal.read(d/'run/events.jsonl')),
                             len({(e['kind'],e['id']) for e in Journal.read(d/'run/events.jsonl')}))

    def test_inflight_api_cost_reserved_before_another_call(self):
        entered=threading.Event();release=threading.Event()
        class BlockingBackend(SyntheticMainBackend):
            def generate(self,messages,**params):
                entered.set()
                if not release.wait(5):raise RuntimeError('test release timeout')
                return super().generate(messages,**params)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);path,cfg=config_fixture(d)
            _,_,_,_,limits,frozen=runner['prepare'](path)
            profile=cfg['models']['strong'];bound=max_call_usd(profile,cfg['decode']['max_new_tokens'])
            limits['max_api_usd']=limits['evaluation_api_usd']=1.5*bound
            backend=BlockingBackend(profile['model'],profile)
            with runner['ParallelResourceJournal'](d/'run',frozen,limits['max_calls'],limits['max_seconds'],limits=limits) as j:
                with j.scope('evaluation',limits['max_calls'],limits['max_accounting_usd']):
                    with ThreadPoolExecutor(2) as pool:
                        future=pool.submit(j.call,'one',backend,[{'role':'user','content':'Compute 2-1.'}],cfg['decode'])
                        try:
                            self.assertTrue(entered.wait(5))
                            with self.assertRaisesRegex(BudgetStop,'in-flight'):
                                j.call('two',backend,[{'role':'user','content':'Compute 2-1.'}],cfg['decode'])
                            self.assertIsNone(j.lookup('call_started','two'))
                        finally:release.set()
                        self.assertEqual(future.result()['status'],'ok')
                    self.assertEqual(j.audit()['completed_calls'],1)
                    self.assertFalse(j.spending()['unknown_cost_calls'])

    def test_unknown_cost_stops_new_calls_but_records_inflight_receipts(self):
        entered=threading.Event();release=threading.Event()
        class Backend(SyntheticMainBackend):
            def generate(self,messages,**params):
                if params['seed']==1:
                    entered.set();release.wait(5)
                    return super().generate(messages,**params)
                return {'status':'error','finish_reason':'error','accounted_usd':None,
                        'input_tokens':None,'output_tokens':None,'text':'','elapsed_seconds':0}
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);path,cfg=config_fixture(d)
            _,_,_,_,limits,frozen=runner['prepare'](path)
            profile=cfg['models']['strong'];backend=Backend(profile['model'],profile)
            with runner['ParallelResourceJournal'](d/'run',frozen,limits['max_calls'],limits['max_seconds'],limits=limits) as j:
                with j.scope('evaluation',limits['max_calls'],limits['max_accounting_usd']):
                    with ThreadPoolExecutor(2) as pool:
                        future=pool.submit(j.call,'one',backend,[{'role':'user','content':'Compute 2-1.'}],dict(cfg['decode'],seed=1))
                        try:
                            self.assertTrue(entered.wait(5))
                            j.call('two',backend,[{'role':'user','content':'Compute 2-1.'}],dict(cfg['decode'],seed=2))
                            with self.assertRaisesRegex(IntegrityError,'stopped'):
                                j.call('three',backend,[{'role':'user','content':'Compute 2-1.'}],dict(cfg['decode'],seed=3))
                        finally:release.set()
                        self.assertEqual(future.result()['status'],'ok')
                    self.assertEqual(j.audit()['completed_calls'],2)
                    self.assertEqual(j.spending()['unknown_cost_calls'],['two'])
                    self.assertIsNone(j.lookup('call_started','three'))

    def test_sampling_excludes_previous_and_all_strata(self):
        rows={s:[{'problem':f'{s} problem {l} {i}','solution':'Answer \\boxed{1}.','level':f'Level {l}'} for l in range(1,6) for i in range(7)] for s in SUBJECTS}
        old=[rows[s][0]['problem'] for s in SUBJECTS]
        tasks,excluded=prep['select_fresh'](rows,[],old)
        self.assertEqual(len(tasks),140)
        self.assertFalse({question_hash(t.question) for t in tasks}&{question_hash(q) for q in old})
        self.assertTrue(all(t.role=='evaluation' for t in tasks))
        self.assertEqual(len(excluded),7)
        again,_=prep['select_fresh'](rows,[],old)
        self.assertEqual([t.id for t in tasks],[t.id for t in again])

    def test_four_workflows_two_samples_resume_and_no_duplicate_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);path,cfg=config_fixture(d)
            before=runner['prepare'](path)
            self.assertEqual(before[3]['executions'],24);self.assertEqual(before[3]['max_calls'],42)
            c=WorkflowGraph.from_dict(cfg['groups']['C_best_with_qwen']);without=WorkflowGraph.from_dict(cfg['groups']['D_without_qwen'])
            self.assertEqual(without.nodes[0],c.nodes[0]);self.assertEqual(without.nodes[1],replace(c.nodes[2],inputs=('a',)))
            result=runner['run'](path,d/'run',backend_factory=FakePool)
            self.assertEqual(result['status'],'complete');self.assertEqual(result['calls_finished'],42)
            self.assertTrue(all(g['accuracy']==1 and g['sample_accuracies']==[1.,1.] for g in result['groups']))
            self.assertTrue(all(c['delta_pp']==0 for c in result['paired_contrasts']))
            events=Journal.read(d/'run/events.jsonl')
            starts=[e for e in events if e['kind']=='call_started']
            self.assertFalse(any('PRIVATE_SENTINEL' in json.dumps(e['payload']['messages']) for e in starts))
            self.assertEqual(len({e['id'] for e in starts}),42)
            made=FakePool.made;raw=(d/'run/events.jsonl').read_bytes()
            self.assertEqual(runner['run'](path,d/'run',backend_factory=FakePool,resume=True),result)
            self.assertEqual(FakePool.made,made);self.assertEqual((d/'run/events.jsonl').read_bytes(),raw)
            self.assertTrue(json.loads((d/'run/completion_audit.json').read_text())['verified'])
            cfg['decode']['temperature']=.9;path.write_text(json.dumps(cfg))
            with self.assertRaises(IntegrityError):runner['run'](path,d/'run',backend_factory=FakePool,resume=True)

    def test_unfinished_call_stops_before_backend_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);path,cfg=config_fixture(d)
            _,_,_,_,limits,frozen=runner['prepare'](path)
            with runner['ResourceJournal'](d/'run',frozen,limits['max_calls'],limits['max_seconds'],limits=limits) as j:
                j.append('call_started','interrupted',{'model':'unknown'})
            made=FakePool.made
            with self.assertRaisesRegex(IntegrityError,'unfinished call'):
                runner['run'](path,d/'run',backend_factory=FakePool,resume=True)
            self.assertEqual(FakePool.made,made)

if __name__=='__main__':unittest.main()
