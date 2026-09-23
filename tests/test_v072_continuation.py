import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch
from nicheflow.datasets import Task
from nicheflow.v072.durable import Store
from scripts import run_v072_continuation as runner


class ContinuationProtocolTests(unittest.TestCase):
    def test_audit_decisions_are_bound_complete_and_separate_from_raw_scores(self):
        cfg=json.loads((runner.ROOT/'configs/multidomain_v072.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            e=runner.Experiment.__new__(runner.Experiment)
            e.store=Store(Path(tmp),{'limits':cfg['limits']})
            key='answer/seeds/math/task/fixed_LM/0'
            raw={'domain':'math','task_id':'task','sample':0,'candidate':'fixed_LM',
                 'score':{'quality':0.,'audit_required':True,'outcome':'complete'}}
            e.store.artifact(key,raw,write=True)
            with self.assertRaises(runner.Paused):e.gate('seeds')
            queue=json.loads((Path(tmp)/'audit_seeds.json').read_text())
            approval={'queue_sha256':runner.digest(queue),'reviewed_rows':{key:{'quality':1.,'reason':'independent equivalence check'}}}
            (Path(tmp)/'audit_seeds.approved.json').write_text(json.dumps(approval))
            e.load_audits();e.gate('seeds')
            self.assertEqual(e.store.artifact(key)['score']['quality'],0.)
            self.assertEqual(e.effective(key,raw)['score']['quality'],1.)
            approval['reviewed_rows'][key]['quality']=0.
            (Path(tmp)/'audit_seeds.approved.json').write_text(json.dumps(approval))
            with self.assertRaises(AssertionError):e.load_audits()
            e.store.close()

    def test_statistics_do_not_depend_on_thread_completion_order(self):
        e=runner.Experiment.__new__(runner.Experiment)
        rows=[{'candidate':'M','task_id':str(i),'sample':0,'score':{'quality':q,'outcome':'complete'},'cost_cny':q}
              for i,q in enumerate([1.,1e-16,1e-16,.3333333333333333,.123456789])]
        self.assertEqual(e.stats('M',runner.seeds()['M'],rows),e.stats('M',runner.seeds()['M'],list(reversed(rows))))

    def test_real_dispatch_concurrency_and_quota_resume(self):
        cfg=json.loads((runner.ROOT/'configs/multidomain_v072.json').read_text())
        attempted=[];guard=threading.Lock();quota={'failed':False}
        def provider(profile,request,key):
            ident=request['messages'][0]['content']
            with guard:
                attempted.append(ident)
                rejected=ident=='job3' and not quota['failed']
                if rejected:quota['failed']=True
            if rejected:return {'status':'http_error','http_status':429,'reference_cost':None,'currency':'CNY'}
            time.sleep(.002)
            return {'status':'ok','finish_reason':'stop','text':'answer','input_tokens':10,'output_tokens':10,
                    'reference_cost':.00001,'currency':'CNY','elapsed_seconds':.002}
        def setup(directory,resume):
            e=runner.Experiment.__new__(runner.Experiment);e.config=cfg;e.keys={'DASHSCOPE_API_KEY':'fake'}
            e.store=Store(directory,{'limits':cfg['limits']},resume=resume)
            e.api=threading.BoundedSemaphore(4);e.local=threading.BoundedSemaphore(1)
            e.stop=threading.Event();e.start=time.monotonic();return e
        def dispatch(e):
            def job(i):
                try:return e.response('job'+str(i),'M',{'messages':[{'content':'job'+str(i)}],'max_tokens':8192},100,'test')
                except runner.Paused:return None
            with ThreadPoolExecutor(10) as pool:return list(pool.map(job,range(100)))
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'call_model',provider):
            e=setup(Path(tmp),False);dispatch(e)
            completed={r[0] for r in e.store.db.execute("SELECT call_id FROM attempts WHERE status='complete'")}
            self.assertTrue(completed);self.assertTrue(e.store.get_meta('pause'));e.store.close()
            e=setup(Path(tmp),True);out=dispatch(e)
            self.assertTrue(all(r and r['status']=='ok' for r in out));self.assertFalse(e.store.unresolved())
            self.assertEqual(e.store.db.execute("SELECT count(*) FROM attempts WHERE status='complete'").fetchone()[0],100)
            self.assertTrue(all(attempted.count(k)==1 for k in completed));e.store.close()

    def test_graph_limits_canonicalization_and_private_separation(self):
        graphs = runner.seeds()
        for g in graphs.values(): self.assertEqual(g, runner.validate(g))
        self.assertEqual(runner.validate({'nodes': [runner.node('L','solve',name='renamed')], 'output':'renamed'}), graphs['L'])
        for bad in [
            {'nodes':[runner.node('H','solve'),runner.node('H','review',inputs=['n0'],name='n1')],'output':'n1'},
            {'nodes':[runner.node('L','solve'),runner.node('M','solve',name='n1')],'output':'n1'},
            {'nodes':[runner.node('L','plan')],'output':'n0'},
        ]:
            with self.assertRaises(ValueError): runner.validate(bad)
        cfg=json.loads((runner.ROOT/'configs/multidomain_v072.json').read_text())
        task=Task('m','math','train','development','Public problem','HIDDEN_GOLD','math_official',private={'solution':'SECRET_SOLUTION'})
        req,_=runner.workflow_payload(cfg,task,graphs['fixed_HL']['nodes'][0],{},0,False)
        self.assertNotIn('HIDDEN_GOLD',json.dumps(req));self.assertNotIn('SECRET_SOLUTION',json.dumps(req))
        self.assertNotIn('Final answer in',req['messages'][1]['content'])
        req,_=runner.workflow_payload(cfg,task,graphs['fixed_HL']['nodes'][1],{'n0':'Visible plan'},0,True)
        self.assertIn('Visible plan',req['messages'][1]['content']);self.assertIn('Final answer in',req['messages'][1]['content'])

    def test_full_stage_sequence_and_resume_no_duplicate_execution(self):
        cfg=json.loads((runner.ROOT/'configs/multidomain_v072.json').read_text())
        cfg['limits']={'models':{m:{'calls':15000,'input_tokens':20000000,'output_tokens':10000000} for m in cfg['models']},'currency':{'CNY':150,'USD':5}}
        calls=[];mutex=threading.Lock()
        class FakeParent:
            def close(self): pass
        def tasks(domain,split):
            return [Task(f'{domain}/{split}/{i}',domain,'test' if split=='final' else 'train','evaluation' if split=='final' else 'development',
                f'Question {i}','private','math_official') for i in range({'development':40,'screening':20,'calibration':40,'final':100}[split])]
        def make(directory,resume):
            e=runner.Experiment.__new__(runner.Experiment); e.config=cfg
            e.store=Store(directory,{'limits':cfg['limits'],'test':True},resume=resume)
            e.parent=FakeParent(); e.completed=0; e.scorers=threading.BoundedSemaphore(2)
            e.manifest={'candidate_feedback_ids':{d:[t.id for t in tasks(d,'development')[:20]] for d in runner.DOMAINS}}
            def load(d,s):
                if s=='final': self.assertIsNotNone(e.store.artifact('final_protocol_frozen'))
                return tasks(d,s)
            e.tasks=load
            def response(ident,model,request,bound,stage):
                with mutex:
                    old=e.store.receipt(ident)
                    if old is not None:return old
                    profile=cfg['models'][model]
                    e.store.begin(ident,model,request,profile,stage,bound,8192)
                    calls.append(ident)
                    text='answer'
                    if stage=='proposal':
                        graph={'nodes':[runner.node('M','solve',prompt='Reusable variant '+ident)],'output':'n0'}
                        text=json.dumps(graph)
                    r={'status':'ok','finish_reason':'stop','text':text,'input_tokens':10,'output_tokens':10,
                       'reference_cost':(10*profile['input_per_million']+10*profile['output_per_million'])/1e6,
                       'currency':profile['currency'],'elapsed_seconds':.1}
                    e.store.finish(ident,r);return r
            e.response=response
            e.reference_checks=lambda split:e.store.artifact('reference_checks/'+split,{'passed':True},write=True)
            real_baseline=e.baseline
            def baseline(stage,t,m,s):
                if stage!='development':return real_baseline(stage,t,m,s)
                return {'candidate':m,'domain':t.dataset,'task_id':t.id,'sample':s,'score':{'quality':.5,'outcome':'complete','audit_required':False},
                        'cost_cny':0 if m=='L' else .001,'calls':[],'elapsed_seconds':.1,'parent_receipt':True}
            e.baseline=baseline
            return e
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'run_score',lambda *args:{'quality':1.,'outcome':'complete','audit_required':False}),contextlib.redirect_stdout(io.StringIO()):
            e=make(Path(tmp)/'run',False)
            # Reduce expensive progress/backup I/O, preserving actual receipts and artifacts.
            e.store.progress=lambda *a,**k:None;e.store.backup=lambda:None
            self.assertEqual(e.run(),0)
            count=len(calls);self.assertEqual(len(set(calls)),count)
            e=make(Path(tmp)/'run',True);e.store.progress=lambda *a,**k:None;e.store.backup=lambda:None
            self.assertEqual(e.run(),0);self.assertEqual(len(calls),count)


if __name__=='__main__':unittest.main()
