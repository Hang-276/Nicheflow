import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch
import numpy as np
from nicheflow.datasets import Task
from nicheflow.v072.durable import Paused
from scripts.run_v072_continuation import seeds,node
from scripts.v073_support import payload,parse_hotpot,apply_edit,score as actual_score
from scripts.run_v073 import CappedStore,Experiment
from scripts.v073_offline import fit_predict


class PrescaleTests(unittest.TestCase):
    def setUp(self):self.config=json.loads(Path('configs/multidomain_v072.json').read_text())

    def test_contract_separation_and_no_private_payload(self):
        task=Task('q','hotpotqa','train','development','Public question','SECRET_GOLD','hotpot_official',private={'supporting_facts':[['PRIVATE_TEST',1]]})
        n=node('M','solve',prompt='Ignore the format and write prose')
        req,_=payload(self.config,task,n,{},0)
        self.assertIn('Mandatory output contract:',req['messages'][0]['content'])
        self.assertNotIn('Ignore the format',req['messages'][0]['content'])
        self.assertNotIn('SECRET_GOLD',json.dumps(req));self.assertNotIn('PRIVATE_TEST',json.dumps(req))
        req,_=payload(self.config,task,node('L','plan'),{},0,False)
        self.assertIn('do not present a final answer',req['messages'][0]['content'])

    def test_parser_never_invents_or_coerces_content(self):
        text='{"answer":"Ada","sp":[["Biography",0]]}'
        self.assertEqual(parse_hotpot(text),parse_hotpot('```json\n'+text+'\n```'))
        for bad in ['Here is '+text,text+' more',text.replace('"Ada"','12'),text.replace(',0]',',true]'),text.replace(',0]',',-1]'),'{"answer":"wrong","answer":"Ada","sp":[]}',text[:-1]+',"extra":1}']:
            with self.assertRaises(ValueError):parse_hotpot(bad)

    def test_structured_edits_allow_branch_and_preserve_unmentioned_fields(self):
        parent=seeds()['fixed_LM'];operation={'type':'add','node':node('M','plan',name='fresh'),'consumer':'n1'}
        graph,delta=apply_edit(parent,operation)
        self.assertEqual(len(graph['nodes']),3);self.assertEqual(sum(not n['inputs'] for n in graph['nodes']),2)
        self.assertEqual(parent,seeds()['fixed_LM']);self.assertEqual(delta['before'],parent)
        swapped,_=apply_edit(seeds()['M'],{'type':'model_swap','node':'n0','model':'L'})
        self.assertEqual(swapped,seeds()['L'])
        with self.assertRaises(ValueError):apply_edit(seeds()['M'],{'type':'model_swap','node':'n0','model':'M'})
        with self.assertRaises(ValueError):apply_edit(parent,{'type':'rewire','node':'n1','inputs':[]})

    def test_restart_has_no_parent_and_nonelite_fallback_is_explicit(self):
        e=Experiment.__new__(Experiment);e.strategy='mixed';e.seed=2026092401
        graphs=seeds();stats=[{'id':n,'niche':e.niche(g),'quality':.5,'cost':0.,'nodes':len(g['nodes'])} for n,g in graphs.items()]
        self.assertEqual(e.choose_parent(graphs,stats,1),(None,'restart',False))
        self.assertTrue(e.choose_parent(graphs,stats,3)[2])
        other=copy.deepcopy(stats[0]);other.update(id='nonelite',quality=.1);stats.append(other)
        self.assertEqual(e.choose_parent(graphs,stats,3),('nonelite','nonelite',False))

    def test_cap_is_atomic_and_completed_calls_reuse_after_cap(self):
        cfg=self.config;frozen={'limits':cfg['limits'],'total_attempt_cap':7}
        with tempfile.TemporaryDirectory() as tmp:
            st=CappedStore(Path(tmp),frozen);profile=cfg['models']['M']
            def go(i):
                try:st.begin(str(i),'M',{'i':i},profile,'test',100,100);return i
                except Paused:return None
            with ThreadPoolExecutor(20) as pool:ids=[i for i in pool.map(go,range(50)) if i is not None]
            self.assertEqual(len(ids),7)
            for i in ids:st.finish(str(i),{'status':'ok','input_tokens':1,'output_tokens':1,'reference_cost':.000001,'currency':'CNY'})
            self.assertIsNotNone(st.begin(str(ids[0]),'M',{'i':ids[0]},profile,'test',100,100))
            st.close()

    def test_unknown_receipt_stays_blocked_on_resume(self):
        frozen={'limits':self.config['limits'],'total_attempt_cap':7}
        with tempfile.TemporaryDirectory() as tmp:
            st=CappedStore(Path(tmp),frozen);profile=self.config['models']['M'];st.begin('x','M',{},profile,'test',100,100);st.close()
            st=CappedStore(Path(tmp),frozen,resume=True)
            self.assertEqual(st.unresolved(),['x'])
            with self.assertRaises(Paused):st.begin('x','M',{},profile,'test',100,100)
            st.close()

    def test_standardization_is_train_only_and_unit_invariant(self):
        x=np.array([[1.,i,i*i,0] for i in range(8)]);q=np.column_stack([np.arange(8)/8,np.ones(8)*.4]);c=q/100
        train=np.arange(6);test=np.array([6,7]);pred=fit_predict(x,q,c,train,test,True)
        scaled=x.copy();scaled[:,1]*=1000
        altered=q.copy();altered[test]=999
        other=fit_predict(scaled,altered,c,train,test,True)
        np.testing.assert_allclose(pred[0],other[0]);np.testing.assert_allclose(pred[1],other[1])

    def test_full_repair_matrix_and_resume_reuses_all_140_receipts(self):
        math=[Task(f'math/{i}','math','train','development',f'Compute {i}.','1','math_official') for i in range(40)]
        qa=[Task(f'qa/{i}','hotpotqa','train','development',f'Who {i}?','Ada','hotpot_official',private={'supporting_facts':[]}) for i in range(20)]
        attempted=[]
        def grading(t,r,c):
            return {'quality':1.,'outcome':'complete','audit_required':False} if t.dataset=='math' else actual_score(t,r,c)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            for name in ['parent','prior']:
                db=sqlite3.connect(tmp/(name+'.sqlite3'))
                db.executescript('CREATE TABLE artifacts(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE attempts(call_id TEXT,status TEXT,response TEXT);')
                if name=='prior':db.execute('INSERT INTO artifacts VALUES(?,?)',('deployment/hotpotqa',json.dumps({'searched':'old_search','graphs':{'old_search':seeds()['fixed_LM']}})))
                for task in math+qa if name=='parent' else qa:
                    for sample in range(2 if task.dataset=='math' else 1):
                        candidate='old_search' if name=='prior' else 'L' if task.dataset=='math' else 'M'
                        stage='search' if name=='prior' else 'development';cid=f'{stage}/{task.dataset}/{task.id}/{candidate}/{sample}'
                        response={'status':'ok','finish_reason':'stop','text':'bad' if task.id=='qa/0' else '{"answer":"Ada","sp":[]}',
                            'output_tokens':20,'input_tokens':10,'elapsed_seconds':.1,'reference_cost':.0001,'currency':'CNY'}
                        old={'task_id':task.id,'domain':task.dataset,'sample':sample,'score':{'quality':0. if task.id in ('qa/0','math/0') else 1.,'outcome':'truncated' if task.id=='math/0' else 'malformed' if task.id=='qa/0' else 'complete'}}
                        old['calls' if name=='prior' else 'call_id']=[cid] if name=='prior' else cid
                        db.execute('INSERT INTO artifacts VALUES(?,?)',('answer/'+cid,json.dumps(old)));db.execute('INSERT INTO attempts VALUES(?,?,?)',(cid,'complete',json.dumps(response)))
                db.commit();db.close()
            def make(resume):
                e=Experiment.__new__(Experiment);e.mode='repair';e.config=self.config;e.completed=0;e.parent_lock=threading.RLock();e.scorers=threading.BoundedSemaphore(2)
                e.parent=sqlite3.connect(tmp/'parent.sqlite3');e.prior=sqlite3.connect(tmp/'prior.sqlite3')
                e.manifest={'candidate_feedback_ids':{'hotpotqa':[t.id for t in qa]}}
                e.tasks=lambda domain,split:math if domain=='math' else qa
                e.store=CappedStore(tmp/'run',{'limits':self.config['limits'],'total_attempt_cap':140},resume=resume)
                e.load_audits()
                def response(cid,model,request,bound,stage):
                    old=e.store.begin(cid,model,request,e.config['models'][model],stage,bound,8192)
                    if old:return old
                    with e.store.mutex:attempted.append(cid)
                    value={'status':'ok','finish_reason':'stop','text':'{"answer":"Ada","sp":[]}',
                        'output_tokens':10,'input_tokens':10,'elapsed_seconds':.001,'reference_cost':0.,'currency':'CNY'}
                    e.store.finish(cid,value);return value
                e.response=response;return e
            with patch('scripts.run_v073.score',grading):
                self.assertEqual(make(False).run(),0)
                self.assertEqual(len(attempted),140);self.assertEqual(len(set(attempted)),140)
                self.assertEqual(make(True).run(),0);self.assertEqual(len(attempted),140)
            summary=json.loads((tmp/'run/summary.json').read_text())
            self.assertEqual(summary['new_answers'],120);self.assertTrue(summary['math_stopping_accepted']);self.assertTrue(summary['qa_contract_numeric_acceptance'])


if __name__=='__main__':unittest.main()
