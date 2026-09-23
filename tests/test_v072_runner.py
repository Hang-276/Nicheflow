"""End-to-end quota interruption and continuation with fake model endpoints."""
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
from nicheflow.datasets import Task
from scripts import run_v072 as runner

ROOT=Path(__file__).resolve().parents[1]

class DevelopmentResumeTests(unittest.TestCase):
    def test_full_matrix_resumes_after_quota_without_repeating_completed_calls(self):
        cfg=json.loads((ROOT/'configs/multidomain_v072.json').read_text())
        frozen={'limits':cfg['limits'],'synthetic':True}
        calls=Counter();mutex=threading.Lock();did_reject=False
        def model(profile,payload,key):
            nonlocal did_reject
            with mutex:
                calls[profile['model']]+=1
                reject=profile['model']==cfg['models']['H']['model'] and calls[profile['model']]==4 and not did_reject
                if reject:did_reject=True
            if reject:return {'status':'http_error','http_status':429,'error_code':'insufficient_quota','reference_cost':None,'currency':profile['currency']}
            return {'status':'ok','finish_reason':'stop','text':'saved answer','input_tokens':20,'output_tokens':10,
                    'reference_cost':(20*profile['input_per_million']+10*profile['output_per_million'])/1e6,
                    'currency':profile['currency'],'elapsed_seconds':.1}
        def tasks(path):
            domain=Path(path).parent.name
            return [Task(f'{domain}/{i}',domain,'train','development',f'Public {domain} task {i}.','private label',
                {'math':'math_official','mbpp':'python_tests','hotpotqa':'hotpot_official'}[domain]) for i in range(40)]
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            for name,value in [('freeze',lambda c:frozen),('credentials',lambda p:{'DASHSCOPE_API_KEY':'test','DEEPSEEK_API_KEY':'test'}),
                               ('load_tasks',tasks),('call_model',model),('run_score',lambda *args:{'quality':1.,'outcome':'complete','audit_required':False})]:
                stack.enter_context(patch.object(runner,name,value))
            stack.enter_context(patch.object(runner.signal,'signal'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            directory=Path(tmp)/'run'
            self.assertEqual(runner.run(cfg,directory,False),2)
            first=json.loads((directory/'progress.json').read_text())
            self.assertIsNotNone(first['pause'])
            self.assertGreater(sum(x['complete'] for x in first['totals']['models'].values()),0)
            self.assertEqual(runner.run(cfg,directory,True),0)
            report=json.loads((directory/'summary.json').read_text())
            self.assertEqual(report['answers'],960);self.assertEqual(len(report['groups']),12)
            self.assertFalse(report['final_evaluation_opened']);self.assertEqual(report['search_rounds_started'],0)
            with sqlite3.connect(directory/'state.sqlite3') as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM attempts WHERE status='complete'").fetchone()[0],972)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM attempts WHERE status='rejected'").fetchone()[0],1)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM calls").fetchone()[0],972)
                self.assertEqual(db.execute("SELECT MAX(n) FROM (SELECT COUNT(*) n FROM attempts WHERE status='complete' GROUP BY call_id)").fetchone()[0],1)
            self.assertEqual(sum(calls.values()),973)
