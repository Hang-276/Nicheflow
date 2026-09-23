import json
from dataclasses import replace
from pathlib import Path
import unittest
from nicheflow.datasets import Task
from scripts.run_v072 import payload

ROOT=Path(__file__).resolve().parents[1]

class V072ProtocolTests(unittest.TestCase):
    def setUp(self):self.cfg=json.loads((ROOT/'configs/multidomain_v072.json').read_text())
    def test_selected_models_and_modes_keep_gold_out_of_requests(self):
        t=Task('m','math','train','development','Public question','PRIVATE_GOLD_SENTINEL','math_official',private={'solution':'PRIVATE_REFERENCE_SENTINEL'})
        for role,profile in self.cfg['models'].items():
            req,bound=payload(self.cfg,profile,t,1)
            raw=json.dumps(req)
            self.assertNotIn('PRIVATE_GOLD_SENTINEL',raw);self.assertNotIn('PRIVATE_REFERENCE_SENTINEL',raw)
            self.assertEqual(req['max_tokens'],8192);self.assertGreater(bound,0)
            if role=='H':self.assertEqual(req['reasoning_effort'],'none');self.assertNotIn('enable_thinking',req)
            elif role=='D':self.assertEqual(req['thinking'],{'type':'disabled'})
            elif role=='L':self.assertEqual(req['chat_template_kwargs'],{'enable_thinking':False})
            else:self.assertFalse(req['enable_thinking'])
    def test_code_exposes_only_declared_public_example(self):
        t=Task('c','mbpp','train','development','Implement f','SECRET_CODE','python_tests',
               public_tests=('assert f(1)==2',),private={'tests':['assert f(99)==100'],'evalplus':{'plus_input':'SECRET_CASES'}})
        req,_=payload(self.cfg,self.cfg['models']['M'],t,0);raw=json.dumps(req)
        self.assertIn('assert f(1)==2',raw)
        for forbidden in ['assert f(99)==100','SECRET_CODE','SECRET_CASES']:self.assertNotIn(forbidden,raw)
    def test_hotpot_context_sentence_indices_are_visible_but_support_gold_is_not(self):
        t=Task('q','hotpotqa','train','development','Question','SECRET_ANSWER','hotpot_official',
            context=[{'title':'A','sentences':[{'id':0,'text':'Sentence'}]}],private={'supporting_facts':[['SECRET_SUPPORT',2]]})
        req,_=payload(self.cfg,self.cfg['models']['H'],t,0);raw=json.dumps(req)
        self.assertIn('Sentence',raw);self.assertNotIn('SECRET_ANSWER',raw);self.assertNotIn('SECRET_SUPPORT',raw)
