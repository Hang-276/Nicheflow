import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from nicheflow.graph import Node, WorkflowGraph, proposal_messages, parse_proposal
from nicheflow.main_entry import run_main
from nicheflow.ledger import Journal
from test_main import ROOT, SyntheticPool


class GenerationContractTests(unittest.TestCase):
    def test_format_is_explicitly_a_paid_llm_and_null_still_rejected(self):
        parent = WorkflowGraph((Node('a', 'Generate'),), 'a')
        messages = proposal_messages([parent], ['local', 'strong'], allowed_operators=['Generate','Format','Identity','Join'])
        contracts = json.loads(messages[1]['content'])['operator_contracts']
        self.assertEqual(contracts['Format']['execution'], 'LLM')
        self.assertEqual(contracts['Format']['model'], ['local','strong'])
        self.assertIsNone(contracts['Identity']['model'])
        graph = {'nodes':[{'id':'a','operator':'Generate','model':'local'},
                          {'id':'f','operator':'Format','inputs':['a'],'model':None}], 'output':'f'}
        with self.assertRaisesRegex(ValueError, 'node f.*Format.*None'):
            parse_proposal(json.dumps(graph), [parent], ['local','strong'])
        graph['nodes'][1]['model'] = 'strong'
        result = parse_proposal(json.dumps(graph), [parent], ['local','strong'])
        self.assertEqual(result.model_calls, 2)
        graph['nodes'][1]['model'] = 'invented-model'
        with self.assertRaises(ValueError): parse_proposal(json.dumps(graph), [parent], ['local','strong'])

    def test_real_data_six_rounds_format_candidates_and_terminal_grid(self):
        class FormatPool(SyntheticPool):
            def __init__(self, config):
                super().__init__(config)
                for backend in self.backends.values():
                    original = backend.generate
                    def generate(messages, _original=original, **params):
                        r = _original(messages, **params)
                        if 'Propose one executable' in messages[0]['content']:
                            r['text'] = json.dumps({'nodes':[
                                {'id':'a','operator':'Generate','model':'local','prompt':f'Solve variant {params["seed"]}'},
                                {'id':'f','operator':'Format','model':'strong','inputs':['a'],'prompt':'Format according to output_contract.'}], 'output':'f'})
                        return r
                    backend.generate = generate
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            args = SimpleNamespace(config=str(ROOT/'configs/main_math.json'), rounds=6, run_dir=d)
            r, code = run_main(args, ROOT, FormatPool)
            self.assertEqual(code,0)
            self.assertEqual(r['rounds_completed'],6)
            self.assertEqual(r['evaluation']['completed_workflows'],105)
            self.assertTrue(r['evaluation']['learning_state_unchanged'])
            self.assertEqual(r['numerical_health']['snapshots'],7)
            self.assertTrue(r['numerical_health']['all_finite_spd'])
            events = Journal.read(Path(d)/'events.jsonl')
            self.assertTrue(any(e['kind']=='generation' and e['payload']['status']=='valid' for e in events))
            self.assertTrue(any(e['kind']=='archive' and e['id'].startswith('round:') for e in events))
            self.assertTrue(any(e['kind']=='call_finished' and e['id'].endswith(':f') for e in events))
            instances=len(FormatPool.instances)
            _,code=run_main(args,ROOT,FormatPool)
            self.assertEqual(code,0)
            self.assertEqual(len(FormatPool.instances),instances)

    def test_six_round_launch_is_forwarded_without_running_real_models(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'scripts').mkdir();(root/'bin').mkdir()
            for name in ['start_main_tmux.sh','run_main_once.sh']:
                shutil.copy(ROOT/'scripts'/name,root/'scripts'/name)
            (root/'scripts/nicheflow.sh').write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> invoked.txt\n')
            tmux=root/'bin/tmux';tmux.write_text('#!/bin/bash\nif [[ "$1" == has-session ]]; then exit 1; fi\nif [[ "$1" == list-sessions ]]; then exit 0; fi\nprintf "%s\\n" "$*" >> tmux_calls.txt\n');tmux.chmod(0o755)
            env={**os.environ,'PATH':str(root/'bin')+':'+os.environ['PATH']}
            r=subprocess.run(['bash','scripts/start_main_tmux.sh','quick','6'],cwd=root,env=env,capture_output=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('--rounds 6',(root/'invoked.txt').read_text())
            self.assertIn('run_main_once.sh quick 6',(root/'tmux_calls.txt').read_text())
            r=subprocess.run(['bash','scripts/run_main_once.sh','quick','6'],cwd=root,env=env,capture_output=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('--rounds 6 --run-dir runs/main_v044_quick_r6',(root/'invoked.txt').read_text())
            self.assertTrue((root/'runs/main_v044_quick_r6/exit_code.txt').exists())
            for bad in ['0','-1','6;echo bad']:
                r=subprocess.run(['bash','scripts/start_main_tmux.sh','quick',bad],cwd=root,env=env,capture_output=True)
                self.assertEqual(r.returncode,2)
