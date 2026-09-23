import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from test_main import fixture, SyntheticPool, ROOT
from nicheflow.graph import Node, WorkflowGraph, proposal_messages
from nicheflow.ledger import Journal
from nicheflow.main_entry import run_main, plan
from nicheflow.mutations import validate_mutation
from nicheflow.research_policy import ParentEstimate, research_seeds
from nicheflow.semantic import SemanticFeatures
from nicheflow.spec import EnvironmentBlocked, IntegrityError, digest


class FakeEncoder:
    """Synthetic feature fixture, never a production semantic fallback."""
    def __init__(self, spec): self.dimension = spec['dimension']
    def encode(self, text):
        value = np.array(list(hashlib.sha256(text.encode()).digest())[:self.dimension], float) - 127
        return (value / np.linalg.norm(value)).tolist()


class Pool(SyntheticPool):
    semantic_encoder_factory = FakeEncoder


def revised_fixture(directory, rounds=2, deferred=True):
    args, config = fixture(directory, rounds, deferred)
    revision = json.loads((ROOT / 'configs/main_math_v050.json').read_text())
    for k in ('models', 'research_revision', 'semantic', 'generation_model', 'workflow_output_policy'):
        config[k] = revision[k]
    # A clearly synthetic identity supplies the unresolved API slot for offline tests.
    config['models']['middle'].update(configuration_status='configured', model='synthetic-middle-fixture',
        accepted_response_models=['synthetic-middle-fixture'], pricing=copy.deepcopy(config['models']['strong']['pricing']))
    Path(args.config).write_text(json.dumps(config))
    args.stop_after_round, args.resume_from = None, None
    return args, config


def state(directory, round_index):
    return json.loads((Path(directory) / 'checkpoints' / f'round_{round_index:04d}.json').read_text())['state']


class ResearchTests(unittest.TestCase):
    def test_semantics_are_public_only_and_model_identity_is_visible(self):
        config = json.loads((ROOT / 'configs/main_math_v050.json').read_text())
        feature = SemanticFeatures(config, FakeEncoder(config['semantic']))
        graph = research_seeds(config)[0]
        a = feature.route({'question':'Find area of a triangle', 'gold':'12', 'private':{'answer':'12'}}, graph)
        b = feature.route({'question':'Find area of a triangle', 'gold':'999', 'private':{'answer':'999'}}, graph)
        c = feature.route({'question':'Count primes in a sequence'}, graph)
        self.assertEqual(a,b)
        self.assertNotEqual(a,c)
        self.assertLessEqual(np.linalg.norm(a), 1.000000001)
        self.assertNotEqual(feature.workflow(graph), feature.workflow(research_seeds(config)[1]))

    def test_mutation_rejects_noop_and_replacement_of_entire_identity(self):
        parent = WorkflowGraph((Node('a','Generate','Solve.'),),'a')
        with self.assertRaisesRegex(ValueError,'unchanged'):
            validate_mutation([parent], parent)
        other = WorkflowGraph((Node('b','Generate','Solve.'),),'b')
        with self.assertRaisesRegex(ValueError,'retain'):
            validate_mutation([parent], other)
        child = WorkflowGraph((Node('a','Generate','Verify arithmetic.'),),'a')
        self.assertEqual(validate_mutation([parent], child)['replaced'], ['a'])

    def test_generation_explicitly_defines_topology_labels(self):
        parent=WorkflowGraph((Node('a','Generate','Solve.'),),'a')
        content=json.loads(proposal_messages([parent],['local','middle','strong'])[1]['content'])
        self.assertEqual(content['topology_contract']['allowed_values'],[None,'NGT-Independent','Subgroup'])
        invalid=WorkflowGraph(parent.nodes,'a',topology_primitive='Sequential')
        with self.assertRaisesRegex(ValueError,'unknown topology'):
            invalid.validate()

    def test_source_feedback_and_rescheduling_are_observable(self):
        with tempfile.TemporaryDirectory() as d:
            args,_ = revised_fixture(d)
            summary, code = run_main(args, ROOT, Pool)
            self.assertEqual(code,0)
            events = Journal.read(Path(args.run_dir)/'events.jsonl')
            final = state(args.run_dir,2)
            credits = [e['payload'] for e in events if e['kind']=='parent_credit']
            self.assertTrue(credits)
            counts={}
            for c in credits: counts[str(c['parent_niche'])]=counts.get(str(c['parent_niche']),0)+1
            self.assertEqual({k:v['count'] for k,v in final['search_estimates'].items() if v['count']}, counts)
            self.assertEqual(len([e for e in events if e['kind']=='scheduler']),4)
            contexts=[e['payload'] for e in events if e['kind']=='block_context']
            self.assertEqual([c['remaining_blocks'] for c in contexts],[4,3,2,1])
            messages=[e['payload']['messages'] for e in events if e['kind']=='call_started' and e['id'].endswith(':generate')]
            feedback=json.loads(messages[0][1]['content'])['parent_development_feedback']
            self.assertEqual(feedback['role'],'development')
            self.assertIn('failure_counts',feedback)
            self.assertFalse(feedback['has_test_labels'])
            boundaries=[e['payload']['boundaries'] for e in events if e['kind']=='archive_rebin' and e['payload']['frozen_after_bootstrap']]
            self.assertTrue(all(x==final['archive']['boundaries'] for x in boundaries))

    def test_pause_restore_matches_uninterrupted_learning_without_duplicate_calls(self):
        with tempfile.TemporaryDirectory() as d:
            args,_ = revised_fixture(d,rounds=4)
            complete=Path(d)/'complete'; args.run_dir=str(complete)
            self.assertEqual(run_main(args,ROOT,Pool)[1],0)
            first=Path(d)/'first'; args.run_dir=str(first);args.stop_after_round=2
            summary,code=run_main(args,ROOT,Pool)
            self.assertEqual((code,summary['status']),(0,'stage_paused'))
            source_bytes=(first/'events.jsonl').read_bytes()
            second=Path(d)/'second';args.run_dir=str(second);args.resume_from=str(first);args.stop_after_round=4
            summary,code=run_main(args,ROOT,Pool)
            self.assertEqual(code,0)
            self.assertEqual(summary['rounds_completed'],4)
            self.assertEqual(digest(state(complete,4)),digest(state(second,4)))
            self.assertEqual(source_bytes,(first/'events.jsonl').read_bytes())
            events=Journal.read(second/'events.jsonl')
            self.assertFalse(any(e['kind']=='call_started' and e['id'].startswith('bootstrap') for e in events))
            actual=[e['id'] for p in (first,second) for e in Journal.read(p/'events.jsonl') if e['kind']=='call_started']
            expected=[e['id'] for e in Journal.read(complete/'events.jsonl') if e['kind']=='call_started']
            self.assertEqual(actual,expected)

    def test_horizon_extension_accepts_only_unchanged_learning_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            args,config=revised_fixture(d,rounds=1)
            first=Path(args.run_dir)
            self.assertEqual(run_main(args,ROOT,Pool)[1],0)
            args.resume_from=str(first);args.run_dir=str(Path(d)/'extension');args.rounds=3
            self.assertEqual(run_main(args,ROOT,Pool)[1],0)
            config['semantic']['dimension']=9
            Path(args.config).write_text(json.dumps(config))
            args.run_dir=str(Path(d)/'bad')
            with self.assertRaisesRegex(IntegrityError,'horizon only'):
                run_main(args,ROOT,Pool)
            self.assertFalse(Path(args.run_dir).exists())

    def test_frozen_evaluation_retains_ucb_and_does_not_update_semantic_heads(self):
        with tempfile.TemporaryDirectory() as d:
            args,_=revised_fixture(d,rounds=1,deferred=False)
            summary,code=run_main(args,ROOT,Pool)
            self.assertEqual(code,0)
            self.assertTrue(summary['evaluation']['learning_state_unchanged'])
            events=Journal.read(Path(args.run_dir)/'events.jsonl')
            route=next(e['payload']['decision'] for e in events if e['kind']=='evaluation_route')
            self.assertTrue(any(v['quality']['width']>0 for v in route['scores'].values()))

    def test_resume_after_paid_call_replays_without_new_duplicate_charge(self):
        with tempfile.TemporaryDirectory() as d:
            args,_=revised_fixture(d)
            append=Journal.append;crashed=[False]
            def interrupt(j,k,key,payload):
                value=append(j,k,key,payload)
                if k=='parent_credit' and not crashed[0]:
                    crashed[0]=True
                    raise KeyboardInterrupt('simulated crash')
                return value
            with patch.object(Journal,'append',interrupt), self.assertRaises(KeyboardInterrupt):
                run_main(args,ROOT,Pool)
            before={e['id'] for e in Journal.read(Path(args.run_dir)/'events.jsonl') if e['kind']=='call_started'}
            self.assertEqual(run_main(args,ROOT,Pool)[1],0)
            events=Journal.read(Path(args.run_dir)/'events.jsonl')
            ids=[e['id'] for e in events if e['kind']=='call_started']
            self.assertEqual(len(ids),len(set(ids)))
            self.assertTrue(before <= set(ids))

    def test_cold_estimate_does_not_invent_observations(self):
        estimate=ParentEstimate()
        self.assertGreaterEqual(estimate.ucb(1,1),1)
        self.assertEqual(estimate.count,0)

    def test_pending_middle_blocks_execution_even_if_credentials_exist(self):
        config_path=ROOT/'configs/main_math_v050.json'
        with tempfile.TemporaryDirectory() as d, patch('nicheflow.main_entry.ModelPool', side_effect=AssertionError('must not load')):
            result=plan(config_path,ROOT)
            self.assertFalse(result['ready_for_configured_run'])
            self.assertIsNone(result['limits'])
            self.assertTrue(any('model selection pending' in p for p in result['problems']))
            args=SimpleNamespace(config=config_path,rounds=1,run_dir=str(Path(d)/'blocked'))
            with self.assertRaisesRegex(EnvironmentBlocked,'model selection pending'):
                run_main(args,ROOT)
            self.assertFalse(Path(args.run_dir).exists())
        from nicheflow.main_models import MainAPI
        with self.assertRaisesRegex(EnvironmentBlocked,'selection is pending'):
            MainAPI(json.loads(config_path.read_text())['models']['middle'])

    def test_reserved_dashscope_interface_payload_identity_and_usage(self):
        from nicheflow.main_models import MainAPI
        with tempfile.TemporaryDirectory() as d:
            _,config=revised_fixture(d)
        profile=config['models']['middle']
        response={'model':'synthetic-middle-fixture','choices':[{'message':{'content':'answer'},'finish_reason':'stop'}],
                  'usage':{'prompt_tokens':10,'completion_tokens':4}}
        with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'synthetic-test-key'}), patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as http:
            result=MainAPI(profile).generate([{'role':'user','content':'hello'}],seed=9,max_new_tokens=20,temperature=.7,top_p=.8)
        request=http.call_args.args[0];sent=json.loads(request.data)
        self.assertEqual(request.full_url,profile['endpoint'])
        self.assertEqual(sent['model'],'synthetic-middle-fixture')
        self.assertEqual(sent['top_p'],.8)
        self.assertNotIn('seed',sent)
        self.assertNotIn('thinking',sent)
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['usage'],response['usage'])
        self.assertGreater(result['accounted_usd'],0)
        response['model']='unapproved-model'
        with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'synthetic-test-key'}), patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
            result=MainAPI(profile).generate([{'role':'user','content':'hello'}],seed=9,max_new_tokens=20,temperature=.7,top_p=.8)
        self.assertEqual(result['status'],'error')
        self.assertGreater(result['accounted_usd'],0)

    def test_config_key_order_cannot_change_seed_order_or_feature_columns(self):
        config=json.loads((ROOT/'configs/main_math_v050.json').read_text())
        reordered=json.loads(json.dumps(config,sort_keys=True))
        self.assertEqual(digest(config),digest(reordered))
        self.assertEqual([g.version for g in research_seeds(config)],[g.version for g in research_seeds(reordered)])
        for graph in research_seeds(config):
            a=SemanticFeatures(config,FakeEncoder(config['semantic']))
            b=SemanticFeatures(reordered,FakeEncoder(reordered['semantic']))
            self.assertEqual(a.route({'question':'Compute 2 + 3'},graph),b.route({'question':'Compute 2 + 3'},graph))

    def test_independent_evaluation_of_deferred_checkpoint_preserves_learning(self):
        from nicheflow.research_eval import evaluate
        with tempfile.TemporaryDirectory() as d:
            args,config=revised_fixture(d,rounds=2,deferred=True)
            self.assertEqual(run_main(args,ROOT,Pool)[1],0)
            original=(Path(args.run_dir)/'events.jsonl').read_bytes()
            config['evaluation']['status']='configured'
            config['limits'].update(evaluation_seconds=3600,evaluation_api_usd=1)
            eval_config=Path(d)/'eval.json';eval_config.write_text(json.dumps(config))
            result=evaluate(ROOT,args.run_dir,eval_config,Path(d)/'eval',backend_factory=Pool)
            self.assertEqual(result['evaluation']['status'],'complete')
            self.assertTrue(result['evaluation']['learning_state_unchanged'])
            self.assertEqual(original,(Path(args.run_dir)/'events.jsonl').read_bytes())
            self.assertEqual(result['coverage']['router_updates'],0)
            with self.assertRaisesRegex(IntegrityError,'output must be new'):
                evaluate(ROOT,args.run_dir,eval_config,Path(d)/'eval',backend_factory=Pool)

    def test_independent_evaluation_reports_even_one_execution_error(self):
        from nicheflow.research_eval import evaluate
        from nicheflow.main_entry import completion_exit_code
        class OneFailurePool(Pool):
            def __init__(self,config):
                super().__init__(config)
                self.failed=False
                for backend in self.backends.values():
                    generate=backend.generate
                    def once(messages,_generate=generate,**params):
                        result=_generate(messages,**params)
                        if not self.failed:
                            self.failed=True
                            result.update(status='error',finish_reason='error',error='synthetic known failure')
                        return result
                    backend.generate=once
        with tempfile.TemporaryDirectory() as d:
            args,config=revised_fixture(d,rounds=1,deferred=True)
            self.assertEqual(run_main(args,ROOT,Pool)[1],0)
            config['evaluation']['status']='configured'
            config['limits'].update(evaluation_seconds=3600,evaluation_api_usd=1)
            cfg=Path(d)/'eval.json';cfg.write_text(json.dumps(config))
            result=evaluate(ROOT,args.run_dir,cfg,Path(d)/'eval',backend_factory=OneFailurePool)
            self.assertEqual(result['status'],'main_completed_with_execution_errors')
            self.assertEqual(completion_exit_code(result['status']),6)
            self.assertEqual(len(result['execution_errors']),1)
