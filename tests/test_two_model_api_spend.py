import copy
import json
import runpy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_main import ROOT, SyntheticMainBackend
from test_research_v050 import revised_fixture, Pool, FakeEncoder, state
from nicheflow.main_config import validate_main
from nicheflow.main_entry import run_main
from nicheflow.main_models import AccountedLocal
from nicheflow.ledger import Journal
from nicheflow.research_policy import research_seeds
from nicheflow.semantic import SemanticFeatures
from nicheflow.spec import digest


def two_model_fixture(directory, rounds=2):
    args, config = revised_fixture(directory, rounds=rounds)
    config['models'].pop('middle')
    config['cost_objective'] = 'external_api_spend'
    config['models']['local']['accounting_usd_per_gpu_hour'] = 0.
    Path(args.config).write_text(json.dumps(config))
    return args, config


class APIOnlyPool(Pool):
    def __init__(self, config):
        super().__init__(config)
        backend = self.backends['local']
        generate = backend.generate
        def free_local(*args, **kwargs):
            result = generate(*args, **kwargs)
            result.update(accounted_usd=0., local_accounting_usd=0.)
            return result
        backend.generate = free_local


class TwoModelTests(unittest.TestCase):
    def test_zero_local_cost_requires_explicit_objective_and_preserves_time(self):
        config = json.loads((ROOT/'configs/main_math_v051_local_flash.json').read_text())
        validate_main(config)
        ambiguous = copy.deepcopy(config)
        ambiguous.pop('cost_objective')
        with self.assertRaisesRegex(ValueError, 'local accounting rate'):
            validate_main(ambiguous)
        config['models']['local']['accounting_usd_per_gpu_hour'] = 1.
        with self.assertRaisesRegex(ValueError, 'explicit zero'):
            validate_main(config)
        config['models']['local']['accounting_usd_per_gpu_hour'] = 0.
        with patch('nicheflow.backend_process.LocalWorker') as worker:
            worker.return_value.generate.return_value = {'status':'ok', 'elapsed_seconds':13.5}
            result = AccountedLocal(config['models']['local']).generate([])
        self.assertEqual(result['accounted_usd'], 0.)
        self.assertEqual(result['elapsed_seconds'], 13.5)
        self.assertEqual(result['local_cost_basis'], 'external_api_spend_only_local_capacity_excluded')

    def test_role_order_and_features_stay_identical_across_json_order(self):
        config = json.loads((ROOT/'configs/main_math_v051_local_flash.json').read_text())
        reordered = {**config, 'models':dict(reversed(list(config['models'].items())))}
        seeds = research_seeds(config)
        self.assertEqual(len(seeds), 6)
        self.assertEqual([g.version for g in seeds], [g.version for g in research_seeds(reordered)])
        self.assertEqual([g.nodes[0].model for g in seeds[:2]], ['local', 'strong'])
        self.assertTrue(all(n.model in {'local','strong'} for g in seeds for n in g.nodes))
        a = SemanticFeatures(config, FakeEncoder(config['semantic']))
        b = SemanticFeatures(reordered, FakeEncoder(config['semantic']))
        self.assertEqual((a.workflow_dimension, a.router_dimension), (18,29))
        for graph in seeds:
            self.assertEqual(a.route({'question':'Compute 2+3'},graph), b.route({'question':'Compute 2+3'},graph))

    def test_two_model_api_only_learning_restores_exactly_and_retains_local_time(self):
        with tempfile.TemporaryDirectory() as directory:
            args, config = two_model_fixture(directory)
            complete = Path(directory)/'complete'
            args.run_dir = str(complete)
            report, code = run_main(args, ROOT, APIOnlyPool)
            self.assertEqual(code, 0)
            self.assertAlmostEqual(report['accounted_usd'], report['api_tariff_usd'])
            self.assertGreater(report['api_tariff_usd'], 0.)
            self.assertEqual(report['local_inference_accounting_usd'], 0.)
            self.assertGreater(report['local_inference_seconds'], 0.)
            self.assertEqual(report['cost_objective'], 'external_api_spend')
            first, second = Path(directory)/'first', Path(directory)/'second'
            args.run_dir, args.stop_after_round = str(first), 1
            self.assertEqual(run_main(args, ROOT, APIOnlyPool)[1], 0)
            args.run_dir, args.stop_after_round, args.resume_from = str(second), 2, str(first)
            self.assertEqual(run_main(args, ROOT, APIOnlyPool)[1], 0)
            self.assertEqual(digest(state(complete,2)), digest(state(second,2)))
            monitor = runpy.run_path(str(ROOT/'scripts/monitor_v051.py'))['snapshot']
            before = (second/'events.jsonl').read_bytes()
            monitored = monitor(second)
            self.assertEqual((monitored['rounds_completed'], monitored['stage_rounds_completed']), (2,1))
            self.assertTrue(monitored['complete_event_hash_chain_valid'])
            self.assertEqual(before, (second/'events.jsonl').read_bytes())
            events = Journal.read(first/'events.jsonl') + Journal.read(second/'events.jsonl')
            calls = [e for e in events if e['kind']=='call_started']
            self.assertEqual({e['payload']['model'] for e in calls}, {'local','strong'})
            self.assertEqual(len(calls), len({e['id'] for e in calls}))
            self.assertFalse(any(e['kind']=='evaluation_started' for e in events))
