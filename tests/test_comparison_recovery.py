import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest

from test_two_model_api_spend import two_model_fixture, APIOnlyPool, ROOT
from test_research_v050 import FakeEncoder
from nicheflow.main_entry import run_main
from nicheflow.comparison_eval import prepare_comparison, execute_comparison
from nicheflow.ledger import Journal
from nicheflow.spec import IntegrityError

mod = runpy.run_path(str(ROOT/'scripts/recover_v052_comparison.py'))


class SimulatedPowerLoss(BaseException):
    pass


class NamedPool(APIOnlyPool):
    def __init__(self, cfg):
        super().__init__(cfg)
        for role, backend in self.backends.items():
            backend.model_id = cfg['models'][role].get('path', cfg['models'][role].get('model'))
            backend.environment['model'] = {'id': backend.model_id}


class InterruptedPool(NamedPool):
    def __init__(self, cfg):
        super().__init__(cfg)
        original = self.backends['local'].generate
        self.count = 0
        def generate(*args, **kwargs):
            self.count += 1
            if self.count == 3:
                raise SimulatedPowerLoss()
            return original(*args, **kwargs)
        self.backends['local'].generate = generate


class RecoveryTests(unittest.TestCase):
    def test_reuses_completed_answers_and_rejects_unsafe_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            args, cfg = two_model_fixture(d, rounds=1)
            cfg['evaluation'].update(samples_per_task=1, pass_k=[1])
            Path(args.config).write_text(json.dumps(cfg))
            self.assertEqual(run_main(args, ROOT, NamedPool)[1], 0)
            cfg['evaluation']['status'] = 'configured'
            ep = d/'eval.json'; ep.write_text(json.dumps(cfg))
            prepared = prepare_comparison(ROOT, args.run_dir, ep, encoder_factory=FakeEncoder)
            # Real run has unused calls after length truncations; simulate one
            # spare reservation while still charging the abandoned attempt.
            prepared['plan']['max_model_calls'] += 1
            source = d/'interrupted'
            with self.assertRaises(SimulatedPowerLoss):
                execute_comparison(prepared, source, ROOT, InterruptedPool)
            before = {p.name:p.read_bytes() for p in source.glob('*.json*')}
            events = Journal.read(source/'events.jsonl')
            prefix, tail = mod['recovery_prefix'](events, cfg)
            unsafe = copy.deepcopy(events)
            unsafe[-1]['payload']['model'] = cfg['models']['strong']['model']
            with self.assertRaisesRegex(IntegrityError, 'cannot retry'):
                mod['recovery_prefix'](unsafe, cfg)
            with self.assertRaises(IntegrityError):
                mod['recovery_prefix'](events[:-1], cfg)
            config_bad = copy.deepcopy(cfg)
            config_bad['models']['local']['accounting_usd_per_gpu_hour'] = 1
            with self.assertRaises(IntegrityError):
                mod['recovery_prefix'](events, config_bad)
            output = d/'recovered'
            plan = mod['recover'](ROOT, source, output, ep, False, NamedPool)
            self.assertFalse(output.exists())
            result = mod['recover'](ROOT, source, output, ep, True, NamedPool)
            self.assertEqual(result['status'], 'complete')
            self.assertFalse(result['unknown_call_ids'])
            after = Journal.read(output/'events.jsonl')
            self.assertEqual(after[:len(prefix)], prefix)
            self.assertEqual(before, {p.name:p.read_bytes() for p in source.glob('*.json*')})
            new_starts = [e['id'] for e in after[len(prefix):] if e['kind']=='call_started']
            finished = {e['id'] for e in prefix if e['kind']=='call_finished'}
            self.assertFalse(set(new_starts) & finished)
            self.assertEqual(new_starts.count(tail[-1]['id']), 1)
            self.assertEqual(result['physical_attempts_including_abandoned'], result['physical_calls']+1)
            self.assertEqual(result['recovery']['completed_executions_reused'], plan['completed_executions_reused'])
            renderer = runpy.run_path(str(ROOT/'scripts/render_v052_comparison.py'))['render']
            self.assertTrue(renderer(output, 0))
            with self.assertRaisesRegex(IntegrityError, 'new output'):
                mod['recover'](ROOT, source, output, ep, True, NamedPool)


if __name__ == '__main__':
    unittest.main()
