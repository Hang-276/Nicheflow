from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from nicheflow.ledger import Journal
from nicheflow.main_entry import run_main
from test_main import fixture, ROOT, SyntheticPool


class LengthPool(SyntheticPool):
    reason = 'length'

    def __init__(self, config):
        super().__init__(config)
        for backend in self.backends.values():
            generate = backend.generate
            def limited(messages, _generate=generate, **params):
                result = _generate(messages, **params)
                if 'Propose one executable' not in messages[0]['content']:
                    result['finish_reason'] = self.reason
                return result
            backend.generate = limited


class LengthOutcomeTests(unittest.TestCase):
    def test_repeated_known_length_is_zero_scored_paid_and_not_retried(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            args, _ = fixture(d)
            summary, code = run_main(args, ROOT, LengthPool)
            self.assertEqual(code, 0)
            self.assertEqual(summary['rounds_completed'], 1)
            self.assertEqual(summary['execution_errors'], [])
            self.assertEqual(len(summary['length_limited_workflows']), summary['execution_count'])
            self.assertEqual(summary['evaluation']['completed_workflows'], 18)
            for weight in summary['evaluation']['weight_results']:
                self.assertEqual(weight['pass_at_k']['1'], 0)
                self.assertEqual(weight['execution_failure_rate'], 1)
                self.assertGreater(weight['mean_accounting_usd_per_query'], 0)
            events = Journal.read(Path(args.run_dir) / 'events.jsonl')
            executions = [e['payload'] for e in events if e['kind'] == 'execution']
            self.assertTrue(all(e['evaluation'] is None and len(e['calls']) == 1 for e in executions))
            self.assertTrue(any(e['kind'] == 'router' for e in events))
            archive = [e['payload'] for e in events if e['kind'] == 'archive']
            self.assertTrue(archive)
            self.assertEqual(summary['calls_attempted'], summary['calls_finished'])
            self.assertEqual(summary['unknown_calls'], [])
            self.assertTrue(summary['evaluation']['learning_state_unchanged'])
            count = len(LengthPool.instances)
            repeat, code = run_main(args, ROOT, LengthPool)
            self.assertEqual(len(LengthPool.instances), count)
            self.assertEqual(repeat['calls_finished'], summary['calls_finished'])

    def test_other_finish_reason_is_not_reclassified_as_length(self):
        class OtherReasonPool(LengthPool):
            reason = 'content_filter'
        with tempfile.TemporaryDirectory() as d:
            args, _ = fixture(d)
            summary, code = run_main(args, ROOT, OtherReasonPool)
            self.assertEqual(code, 6)
            self.assertEqual(summary['rounds_completed'], 0)
            self.assertEqual(summary['calls_attempted'], 1)
            self.assertEqual(summary['length_limited_workflows'], [])
            self.assertEqual(len(summary['execution_errors']), 1)
