"""Safety checks for model screening: costs, private labels, receipts, resume."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import model_selection_v070 as m
from nicheflow.datasets import Task


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((m.ROOT / 'configs/model_selection_v070.plan.json').read_text())
        self.task = Task('test', 'math', 'train', 'evaluation', 'What is 2+2?', 'SECRET_GOLD',
                         'rational', private={'solution': 'SECRET_SOLUTION'})

    def test_payload_hides_answers_and_disables_thinking(self):
        for name, profile in self.config['models'].items():
            payload, bound = m.request_payload(self.config, profile, self.task.model_input(),
                                              {'max_tokens': 4096}, 17)
            self.assertNotIn('SECRET', json.dumps(payload))
            self.assertGreater(bound, len(json.dumps(self.task.model_input())))
            self.assertEqual('seed' in payload, name == 'local')
            if name == 'local':
                self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
            elif name == 'deepseek':
                self.assertEqual(payload['thinking'], {'type': 'disabled'})
            else:
                self.assertFalse(payload['enable_thinking'])

    def test_identity_and_thinking_contract(self):
        profile = self.config['models']['qwen_flash']
        response = {'model': profile['model'], 'choices': [{'message': {'content': '4'},
                    'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        for change, expected in [('none', 'ok'), ('wrong_model', 'protocol_error'), ('thinking', 'protocol_error')]:
            data = copy.deepcopy(response)
            if change == 'wrong_model': data['model'] = 'different-model'
            if change == 'thinking': data['choices'][0]['message']['reasoning_content'] = 'reason'
            with patch.object(m.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(data).encode())):
                result = m.call_model(profile, {}, 'DUMMY')
            self.assertEqual(result['status'], expected)
            self.assertAlmostEqual(result['reference_cost'], 6e-6)

    def test_quota_errors_do_not_expose_body(self):
        exc = m.urllib.error.HTTPError('https://example.invalid', 429, 'rate', {},
                                      io.BytesIO(b'{"error":{"code":"insufficient_quota","message":"PRIVATE"}}'))
        with patch.object(m.urllib.request, 'urlopen', side_effect=exc):
            result = m.call_model(self.config['models']['qwen_plus'], {}, 'DUMMY')
        self.assertTrue(result['quota_or_rate_limit'])
        self.assertIsNone(result['reference_cost'])
        self.assertNotIn('PRIVATE', json.dumps(result))

    def test_scoring_failure_retains_billable_receipt(self):
        response = {'status': 'ok', 'finish_reason': 'stop', 'text': '4'}
        with patch.object(m, 'evaluate', side_effect=RuntimeError('private diagnostic')):
            score = m.score_result(self.task, response)
        self.assertEqual(score['outcome'], 'scoring_error')
        self.assertIsNone(score['quality'])
        response['finish_reason'] = 'length'
        self.assertEqual(m.score_result(self.task, response)['quality'], 0.)

    def test_reservations_stop_overspend_and_completed_calls_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'run'
            ledger = m.Ledger(directory, {'frozen': True}, {'CNY': 1.}, 2, 60)
            try:
                ledger.begin('one', {'request': 1}, 'CNY', .8)
                with self.assertRaisesRegex(ValueError, 'budget'):
                    ledger.begin('two', {'request': 2}, 'CNY', .3)
                record = {'response': {'reference_cost': .6, 'currency': 'CNY', 'status': 'ok'},
                          'score': {'quality': 1., 'outcome': 'complete'}}
                ledger.finish('one', record)
            finally: ledger.close()
            ledger = m.Ledger(directory, {'frozen': True}, {'CNY': 1.}, 2, 60, resume=True)
            try:
                self.assertEqual(ledger.begin('one', {'request': 1}, 'CNY', .8), record)
                self.assertEqual(len(ledger.starts), 1)
                with self.assertRaisesRegex(ValueError, 'changed input'):
                    ledger.begin('one', {'request': 2}, 'CNY', .8)
            finally: ledger.close()

    def test_unknown_cost_stops_further_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = m.Ledger(Path(temp) / 'run', {}, {'USD': 1.}, 3, 60)
            try:
                ledger.begin('one', {}, 'USD', .1)
                ledger.finish('one', {'response': {'reference_cost': None, 'currency': 'USD', 'status': 'unknown'},
                                      'score': {'quality': 0., 'outcome': 'unknown'}})
                self.assertTrue(ledger.stopped)
                with self.assertRaisesRegex(ValueError, 'stopped'):
                    ledger.begin('two', {}, 'USD', .1)
            finally: ledger.close()


if __name__ == '__main__':
    unittest.main()
