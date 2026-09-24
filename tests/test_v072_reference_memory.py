import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import v072_code_worker as worker
from scripts import run_v072_continuation as continuation
from nicheflow.spec import file_hash


class ReferenceMemoryTests(unittest.TestCase):
    def test_reference_check_resume_reuses_completed_cases(self):
        tasks = [SimpleNamespace(id=str(i), gold='def f(): return 1', record=lambda i=i: {'id': str(i)}) for i in range(2)]
        records = {'reference_case/final/0': {'passed': True, 'task_sha256': continuation.reference_task_digest(tasks[0])}}
        def artifact(key, value=None, write=False):
            if write:
                records[key] = value
            return records.get(key)
        experiment = SimpleNamespace(tasks=lambda *args: tasks, config={}, store=SimpleNamespace(artifact=artifact))
        with patch.object(continuation, 'run_score', return_value={'quality': 1}) as score:
            continuation.Experiment.reference_checks(experiment, 'final')
            self.assertEqual(score.call_count, 1)
            self.assertEqual(score.call_args.args[0].id, '1')
            continuation.Experiment.reference_checks(experiment, 'final')
            self.assertEqual(score.call_count, 1)

    def test_reference_identity_supports_nonfinite_benchmark_inputs(self):
        task = SimpleNamespace(record=lambda: {'inputs': [float('inf'), -float('inf'), float('nan')]})
        self.assertEqual(continuation.reference_task_digest(task), continuation.reference_task_digest(task))
        different = SimpleNamespace(record=lambda: {'inputs': ['Infinity', '-Infinity', 'NaN']})
        self.assertNotEqual(continuation.reference_task_digest(task), continuation.reference_task_digest(different))

    def test_migration_does_not_allow_unrelated_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ['scripts/v072_code_worker.py', 'scripts/v072_code_sandbox.py', 'nicheflow/other.py']
            for name in names:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original')
            old = {name: file_hash(root/name) for name in names}
            with patch.object(continuation, 'ROOT', root):
                self.assertIsNone(continuation.verify_parent_sources(old))
                for name in names[:2]:
                    (root/name).write_text('repair')
                with self.assertRaises(AssertionError):
                    continuation.verify_parent_sources(old)
                migration = {'version': 'v072-reference-memory-1', 'validation': {'passed': True},
                    'source_changes': {name: {'old_sha256': old[name], 'new_sha256': file_hash(root/name)} for name in names[:2]}}
                path = root/'setup/v072/scoring_memory_migration_20260924.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(migration))
                self.assertEqual(continuation.verify_parent_sources(old), file_hash(path))
                (root/names[2]).write_text('unrelated change')
                with self.assertRaises(AssertionError):
                    continuation.verify_parent_sources(old)

    def test_reference_size_counts_aliases_once_and_handles_cycles(self):
        shared = [1, 2]
        value = [shared, shared]
        expected = sum(sys.getsizeof(x) for x in (value, shared, 1, 2))
        self.assertEqual(worker.reference_size(value), expected)
        value.append(value)
        self.assertEqual(worker.reference_size(value), expected + sys.getsizeof(value) - sys.getsizeof([shared, shared]))

    def test_original_candidate_guard_is_unchanged_without_recovery(self):
        original = worker.eval_runtime.query_maximum_memory_bytes
        def fake_check(*args, **kwargs):
            self.assertIs(worker.eval_runtime.query_maximum_memory_bytes, original)
            return worker.PASS, []
        with patch.object(worker, 'untrusted_check', side_effect=fake_check):
            self.assertTrue(worker.check('', [[]], 'f', [1], [.01])[0])

    def test_reference_allowance_is_scoped_and_restored_on_failure(self):
        original = worker.eval_runtime.query_maximum_memory_bytes
        def fake_check(*args, **kwargs):
            self.assertEqual(worker.eval_runtime.query_maximum_memory_bytes(), worker.CANDIDATE_MEMORY + 12345)
            raise RuntimeError('infrastructure failure')
        with patch.object(worker, 'untrusted_check', side_effect=fake_check):
            with self.assertRaises(RuntimeError):
                worker.check('', [[]], 'f', [1], [.01], reference_bytes=12345)
        self.assertIs(worker.eval_runtime.query_maximum_memory_bytes, original)

    def test_oracle_oom_is_not_a_wrong_answer(self):
        record = {'task': {'private': {'evalplus': {'entry_point': 'f', 'prompt': '',
            'canonical_solution': '', 'task_id': 'Mbpp/1', 'base_input': [[]]}}},
            'response': {'status': 'ok', 'finish_reason': 'stop', 'text': 'def f(): return 1'}}
        with patch.object(worker, 'mbpp_deserialize_inputs', return_value=[[]]), \
             patch.object(worker, 'trusted_exec', side_effect=MemoryError):
            with self.assertRaises(worker.ReferenceMemoryLimit):
                worker.score(record)


if __name__ == '__main__':
    unittest.main()
