import copy
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from nicheflow.ledger import Journal
from nicheflow.main_config import load_protocol, derive_limits
from nicheflow.main_entry import run_main
from nicheflow.main_policy import main_seeds
from nicheflow.math_protocol import SUBJECTS, quick35_indices
from nicheflow.spec import IntegrityError, file_hash
from test_main import ROOT, SyntheticPool


class QuickEvaluationTests(unittest.TestCase):
    def test_frozen_subset_balanced_unchanged_and_independent_of_answers(self):
        _, quick, provenance = load_protocol(ROOT / 'configs/main_math.json', ROOT)
        _, full, _ = load_protocol(ROOT / 'configs/main_math_full.json', ROOT)
        selected = provenance['evaluation_source_indices']
        self.assertEqual(len(selected), 35)
        self.assertEqual([t.record() for t in quick['evaluation']],
                         [full['evaluation'][i].record() for i in selected])
        self.assertEqual(quick35_indices([replace(t, gold='changed', private={**t.private, 'solution': 'changed'})
                                         for t in full['evaluation']]), selected)
        for subject in SUBJECTS:
            for level in range(1, 6):
                self.assertEqual(sum(t.private['subject'] == subject and t.private['level'] == level
                                     for t in quick['evaluation']), 1)
        self.assertEqual(sum('[asy]' in t.question for t in quick['evaluation']), 7)

    def test_learning_unchanged_and_quick_budget_fixed_across_round_counts(self):
        q, qt, qp = load_protocol(ROOT / 'configs/main_math.json', ROOT)
        f, ft, _ = load_protocol(ROOT / 'configs/main_math_full.json', ROOT)
        q100, _, qp100 = load_protocol(ROOT / 'configs/main_math.json', ROOT, 100)
        qlim, flim, q100lim = [derive_limits(c, ts, main_seeds(c)) for c, ts in [(q, qt), (f, ft), (q100, qt)]]
        self.assertEqual(qp, qp100)
        for role in ('development', 'calibration'):
            self.assertEqual(qt[role], ft[role])
        for c in (q, f):
            c.pop('evaluation')
            c['limits'].pop('evaluation_seconds')
            c['limits'].pop('evaluation_api_usd')
        self.assertEqual(q, f)
        for key in ('bootstrap_calls', 'block_calls', 'learning_calls', 'learning_seconds', 'learning_api_usd'):
            self.assertEqual(qlim[key], flim[key])
        for key in ('evaluation_workflows', 'evaluation_calls', 'evaluation_seconds', 'evaluation_api_usd'):
            self.assertEqual(qlim[key], q100lim[key])
        self.assertEqual(qlim['evaluation_workflows'], 105)
        self.assertEqual(qlim['max_calls'], 2629)
        self.assertEqual(qlim['max_api_usd'], 7.)
        self.assertEqual(flim['evaluation_workflows'], 1500)
        self.assertEqual(flim['max_calls'], 19369)

    def test_rehashed_selection_cannot_change_frozen_ids(self):
        with tempfile.TemporaryDirectory() as d:
            c = json.loads((ROOT / 'configs/main_math.json').read_text())
            source = ROOT / c['evaluation']['subset_manifest']
            selection = json.loads(source.read_text())
            selection['task_ids'][0] = selection['task_ids'][1]
            p = Path(d) / 'selection.json'; p.write_text(json.dumps(selection))
            c['evaluation'].update(subset_manifest=str(p), subset_manifest_sha256=file_hash(p))
            config = Path(d) / 'config.json'; config.write_text(json.dumps(c))
            with self.assertRaises(IntegrityError): load_protocol(config, ROOT)
            c['evaluation']['subset_manifest_sha256'] = '0' * 64
            config.write_text(json.dumps(c))
            with self.assertRaises(IntegrityError): load_protocol(config, ROOT)

    def test_cannot_defer_quick_or_mislabel_subset_as_full(self):
        with tempfile.TemporaryDirectory() as d:
            original = json.loads((ROOT / 'configs/main_math.json').read_text())
            for override in ({'status': 'deferred'}, {'benchmark': 'math500_full'}):
                c = copy.deepcopy(original); c['evaluation'].update(override)
                p = Path(d) / 'config.json'; p.write_text(json.dumps(c))
                with self.assertRaises(IntegrityError): load_protocol(p, ROOT)

    def test_both_profiles_complete_and_keep_learning_routes_and_seeds_consistent(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            evidence = {}
            for name, filename, count in [('quick', 'main_math.json', 35), ('full', 'main_math_full.json', 500)]:
                args = SimpleNamespace(config=str(ROOT / 'configs' / filename), rounds=1,
                                       run_dir=str(Path(d) / name))
                summary, code = run_main(args, ROOT, SyntheticPool)
                self.assertEqual(code, 0)
                self.assertTrue(summary['synthetic_backend'])
                self.assertFalse(summary['synthetic_data'])
                self.assertEqual(summary['evaluation']['task_count'], count)
                self.assertEqual(summary['evaluation']['completed_workflows'], count * 3)
                self.assertEqual(summary['evaluation']['diagnostic_subset'], name == 'quick')
                self.assertTrue(summary['evaluation']['learning_state_unchanged'])
                self.assertTrue(summary['numerical_health']['all_finite_spd'])
                events = Journal.read(Path(args.run_dir) / 'events.jsonl')
                state = json.loads((Path(args.run_dir) / 'trained_state.json').read_text())['state']
                state.pop('policy_config_digest')  # Profile metadata differs; learned parameters must not.
                evidence[name] = (events, state)
                before = file_hash(Path(args.run_dir) / 'events.jsonl')
                instances = len(SyntheticPool.instances)
                _, repeat_code = run_main(args, ROOT, SyntheticPool)
                self.assertEqual(repeat_code, 0)
                self.assertEqual(len(SyntheticPool.instances), instances)
                self.assertEqual(file_hash(Path(args.run_dir) / 'events.jsonl'), before)
            self.assertEqual(evidence['quick'][1], evidence['full'][1])
            full_routes = {e['id']: e['payload']['decision'] for e in evidence['full'][0] if e['kind'] == 'evaluation_route'}
            full_calls = {e['id']: e['payload'] for e in evidence['full'][0] if e['kind'] == 'call_finished'}
            for e in evidence['quick'][0]:
                if e['kind'] == 'evaluation_route':
                    self.assertEqual(e['payload']['decision'], full_routes[e['id']])
                if e['kind'] == 'call_finished':
                    self.assertEqual(e['payload'], full_calls[e['id']])
