import copy
import json
from pathlib import Path
import tempfile
import unittest
from nicheflow.datasets import load_tasks
from nicheflow.ledger import Journal
from nicheflow.main_budget import ResourceJournal
from nicheflow.main_config import load_protocol, derive_limits, max_call_usd
from nicheflow.main_entry import run_main
from nicheflow.main_policy import main_seeds
from nicheflow.main_reporting import main_report
from nicheflow.math_protocol import math500_tasks, question_hash, SUBJECTS, TEXT_PROTOCOL
from nicheflow.scoring import evaluate
from nicheflow.spec import BudgetStop, IntegrityError, digest, file_hash
from test_main import fixture, SyntheticPool, ROOT


class Math500Tests(unittest.TestCase):
    def test_all_published_rows_order_answers_and_inline_diagrams_are_preserved(self):
        rows = [json.loads(line) for line in (ROOT / 'data/source_math_v041/math500.jsonl').read_text().splitlines()]
        tasks = load_tasks(ROOT / 'data/main_math500_v1/evaluation.jsonl')
        self.assertEqual(len(tasks), 500)
        self.assertEqual(sum('[asy]' in t.question for t in tasks), 42)
        self.assertEqual([t.record() for t in tasks], [t.record() for t in math500_tasks(rows)])
        for row, task in zip(rows, tasks):
            self.assertEqual(task.question, row['problem'])
            self.assertEqual(task.gold, row['answer'])
            self.assertEqual(task.private['source_id'], row['unique_id'])
            self.assertEqual(task.private['input_protocol'], TEXT_PROTOCOL)
            self.assertFalse(task.attachments)
            self.assertNotIn('gold', task.model_input())
            self.assertNotIn('solution', task.model_input())
            self.assertEqual(evaluate(task, '\\boxed{' + task.gold + '}')['quality'], 1.)

    def test_learning_covers_all_35_strata_and_is_disjoint_from_full_test(self):
        c, parts, _ = load_protocol(ROOT / 'configs/main_math_full.json', ROOT)
        hashes = set()
        for role, tasks in parts.items():
            for t in tasks:
                h = question_hash(t.question)
                self.assertNotIn(h, hashes); hashes.add(h)
                self.assertEqual(t.split, 'test' if role == 'evaluation' else 'train')
            if role != 'evaluation':
                for subject in SUBJECTS:
                    for level in range(1,6):
                        self.assertEqual(sum(t.private['subject'] == subject and t.private['level'] == level for t in tasks),
                                         1 if role == 'development' else 2)
        self.assertEqual({k:len(v) for k,v in parts.items()}, {'development':35,'calibration':70,'evaluation':500})

    def test_full_math500_rejects_deferred_status_instead_of_silently_skipping(self):
        with tempfile.TemporaryDirectory() as d:
            c=json.loads((ROOT/'configs/main_math_full.json').read_text())
            c['evaluation']['status']='deferred'
            p=Path(d)/'c.json';p.write_text(json.dumps(c))
            with self.assertRaises(IntegrityError):load_protocol(p,ROOT)

    def test_eval_extent_and_budget_are_identical_for_one_and_hundred_rounds(self):
        a, tasks, pa=load_protocol(ROOT/'configs/main_math_full.json',ROOT,1)
        b, _, pb=load_protocol(ROOT/'configs/main_math_full.json',ROOT,100)
        la,lb=[derive_limits(c,tasks,main_seeds(c)) for c in (a,b)]
        self.assertEqual(pa,pb)
        for key in ['evaluation_workflows','evaluation_calls','evaluation_seconds','evaluation_api_usd']:
            self.assertEqual(la[key],lb[key])
        self.assertEqual(la['evaluation_workflows'],1500)
        self.assertEqual(la['evaluation_calls'],18000)
        self.assertEqual(la['learning_calls'],1369)
        self.assertEqual(la['max_api_usd'],20.)

    def test_learning_cannot_spend_evaluation_api_allowance(self):
        with tempfile.TemporaryDirectory() as d:
            args,c=fixture(d)
            _, tasks, _=load_protocol(args.config,ROOT)
            lim=derive_limits(c,tasks,main_seeds(c))
            backend=SyntheticPool(c).backends['strong']
            bound=max_call_usd(backend.profile,c['decode']['max_new_tokens'])
            lim.update(learning_api_usd=bound+.00001,evaluation_api_usd=10.,max_api_usd=10.+bound+.00001)
            with ResourceJournal(Path(d)/'j',{},100,lim['max_seconds'],limits=lim) as j:
                with j.scope('bootstrap',10,10.):
                    j.call('first',backend,[{'role':'user','content':'test'}],c['decode'])
                    with self.assertRaises(BudgetStop):
                        j.call('denied',backend,[{'role':'user','content':'test'}],c['decode'])
                with j.scope('evaluation',10,10.):
                    j.call('eval',backend,[{'role':'user','content':'test'}],c['decode'])
                self.assertEqual(backend.calls,2)
                self.assertIsNone(j.lookup('call_started','denied'))

    def test_evaluation_has_separate_clock_and_resume_does_not_reset_it(self):
        with tempfile.TemporaryDirectory() as d:
            args,c=fixture(d);_,tasks,_=load_protocol(args.config,ROOT)
            lim=derive_limits(c,tasks,main_seeds(c));lim.update(learning_seconds=100,evaluation_seconds=1000,max_seconds=1100)
            now=[0.]
            with ResourceJournal(Path(d)/'j',{},100,1100,limits=lim,clock=lambda:now[0]) as j:
                now[0]=80
                self.assertEqual(j.remaining_seconds(),20)
                with j.scope('evaluation',10,10.):
                    self.assertEqual(j.remaining_seconds(),1000)
                now[0]=1030
                self.assertEqual(j.remaining_seconds(),50)
            with ResourceJournal(Path(d)/'j',{},100,1100,limits=lim,clock=lambda:now[0]) as j:
                self.assertEqual(j.remaining_seconds(),50)
                now[0]=1081
                with self.assertRaises(BudgetStop):j.reserve(0)

    def test_complete_report_requires_every_frozen_evaluation_slot(self):
        with tempfile.TemporaryDirectory() as d:
            args,_=fixture(d);run_main(args,ROOT,SyntheticPool)
            path=Path(args.run_dir)/'events.jsonl';events=Journal.read(path)
            missing=next(e['id'] for e in events if e['kind']=='evaluation_route')
            events=[e for e in events if not(e['kind']=='evaluation_route' and e['id']==missing)]
            prev='0'*64
            for seq,e in enumerate(events):
                e.pop('hash');e.update(seq=seq,previous=prev);e['hash']=digest(e);prev=e['hash']
            path.write_text(''.join(json.dumps(e)+'\n' for e in events))
            with self.assertRaises(IntegrityError):main_report(args.run_dir)

    def test_changed_metadata_cannot_hide_an_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            args,c=fixture(d)
            development=load_tasks(Path(d)/'development.jsonl')
            cal=Path(d)/'calibration.jsonl';rows=[json.loads(s) for s in cal.read_text().splitlines()]
            rows[0]['question']='  '+development[0].question.replace(' ','  ')+'  '
            rows[0]['context']={'changed':'metadata'}
            cal.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            p=Path(c['data_manifest']);m=json.loads(p.read_text());m['partitions']['calibration']['sha256']=file_hash(cal)
            p.write_text(json.dumps(m));c['data_manifest_sha256']=file_hash(p);Path(args.config).write_text(json.dumps(c))
            with self.assertRaises(IntegrityError):load_protocol(args.config,ROOT)
