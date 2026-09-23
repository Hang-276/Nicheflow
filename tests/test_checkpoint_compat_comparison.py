import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from test_two_model_api_spend import two_model_fixture, APIOnlyPool, ROOT
from test_research_v050 import FakeEncoder
from nicheflow.checkpoint_compat import verify_checkpoint, verify_code_transition
from nicheflow.cli import code_identity
from nicheflow.comparison_eval import prepare_comparison, execute_comparison
from nicheflow.main_entry import run_main
from nicheflow.research_eval import evaluate
from nicheflow.research_policy import ResearchPolicy
from nicheflow.spec import IntegrityError, digest


def manifest(path, old, new):
    data={'schema':'nicheflow_code_compat_v1','purpose':'checkpoint_serialization_and_frozen_comparison_only',
          'source_code':old,'target_code':new,
          'changed_files':sorted(k for k in old.keys()|new.keys() if old.get(k)!=new.get(k))}
    path.write_text(json.dumps(data)); return path


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_legacy_digest_is_verified_without_bypassing_corruption(self):
        raw={'schema':'nicheflow_state_v2','research':{'search_visits':{3:2,22:1}},'weight':.25}
        checkpoint={'state':json.loads(json.dumps(raw)),'digest':digest(raw)}
        self.assertNotEqual(digest(checkpoint['state']),checkpoint['digest'])
        self.assertEqual(verify_checkpoint(checkpoint,copy.deepcopy(checkpoint),allow_legacy=True)['method'],
                         'legacy_integer_search_visits')
        with self.assertRaises(IntegrityError):verify_checkpoint(checkpoint,checkpoint)
        changed=copy.deepcopy(checkpoint);changed['state']['weight']=.5
        with self.assertRaises(IntegrityError):verify_checkpoint(changed,changed,allow_legacy=True)
        with self.assertRaises(IntegrityError):verify_checkpoint(changed,checkpoint,allow_legacy=True)
        changed=copy.deepcopy(checkpoint);changed['state']['research']['search_visits']={'03':1,'22':1}
        with self.assertRaises(IntegrityError):verify_checkpoint(changed,changed,allow_legacy=True)

    def test_code_migration_requires_exact_manifest_and_target(self):
        with tempfile.TemporaryDirectory() as d:
            old={'nicheflow/research_policy.py':'old','nicheflow/router.py':'same'};new={'nicheflow/research_policy.py':'new','nicheflow/router.py':'same'}
            with self.assertRaises(IntegrityError):verify_code_transition(old,new)
            p=manifest(Path(d)/'compat.json',old,new)
            self.assertEqual(verify_code_transition(old,new,p)['kind'],'explicit_compatibility_migration')
            with self.assertRaises(IntegrityError):verify_code_transition(old,{**new,'nicheflow/router.py':'changed'},p)
            with self.assertRaises(IntegrityError):verify_code_transition({**old,'nicheflow/router.py':'changed'},new,p)
            changed={**new,'nicheflow/router.py':'changed'}
            manifest(p,old,changed)
            with self.assertRaises(IntegrityError):verify_code_transition(old,changed,p)

    def test_actual_legacy_states_resume_and_evaluate_without_source_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);args,cfg=two_model_fixture(d,rounds=1)
            cfg['evaluation'].update(samples_per_task=1,pass_k=[1]);Path(args.config).write_text(json.dumps(cfg))
            current=code_identity(ROOT);old={**current,'nicheflow/research_policy.py':'synthetic-legacy-integer-export'}
            normal=ResearchPolicy.export_state;finish=ResearchPolicy.finish_bootstrap
            def legacy(self):
                out=normal(self)
                out['research']['search_visits']={int(k):v for k,v in out['research']['search_visits'].items()}
                return out
            def seed_visits(self):
                finish(self);self.search_visits.update({3:2,22:1})
            with patch('nicheflow.cli.code_identity',return_value=old),patch.object(ResearchPolicy,'export_state',legacy),patch.object(ResearchPolicy,'finish_bootstrap',seed_visits):
                self.assertEqual(run_main(args,ROOT,APIOnlyPool)[1],0)
            source=Path(args.run_dir);before={p.name:p.read_bytes() for p in source.glob('*.json*')}
            compat=manifest(d/'compat.json',old,current)
            monitor=runpy.run_path(str(ROOT/'scripts/monitor_v051.py'))['snapshot']
            self.assertEqual(monitor(source)['checkpoint_validation']['method'],'legacy_integer_search_visits')
            eval_cfg=copy.deepcopy(cfg);eval_cfg['evaluation']['status']='configured'
            ep=d/'eval.json';ep.write_text(json.dumps(eval_cfg))
            with self.assertRaises(IntegrityError):prepare_comparison(ROOT,source,ep,encoder_factory=FakeEncoder)
            prepared=prepare_comparison(ROOT,source,ep,compat,FakeEncoder)
            self.assertEqual(prepared['plan']['checkpoint_checks']['after']['method'],'legacy_integer_search_visits')
            result=execute_comparison(prepared,d/'comparison',ROOT,APIOnlyPool)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(result['groups']),8)
            self.assertEqual(len(result['paired_comparisons']),3)
            self.assertLess(result['unique_executions_planned'],prepared['plan']['logical_executions'])
            self.assertTrue(all(g['tasks']==3 for g in result['groups']))
            self.assertFalse(result['unknown_call_ids']);self.assertFalse(result['unknown_cost_ids'])
            self.assertTrue(result['learning_state_unchanged'])
            events=[json.loads(x) for x in (d/'comparison/events.jsonl').read_text().splitlines()]
            self.assertFalse(any(e['kind'] in ['router','feedback','generation','scheduler'] for e in events))
            self.assertEqual(len([e for e in events if e['kind']=='execution']),prepared['plan']['unique_executions'])
            # The original independent evaluator also supports both legacy checkpoints.
            standalone=evaluate(ROOT,source,ep,d/'independent',0,APIOnlyPool,compat)
            self.assertEqual(standalone['evaluation']['status'],'complete')
            args.run_dir=str(d/'resumed');args.resume_from=str(source);args.rounds=2;args.compatibility_manifest=str(compat)
            self.assertEqual(run_main(args,ROOT,APIOnlyPool)[1],0)
            cp=json.loads((d/'resumed/checkpoints/round_0002.json').read_text())
            self.assertEqual(cp['digest'],digest(cp['state']))
            self.assertTrue(all(isinstance(k,str) for k in cp['state']['research']['search_visits']))
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.glob('*.json*')})
            # Snapshot mismatch is rejected before any paid evaluation can occur.
            cp_path=source/'checkpoints/round_0001.json';bad=json.loads(cp_path.read_text())
            bad['state']['research']['search_visits']['3']+=1;cp_path.write_text(json.dumps(bad))
            with self.assertRaises(IntegrityError):prepare_comparison(ROOT,source,ep,compat,FakeEncoder)
