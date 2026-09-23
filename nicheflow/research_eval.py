"""Independent v2 held-out evaluation, preserving the exact learning checkpoint."""
import copy
import json
from pathlib import Path
from .cli import code_identity
from .ledger import Journal, atomic_json
from .main_budget import ResourceJournal
from .main_config import load_protocol, derive_limits
from .main_driver import MainDriver
from .main_models import ModelPool
from .main_reporting import main_report
from .research_policy import ResearchPolicy, research_seeds
from .research_state import read_stage_source, restore
from .runtime import GraphExecutor
from .spec import IntegrityError, digest, file_hash
from .checkpoint_compat import verify_checkpoint


def evaluate(root, source_dir, evaluation_config, out_dir, round_index=None, backend_factory=ModelPool, compatibility_manifest=None):
    root, source_dir, out_dir = Path(root), Path(source_dir), Path(out_dir)
    if out_dir.exists():
        raise IntegrityError('evaluation output must be new; no overwrite or implicit retry')
    frozen_source = json.loads((source_dir/'config.json').read_text())['config']
    learning = frozen_source['settings']
    check = copy.deepcopy(learning); check['rounds'] += 1
    source = read_stage_source(source_dir, check, code_identity(root), compatibility_manifest)
    config, tasks, provenance = load_protocol(evaluation_config, root, learning['rounds'])
    if config['evaluation']['status'] != 'configured':
        raise IntegrityError('explicit held-out evaluation protocol required')
    # Evaluation scope and its reserved caps may differ; learning/model/features cannot.
    def core(c):
        c=copy.deepcopy(c);c.pop('evaluation');c['limits'].pop('evaluation_seconds',None);c['limits'].pop('evaluation_api_usd',None)
        return c
    if core(config) != core(learning):
        raise IntegrityError('evaluation config changed learning protocol')
    round_index = source['round'] if round_index is None else round_index
    path = source_dir/'checkpoints'/f'round_{round_index:04d}.json'
    checkpoint=json.loads(path.read_text())
    records=Journal.read(source_dir/'events.jsonl')
    expected=next((e['payload'] for e in records if e['kind']=='state_snapshot' and e['id']==f'round:{round_index}'),None)
    validation = verify_checkpoint(checkpoint, expected, allow_legacy=True)
    limits=derive_limits(config,tasks,research_seeds(config))
    limits.update(max_calls=limits['evaluation_calls'],max_seconds=limits['evaluation_seconds'],
                  max_api_usd=limits['evaluation_api_usd'],max_accounting_usd=limits['evaluation_calls']*limits['max_call_accounting_usd'],
                  learning_calls=0,learning_seconds=0,learning_api_usd=0)
    synthetic=bool(getattr(backend_factory,'is_synthetic',False))
    if synthetic != frozen_source['synthetic']:
        raise IntegrityError('cannot mix synthetic and real checkpoint evidence')
    frozen={'mode':'main','settings':config,'code':code_identity(root),'provenance':provenance,'limits':limits,'synthetic':synthetic,
            'evaluation_kind':'v2_frozen_checkpoint','learning_updates_allowed':False,
            'source':{k:v for k,v in source['evidence'].items() if k not in {'old_horizon','new_horizon','budget_context'}},
            'checkpoint_round':round_index,'checkpoint_digest':checkpoint['digest'], 'checkpoint_validation':validation}
    with ResourceJournal(out_dir,frozen,limits['max_calls'],limits['max_seconds'],limits=limits) as journal:
        pool=backend_factory(config)
        try:
            identity={k:v.environment for k,v in pool.backends.items()}
            if any(identity[k]['model'] != source['environment'][k]['model'] for k in identity):
                raise IntegrityError('evaluation model environment changed')
            atomic_json(out_dir/'environment.json',identity)
            executor=GraphExecutor(journal,pool.backends,config['decode'],workflow_output_policy=config.get('workflow_output_policy','legacy'))
            executor.length_limit_as_zero=config.get('length_limit_outcome')=='zero_quality_no_retry'
            factory=getattr(backend_factory,'semantic_encoder_factory',None) if synthetic else None
            policy=ResearchPolicy(executor,learning,tasks['development'],synthetic=synthetic,encoder=factory(learning['semantic']) if factory else None)
            restore(policy,checkpoint['state'])
            driver=MainDriver(policy,tasks,provenance)
            # Driver reads only evaluation sampling from its protocol; policy retains original digest.
            driver.config=config
            journal.append('main_contract','main',{'partitions':{k:[t.record() for t in ts] for k,ts in tasks.items()},'provenance':provenance})
            driver.evaluate_frozen(checkpoint['state'])
            failures=[e['id'] for e in journal.events if e['kind']=='execution' and not executor.assessment_complete(e['payload'])]
            journal.append('run_finished','main:done',{'status':'main_completed_with_execution_errors' if failures else 'main_run_complete',
                'rounds_completed':round_index,'evaluation_complete':True,'failed_executions':failures})
        finally:
            pool.close()
    if file_hash(source_dir/'events.jsonl') != source['evidence']['events_sha256']:
        raise IntegrityError('source changed during independent evaluation')
    result=main_report(out_dir)
    atomic_json(out_dir/'checkpoint_eval.json',result)
    return result
