#!/usr/bin/env python3
"""Apply the validated scoring-only repair with an immutable pre-migration backup."""
import copy
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nicheflow.datasets import load_tasks
from nicheflow.ledger import atomic_json
from nicheflow.spec import digest, file_hash


def main():
    staging = ROOT/'.scoring-migration-staging'
    directory = ROOT/'runs/multidomain_v072_continuation'
    migration_path = ROOT/'setup/v072/scoring_memory_migration_20260924.json'
    assert not migration_path.exists(), 'migration already exists; do not overwrite it'
    lock = (directory/'writer.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    db = sqlite3.connect(directory/'state.sqlite3')
    assert db.execute("SELECT count(*) FROM attempts WHERE status NOT IN ('complete','rejected')").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM artifacts WHERE key LIKE 'answer/final/%'").fetchone()[0] == 0
    old = json.loads(db.execute("SELECT value FROM meta WHERE key='frozen'").fetchone()[0])
    parent = sqlite3.connect(f"file:{ROOT}/runs/multidomain_v072/checkpoint.sqlite3?mode=ro", uri=True)
    parent_code = json.loads(parent.execute("SELECT value FROM meta WHERE key='frozen'").fetchone()[0])['code']
    sources = ['scripts/v072_code_worker.py', 'scripts/v072_code_sandbox.py', 'scripts/run_v072_continuation.py']
    for name in sources:
        assert file_hash(ROOT/name) == old['source_hashes'][name], name
    checks = {}
    for name, expected in [('canonical', 1), ('wrong', 0)]:
        path = staging/(name+'_validation_final.json')
        record = json.loads(path.read_text())
        assert record['returncode'] == 0 and json.loads(record['stdout'])['quality'] == expected
        assert record['worker_sha256'] == file_hash(staging/'v072_code_worker.py')
        assert record['launcher_sha256'] == file_hash(staging/'v072_code_sandbox.py')
        checks[name] = {'passed': True, 'evidence_sha256': file_hash(path)}
    parity_path = staging/'ordinary_parity_validation.json'
    parity = json.loads(parity_path.read_text())
    assert parity['passed'] and len(parity['cases']) == 4
    checks['ordinary_parity'] = {'passed': True, 'evidence_sha256': file_hash(parity_path)}
    result = subprocess.run([sys.executable, str(staging/'v072_code_sandbox.py'), '--root',
                             str(ROOT/'.code-sandbox-memoryfix'), '--selftest'], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0 and json.loads(result.stdout)['passed'], result.stderr
    checks['isolation'] = json.loads(result.stdout)
    backup = directory/'before_reference_memory_migration'
    assert not backup.exists()
    backup.mkdir()
    with sqlite3.connect(backup/'state.sqlite3') as target:
        db.backup(target)
    config = json.loads((ROOT/'configs/multidomain_v072.json').read_text())
    box = Path(config['code_sandbox_root'])
    for name in sources:
        target = backup/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    shutil.copy2(box/'worker.py', backup/'sandbox-worker.py')
    shutil.copy2(box/'sandbox-manifest.json', backup/'sandbox-manifest.json')
    sandbox_manifest = json.loads((box/'sandbox-manifest.json').read_text())
    sandbox_manifest['worker_sha256'] = file_hash(staging/'v072_code_worker.py')
    sandbox_manifest['files']['worker.py'] = sandbox_manifest['worker_sha256']
    new_manifest = json.dumps(sandbox_manifest, indent=2)+'\n'
    tasks = {task.id: task for task in load_tasks(ROOT/'data/multidomain_v072/mbpp/final.jsonl')}
    evidence = directory/'diagnostics_20260924/canonical_reference_checks.jsonl'
    reuse = []
    for line in evidence.read_text().splitlines():
        row = json.loads(line)
        if row['returncode'] == 0 and json.loads(row['stdout'])['quality'] == 1:
            reuse.append(row['task_id'])
    assert len(reuse) == len(set(reuse)) == 66
    reuse.append('mbpp/test/255')
    reference_records = {task_id: {'passed': True, 'task_sha256': digest(json.dumps(
        tasks[task_id].record(), sort_keys=True, ensure_ascii=False, separators=(',', ':')))} for task_id in reuse}
    migration = {
        'version': 'v072-reference-memory-1', 'created': time.time(),
        'reason': 'trusted oracle exceeded 2 GiB, including unused Arrow virtual reservation; no final model responses existed',
        'source_changes': {name: {'old_sha256': parent_code[name],
                                 'new_sha256': file_hash(staging/Path(name).name)} for name in sources[:2]},
        'sandbox_manifest': {'old_sha256': file_hash(box/'sandbox-manifest.json')},
        'validation': {'passed': True, 'checks': checks},
        'reused_reference_cases': reuse, 'old_reference_evidence_sha256': file_hash(evidence),
        'protocol': {'ordinary_path': 'unchanged 2 GiB and original allocator',
                     'retry_trigger': 'MemoryError while generating trusted reference only',
                     'trusted_recovery_limit_gib': 8, 'candidate_base_limit_gib': 2,
                     'reference_overhead': 'retained oracle object bytes added explicitly',
                     'recovery_arrow_allocator': 'system', 'maximum_scoring_attempts': 2,
                     'previous_output_disposal_outside_timed_call': True,
                     'models_prompts_tasks_tests_answers_selection': 'unchanged'},
        'old_frozen_sha256': digest(old),
    }
    for name in sources:
        path = ROOT/name
        temp = path.with_name(path.name+'.migration-new')
        temp.write_bytes((staging/path.name).read_bytes())
        temp.chmod(path.stat().st_mode & 0o777)
        temp.replace(path)
    for name, value in [('worker.py', (staging/'v072_code_worker.py').read_text()), ('sandbox-manifest.json', new_manifest)]:
        temp = box/(name+'.migration-new')
        temp.write_text(value); temp.chmod(0o444); temp.replace(box/name)
    migration['sandbox_manifest']['new_sha256'] = file_hash(box/'sandbox-manifest.json')
    atomic_json(migration_path, migration)
    frozen = copy.deepcopy(old)
    for name in sources:
        frozen['source_hashes'][name] = file_hash(ROOT/name)
    frozen['scoring_memory_migration_sha256'] = file_hash(migration_path)
    with db:
        db.execute("UPDATE meta SET value=? WHERE key='frozen'", (json.dumps(frozen, sort_keys=True),))
        db.execute('INSERT INTO events(created,kind,payload) VALUES(?,?,?)',
                   (time.time(), 'audited_scoring_memory_migration', json.dumps({'old_frozen': old,
                     'new_frozen_sha256': digest(frozen), 'migration_sha256': file_hash(migration_path)})))
        for task_id in reuse:
            value = reference_records[task_id]
            db.execute('INSERT INTO artifacts VALUES(?,?)', ('reference_case/final/'+task_id, json.dumps(value, sort_keys=True)))
    with sqlite3.connect(directory/'checkpoint.sqlite3') as target:
        db.backup(target)
    print(json.dumps({'migration': str(migration_path), 'reference_cases_reused': len(reuse),
                      'successful_receipts_preserved': db.execute("SELECT count(*) FROM attempts WHERE status='complete'").fetchone()[0]}))
    db.close(); parent.close(); lock.close()


if __name__ == '__main__':
    main()
