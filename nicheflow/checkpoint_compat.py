"""Narrow, audited compatibility for v0.5.1 integer niche-key digests."""
import copy
import json
from pathlib import Path
from .spec import IntegrityError, digest, file_hash

COMPATIBLE_FILES = {
    'nicheflow/checkpoint_compat.py', 'nicheflow/comparison_eval.py',
    'nicheflow/research_policy.py', 'nicheflow/research_state.py',
    'nicheflow/research_eval.py', 'nicheflow/main_entry.py', 'nicheflow/cli.py',
}

def verify_code_transition(source, target, manifest_path=None):
    if source == target:
        return {'kind': 'identical_code'}
    if manifest_path is None:
        raise IntegrityError('stage extension may change horizon only; code/models/features/data must match')
    manifest = json.loads(Path(manifest_path).read_text())
    changed = sorted(k for k in source.keys() | target.keys() if source.get(k) != target.get(k))
    if (manifest.get('schema') != 'nicheflow_code_compat_v1'
            or manifest.get('source_code') != source or manifest.get('target_code') != target
            or manifest.get('changed_files') != changed
            or not set(changed) <= COMPATIBLE_FILES
            or manifest.get('purpose') != 'checkpoint_serialization_and_frozen_comparison_only'):
        raise IntegrityError('code compatibility manifest does not match exact source and target')
    return {'kind': 'explicit_compatibility_migration', 'manifest_sha256': file_hash(Path(manifest_path)),
            'source_code_digest': digest(source), 'target_code_digest': digest(target),
            'changed_files': changed}


def verify_checkpoint(checkpoint, snapshot, *, allow_legacy=False):
    if checkpoint != snapshot:
        raise IntegrityError('source checkpoint does not match immutable journal')
    state = checkpoint['state']
    if state.get('schema') != 'nicheflow_state_v2':
        raise IntegrityError('only v2 learning checkpoints can be extended')
    visits = state['research']['search_visits']
    if any(not isinstance(k, str) or not k.isdecimal() or str(int(k)) != k
           or not 0 <= int(k) < 100 or type(v) is not int or v < 0 for k, v in visits.items()):
        raise IntegrityError('invalid canonical niche visit mapping')
    canonical = digest(state)
    if canonical == checkpoint['digest']:
        method = 'canonical_json'
    elif allow_legacy:
        old = copy.deepcopy(state)
        old['research']['search_visits'] = {int(k): v for k, v in visits.items()}
        if digest(old) != checkpoint['digest']:
            raise IntegrityError('source checkpoint digest mismatch, including legacy integer-key format')
        method = 'legacy_integer_search_visits'
    else:
        raise IntegrityError('source checkpoint digest mismatch')
    return {'method': method, 'original_digest': checkpoint['digest'],
            'canonical_digest': canonical, 'source_rewritten': False}
