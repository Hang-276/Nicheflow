#!/usr/bin/env python3
"""Start the pinned local model on localhost; verify weights before loading."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    identity = json.loads((ROOT / 'setup/local_model_identity.json').read_text())
    snapshot = Path(identity['snapshot_path'])
    source = {row['path']: row for row in json.loads((ROOT / 'setup/huggingface_files.json').read_text())}
    for name, expected in {**identity['configuration_sha256'], **identity['weight_sha256']}.items():
        with (snapshot / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, f'model hash mismatch: {name}'
    for name in identity['configuration_sha256']:
        row = source[name]
        if row['lfs']:
            assert identity['configuration_sha256'][name] == row['lfs']['sha256']
        else:
            content = (snapshot / name).read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            assert blob == row['blob_id'], f'mirror differs from pinned source: {name}'
    identity['source_config_git_blobs_verified'] = True
    runtime = ROOT / '.serve-venv'
    lock = subprocess.check_output([str(runtime / 'bin/python'), '-m', 'pip', 'freeze'], text=True)
    (ROOT / 'setup/requirements.lock').write_text(lock)
    command = [str(runtime / 'bin/vllm'), 'serve', str(snapshot),
               '--served-model-name', 'nicheflow-qwen35-9b-awq-v070',
               '--host', '127.0.0.1', '--port', '8070',
               '--tensor-parallel-size', '1', '--max-model-len', '24576',
               '--max-num-seqs', '2', '--max-num-batched-tokens', '2048',
               '--gpu-memory-utilization', '0.88', '--reasoning-parser', 'qwen3',
               '--language-model-only', '--no-enable-prefix-caching',
               '--generation-config', 'vllm']
    (ROOT / 'setup/serve_command.json').write_text(json.dumps(command, indent=2) + '\n')
    identity['runtime_lock_sha256'] = hashlib.sha256(lock.encode()).hexdigest()
    identity['serve_command'] = command
    (ROOT / 'setup/local_model_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    # All model assets are pinned locally; no remote repository Python is used.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['VLLM_NO_USAGE_STATS'] = '1'
    os.environ['PATH'] = str(runtime / 'bin') + os.pathsep + os.environ.get('PATH', '')
    cuda_home = Path(sysconfig.get_paths()['purelib']) / 'nvidia/cu13'
    assert (cuda_home / 'bin/nvcc').is_file(), 'isolated CUDA compiler missing'
    # NVIDIA's CUDA 13 wheels use lib/, while the JIT linker expects lib64/.
    assert (cuda_home / 'lib/libcudart.so.13').is_file(), 'CUDA runtime library missing'
    if not (cuda_home / 'lib64').exists():
        (cuda_home / 'lib64').symlink_to('lib', target_is_directory=True)
    if not (cuda_home / 'lib/libcudart.so').exists():
        (cuda_home / 'lib/libcudart.so').symlink_to('libcudart.so.13')
    os.environ['CUDA_HOME'] = str(cuda_home)
    os.environ['PATH'] = str(cuda_home / 'bin') + os.pathsep + os.environ['PATH']
    identity['serve_environment'] = {'CUDA_HOME': str(cuda_home),
                                   'PATH_prepend': [str(cuda_home / 'bin'), str(runtime / 'bin')]}
    (ROOT / 'setup/local_model_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print('MODEL_HASHES_VERIFIED_STARTING_LOCAL_SERVER', flush=True)
    os.execv(command[0], command)


if __name__ == '__main__':
    main()
