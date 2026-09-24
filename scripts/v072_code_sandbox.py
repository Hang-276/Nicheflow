#!/usr/bin/env python3
"""Launch a non-root, syscall-restricted evaluator inside a prepared Linux root."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import resource
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--selftest',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve()
    if sys.platform!='linux' or os.geteuid()!=0:raise RuntimeError('Linux chroot launcher requires server root')
    if not (root/'sandbox-manifest.json').is_file():raise RuntimeError('prepared isolation root absent')
    record={'selftest':True} if a.selftest else json.load(sys.stdin)
    def isolate(memory_limit):
        os.setsid()
        os.chroot(root);os.chdir('/tmp')
        os.setgroups([]);os.setgid(65534);os.setuid(65534)
        libc=ctypes.CDLL(None,use_errno=True)
        if libc.prctl(38,1,0,0,0)!=0:raise RuntimeError('no_new_privs failed')
        resource.setrlimit(resource.RLIMIT_CPU,(120,120))
        resource.setrlimit(resource.RLIMIT_AS,(memory_limit,memory_limit))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,8*1024**2))
        resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
        resource.setrlimit(resource.RLIMIT_NPROC,(64,64))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    env={'PATH':'/opt/python/bin','PYTHONHOME':'/opt/python','PYTHONPATH':'/opt/site','HOME':'/tmp','LD_LIBRARY_PATH':'/opt/python/lib',
         'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TMPDIR':'/tmp','LANG':'C.UTF-8'}
    for headroom in (False, True):
        # The request cannot grant itself extra resources. Only an observed
        # trusted-reference MemoryError enables one fresh scoring attempt.
        record['reference_headroom'] = headroom
        memory_limit = (8 if headroom else 2) * 1024**3
        # PyArrow's default allocator reserves 1 GiB of otherwise unused
        # virtual address space on this image. The recovery worker uses the
        # system allocator; the ordinary frozen path remains unchanged.
        worker_env = {**env, 'ARROW_DEFAULT_MEMORY_POOL': 'system'} if headroom else env
        proc=subprocess.Popen(['/opt/python/bin/python','/worker.py'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,text=True,env=worker_env,preexec_fn=lambda: isolate(memory_limit),close_fds=True)
        try:out,err=proc.communicate(json.dumps(record),timeout=150)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid,signal.SIGKILL);proc.communicate()
            raise RuntimeError('sandbox worker time limit; do not relabel infrastructure timeout as a wrong answer')
        if proc.returncode:
            # No API keys or host environment enter the child.
            print(err[-3000:],file=sys.stderr)
            return 3
        result=json.loads(out)
        if result.get('infrastructure_error') == 'reference_memory_limit':
            if not headroom:
                continue
            raise RuntimeError('trusted reference exceeds bounded 8 GiB preparation limit')
        if headroom:
            result['reference_memory_recovery'] = {
                'trusted_limit_bytes': memory_limit, 'candidate_base_limit_bytes': 2*1024**3,
                'arrow_memory_pool': 'system',
                'previous_output_disposal_outside_timed_call': True,
                'policy': 'base allowance plus retained trusted-reference object bytes',
                'scoring_attempts': 2}
        print(json.dumps(result,ensure_ascii=False))
        return 0


if __name__=='__main__':raise SystemExit(main())
