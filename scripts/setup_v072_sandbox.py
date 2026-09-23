#!/usr/bin/env python3
"""Prepare a credential-free, read-only runtime tree; no model code is executed."""
import argparse
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import sysconfig


def copy_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target,follow_symlinks=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root).resolve()
    if sys.platform!='linux' or os.geteuid()!=0:raise RuntimeError('prepare on the Linux server as root')
    if root.exists():raise RuntimeError('existing isolation root; verify it instead of overwriting')
    root.mkdir(parents=True)
    copy_file(Path(sys.executable).resolve(),root/'opt/python/bin/python')
    version=f'python{sys.version_info.major}.{sys.version_info.minor}'
    stdlib=Path(sysconfig.get_path('stdlib'))
    shutil.copytree(stdlib,root/'opt/python/lib'/version,ignore=shutil.ignore_patterns('site-packages','__pycache__','test','tests','idlelib','tkinter','ensurepip'))
    site=Path(sysconfig.get_path('purelib'))
    shutil.copytree(site,root/'opt/site',ignore=shutil.ignore_patterns('__pycache__','tests','test','pip','pip-*','setuptools','setuptools-*'))
    copy_file(Path(__file__).with_name('v072_code_worker.py'),root/'worker.py')
    shared=set()
    for binary in [Path(sys.executable).resolve(),*root.rglob('*.so'),*root.rglob('*.so.*')]:
        r=subprocess.run(['ldd',str(binary)],capture_output=True,text=True)
        for line in r.stdout.splitlines():
            for match in re.findall(r'(?:=>\s+)?(/[^\s()]+)',line):
                path=Path(match)
                if path.is_file() and not path.is_relative_to(root):shared.add(path)
    cache=subprocess.check_output(['ldconfig','-p'],text=True)
    seccomp=[Path(line.split('=>')[-1].strip()) for line in cache.splitlines() if 'libseccomp.so.2 ' in line]
    if not seccomp:raise RuntimeError('libseccomp.so.2 required')
    shared.update(seccomp)
    for path in shared:
        copy_file(path,root/str(path).lstrip('/'))
        copy_file(path,root/'opt/python/lib'/path.name)
    for path in root.rglob('*'):
        if path.is_dir():path.chmod(0o555)
        elif path.is_file():path.chmod(0o555 if os.access(path,os.X_OK) else 0o444)
    for name in ['tmp','dev','dev/shm']:(root/name).mkdir(parents=True,exist_ok=True)
    (root/'tmp').chmod(0o1777);(root/'dev/shm').chmod(0o1777)
    devices={'null':(1,3),'zero':(1,5),'urandom':(1,9)}
    for name,(major,minor) in devices.items():
        os.mknod(root/'dev'/name,stat.S_IFCHR|0o666,os.makedev(major,minor))
        (root/'dev'/name).chmod(0o666)
    # EvalPlus queries memory via psutil. Expose only a static limit description,
    # never mount host procfs or expose host processes / descriptors.
    (root/'proc').mkdir()
    (root/'proc/meminfo').write_text('MemTotal: 2097152 kB\nMemFree: 2097152 kB\nMemAvailable: 2097152 kB\nBuffers: 0 kB\nCached: 0 kB\nActive: 0 kB\nInactive: 0 kB\nShmem: 0 kB\nSlab: 0 kB\nSReclaimable: 0 kB\n')
    (root/'proc').chmod(0o555);(root/'proc/meminfo').chmod(0o444)
    hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file()}
    manifest={'python':sys.version,'files':hashes,'credential_files_copied':False,
              'isolation':'chroot+uid65534+no_new_privs+seccomp+rlimits','worker_sha256':hashes['worker.py'],'devices':devices}
    (root/'sandbox-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(root/'sandbox-manifest.json').chmod(0o444);root.chmod(0o555)
    print(json.dumps({'root':str(root),'files':len(hashes),'manifest_sha256':hashlib.sha256((root/'sandbox-manifest.json').read_bytes()).hexdigest()}))


if __name__=='__main__':main()
