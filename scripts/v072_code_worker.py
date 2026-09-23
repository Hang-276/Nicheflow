#!/usr/bin/env python3
"""Trusted worker inside a credential-free chroot, before untrusted code runs."""
import contextlib
import ctypes
import errno
import io
import json
import os
import re
import socket
import sys
from evalplus.eval import untrusted_check,PASS
from evalplus.gen.util import trusted_exec
from evalplus.data.mbpp import mbpp_deserialize_inputs
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS


def restrict_syscalls():
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=lib.seccomp_init(0x7fff0000)
    if not ctx:raise RuntimeError('seccomp initialization failed')
    names=('socket socketpair connect bind listen accept accept4 sendto sendmsg sendmmsg recvfrom recvmsg recvmmsg '
           'mount umount2 pivot_root chroot unshare setns ptrace process_vm_readv process_vm_writev pidfd_getfd '
           'bpf perf_event_open keyctl add_key request_key open_by_handle_at name_to_handle_at userfaultfd '
           'execve execveat reboot swapon swapoff kexec_load init_module finit_module delete_module '
           'io_uring_setup io_uring_enter io_uring_register').split()
    for name in names:
        nr=lib.seccomp_syscall_resolve_name(name.encode())
        if nr>=0 and lib.seccomp_rule_add(ctx,0x00050000|errno.EPERM,nr,0)!=0:raise RuntimeError('seccomp rule failed')
    if lib.seccomp_load(ctx)!=0:raise RuntimeError('seccomp load failed')
    lib.seccomp_release(ctx)


def check(code,inputs,entry,expected,times,atol=0):
    status,_=untrusted_check('mbpp',code,inputs,entry,expected=expected,atol=atol,ref_time=times,
                              fast_check=True,min_time_limit=.2,gt_time_limit_factor=4.)
    return status==PASS,status


def selftest():
    assert os.getuid()==65534 and os.getgid()==65534
    assert not os.path.exists('/root/.config/nicheflow/credentials.env')
    assert os.listdir('/proc')==['meminfo'] and not os.path.exists('/proc/self')
    blocked=False
    try:socket.socket()
    except PermissionError:blocked=True
    assert blocked,'network must be blocked'
    try:open('/runtime-marker','w')
    except PermissionError:pass
    else:raise AssertionError('runtime root must be read-only to worker')
    good=check('def f(x): return x+1',[[1],[2]],'f',[2,3],[.01,.01])
    bad=check('def f(x): return x',[[1],[2]],'f',[2,3],[.01,.01])
    assert good[0] and not bad[0],(good,bad)
    return {'passed':True,'uid':os.getuid(),'network_blocked':blocked,'host_credentials_absent':True,
            'read_only_runtime':True,'good_and_bad_code_checks':True,'isolation':'chroot+uid_drop+no_new_privs+seccomp+rlimits'}


def score(record):
    task,response=record['task'],record['response']
    if response.get('status')!='ok':raise ValueError('provider failure is not a code failure')
    if response['finish_reason']=='length':return {'quality':0.,'outcome':'truncated','audit_required':False}
    if response['finish_reason']!='stop':raise ValueError('unexpected finish')
    blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',response['text'],re.S)
    code=blocks[-1] if blocks else response['text'].strip()
    private=task['private'];problem=private['evalplus'];entry=problem['entry_point']
    code=private.get('setup','')+'\n'+code
    oracle=problem['prompt']+problem['canonical_solution']
    stats={}
    for group in ['base','plus']:
        inputs=mbpp_deserialize_inputs(problem['task_id'],problem[group+'_input'])
        expected,times=trusted_exec(oracle,inputs,entry,record_time=True,
                                   output_not_none=entry in MBPP_OUTPUT_NOT_NONE_TASKS)
        passed,status=check(code,inputs,entry,expected,times,problem['atol'])
        stats[group+'_passed']=passed;stats[group+'_status']=status;stats[group+'_tests']=len(inputs)
    tests=private['tests']
    wrapped=code+'\ndef __nicheflow_private_tests():\n'+''.join('    '+line+'\n' for test in tests for line in test.splitlines())+'    return True\n'
    source_pass,status=check(wrapped,[[]],'__nicheflow_private_tests',[True],[.01])
    stats.update(source_hidden_passed=source_pass,source_hidden_status=status,source_hidden_tests=len(tests))
    return {'quality':float(source_pass and stats['base_passed'] and stats['plus_passed']),
            'outcome':'complete','quality_metric':'source_hidden_and_evalplus_base_plus','metrics':stats,'audit_required':False}


record=json.load(sys.stdin)
restrict_syscalls()
with contextlib.redirect_stdout(io.StringIO()):
    result=selftest() if record.get('selftest') else score(record)
print(json.dumps(result,ensure_ascii=False,allow_nan=False))
