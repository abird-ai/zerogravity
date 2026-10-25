#!/usr/bin/env python3
"""Bounded original local owner for fresh source20 upload and remote bootstrap."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from pins import Held, alarm_until, check_closure, parsed, pin, require
from bootstrap_receipt import parse_closed_bootstrap

PYTHON = '/nix/store/jy00n3jp5xj8s40n1fbldmkcr5n7avxc-python3-3.13.15/bin/python3'
OWNER_SHA = '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'
CACHE_SHA = '42ac6bf6f82e61adfe1cff7c565076361f222c51792f02df09337ff1779a791a'
BASE = 'hip-multiset-natural-x2-20261024-1'
PREDECESSOR_BASES = ('hip-multiset-source12-20261024-4', 'hip-multiset-controls-v5-20261024-1',
                     'hip-multiset-controls-v6-20261024-1', 'hip-multiset-controls-v7-20261024-1',
                     'hip-multiset-source20-20261024-1')
assert BASE not in PREDECESSOR_BASES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('payload', 'archive', 'remote-base', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    require(args.remote_base.parent == Path('/home/pvl/.cache/abird/zerogravity/qualification')
            and args.remote_base.name == BASE
            and all(v.is_absolute() for v in (args.payload, args.archive, args.output))
            and not args.output.exists(), 'fixed owned source20 namespace')
    args.output.mkdir(mode=0o700)
    summary = {'status': 'UNQUALIFIED', 'errors': [], 'steps': []}
    held = []
    control = control_pin = closure = closure_pin = None
    def stop(sig, _frame):
        message = 'admin signal '+str(sig)
        summary['errors'].append(message)
        raise InterruptedError(message)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        signal.signal(sig, stop)
    # All prehashes/stages share this absolute bound; no per-stage reset.
    work_end = time.monotonic() + 1830
    summary['local_work_seconds'] = 1830
    summary['local_finalization_seconds'] = 120
    summary['local_receipt_seconds'] = 5
    try:
        alarm_until(work_end)
        helper = args.payload/'source/tools/qwen27b-full-window-reference/owned_container_v4.py'
        require(pin(helper)['sha256'] == OWNER_SHA, 'unchanged original local owner')
        spec = importlib.util.spec_from_file_location('admin_original_owner', helper)
        owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner)
        control, control_pin = parsed(args.payload/'control.json')
        check_closure(args.payload, control)
        closure, closure_pin = parsed(args.payload/'source-closure.json')
        archive = Held(args.archive, limit=192*1024**2)
        held.append(archive)
        for name in ('bootstrap.py', 'pins.py'):
            held.append(Held(args.payload/'runner'/name))
        ssh = ['ssh', '-o', 'HostName=100.100.1.1', '-o', 'BindInterface=tailscale0',
               '-o', 'PermitLocalCommand=no', '-o', 'StrictHostKeyChecking=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
               '-o', 'ServerAliveCountMax=3', 'pvl-x2']
        create = 'from pathlib import Path; p=Path('+repr(str(args.remote_base))+'); p.mkdir(mode=0o700); print("FRESH_OWNED_BASE_CREATED")'
        stages = [('mkdir', ssh+[shlex.join([PYTHON, '-B', '-c', create])], 30)]
        paths = ['/proc/'+str(os.getpid())+'/fd/'+str(h.file.fileno()) for h in held]
        # One held original FD source per scp call; remote leaf names are fixed.
        for index, leaf in enumerate(('source-payload.tar', 'bootstrap.py', 'pins.py')):
            stages.append(('upload-'+leaf, ['scp', '-o', 'HostName=100.100.1.1', '-o', 'BindInterface=tailscale0',
                           '-o', 'PermitLocalCommand=no', '-o', 'StrictHostKeyChecking=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '--',
                           paths[index], 'pvl-x2:'+str(args.remote_base/leaf)], 180))
        state = Path('/home/pvl/.local/state/abird/zerogravity')/args.remote_base.name/'bootstrap-1'
        stages.append(('bootstrap', ssh+[shlex.join([PYTHON, '-B', str(args.remote_base/'bootstrap.py'),
            '--base', str(args.remote_base), '--output', str(state), '--archive-sha', archive.proof['sha256'],
            '--control-sha', control_pin['sha256']])], 960))
        for name, argv, wall in stages:
            receipt = {'name': name, 'argv': argv, 'errors': [], 'wall_seconds': wall}
            child = log = None
            try:
                alarm_until(work_end)
                child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                receipt['pid'] = child.pid
                log = owner.Log(child.stdout, args.output/(name+'.log'))
                receipt['observed_exit'] = log.observe(child, wall)
            except BaseException as error:
                receipt['errors'].append(repr(error))
            finally:
                # Never interrupt the original TERM/twoKILL-before-reap stages.
                signal.setitimer(signal.ITIMER_REAL, 0)
                if child is not None:
                    try:
                        owner.cleanup_group(child, receipt)
                    except BaseException as error:
                        receipt['errors'].append('owner: '+repr(error))
                if log is not None:
                    try:
                        alarm_until(min(time.monotonic(), work_end) + 30)
                        log.finish()
                    except BaseException as error:
                        receipt['errors'].append('EOF: '+repr(error))
                    receipt['log'] = {'eof': log.eof, 'closed': log.closed, 'bytes': log.total}
                if name == 'bootstrap' and receipt.get('observed_exit') == 0 and not receipt['errors']:
                    try:
                        summary['remote_bootstrap'] = parse_closed_bootstrap(
                            (args.output/(name+'.log')).read_bytes(), control_pin, archive.proof,
                            closure, closure_pin, CACHE_SHA)
                        receipt['remote_bootstrap'] = 'CLOSED_PASS_SCOPED_SAME_AUTHORITY'
                    except BaseException as error:
                        receipt['errors'].append('remote receipt: '+repr(error))
                summary['steps'].append(receipt)
            alarm_until(work_end)
            require(receipt.get('observed_exit') == 0 and not receipt['errors'], 'source/bootstrap stage failed: '+name)
    except BaseException as error:
        summary['errors'].append(repr(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        final_end = min(time.monotonic(), work_end) + 120
        for h in held:
            try:
                alarm_until(final_end)
                h.after()
            except BaseException as error:
                summary['errors'].append('held upload: '+repr(error))
            finally:
                h.close()
        try:
            alarm_until(final_end)
            require(control_pin is not None and pin(args.payload/'control.json') == control_pin,
                    'same local controller authority')
            check_closure(args.payload, control)
        except BaseException as error:
            summary['errors'].append('finalization: '+repr(error))
        summary['status'] = 'CLOSED_PASS_SCOPED' if not summary['errors'] else 'FAILED_CLOSED'
        signal.setitimer(signal.ITIMER_REAL, 5)
        (args.output/'invocation.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'status': summary['status'], 'errors': summary['errors']}, indent=2), flush=True)
    signal.setitimer(signal.ITIMER_REAL, 0)
    return 0 if summary['status'] == 'CLOSED_PASS_SCOPED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
