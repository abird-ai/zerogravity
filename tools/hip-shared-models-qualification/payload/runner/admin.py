#!/usr/bin/env python3
"""Bounded original local owner for fresh H upload and remote bootstrap."""
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
BASE = 'resume-all-five-hip-20261025-2'
PREDECESSOR_BASES = ('hip-multiset-source12-20261024-4', 'hip-multiset-controls-v5-20261024-1',
                     'hip-multiset-controls-v6-20261024-1', 'hip-multiset-controls-v7-20261024-1',
                     'hip-multiset-source20-20261024-2')
assert BASE not in PREDECESSOR_BASES


class ManagedSignals:
    """Record TERM/INT/HUP/ALRM only inside ownership and cleanup critical sections."""
    SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)

    def __init__(self, receipt, prefix, on_signal=None):
        self.receipt, self.prefix = receipt, prefix
        self.on_signal = on_signal
        self.depth = 0
        self.pending = []
        self.previous = {}

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, kind, value, traceback):
        self.depth -= 1
        return False

    def _handle(self, signum, _frame):
        # Record-only while a mandatory original close/ownership step is in flight;
        # the catching phase records the raise outside those sections.
        if self.on_signal is not None:
            self.on_signal()
        if self.depth:
            self.pending.append(signum)
            return
        message = self.prefix+' signal '+str(signum)
        raise (TimeoutError(message) if signum == signal.SIGALRM else InterruptedError(message))

    def install(self):
        try:
            for signum in self.SIGNALS:
                self.previous[signum] = signal.signal(signum, self._handle)
        except BaseException:
            # A partial install still restores every handler it replaced.
            try:
                self.restore()
            except BaseException:
                signal.setitimer(signal.ITIMER_REAL, 0)
            raise
        return self

    def restore(self):
        # Unconditional timer cancel and handler restore, even after a failed write.
        signal.setitimer(signal.ITIMER_REAL, 0)
        first_error = None
        for signum, handler in list(self.previous.items()):
            try:
                signal.signal(signum, handler)
            except BaseException as error:
                if first_error is None:
                    first_error = error
            else:
                self.previous.pop(signum, None)
        if first_error is not None:
            raise first_error

    def rethrow(self):
        # Ownership is already acquired, so a deferred signal still fails the run.
        while self.pending:
            signum = self.pending.pop(0)
            message = self.prefix+' signal '+str(signum)
            raise (TimeoutError(message) if signum == signal.SIGALRM else InterruptedError(message))

    def note(self):
        # Mandatory cleanup completed; record the signal without abandoning
        # any remaining original close.
        while self.pending:
            self.receipt['errors'].append(self.prefix+' signal '+str(self.pending.pop(0)))


_LIVE = {}


def _emergency_close():
    """Outer safety net: close/reap owned objects and restore handlers."""
    guard = _LIVE.get('guard')
    child = _LIVE.get('child')
    log = _LIVE.get('log')
    owner = _LIVE.get('owner')
    held = list(_LIVE.get('held', []))
    receipt = _LIVE.get('receipt') or {'errors': []}
    try:
        if guard is not None:
            with guard:
                if (child is not None and not _LIVE.get('reaped')
                        and not _LIVE.get('cleanup_attempted') and owner is not None):
                    try:
                        owner.cleanup_group(child, receipt)
                        _LIVE['reaped'] = True
                    except BaseException:
                        pass
                if log is not None:
                    try:
                        log.finish()
                    except BaseException:
                        pass
                if child is not None and child.stdout is not None and not child.stdout.closed:
                    try:
                        child.stdout.close()
                    except BaseException:
                        pass
                for item in held:
                    try:
                        item.close()
                    except BaseException:
                        pass
    finally:
        if guard is not None:
            try:
                guard.restore()
            except BaseException:
                signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    _LIVE.clear()
    try:
        return _main()
    finally:
        _emergency_close()


def _main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('payload', 'archive', 'remote-base', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    require(args.remote_base.parent == Path('/home/pvl/.cache/abird/zerogravity/qualification')
            and args.remote_base.name == BASE
            and all(v.is_absolute() for v in (args.payload, args.archive, args.output))
            and not args.output.exists(), 'fixed owned H namespace')
    args.output.mkdir(mode=0o700)
    summary = {'status': 'UNQUALIFIED', 'errors': [], 'steps': []}
    held = []
    control = control_pin = closure = closure_pin = None
    guard = ManagedSignals(summary, 'admin signal')
    _LIVE['guard'] = guard
    _LIVE['receipt'] = summary
    _LIVE['held'] = held
    guard.install()
    # All prehashes/stages share this absolute bound; no per-stage reset.
    work_end = time.monotonic() + 1830
    summary['local_work_seconds'] = 1830
    summary['local_finalization_seconds'] = 120
    summary['local_receipt_seconds'] = 5
    receipt_end = None
    try:
        try:
            alarm_until(work_end)
            helper = args.payload/'source/tools/qwen27b-full-window-reference/owned_container_v4.py'
            require(pin(helper)['sha256'] == OWNER_SHA, 'unchanged original local owner')
            spec = importlib.util.spec_from_file_location('admin_original_owner', helper)
            owner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(owner)
            _LIVE['owner'] = owner
            control, control_pin = parsed(args.payload/'control.json')
            check_closure(args.payload, control)
            closure, closure_pin = parsed(args.payload/'source-closure.json')
            with guard:
                archive = Held(args.archive, limit=192*1024**2)
                held.append(archive)
                _LIVE['held'] = list(held)
                for name in ('bootstrap.py', 'pins.py'):
                    held.append(Held(args.payload/'runner'/name))
                    _LIVE['held'] = list(held)
            guard.rethrow()
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
                # Register before Popen/acquisition so an asynchronous signal after
                # real cleanup still retains the actual child/cleanup receipt.
                summary['steps'].append(receipt)
                _LIVE['receipt'] = receipt
                child = log = None
                _LIVE.update(child=None, log=None, cleanup_attempted=False, reaped=False)
                try:
                    alarm_until(work_end)
                    with guard:
                        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                        _LIVE['child'] = child
                        receipt['pid'] = child.pid
                        log = owner.Log(child.stdout, args.output/(name+'.log'))
                        _LIVE['log'] = log
                    guard.rethrow()
                    receipt['observed_exit'] = log.observe(child, wall)
                except BaseException as error:
                    receipt['errors'].append(repr(error))
                finally:
                    # Never interrupt the original TERM/twoKILL-before-reap stages.
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    try:
                        with guard:
                            if child is not None:
                                # Same-child cleanup attempt precedes the original call; never replayed.
                                _LIVE['cleanup_attempted'] = True
                                try:
                                    owner.cleanup_group(child, receipt)
                                    _LIVE['reaped'] = True
                                except BaseException as error:
                                    receipt['errors'].append('owner: '+repr(error))
                            if child is not None and log is None:
                                # Original Log never constructed: close the raw stdout pipe.
                                try:
                                    child.stdout.close()
                                except BaseException as error:
                                    receipt['errors'].append('raw stdout: '+repr(error))
                            if log is not None:
                                try:
                                    alarm_until(min(time.monotonic(), work_end) + 30)
                                except BaseException as error:
                                    receipt['errors'].append('EOF deadline: '+repr(error))
                                try:
                                    log.finish()
                                except BaseException as error:
                                    receipt['errors'].append('EOF: '+repr(error))
                                receipt['log'] = {'eof': log.eof, 'closed': log.closed, 'bytes': log.total}
                            # Independently close the raw pipe even if EOF/finish failed.
                            if child is not None and child.stdout is not None and not child.stdout.closed:
                                try:
                                    child.stdout.close()
                                except BaseException as error:
                                    receipt['errors'].append('raw stdout: '+repr(error))
                            guard.note()
                    finally:
                        _LIVE['child'] = None
                        _LIVE['log'] = None
                if name == 'bootstrap' and receipt.get('observed_exit') == 0 and not receipt['errors']:
                    try:
                        summary['remote_bootstrap'] = parse_closed_bootstrap(
                            (args.output/(name+'.log')).read_bytes(), control_pin, archive.proof,
                            closure, closure_pin, CACHE_SHA)
                        receipt['remote_bootstrap'] = 'CLOSED_PASS_SCOPED_SAME_AUTHORITY'
                    except BaseException as error:
                        receipt['errors'].append('remote receipt: '+repr(error))
                alarm_until(work_end)
                require(receipt.get('observed_exit') == 0 and not receipt['errors']
                        and not summary['errors'], 'source/bootstrap stage failed: '+name)
        except BaseException as error:
            summary['errors'].append(repr(error))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            final_end = min(time.monotonic(), work_end) + 120
            try:
                with guard:
                    for h in held:
                        try:
                            alarm_until(final_end)
                            h.after()
                        except BaseException as error:
                            summary['errors'].append('held upload: '+repr(error))
                        finally:
                            try:
                                h.close()
                            except BaseException as error:
                                summary['errors'].append('held close: '+repr(error))
                    guard.note()
                try:
                    alarm_until(final_end)
                    require(control_pin is not None and pin(args.payload/'control.json') == control_pin,
                            'same local controller authority')
                    check_closure(args.payload, control)
                except BaseException as error:
                    summary['errors'].append('finalization: '+repr(error))
            except BaseException as error:
                # A signal in a gap between finalization phases never skips the failure receipt.
                summary['errors'].append('finalization cleanup: '+repr(error))
            summary['status'] = 'CLOSED_PASS_SCOPED' if not summary['errors'] else 'FAILED_CLOSED'
            receipt_end = time.monotonic() + 5
            try:
                alarm_until(receipt_end)
                (args.output/'invocation.json').write_text(json.dumps(summary, indent=2)+'\n')
            except BaseException as error:
                summary['errors'].append('receipt write: '+repr(error))
                summary['status'] = 'FAILED_CLOSED'
                try:
                    # Only the remaining part of the same absolute receipt budget.
                    alarm_until(receipt_end)
                    (args.output/'invocation.json').write_text(json.dumps(summary, indent=2)+'\n')
                except BaseException as error:
                    summary['errors'].append('receipt rewrite: '+repr(error))
                    summary['status'] = 'FAILED_CLOSED'
    finally:
        try:
            # Final stdout shares the one five-second receipt deadline. Handlers
            # and the timer stay installed until print/flush returns or fails.
            if receipt_end is None:
                receipt_end = time.monotonic()
            alarm_until(receipt_end)
            print(json.dumps({'status': summary['status'], 'errors': summary['errors']}, indent=2), flush=True)
        except BaseException as error:
            summary['errors'].append('stdout: '+repr(error))
            summary['status'] = 'FAILED_CLOSED'
        finally:
            guard.restore()
    return 0 if summary['status'] == 'CLOSED_PASS_SCOPED' and not summary['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
