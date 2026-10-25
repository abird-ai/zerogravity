#!/usr/bin/env python3
"""Local original SSH owner for one H build, finite lifecycle, or source-page event-probe case."""
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
from bootstrap_receipt import _one, _owner
from sigterm_cancellation import SigtermCancellation

PYTHON = '/nix/store/jy00n3jp5xj8s40n1fbldmkcr5n7avxc-python3-3.13.15/bin/python3'
OWNER_SHA = '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'
BASE = 'resume-all-five-hip-20261025-3'
PREDECESSOR_BASES = ('hip-multiset-source12-20261024-4', 'hip-multiset-controls-v5-20261024-1',
                     'hip-multiset-controls-v6-20261024-1', 'hip-multiset-controls-v7-20261024-1',
                     'hip-multiset-source20-20261024-2')
assert BASE not in PREDECESSOR_BASES
MODES = ('known-qwen-reload', 'qwen-final-reader-unknown', 'gemma-feedback-unknown',
         'source-page-query-unknown', 'source-page-wait-unknown',
         'source-page-destroy-unknown')
LIMITS = {'build': 3600, 'known-qwen-reload': 7200,
          'qwen-final-reader-unknown': 1800, 'gemma-feedback-unknown': 1800,
          'source-page-query-unknown': 1800, 'source-page-wait-unknown': 1800,
          'source-page-destroy-unknown': 1800}
ARTIFACT = 'target/release/examples/hip_shared_models'


def validate_binding(value, closure_pin, remote_base):
    def proof(p):
        require(type(p) is dict and set(p) == {'bytes', 'sha256'} and type(p['bytes']) is int
                and 0 < p['bytes'] <= 2*1024**3 and len(p['sha256']) == 64
                and p['sha256'] == p['sha256'].lower(), 'bounded reviewed build pin')
    require(set(value) == {'schema', 'source_base', 'source_closure', 'binary', 'build_receipt', 'provenance'}
            and value['schema'] == 'zg.hip-source20-build-binding.v1'
            and value['source_base'] == str(remote_base)
            and value['source_closure'] == closure_pin, 'reviewed H build binding only')
    for key in ('binary', 'build_receipt'):
        proof(value[key])
    proof(value['provenance'])
    return value


def remote_argv(mode, remote_base, remote_output, control_sha, binding=None):
    argv = [PYTHON, '-B', str(remote_base/'physical-runner/run.py'), '--base', str(remote_base),
            '--output', str(remote_output), '--mode', mode, '--control-sha', control_sha]
    if mode != 'build':
        require(binding is not None and set(binding) == {'binary', 'build_receipt'}, 'bound H build required')
        argv += ['--binary-sha', binding['binary']['sha256'],
                 '--binary-bytes', str(binding['binary']['bytes']),
                 '--build-receipt-sha', binding['build_receipt']['sha256'],
                 '--build-receipt-bytes', str(binding['build_receipt']['bytes'])]
    else:
        require(binding is None, 'a fresh build passes no artifact/receipt pins')
    return argv


def unchanged_local_package(payload, control, control_pin, packaged_closure, closure_pin):
    """Final local check: same bounded controller and every payload byte."""
    require(pin(payload/'control.json') == control_pin, 'local frozen authority unchanged')
    require(pin(payload/'source-closure.json') == closure_pin, 'local packaged source closure unchanged')
    check_closure(payload, control)
    check_closure(payload, packaged_closure)
    return 'UNCHANGED_FULL_SHA'


def remote_receipt(mode, path, closure_pin, control_pin, binding=None):
    remote = _one(Path(path).read_bytes(), False)
    _owner(remote)
    hydrated = remote.get('hydrated_closure')
    require(remote['mode'] == mode and remote['status'] == 'CLOSED_PASS_SCOPED'
            and remote['control'] == control_pin and remote['control_after'] == control_pin
            and remote['source_closure'] == closure_pin
            and remote.get('source_after') == 'UNCHANGED',
            'closed remote semantic/source/control closure')
    require(type(hydrated) is dict and set(hydrated) == {'bytes', 'sha256'}
            and type(hydrated['bytes']) is int and hydrated['bytes'] > closure_pin['bytes']
            and len(hydrated['sha256']) == 64 and hydrated['sha256'] != closure_pin['sha256'],
            'separate authenticated hydrated source closure')
    if mode == 'build':
        require(binding is None and remote['build_artifact']['path'].endswith(ARTIFACT)
                and remote['build_artifact']['source_closure'] == closure_pin
                and remote['build_artifact']['hydrated_closure'] == hydrated
                and type(remote['build_receipt']) is dict and set(remote['build_receipt']) == {'bytes', 'sha256'},
                'actual H build artifact/receipt record')
    else:
        require(binding is not None and remote['binary'] == binding['binary']
                and remote['build_binding']['build_receipt'] == binding['build_receipt']
                and remote.get('held_data_files') == 70
                and remote['data_after'] == 'UNCHANGED_HELD_FULL_SHA'
                and remote['semantic']['mode'] == mode
                and remote['semantic']['math_state_owner_scope_only'] is True
                and remote['semantic']['serving_speed_or_general_fit'] is False,
                'bound H build, exact held_data_files70 and finite/probe semantics only')
    return remote


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--remote-base', type=Path, required=True)
    parser.add_argument('--remote-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('build',)+MODES, required=True)
    parser.add_argument('--binding', type=Path)
    args = parser.parse_args()
    require(all(p.is_absolute() for p in (args.payload, args.remote_base, args.remote_output, args.output))
            and not args.output.exists(), 'absolute fresh own output paths')
    require(args.remote_base.parent == Path('/home/pvl/.cache/abird/zerogravity/qualification')
            and args.remote_base.name == BASE
            and args.remote_output.parent == Path('/home/pvl/.local/state/abird/zerogravity')/args.remote_base.name,
            'fixed owned H remote namespaces')
    args.output.mkdir(mode=0o700)
    child = log = None
    receipt = {'mode': args.mode, 'status': 'UNQUALIFIED', 'errors': [],
               'remote_base': str(args.remote_base), 'remote_output': str(args.remote_output)}
    guard = ManagedSignals(receipt, 'local SSH owner')
    _LIVE['guard'] = guard
    _LIVE['receipt'] = receipt
    guard.install()
    started = time.monotonic()
    receipt_end = None
    owner = control = control_pin = packaged_closure = closure_pin = binding = None
    work_end = started + LIMITS[args.mode] + 1800
    receipt['local_work_seconds'] = LIMITS[args.mode] + 1800
    receipt['local_finalization_seconds'] = 900
    receipt['local_receipt_seconds'] = 5
    try:
        alarm_until(work_end)
        helper = args.payload/'source/tools/qwen27b-full-window-reference/owned_container_v4.py'
        require(pin(helper)['sha256'] == OWNER_SHA, 'unchanged original local group/EOF owner')
        spec = importlib.util.spec_from_file_location('local_original_owner', helper)
        owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner)
        _LIVE['owner'] = owner
        control, control_pin = parsed(args.payload/'control.json')
        check_closure(args.payload, control)
        packaged_closure, closure_pin = parsed(args.payload/'source-closure.json')
        binding = None
        if args.mode != 'build':
            require(args.binding is not None and args.binding.is_absolute(), 'reviewed binding file required')
            value, binding_pin = parsed(args.binding)
            validate_binding(value, closure_pin, args.remote_base)
            binding = {'binary': value['binary'], 'build_receipt': value['build_receipt']}
            receipt['binding'] = binding_pin
        remote = remote_argv(args.mode, args.remote_base, args.remote_output,
                             control_pin['sha256'], binding)
        argv = ['ssh', '-o', 'HostName=100.100.1.1', '-o', 'BindInterface=tailscale0',
                '-o', 'PermitLocalCommand=no', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
                '-o', 'ServerAliveCountMax=3', 'pvl-x2', shlex.join(remote)]
        receipt['argv'] = argv
        with guard:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            _LIVE['child'] = child
            receipt['pid'] = child.pid
            log = owner.Log(child.stdout, args.output/'execution.log')
            _LIVE['log'] = log
        guard.rethrow()
        with SigtermCancellation() as cancellation:
            receipt['observed_exit'] = cancellation.observe(log, child, LIMITS[args.mode]+900)
    except BaseException as error:
        receipt['errors'].append(repr(error))
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                # Mandatory original group/EOF/pipe closes: record-only managed signals.
                with guard:
                    if child is not None:
                        # Same-child cleanup attempt precedes the original call; never replayed.
                        _LIVE['cleanup_attempted'] = True
                        try:
                            owner.cleanup_group(child, receipt)
                            _LIVE['reaped'] = True
                        except BaseException as error:
                            receipt['errors'].append('local owner: '+repr(error))
                    if child is not None and log is None:
                        # Original Log never constructed: close the raw stdout pipe.
                        try:
                            child.stdout.close()
                        except BaseException as error:
                            receipt['errors'].append('local raw stdout: '+repr(error))
                    if log is not None:
                        try:
                            log.finish()
                        except BaseException as error:
                            receipt['errors'].append('local EOF: '+repr(error))
                        receipt['log'] = {'total': log.total, 'retained': log.retained,
                                          'eof': log.eof, 'closed': log.closed}
                    # Independently close the raw pipe even if EOF/finish failed.
                    if child is not None and child.stdout is not None and not child.stdout.closed:
                        try:
                            child.stdout.close()
                        except BaseException as error:
                            receipt['errors'].append('local raw stdout: '+repr(error))
                guard.note()
                final_end = min(time.monotonic(), work_end) + 900
                try:
                    alarm_until(final_end)
                    if all(v is not None for v in (control, control_pin, packaged_closure, closure_pin)):
                        receipt['local_package_after'] = unchanged_local_package(
                            args.payload, control, control_pin, packaged_closure, closure_pin)
                except BaseException as error:
                    receipt['errors'].append('local package: '+repr(error))
                try:
                    alarm_until(final_end)
                    if receipt.get('observed_exit') == 0 and not receipt['errors']:
                        remote = remote_receipt(args.mode, args.output/'execution.log',
                                                closure_pin, control_pin, binding)
                        receipt['remote_invocation'] = remote
                        require(remote['status'] == 'CLOSED_PASS_SCOPED', 'actual remote closure')
                except BaseException as error:
                    receipt['errors'].append('local receipt: '+repr(error))
            except BaseException as error:
                # A signal in a gap between phases never skips the failure receipt.
                receipt['errors'].append('local cleanup: '+repr(error))
            receipt['seconds'] = time.monotonic()-started
            receipt['status'] = ('CLOSED_PASS_SCOPED'
                                 if not receipt['errors'] and receipt.get('observed_exit') == 0
                                 else 'FAILED_CLOSED')
            # One absolute receipt budget covers the attempt, optional failure
            # rewrite and the final stdout; a failed or retried write can never
            # upgrade the recorded failure to a pass.
            receipt_end = time.monotonic() + 5
            try:
                alarm_until(receipt_end)
                (args.output/'invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
            except BaseException as error:
                receipt['errors'].append('local receipt write: '+repr(error))
                receipt['status'] = 'FAILED_CLOSED'
                try:
                    alarm_until(receipt_end)
                    (args.output/'invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
                except BaseException as error:
                    receipt['errors'].append('local receipt rewrite: '+repr(error))
                    receipt['status'] = 'FAILED_CLOSED'
        finally:
            try:
                # Final stdout shares the one five-second receipt deadline.
                # Handlers and the timer stay installed until print/flush returns or fails.
                if receipt_end is None:
                    receipt_end = time.monotonic()
                alarm_until(receipt_end)
                print(json.dumps({k: v for k, v in receipt.items() if k not in ('argv', 'remote_invocation')},
                                 indent=2), flush=True)
            except BaseException as error:
                receipt['errors'].append('local stdout: '+repr(error))
                receipt['status'] = 'FAILED_CLOSED'
            finally:
                guard.restore()
    return 0 if receipt['status'] == 'CLOSED_PASS_SCOPED' and not receipt['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
