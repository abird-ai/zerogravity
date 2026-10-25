#!/usr/bin/env python3
"""Local original group/EOF owner for one fresh Vulkan Gemma physical case."""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from pins import alarm_until, parsed, pin, require
from sigterm_cancellation import SigtermCancellation

PYTHON = '/nix/store/jy00n3jp5xj8s40n1fbldmkcr5n7avxc-python3-3.13.15/bin/python3'
OWNER_SHA = '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'
BASE = 'resume-all-five-vulkan-20261025-2'
MODES = ('build', 'resident', 'resident-unknown', 'resident-row-unknown',
         'session-known', 'session-release-fault', 'snapshot-known',
         'snapshot-release-fault', 'partial-native',
         'peer-vulkan-then-hip', 'peer-hip-then-vulkan')
LIMITS = {'build': 3600}
LIMITS.update({mode: 7200 for mode in ('resident', 'resident-unknown', 'resident-row-unknown')})
LIMITS.update({mode: 1800 for mode in MODES if mode != 'build' and mode not in LIMITS})


def write_receipt(path, receipt, seconds):
    """One bounded receipt attempt; the timer is cancelled on every path."""
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        path.write_text(json.dumps(receipt, indent=2) + '\n')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class ManagedSignals:
    """Record signals across owned child acquisition/cleanup; ALRM raises elsewhere."""
    def __init__(self):
        self.cancel = SigtermCancellation()
        self.previous = {}
        self.critical_depth = 0
        self.pending_alarm = None

    def _record(self, _signum, _frame):
        self.cancel.received = True

    def _alarm(self, _signum, _frame):
        error = TimeoutError('local supervisor phase deadline')
        if self.critical_depth:
            self.pending_alarm = error
        else:
            raise error

    def __enter__(self):
        self.cancel.__enter__()
        for signum, handler in ((signal.SIGHUP, self._record),
                                (signal.SIGALRM, self._alarm)):
            self.previous[signum] = signal.signal(signum, handler)
        return self

    def __exit__(self, kind, value, traceback):
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)
        self.previous.clear()
        return self.cancel.__exit__(kind, value, traceback)

    @contextmanager
    def critical(self):
        self.critical_depth += 1
        try:
            yield
        finally:
            self.critical_depth -= 1

    def raise_pending_alarm(self):
        error = self.pending_alarm
        if error is not None:
            self.pending_alarm = None
            raise error

    def take_pending_alarm(self):
        error = self.pending_alarm
        self.pending_alarm = None
        return error

    @property
    def received(self):
        return self.cancel.received or self.pending_alarm is not None

    def exception(self):
        return self.pending_alarm or self.cancel.exception()

    def check(self):
        self.raise_pending_alarm()
        self.cancel.check()

    def observe(self, log, process, seconds):
        return self.cancel.observe(log, process, seconds)


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
            with guard.critical():
                signal.setitimer(signal.ITIMER_REAL, 0)
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
                guard.__exit__(None, None, None)
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
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    args = parser.parse_args()
    require(args.base.is_absolute()
            and args.base.parent == Path('/home/pvl/.cache/abird/zerogravity/qualification')
            and args.base.name == BASE and args.output.is_absolute()
            and args.output.parent == Path('/home/pvl/.local/state/abird/zerogravity') / BASE
            and not args.output.exists() and args.receipt.is_absolute()
            and not args.receipt.exists(), 'fresh exact owned local case namespace')
    source = args.base / 'source'
    receipt = {'mode': args.mode, 'status': 'UNQUALIFIED', 'errors': [],
               'worker': 'fresh local physical-runner child',
               'supervisor': 'original bounded group/twoKILL-before-oneReap/EOF owner'}
    child = log = None
    owner = None
    started = time.monotonic()
    guard = ManagedSignals()
    _LIVE['guard'] = guard
    _LIVE['receipt'] = receipt
    guard.__enter__()
    work_end = started + LIMITS[args.mode] + 600
    try:
        alarm_until(work_end)
        helper = source / 'tools/qwen27b-full-window-reference/owned_container_v4.py'
        require(pin(helper)['sha256'] == OWNER_SHA,
                'unchanged original local group/EOF owner')
        spec = importlib.util.spec_from_file_location('local_original_owner', helper)
        owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner)
        _LIVE['owner'] = owner
        control, control_pin = parsed(args.base / 'control.json')
        argv = [PYTHON, '-B',
                str(source / 'tools/vulkan-gemma-qualification/physical-runner/run.py'),
                '--base', str(args.base), '--output', str(args.output),
                '--mode', args.mode, '--control-sha', control_pin['sha256']]
        receipt.update(argv=argv, control=control_pin)
        with guard.critical():
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ,
                                          'PYTHONDONTWRITEBYTECODE': '1'})
            _LIVE['child'] = child
            receipt['pid'] = child.pid
            log = owner.Log(child.stdout, args.receipt.parent / (args.mode + '-launch.log'))
            _LIVE['log'] = log
        guard.check()
        receipt['observed_exit'] = guard.observe(log, child, LIMITS[args.mode] + 300)
    except BaseException as error:
        receipt['errors'].append(repr(error))
    finally:
        with guard.critical():
            signal.setitimer(signal.ITIMER_REAL, 0)
            if child is not None and owner is not None:
                # Same-child cleanup attempt precedes the original call; never replayed.
                _LIVE['cleanup_attempted'] = True
                try:
                    owner.cleanup_group(child, receipt)
                    _LIVE['reaped'] = True
                except BaseException as error:
                    receipt['errors'].append('owner: ' + repr(error))
        if guard.cancel.received:
            receipt['errors'].append(repr(guard.exception()))
        pending = guard.take_pending_alarm()
        if pending is not None:
            receipt['errors'].append('owner: ' + repr(pending))
        final_end = min(time.monotonic(), work_end) + 300
        try:
            if log is not None:
                try:
                    alarm_until(final_end)
                except BaseException as error:
                    receipt['errors'].append('EOF deadline: ' + repr(error))
                try:
                    # Mandatory original close even when the absolute EOF
                    # deadline has already expired; finish() owns its bounded
                    # drain and unconditional selector/file/pipe closes.
                    log.finish()
                except BaseException as error:
                    receipt['errors'].append('EOF: ' + repr(error))
                receipt['log'] = {'retained': log.retained, 'total': log.total,
                                  'eof': log.eof, 'closed': log.closed}
        except BaseException as error:
            receipt['errors'].append('EOF cleanup: ' + repr(error))
        finally:
            # Unconditional original closure: an ALRM between stages or a
            # skipped finish() must not leave the Log file/selector/pipe or a
            # never-constructed raw PIPE open. Each close independently tried.
            try:
                if log is not None and not log.closed:
                    try:
                        log.finish()
                    except BaseException as error:
                        receipt['errors'].append('EOF fallback: ' + repr(error))
            finally:
                if child is not None and child.stdout is not None \
                        and not child.stdout.closed:
                    with guard.critical():
                        try:
                            child.stdout.close()
                        except BaseException as error:
                            receipt['errors'].append('raw stdout: ' + repr(error))
        try:
            if receipt.get('observed_exit') == 0 and not receipt['errors']:
                value, proof = parsed(args.output / 'invocation.json')
                if 'worker' in value:
                    raise ValueError('nested launch receipt')
                receipt['child_invocation'] = proof
                require(value.get('status') == 'CLOSED_PASS_SCOPED',
                        'actual closed physical child')
        except BaseException as error:
            receipt['errors'].append('child receipt: ' + repr(error))
        receipt['seconds'] = time.monotonic() - started
        receipt['status'] = 'CLOSED_PASS_SCOPED' if not receipt['errors'] \
            and receipt.get('observed_exit') == 0 else 'FAILED_CLOSED'
        try:
            try:
                write_receipt(args.receipt, receipt, 5)
            except BaseException as error:
                # An expired receipt deadline is an honest failure; the one
                # optional retry has its own bounded deadline and can never
                # turn the original failure into a pass.
                receipt['errors'].append('receipt: ' + repr(error))
                receipt['status'] = 'FAILED_CLOSED'
                try:
                    write_receipt(args.receipt, receipt, 5)
                except BaseException as retry:
                    receipt['errors'].append('receipt retry: ' + repr(retry))
        finally:
            with guard.critical():
                signal.setitimer(signal.ITIMER_REAL, 0)
            guard.__exit__(None, None, None)
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ('argv', 'child_invocation')}, indent=2), flush=True)
    return 0 if receipt['status'] == 'CLOSED_PASS_SCOPED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
