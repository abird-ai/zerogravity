#!/usr/bin/env python3
"""Verify the fresh local Vulkan Gemma source/data namespace before any build."""
import argparse
from contextlib import contextmanager
import json
import signal
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

from pins import alarm_until, check_closure, parsed, pin, require
from sigterm_cancellation import SigtermCancellation

BASE = 'resume-all-five-vulkan-20261025-1'
WORK_SECONDS = 120
FINAL_SECONDS = 60
RECEIPT_SECONDS = 5


class ManagedSignals:
    """Record TERM/INT/HUP/ALRM across verification and every mandatory close."""
    SIGNALS = (signal.SIGHUP, signal.SIGALRM)

    def __init__(self, receipt, prefix):
        self.cancel = SigtermCancellation()
        self.receipt, self.prefix = receipt, prefix
        self.pending_alarm = None
        self.depth = 0
        self.previous = {}

    def __enter__(self):
        try:
            self.cancel.__enter__()
            for signum in self.SIGNALS:
                self.previous[signum] = signal.signal(signum, self._handle)
        except BaseException:
            # Partial guard setup restores both term signals and HUP/ALRM.
            try:
                self.restore()
            except BaseException:
                signal.setitimer(signal.ITIMER_REAL, 0)
            raise
        return self

    def __exit__(self, kind, value, traceback):
        self.restore()
        return False

    def _handle(self, signum, _frame):
        if signum == signal.SIGALRM:
            error = TimeoutError(self.prefix + ' phase deadline')
            if self.depth:
                self.pending_alarm = error
            else:
                raise error
        else:
            self.cancel.received = True

    @contextmanager
    def critical(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def check(self):
        if self.pending_alarm is not None:
            error = self.pending_alarm
            self.pending_alarm = None
            raise error
        if self.cancel.received:
            self.cancel.received = False
            raise self.cancel.exception()

    def note(self):
        if self.pending_alarm is not None:
            self.receipt['errors'].append(repr(self.pending_alarm))
            self.pending_alarm = None
        if self.cancel.received:
            self.receipt['errors'].append(repr(self.cancel.exception()))
            self.cancel.received = False

    def restore(self):
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
        try:
            self.cancel.__exit__(None, None, None)
        except BaseException as error:
            if first_error is None:
                first_error = error
        else:
            self.cancel.previous.clear()
        if first_error is not None:
            raise first_error


def write_receipt(path, receipt, deadline):
    """One receipt attempt inside the shared absolute finalization deadline."""
    alarm_until(deadline)
    path.write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.base.is_absolute()
            and args.base.parent == Path('/home/pvl/.cache/abird/zerogravity/qualification')
            and args.base.name == BASE and args.output.is_absolute()
            and not args.output.exists(), 'fresh exact owned bootstrap namespace')
    args.output.mkdir(mode=0o700)
    receipt = {'schema': 'zg.vulkan-gemma-bootstrap.v1', 'status': 'UNQUALIFIED',
               'errors': [], 'gpu_or_model_executed': False, 'network_transfer': False}
    receipt['local_work_seconds'] = WORK_SECONDS
    receipt['local_finalization_seconds'] = FINAL_SECONDS
    receipt['local_receipt_seconds'] = RECEIPT_SECONDS
    started = time.monotonic()
    work_end = started + WORK_SECONDS
    receipt_end = None
    guard = ManagedSignals(receipt, 'bootstrap')
    closure = control = None
    closure_pin = control_pin = assembly_pin = None
    source_before = source_after = None
    try:
        guard.__enter__()
        alarm_until(work_end)
        with guard.critical():
            closure, closure_pin = parsed(args.base/'source-closure.json')
            check_closure(args.base/'source', closure)
            control, control_pin = parsed(args.base/'control.json')
            check_closure(args.base, control)
            guard.check()
            require(control.get('source_closure') == closure_pin,
                    'control/source closure authority differs')
            assembly_pin = pin(args.base/'assembled-source.json',
                               limit=16 * 1024**2)
            require(control.get('source_manifest') == assembly_pin,
                    'control binds the exact supplied ROOT manifest')
            require(closure.get('source_manifest') == assembly_pin,
                    'source closure binds the exact supplied ROOT manifest')
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                                   / 'physical-runner'))
            from modes import check_source
            source_before = check_source(args.base)
        guard.check()
        with guard.critical():
            source_after = check_source(args.base)
            require(source_after == source_before,
                    'complete source authority changed')
            require(pin(args.base/'assembled-source.json') == assembly_pin,
                    'assembled source manifest changed')
            require(pin(args.base/'source-closure.json') == closure_pin,
                    'source closure authority changed')
            require(pin(args.base/'control.json') == control_pin,
                    'controller authority changed')
            check_closure(args.base, control)
            check_closure(args.base/'source', closure)
            guard.check()
            for name in ('gemma', 'qwen'):
                guard.check()
                data, data_pin = parsed(args.base/name/'data-closure.json')
                require(data.get('new_download') is False, 'no new acquisition')
                receipt[name+'_data'] = data_pin
        guard.check()
        receipt.update(source_closure=closure_pin, control=control_pin,
                       assembled_source=assembly_pin,
                       source_manifest=source_before['source_manifest'],
                       source_count=source_before['source_count'],
                       files=source_before['files'],
                       bytes=closure['bytes'],
                       source_before='UNCHANGED', source_after='UNCHANGED')
        receipt['status'] = 'CLOSED_PASS_SCOPED'
    except BaseException as error:
        receipt['errors'].append(repr(error))
    finally:
        try:
            with guard.critical():
                signal.setitimer(signal.ITIMER_REAL, 0)
                final_end = min(time.monotonic(), work_end) + FINAL_SECONDS
                try:
                    alarm_until(final_end)
                    require(assembly_pin is not None
                            and pin(args.base/'assembled-source.json') == assembly_pin,
                            'assembled source manifest changed')
                    require(closure_pin is not None
                            and pin(args.base/'source-closure.json') == closure_pin,
                            'source closure authority changed')
                    require(control_pin is not None
                            and pin(args.base/'control.json') == control_pin,
                            'controller authority changed')
                except BaseException as error:
                    receipt['errors'].append('finalization: '+repr(error))
        finally:
            guard.note()
            receipt['seconds'] = time.monotonic() - started
            receipt['status'] = ('CLOSED_PASS_SCOPED'
                                 if not receipt['errors'] else 'FAILED_CLOSED')
            receipt_end = time.monotonic() + RECEIPT_SECONDS
            try:
                try:
                    write_receipt(args.output/'bootstrap.json', receipt,
                                  receipt_end)
                except BaseException as error:
                    # Only the remaining part of the same absolute receipt budget.
                    receipt['errors'].append('receipt: '+repr(error))
                    receipt['status'] = 'FAILED_CLOSED'
                    try:
                        write_receipt(args.output/'bootstrap.json', receipt,
                                      receipt_end)
                    except BaseException as retry:
                        receipt['errors'].append('receipt retry: '+repr(retry))
                        receipt['status'] = 'FAILED_CLOSED'
                try:
                    # Final stdout shares that deadline; handlers/timer remain
                    # installed through print/flush.
                    alarm_until(receipt_end)
                    print(json.dumps({'status': receipt['status'],
                                      'errors': receipt['errors']},
                                     indent=2), flush=True)
                except BaseException as error:
                    receipt['errors'].append('stdout: '+repr(error))
                    receipt['status'] = 'FAILED_CLOSED'
            finally:
                guard.restore()
    return (0 if receipt['status'] == 'CLOSED_PASS_SCOPED'
            and not receipt['errors'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
