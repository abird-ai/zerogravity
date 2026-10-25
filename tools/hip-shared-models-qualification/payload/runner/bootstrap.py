#!/usr/bin/env python3
"""Fresh H extraction and offline hydration; no model/GPU call."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import tarfile
import time

from pins import Held, alarm_until, check_closure, parsed, pin, relative, require

PYTHON = '/nix/store/jy00n3jp5xj8s40n1fbldmkcr5n7avxc-python3-3.13.15/bin/python3'
CACHE = Path('/home/pvl/.cache/abird/zerogravity/qualification/portable-service-20261023-1')
OWNER_SHA = '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'
CACHE_SHA = '42ac6bf6f82e61adfe1cff7c565076361f222c51792f02df09337ff1779a791a'
BASE = 'resume-all-five-hip-20261025-1'
PREDECESSOR_BASES = ('hip-multiset-source12-20261024-4', 'hip-multiset-controls-v5-20261024-1',
                     'hip-multiset-controls-v6-20261024-1', 'hip-multiset-controls-v7-20261024-1',
                     'hip-multiset-source20-20261024-2')
assert BASE not in PREDECESSOR_BASES


def extract_archive(tar, base, forbidden=()):
    """Extract contained regular members with their authenticated archive modes."""
    seen = set()
    total = 0
    for entry in tar:
        name = relative(entry.name)
        require(entry.isfile() and not entry.issparse() and entry.name not in seen
                and entry.name not in forbidden, 'fresh contained regular source member')
        total += entry.size
        require(entry.size <= 64*1024**2 and total <= 192*1024**2, 'bounded complete source archive')
        destination = base/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        mode = stat.S_IMODE(entry.mode)
        with tar.extractfile(entry) as source, destination.open('xb') as target:
            os.fchmod(target.fileno(), mode)
            require(stat.S_IMODE(os.fstat(target.fileno()).st_mode) == mode, 'archive member mode')
            while block := source.read(1024**2):
                target.write(block)
        seen.add(entry.name)
    return seen


def hydrate_import(base):
    """Finalization parser import; runner/ is archived off the bootstrap startup path."""
    sys.path.insert(0, str(base/'runner'))
    from bootstrap_receipt import parse_hydrate
    return parse_hydrate


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
    archive = _LIVE.get('archive')
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
                if archive is not None:
                    try:
                        archive.close()
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
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive-sha', required=True)
    parser.add_argument('--control-sha', required=True)
    args = parser.parse_args()
    require(args.base.parent == Path('/home/pvl/.cache/abird/zerogravity/qualification')
            and args.base.name == BASE and not (args.base/'source').exists()
            and not args.output.exists(), 'fresh owned remote source/output')
    args.output.mkdir(parents=True, mode=0o700)
    resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
    receipt = {'status': 'UNQUALIFIED', 'errors': [], 'gpu_or_model_executed': False}
    archive = child = log = owner = None
    started = time.monotonic()
    work_end = started + 900
    receipt_end = None
    closure = closure_pin = source_pin = None
    guard = ManagedSignals(receipt, 'bootstrap owner')
    _LIVE['guard'] = guard
    _LIVE['receipt'] = receipt
    guard.install()
    try:
        alarm_until(work_end)
        with guard:
            archive = Held(args.base/'source-payload.tar', limit=192*1024**2)
            _LIVE['archive'] = archive
        guard.rethrow()
        require(archive.proof['sha256'] == args.archive_sha, 'exact source payload SHA')
        with tarfile.open(fileobj=archive.file, mode='r:') as tar:
            seen = extract_archive(tar, args.base,
                                   ('pins.py', 'bootstrap.py', 'source-payload.tar'))
        closure, closure_pin = parsed(args.base/'source-closure.json')
        require(seen == set(closure['records']) | {'source-closure.json'}, 'exact source archive member union')
        check_closure(args.base, closure)
        control, source_pin = parsed(args.base/'control.json')
        require(source_pin['sha256'] == args.control_sha, 'exact separately approved controller')
        check_closure(args.base, control)
        sys.path.insert(0, str(args.base/'physical-runner'))
        from modes import check_source
        check_source(args.base)
        helper = args.base/'source/tools/qwen27b-full-window-reference/owned_container_v4.py'
        require(pin(helper)['sha256'] == OWNER_SHA, 'unchanged original remote group/EOF owner')
        spec = importlib.util.spec_from_file_location('bootstrap_original_owner', helper)
        owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner)
        _LIVE['owner'] = owner
        cache = pin(CACHE/'closure.json')
        require(cache['sha256'] == CACHE_SHA, 'same approved offline registry closure')
        argv = [PYTHON, '-B', str(args.base/'source/scripts/native-gpu.py'), '--', PYTHON, '-B',
                str(args.base/'hydrate.py'), '--base', str(args.base), '--cache-base', str(CACHE)]
        receipt.update(argv=argv, source_archive=archive.proof, source_closure=closure_pin,
                       control=source_pin, cache_authority=cache, child_wall_seconds=600,
                       whole_supervisor_alarm_seconds=900)
        with guard:
            child = subprocess.Popen(argv, cwd=args.base, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CARGO_NET_OFFLINE': 'true'})
            _LIVE['child'] = child
            receipt['pid'] = child.pid
            log = owner.Log(child.stdout, args.output/'execution.log')
            _LIVE['log'] = log
        guard.rethrow()
        receipt['observed_exit'] = log.observe(child, 600)
    except BaseException as error:
        receipt['errors'].append(repr(error))
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                try:
                    with guard:
                        if child is not None and owner is not None:
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
                                alarm_until(min(time.monotonic(), work_end) + 60)
                            except BaseException as error:
                                receipt['errors'].append('EOF deadline: '+repr(error))
                            try:
                                log.finish()
                            except BaseException as error:
                                receipt['errors'].append('EOF: '+repr(error))
                            receipt['log'] = {'retained': log.retained, 'total': log.total, 'eof': log.eof, 'closed': log.closed}
                        # Independently close the raw pipe even if EOF/finish failed.
                        if child is not None and child.stdout is not None and not child.stdout.closed:
                            try:
                                child.stdout.close()
                            except BaseException as error:
                                receipt['errors'].append('raw stdout: '+repr(error))
                        if archive is not None:
                            try:
                                archive.after()
                                receipt['source_archive_after'] = 'UNCHANGED_HELD_FULL_SHA'
                            except BaseException as error:
                                receipt['errors'].append('source archive: '+repr(error))
                            finally:
                                try:
                                    archive.close()
                                except BaseException as error:
                                    receipt['errors'].append('source archive close: '+repr(error))
                        guard.note()
                finally:
                    _LIVE['child'] = None
                    _LIVE['log'] = None
                try:
                    alarm_until(min(time.monotonic(), work_end) + 60)
                    if receipt.get('observed_exit') == 0 and not receipt['errors']:
                        parse_hydrate = hydrate_import(args.base)
                        receipt['hydrate'] = parse_hydrate((args.output/'execution.log').read_bytes(),
                                                           receipt, closure, CACHE_SHA)
                        hydrated, hydrated_pin = parsed(args.base/'closure.json')
                        check_closure(args.base, hydrated)
                        require(hydrated_pin['sha256'] != closure_pin['sha256'] and hydrated_pin['bytes'] > closure_pin['bytes'],
                                'fresh hydrated closure, not the packaged one')
                        for name, proof in closure['records'].items():
                            require(hydrated['records'][name] == proof, 'hydrated member changed: '+name)
                        require(hydrated['files'] > closure['files'] and hydrated['reused_cache_closure_sha256'] == CACHE_SHA,
                                'complete hydrated cache closure')
                        receipt['hydrated_closure'] = hydrated_pin
                        control_now, proof = parsed(args.base/'control.json')
                        require(proof == source_pin, 'same approved controller during hydration')
                        check_closure(args.base, control_now)
                        receipt['control_after'] = proof
                except BaseException as error:
                    receipt['errors'].append('hydrated closure: '+repr(error))
            except BaseException as error:
                # A signal in a gap between finalization phases never skips the failure receipt.
                receipt['errors'].append('finalization: '+repr(error))
            receipt['seconds'] = time.monotonic()-started
            receipt['status'] = 'CLOSED_PASS_SCOPED' if not receipt['errors'] and receipt.get('observed_exit') == 0 else 'FAILED_CLOSED'
            receipt_end = time.monotonic() + 5
            try:
                alarm_until(receipt_end)
                (args.output/'invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
            except BaseException as error:
                receipt['errors'].append('receipt write: '+repr(error))
                receipt['status'] = 'FAILED_CLOSED'
                try:
                    # Only the remaining part of the same absolute receipt budget.
                    alarm_until(receipt_end)
                    (args.output/'invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
                except BaseException as error:
                    receipt['errors'].append('receipt rewrite: '+repr(error))
                    receipt['status'] = 'FAILED_CLOSED'
        finally:
            try:
                # Final stdout shares the one five-second receipt deadline. Handlers
                # and the timer stay installed until print/flush returns or fails.
                if receipt_end is None:
                    receipt_end = time.monotonic()
                alarm_until(receipt_end)
                print(json.dumps(receipt, indent=2), flush=True)
            except BaseException as error:
                receipt['errors'].append('stdout: '+repr(error))
                receipt['status'] = 'FAILED_CLOSED'
            finally:
                guard.restore()
    return 0 if receipt['status'] == 'CLOSED_PASS_SCOPED' and not receipt['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
