#!/usr/bin/env python3
"""Local original SSH owner for one source20 x2 build or lifecycle case."""
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
BASE = 'hip-multiset-natural-x2-20261024-1'
PREDECESSOR_BASES = ('hip-multiset-source20-20261024-2', 'hip-multiset-source20-20261024-1',
                     'hip-multiset-source12-20261024-4', 'hip-multiset-controls-v5-20261024-1')
assert BASE not in PREDECESSOR_BASES
MODES = ('healthy', 'cached-unknown', 'session-known', 'session-release-fault',
         'snapshot-known', 'snapshot-release-fault')
LIMITS = {'build': 3600, 'healthy': 7200}
for _mode in MODES:
    LIMITS.setdefault(_mode, 1800)
ARTIFACT = 'target/release/examples/qwen27b_hip_natural'


def check_pending(state):
    while state['pending']:
        sig = state['pending'].pop(0)
        if sig == signal.SIGALRM:
            raise TimeoutError('local owner deadline '+str(sig))
        raise InterruptedError('local SSH owner signal '+str(sig))


class Hold:
    """Record managed signals only; mandatory closes must always finish."""
    def __init__(self, state):
        self.state = state
    def __enter__(self):
        self.state['depth'] = self.state.get('depth', 0)+1
        self.state['critical'] = True
        return self
    def __exit__(self, kind, value, traceback):
        self.state['depth'] = max(0, self.state.get('depth', 1)-1)
        self.state['critical'] = self.state['depth'] > 0
        return False


class Ownership(Hold):
    """Record through Popen/Log ownership; check pending once the child is held."""
    def __exit__(self, kind, value, traceback):
        super().__exit__(kind, value, traceback)
        if kind is None:
            check_pending(self.state)
        return False


def validate_binding(value, closure_pin, remote_base):
    def proof(p):
        require(type(p) is dict and set(p) == {'bytes', 'sha256'} and type(p['bytes']) is int
                and 0 < p['bytes'] <= 2*1024**3 and len(p['sha256']) == 64
                and p['sha256'] == p['sha256'].lower(), 'bounded reviewed build pin')
    require(set(value) == {'schema', 'source_base', 'source_closure', 'binary', 'build_receipt', 'provenance'}
            and value['schema'] == 'zg.hip-source20-natural-x2-build-binding.v1'
            and value['source_base'] == str(remote_base)
            and value['source_closure'] == closure_pin, 'reviewed new natural x2 build binding only')
    for key in ('binary', 'build_receipt'):
        proof(value[key])
    proof(value['provenance'])
    return value


def remote_argv(mode, remote_base, remote_output, control_sha, binding=None):
    argv = [PYTHON, '-B', str(remote_base/'physical-runner/run.py'), '--base', str(remote_base),
            '--output', str(remote_output), '--mode', mode, '--control-sha', control_sha]
    if mode != 'build':
        require(binding is not None and set(binding) == {'binary', 'build_receipt'}, 'bound new build required')
        argv += ['--binary-sha', binding['binary']['sha256'],
                 '--binary-bytes', str(binding['binary']['bytes']),
                 '--build-receipt-sha', binding['build_receipt']['sha256'],
                 '--build-receipt-bytes', str(binding['build_receipt']['bytes'])]
    else:
        require(binding is None, 'a fresh build passes no artifact/receipt pins')
    return argv


def unchanged_local_package(payload, control, control_pin, packaged_closure, closure_pin):
    """v7 final local check: the same bounded controller and every payload byte."""
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
                'actual new build artifact/receipt record')
    else:
        require(binding is not None and remote['binary'] == binding['binary']
                and remote['build_binding']['build_receipt'] == binding['build_receipt']
                and remote.get('held_data_files') == 70
                and remote.get('held_natural_reference_files') == 5
                and remote.get('qwen_only') is True and remote.get('gemma_executed') is False
                and remote.get('natural_authority_status') == 'ACCEPTED_ORIGINAL_CPU_CAPTURE'
                and remote.get('natural_feedback_qualified') is False
                and remote.get('teacher_forcing_accepted') is False
                and remote['data_after'] == 'UNCHANGED_HELD_FULL_SHA'
                and remote.get('natural_after') == 'UNCHANGED_HELD_FULL_SHA'
                and remote['semantic']['mode'] == mode
                and remote['semantic'].get(
                    'trained_native_natural_serving_or_performance_qualified') is False,
                'bound new natural build, exactly 70+5 held files, Q-only natural semantics only')
    return remote


_LIVE = {}


def _restore(state):
    signal.setitimer(signal.ITIMER_REAL, 0)
    for sig, previous in state.get('previous', {}).items():
        try:
            signal.signal(sig, previous)
        except BaseException:
            pass


def _emergency_close():
    """Outer safety net: close/reap owned objects and restore on every exit."""
    state = _LIVE.get('state')
    child = _LIVE.get('child')
    log = _LIVE.get('log')
    owner = _LIVE.get('owner')
    receipt = _LIVE.get('receipt') or {'errors': []}
    try:
        if state is not None:
            state['critical'] = True
            with Hold(state):
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
        if state is not None:
            _restore(state)


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
            'fixed owned source20 remote namespaces')
    args.output.mkdir(mode=0o700)
    child = log = None
    receipt_written = False
    receipt = {'mode': args.mode, 'status': 'UNQUALIFIED', 'errors': [],
               'remote_base': str(args.remote_base), 'remote_output': str(args.remote_output)}
    state = {'critical': False, 'depth': 0, 'pending': [], 'previous': {}}
    _LIVE['state'] = state
    _LIVE['receipt'] = receipt
    def stop(sig, _frame):
        if state['critical']:
            state['pending'].append(sig)
            return
        message = 'local SSH owner signal '+str(sig)
        if sig == signal.SIGALRM:
            raise TimeoutError('local owner deadline '+str(sig))
        raise InterruptedError(message)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        state['previous'][sig] = signal.signal(sig, stop)
    started = time.monotonic()
    owner = control = control_pin = packaged_closure = closure_pin = binding = None
    # Prehash + original SSH/remote owner. Final hash/EOF have a separate finite
    # allowance; only original bounded group cleanup runs with alarm disarmed.
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
            binding, binding_pin = parsed(args.binding)
            validate_binding(binding, closure_pin, args.remote_base)
            binding = {'binary': binding['binary'], 'build_receipt': binding['build_receipt']}
            receipt['binding'] = binding_pin
        remote = remote_argv(args.mode, args.remote_base, args.remote_output,
                             control_pin['sha256'], binding)
        argv = ['ssh', '-o', 'HostName=100.100.1.1', '-o', 'BindInterface=tailscale0',
                '-o', 'PermitLocalCommand=no', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
                '-o', 'ServerAliveCountMax=3', 'pvl-x2', shlex.join(remote)]
        receipt['argv'] = argv
        with Ownership(state):
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            _LIVE['child'] = child
            receipt['pid'] = child.pid
            log = owner.Log(child.stdout, args.output/'execution.log')
            _LIVE['log'] = log
        guarded = SigtermCancellation()
        guarded.__enter__()
        try:
            check_pending(state)
            receipt['observed_exit'] = guarded.observe(log, child, LIMITS[args.mode]+900)
            if guarded.received:
                raise guarded.exception()
        finally:
            guarded.__exit__(None, None, None)
    except BaseException as error:
        receipt['errors'].append(repr(error))
    finally:
        # Original bounded cleanup: record managed signals, close every owned
        # object independently, then report any pending signal after cleanup.
        state['critical'] = True
        with Hold(state):
            if child is not None:
                # Same-child cleanup attempt precedes the original call; never replayed.
                _LIVE['cleanup_attempted'] = True
                try:
                    owner.cleanup_group(child, receipt)
                    _LIVE['reaped'] = True
                except BaseException as error:
                    receipt['errors'].append('local owner: '+repr(error))
            if log is not None:
                try:
                    log.finish()
                except BaseException as error:
                    receipt['errors'].append('local EOF: '+repr(error))
                receipt['log'] = {'total': log.total, 'retained': log.retained, 'eof': log.eof, 'closed': log.closed}
            elif child is not None and child.stdout is not None:
                try:
                    child.stdout.close()
                except BaseException as error:
                    receipt['errors'].append('local stdout: '+repr(error))
        try:
            check_pending(state)
        except BaseException as error:
            receipt['errors'].append('deadline: '+repr(error))
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
        receipt['seconds'] = time.monotonic()-started
        receipt['status'] = 'CLOSED_PASS_SCOPED' if not receipt['errors'] and receipt.get('observed_exit') == 0 else 'FAILED_CLOSED'
        try:
            signal.setitimer(signal.ITIMER_REAL, 5)
            (args.output/'invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
            receipt_written = True
        except BaseException as error:
            receipt['errors'].append('local receipt: '+repr(error))
            receipt['status'] = 'FAILED_CLOSED'
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, previous in state['previous'].items():
                try:
                    signal.signal(sig, previous)
                except BaseException:
                    pass
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('argv', 'remote_invocation')}, indent=2), flush=True)
    return 0 if receipt_written and receipt['status'] == 'CLOSED_PASS_SCOPED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
