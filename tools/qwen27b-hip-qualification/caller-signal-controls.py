#!/usr/bin/env python3
"""CPU-only controls for the run_reference caller signal lifecycle.

Imports the actual frozen caller and replaces only external I/O boundaries
(Podman image inspect, Popen argv, source pin verification) with CPU children.
No Podman, image, model, GPU or network is touched, and no capture is written.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

OWNER_SHA = '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'
MODE_SIGNALS = {'TERM': signal.SIGTERM, 'INT': signal.SIGINT,
                'HUP': signal.SIGHUP, 'ALRM': signal.SIGALRM}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invocation(out, why):
    data = json.loads((out/'invocation.json').read_text())
    assert data.get('status') == 'FAIL_UNQUALIFIED', 'rejected signal status '+str(why)
    return data


def run_case(caller, owner, tool, signal_name, stage, root):
    out = root/f'{signal_name}-{stage}'
    sig = MODE_SIGNALS[signal_name]
    original_popen = caller.subprocess.Popen

    class FakeHelper:
        PODMAN = ['cpu-control']
        LABEL = 'cpu-control'
        Log = owner.Log

        @staticmethod
        def command(_argv, _timeout):
            return type('Done', (), {
                'returncode': 0,
                'stdout': json.dumps([{
                    'Id': caller.contract.IMAGE.removeprefix('sha256:'),
                    'Digest': caller.contract.IMAGE_DIGEST}]).encode(),
                'stderr': b''})()

        @staticmethod
        def cleanup(child, _name, _nonce, _cidfile, receipt):
            if stage == 'cleanup':
                os.kill(os.getpid(), sig)
                time.sleep(0.02)
            owner.cleanup_group(child, receipt)
            return []

    def fake_popen(_argv, **kwargs):
        child = original_popen(
            [sys.executable, '-c',
             'import sys,time; sys.stdout.write("cpu-child\\n"); sys.stdout.flush(); time.sleep(8)'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        if stage == 'acquire':
            os.kill(os.getpid(), sig)
            time.sleep(0.02)
        elif stage == 'drain':
            timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), sig))
            timer.daemon = True
            timer.start()
        return child

    caller.metadata = lambda path: ({'source_directory': str(root/'source'),
                                     'scope': {'original_full_text_forward': True}}
                                    if path.name == 'inputs.json' else {})
    caller.specs = lambda *_args: None
    caller.package = lambda: {'package': 'cpu-control'}
    caller.output_path = lambda _plan, _out: out
    caller.source_open = lambda _plan, _pins, *_args: ({}, [])
    caller.source_after = lambda *_args: []
    caller.owned_helper = lambda: FakeHelper
    caller.subprocess.Popen = fake_popen
    old_argv = sys.argv
    sys.argv = ['run_reference.py', '--new-output', str(out)]
    try:
        caller.main()
    except BaseException:
        pass
    finally:
        sys.argv = old_argv
        caller.subprocess.Popen = original_popen
        signal.alarm(0)
    data = invocation(out, (signal_name, stage))
    cleanup = data.get('foreground_cleanup', {})
    assert cleanup.get('term_sent') is True, 'TERM sent'
    assert cleanup.get('kill_sent_before_reap') is True, 'KILL before reap'
    assert cleanup.get('leader_reaped') is True, 'leader reaped'
    assert cleanup.get('group_absent') is True, 'group absent'
    assert cleanup.get('kill_attempts_before_reap') == 2, 'two authenticated KILLs'
    assert cleanup.get('leader_reap_attempts') == 1, 'one reap'
    assert cleanup.get('stage_errors') == [], 'cleanup stages clean'
    log_cleanup = data.get('log_cleanup')
    assert isinstance(log_cleanup, dict), 'actual original Log cleanup recorded'
    assert log_cleanup.get('pipe_eof') is True, 'original stdout pipe EOF'
    assert log_cleanup.get('all_closed') is True, 'selector/log/pipe all closed'
    assert data.get('source_FDs_closed') is True, 'every original source FD closed'
    pid = data.get('owned_cli_pid')
    assert isinstance(pid, int) and pid > 1, 'owned pid recorded'
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError('owned group still present')
    return (signal_name, stage, data['status'], log_cleanup['pipe_eof'],
            log_cleanup['all_closed'], cleanup['exit_code'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.tool))
    caller = load('caller_under_control', args.tool/'run_reference.py')
    owner = load('owner_under_control', args.tool/'owned_container_v4.py')
    contract = load('contract_under_control', args.tool/'contract.py')
    caller.contract = contract
    assert hashlib.sha256((args.tool/'owned_container_v4.py').read_bytes()).hexdigest() == OWNER_SHA
    source = json.loads((args.tool/'source-pin.json').read_text())
    assert source.get('files') and len(source['files']) == 31, 'original source pin shape'
    results = []
    with tempfile.TemporaryDirectory(prefix='caller-signal-', dir=args.output) as scratch:
        root = Path(scratch)
        (root/'source').mkdir()
        for signal_name in ('TERM', 'INT', 'HUP', 'ALRM'):
            for stage in ('acquire', 'drain', 'cleanup'):
                results.append(run_case(caller, owner, args.tool,
                                        signal_name, stage, root))
    print('CALLER-SIGNAL-CONTROLS PASS cases=%d' % len(results))
    (args.output/'CALLER-SIGNAL-RESULT.json').write_text(json.dumps({
        'schema': 'zg.qwen27b-caller-signal-control.v1',
        'status': 'PASS_CPU_ONLY_CALLER_SIGNAL_CONTROL',
        'cases': results, 'GPU_or_model': False, 'podman': False,
        'network': False, 'capture_written': False}, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
