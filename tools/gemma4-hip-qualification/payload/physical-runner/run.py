#!/usr/bin/env python3
"""One bounded resume-all-five-gemma4 two-artifact build or one of eight Gemma PLE modes."""
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import shutil
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from pins import Held, alarm_until, check_closure, parsed, pin, require
from joint_data import JointDataLease, Q_CACHE, G_CACHE, q, g, enclosed, enclosed_build
from joint_data import HELD_BINARY
from modes import check_source
from fixture_semantic import (MODES, MATRIX_MODES, validate_fixture, expected_cells,
                               strict_report, check_seams)

HIP = '/nix/store/q4xzdr7070dwbcfh07f7d85zr3m9kpkr-clr-7.2.4'
CARGO = '/nix/store/4skfxmk429bn3cb81q1nv0jm36idnijv-cargo-1.95.0/bin'
RUST = '/nix/store/khz08fcmpq1rp5lhpkzvqv8icnfykv79-rustc-1.95.0/bin'
CLANG = '/nix/store/7jq50q3v4nim9dc24sl2qpdk9pfh6jqj-hipClang/bin'
CC = '/nix/store/79mr0jw3qccq7hhf1hh62knxd88dwazc-gcc-wrapper-15.3.0/bin'
PKG = '/nix/store/pc08q4ikl78wsnr8xqb5hgyw2fxz2v2v-pkg-config-wrapper-0.29.2/bin'
PYTHON = '/nix/store/jy00n3jp5xj8s40n1fbldmkcr5n7avxc-python3-3.13.15/bin/python3'
LIBC = '/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib'
LOADER = LIBC+'/ld-linux-x86-64.so.2'
LIBRARY_PATH = ':'.join((LIBC, '/nix/store/vjdqsn0bz2llabfqa746zfgk4kmswba5-vulkan-loader-1.4.357.0/lib', HIP+'/lib'))
OWNER_SHA = '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'
GIB = 1024**3
MIB = 1024**2
BASE = 'resume-all-five-gemma4-20261025-1'
ARTIFACTS = {
    'gemma4_hip_matrix': 'release/examples/gemma4_hip_matrix',
    'gemma4_hip_final_owner': 'release/examples/gemma4_hip_final_owner',
}
MODE_ARTIFACT = {
    'resident': 'gemma4_hip_matrix',
    'resident-row-unknown': 'gemma4_hip_matrix',
    'unknown': 'gemma4_hip_matrix',
    'resident-unknown': 'gemma4_hip_matrix',
    'session-known': 'gemma4_hip_final_owner',
    'session-release-fault': 'gemma4_hip_final_owner',
    'snapshot-known': 'gemma4_hip_final_owner',
    'snapshot-release-fault': 'gemma4_hip_final_owner',
}
LIMITS = {
    'build': (3600,3600,32*GIB),
    'resident': (7200,7200,64*GIB),
    'resident-row-unknown': (1800,1800,64*GIB),
    'unknown': (1800,1800,64*GIB),
    'resident-unknown': (1800,1800,64*GIB),
    'session-known': (7200,7200,64*GIB),
    'session-release-fault': (1800,1800,64*GIB),
    'snapshot-known': (7200,7200,64*GIB),
    'snapshot-release-fault': (1800,1800,64*GIB),
}
assert set(MODE_ARTIFACT) == set(MODES)
# Mode-exact actual fixture calls: the final-owner warmup plus committed hit and
# the cached-row warmup plus armed refusal are two calls each, a fresh armed
# refusal is one, and the seven-cell resident schedule is source-cell derived.
PLANNED_CALLS = {
    'resident': None,
    'resident-row-unknown': 2,
    'unknown': 1,
    'resident-unknown': 1,
    'session-known': 2,
    'session-release-fault': 2,
    'snapshot-known': 2,
    'snapshot-release-fault': 2,
}
assert set(PLANNED_CALLS) == set(MODES)
CARGO_ARGV = [CARGO+'/cargo', 'build', '-p', 'zerogravity',
              '--example', 'gemma4_hip_matrix', '--example', 'gemma4_hip_final_owner',
              '--release', '--features', 'gemma4-hip-qualification,zerogravity-gpu/hardware-abort-probe',
              '--offline', '--locked', '--message-format=json']
BUILD_RECEIPT = 'build-receipt.json'
BINARY_RECORD = 'model-binaries.json'


def cargo_argv_sha256():
    return hashlib.sha256(json.dumps(CARGO_ARGV,separators=(',',':')).encode()).hexdigest()


class Sampler:
    """Read-only observations; no peak/admission inference from free memory."""
    def __init__(self):
        self.stop = threading.Event()
        self.rows = []
        self.errors = []
        self.thread = None

    def one(self, group):
        rss = hwm = members = 0
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry/'stat').read_text().rsplit(') ', 1)[1].split()
                if int(fields[2]) != group:
                    continue
                members += 1
                for line in (entry/'status').read_text().splitlines():
                    if line.startswith('VmRSS:'):
                        rss += int(line.split()[1])*1024
                    elif line.startswith('VmHWM:'):
                        hwm += int(line.split()[1])*1024
            except (OSError, ValueError, IndexError):
                pass
        try:
            used = int(Path('/sys/bus/pci/devices/0000:c6:00.0/mem_info_vram_used').read_text())
        except (OSError, ValueError):
            used = None
        self.rows.append({'t': time.monotonic(), 'global_vram': used,
                          'owned_group_rss': rss, 'sum_member_hwm': hwm, 'members': members,
                          'phase': 'before_launch' if group == -1 else 'original_group_unreaped'})

    def begin(self, group):
        self.one(group)
        def run():
            try:
                while not self.stop.wait(.02):
                    self.one(group)
                    require(len(self.rows) <= 370000, 'finite sampler record bound')
            except BaseException as error:
                self.errors.append(repr(error))
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def finish_before_reap(self, group):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
            require(not self.thread.is_alive(), 'sampler joined before original leader reap')
        self.one(group)
        require(not self.errors, 'sampler failed: '+repr(self.errors))

    def report(self, output):
        require(self.rows, 'actual sampling required')
        (output/'samples.json').write_text(json.dumps(self.rows, separators=(',', ':'))+'\n')
        used = [v['global_vram'] for v in self.rows if v['global_vram'] is not None]
        gaps = [b['t']-a['t'] for a, b in zip(self.rows, self.rows[1:])]
        return {'count': len(self.rows), 'readable_vram_samples': len(used),
                'period_requested_ms': 20, 'first_t': self.rows[0]['t'], 'last_t': self.rows[-1]['t'],
                'max_gap_seconds': max(gaps, default=0), 'global_vram_first': used[0] if used else None,
                'global_vram_last': used[-1] if used else None, 'global_vram_max': max(used, default=0),
                'global_vram_sampled_delta': max(used)-used[0] if used else None,
                'owned_group_rss_max': max(v['owned_group_rss'] for v in self.rows),
                'sum_member_hwm_max': max(v['sum_member_hwm'] for v in self.rows),
                'background_shared': True, 'isolated_peak': False, 'worst_case_bound': False,
                'sum_hwm_is_simultaneous_peak': False, 'joined_before_leader_reap': True,
                'last_sample_is_after_retirement': False,
                'post_retirement_is_separate_observation': True}


def semantic(mode, output, base):
    log = (output/'execution.log').read_text()
    raw = (output/'model-report.json').read_bytes()
    result, result_pin = parsed(output/'model-report.json')
    require(len(raw) == result_pin['bytes']
            and hashlib.sha256(raw).hexdigest() == result_pin['sha256'],
            'one bounded authentic fixture report extent')
    result = strict_report(raw)
    check_seams(base/'source/crates/zerogravity/examples')
    expected = None
    if mode in MATRIX_MODES:
        expected = expected_cells(g.DATA_ROOT/'expected/invocation.json')
    return {'report':result_pin, **validate_fixture(mode, result, log, expected)}


def after_retirement():
    device = Path('/sys/bus/pci/devices/0000:c6:00.0')
    used = int((device/'mem_info_vram_used').read_text())
    nodes = ['/dev/kfd'] + ['/dev/dri/'+p.name for p in (device/'drm').glob('renderD*')]
    holders = []
    scanned = 0
    started = time.monotonic()
    truncated = False
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        if scanned >= 4096 or time.monotonic()-started > 2:
            truncated = True
            break
        scanned += 1
        try:
            before = (entry/'stat').read_text().rsplit(') ', 1)[1].split()[19]
            fds = []
            for index, fd in enumerate((entry/'fd').iterdir()):
                if index >= 1024:
                    truncated = True
                    break
                if time.monotonic()-started > 2:
                    truncated = True
                    break
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target in nodes:
                    fds.append({'fd': fd.name, 'device': target})
            if not fds:
                continue
            rss = None
            for line in (entry/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    rss = int(line.split()[1])*1024
            after = (entry/'stat').read_text().rsplit(') ', 1)[1].split()[19]
            if before == after:
                holders.append({'pid': int(entry.name), 'starttime_ticks': before,
                                'rss': rss, 'device_fds': fds,
                                'ownership': 'unattributed current visible process'})
        except (OSError, ValueError, IndexError):
            continue
    return {'t': time.monotonic(), 'phase': 'after_original_group_absent_and_leader_reaped',
            'pci': '0000:c6:00.0', 'global_vram': used, 'device_nodes': nodes,
            'visible_current_device_holders': holders, 'scanned_pids': scanned,
            'scan_truncated': truncated, 'permission_or_race_omissions_possible': True,
            'original_group_owned_rss': 0,
            'rss_zero_reason': 'original group absence, no recycled-PID lookup',
            'shared_background': True, 'isolated_restoration_or_context_bytes_proven': False}


def bind_build(base, closure_pin, hydrated_closure_pin, artifact_name, binding, receipt_pin, held):
    require(receipt_pin is not None and binding is not None, 'actual new resume-all-five-gemma4 build binding required')
    require(artifact_name in ARTIFACTS, 'exact selected artifact name')
    build_receipt, build_receipt_pin = parsed(base/BUILD_RECEIPT)
    require(build_receipt_pin == receipt_pin, 'supplied new build receipt pin differs')
    require(build_receipt.get('schema') == 'zg.gemma-ple-source21-build-receipt.v1'
            and build_receipt['mode'] == 'build' and build_receipt['status'] == 'CLOSED_PASS_SCOPED'
            and build_receipt['errors'] == [] and build_receipt['observed_exit'] == 0
            and build_receipt['source_after'] == 'UNCHANGED'
            and build_receipt['source_closure'] == closure_pin
            and build_receipt['hydrated_closure'] == hydrated_closure_pin
            and build_receipt['artifact_count'] == 2
            and set(build_receipt['artifacts']) == set(ARTIFACTS)
            and build_receipt['cargo_argv_sha256'] == cargo_argv_sha256()
            and build_receipt['cwd'] == str(base/'source'),
            'actual CLOSED new resume-all-five-gemma4 two-artifact build receipt')
    record, record_pin = parsed(base/BINARY_RECORD)
    require(record == {'artifacts': build_receipt['artifacts'],
                       'source_closure': closure_pin,
                       'hydrated_closure': hydrated_closure_pin},
            'new build artifact record')
    proof = build_receipt['artifacts'][artifact_name]['proof']
    require(proof == binding, 'selected artifact proof differs from reviewed binding')
    path = base/'target'/ARTIFACTS[artifact_name]
    require(build_receipt['artifacts'][artifact_name]['path'] == str(path),
            'selected artifact path exact')
    binary = Held(path, binding, limit=2*GIB)
    held.append(binary)
    return binary, {'build_receipt': build_receipt_pin, 'binary_record': record_pin,
                    'artifact_name': artifact_name, 'binary': binding,
                    'cargo_argv_sha256': cargo_argv_sha256(),
                    'hydrated_closure': hydrated_closure_pin}


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
        for signum in self.SIGNALS:
            self.previous[signum] = signal.signal(signum, self._handle)
        return self

    def restore(self):
        # Unconditional timer cancel and handler restore, even after a failed write.
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)

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
    """Outer safety net: close/reap every owned object and restore handlers."""
    guard = _LIVE.get('guard')
    child = _LIVE.get('child')
    log = _LIVE.get('log')
    owner = _LIVE.get('owner')
    sampler = _LIVE.get('sampler')
    data_lease = _LIVE.get('data_lease')
    binary = _LIVE.get('binary')
    held = list(_LIVE.get('held_data', [])) + list(_LIVE.get('held_cache', []))         + list(_LIVE.get('held_binary', []))
    if binary is not None:
        held.append(binary)
    receipt = _LIVE.get('receipt') or {'errors': []}
    try:
        if guard is not None:
            with guard:
                if sampler is not None:
                    try:
                        sampler.finish_before_reap(child.pid if child is not None else -1)
                    except BaseException:
                        pass
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
                    if item is not None:
                        try:
                            item.close()
                        except BaseException:
                            pass
                if data_lease is not None:
                    try:
                        data_lease.close()
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
    parser.add_argument('--mode', choices=tuple(LIMITS), required=True)
    parser.add_argument('--control-sha', required=True)
    parser.add_argument('--artifact', choices=tuple(ARTIFACTS))
    parser.add_argument('--binary-sha')
    parser.add_argument('--binary-bytes', type=int)
    parser.add_argument('--build-receipt-sha')
    parser.add_argument('--build-receipt-bytes', type=int)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (512*MIB, 64*GIB))
    base, output = args.base, args.output
    require(base.is_absolute() and base.parent == Path('/home/pvl/.cache/abird/zerogravity/qualification')
            and base.name == BASE and output.parent == Path('/home/pvl/.local/state/abird/zerogravity')/BASE
            and not output.exists(), 'fresh exact owned Gemma PLE namespace/output')
    require((args.binary_sha is None) == (args.binary_bytes is None)
            and (args.build_receipt_sha is None) == (args.build_receipt_bytes is None)
            and (args.binary_sha is None) == (args.build_receipt_sha is None),
            'build artifact/receipt binding is both bytes and SHA or absent')
    if args.mode == 'build':
        require(args.artifact is None and args.binary_sha is None,
                'a fresh build passes no artifact/receipt pins')
    else:
        require(args.artifact == MODE_ARTIFACT[args.mode], 'exact mode to artifact mapping')
    binding = None if args.binary_sha is None else {'bytes': args.binary_bytes, 'sha256': args.binary_sha}
    binding_receipt = None if args.build_receipt_sha is None else {'bytes': args.build_receipt_bytes,
                                                                   'sha256': args.build_receipt_sha}
    output.mkdir(parents=True, mode=0o700)
    source = base/'source'
    receipt = {'mode': args.mode, 'status': 'UNQUALIFIED', 'errors': [],
               'scope': 'actual resume-all-five-gemma4 Gemma E2B resident PLE healthy/fresh/cached-row/final-owner modes',
               'bootstrap': 'Client precedes original context partition; any setup failure retires process',
               'model_close_is_global_zero': False, 'policy_is_provisional': True,
               'observations': 'shared VRAM/background and sampled RSS; unsampled peaks remain'}
    child = log = binary = sampler = None
    held_data = []
    held_cache = []
    held_binary = []
    data_lease = None
    started = time.monotonic()
    closure = packaged_closure = closure_pin = packaged_closure_pin = data = data_pin = None
    aborted = False
    def mark_aborted():
        nonlocal aborted
        aborted = True
    guard = ManagedSignals(receipt, 'owned supervisor', mark_aborted).install()
    _LIVE['guard'] = guard
    _LIVE['receipt'] = receipt
    work_end = started + LIMITS[args.mode][0] + 600
    alarm_until(work_end)
    owner = None
    try:
        closure, closure_pin = parsed(base/'closure.json')
        check_closure(base, closure)
        packaged_closure, packaged_closure_pin = parsed(base/'source-closure.json')
        check_closure(base, packaged_closure)
        control, control_pin = parsed(base/'control.json')
        require(control_pin['sha256'] == args.control_sha, 'exact separately approved controller')
        check_closure(base, control)
        check_source(base)
        receipt.update(control=control_pin)
        require(shutil.disk_usage(base).free >= 24*GIB, 'same24GiB build/output free-disk bound')
        helper = source/'tools/qwen27b-full-window-reference/owned_container_v4.py'
        require(pin(helper)['sha256'] == OWNER_SHA, 'unchanged original group/EOF owner')
        spec = importlib.util.spec_from_file_location('original_owned', helper)
        owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner)
        _LIVE['owner'] = owner
        env = os.environ.copy()
        env.update(HIP_PATH=HIP, ROCM_PATH=HIP, CARGO_HOME=str(base/'cargo'),
                   CARGO_TARGET_DIR=str(base/'target'), CARGO_NET_OFFLINE='true',
                   CARGO_BUILD_JOBS='2', RAYON_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
                   TMPDIR=str(base/'tmp'), XDG_DATA_HOME=str(base/'xdg-data'),
                   XDG_STATE_HOME=str(base/'xdg-state'), XDG_CONFIG_HOME=str(base/'xdg-config'),
                   MESA_SHADER_CACHE_DIR=str(base/'mesa'), AMD_COMGR_CACHE_DIR=str(base/'comgr'),
                   LD_PRELOAD='', LD_AUDIT='',
                   PATH=':'.join((HIP+'/bin', CLANG, CARGO, RUST, CC, PKG, '/run/current-system/sw/bin')))
        (base/'tmp').mkdir(exist_ok=True)
        cpus = sorted(os.sched_getaffinity(0))[:4]
        require(len(cpus) == 4, 'four actual allowed CPU threads')
        os.sched_setaffinity(0, cpus)
        deadline, cpu, space = LIMITS[args.mode]
        if args.mode == 'build':
            env.update(XDG_CACHE_HOME=str(base/'xdg-cache'))
            command = CARGO_ARGV
        else:
            data_lease = JointDataLease(base)
            _LIVE['data_lease'] = data_lease
            held_data = data_lease.held
            held_cache = data_lease.cache
            _LIVE['held_data'] = held_data
            _LIVE['held_cache'] = held_cache
            data_pin = {'qwen':q.DATA_PROOF,'gemma':g.DATA_PROOF}
            data = {'records':data_lease.records}
            receipt.update(read_only_original_data_roots=[str(q.DATA_ROOT),str(g.DATA_ROOT)],
                           original_data_authorities=data_pin, network_payload_transfer=False,
                           original_read_only_FD_bindings=True,
                           cache_writable_subtrees=[str(Q_CACHE/'locks'),str(G_CACHE/'locks')],
                           cache_reuse_read_only=False, genuine_original_imports_revalidated=True)
            require(len(held_data) == 70 and len({v.before[:2] for v in held_data}) == 70,
                    'original seventy distinct data/cache FDs')
            _LIVE['held_binary'] = held_binary
            binary, binding_pins = bind_build(base, packaged_closure_pin, closure_pin,
                                              args.artifact, binding, binding_receipt, held_binary)
            _LIVE['binary'] = binary
            receipt.update(binary=binding_pins['binary'], binary_record=binding_pins['binary_record'],
                           build_binding={'build_receipt': binding_pins['build_receipt'],
                                          'cargo_argv_sha256': binding_pins['cargo_argv_sha256']},
                           artifact_name=args.artifact,
                           binary_held_fd=binary.file.fileno(),
                           data_closure=data_pin, held_data_files=len(held_data),
                           loader={'path': LOADER, **pin(Path(LOADER).resolve(strict=True))},
                           library_path=LIBRARY_PATH)
            executable = HELD_BINARY
            command = [LOADER, '--library-path', LIBRARY_PATH, executable]
            if args.mode in MATRIX_MODES:
                command += [str(g.DATA_ROOT/'expected/invocation.json'),
                            str(output/'model-report.json'), args.mode]
            else:
                command += [args.mode, str(output/'model-report.json')]
        if args.mode == 'build':
            isolated = [PYTHON, '-B', str(source/'scripts/native-gpu.py'), '--',
                        *enclosed_build(base, source, command)]
        else:
            # The canonical Gemma cache is the child's XDG root; only its two lock
            # subtrees are writable, and all data FDs remain read-only.
            env['XDG_CACHE_HOME'] = str(G_CACHE.parents[2])
            isolated = [PYTHON, '-B', str(source/'scripts/native-gpu.py'), '--',
                        *enclosed(base, source, binary, data_lease, command, True, output)]
        argv = ['/run/current-system/sw/bin/prlimit', '--as='+str(space), '--cpu='+str(cpu),
                '--nofile=1024', '--', *isolated]
        provider = (16 if args.mode == 'unknown' else 20)*GIB
        receipt.update(argv=argv, cwd=str(source), actual_cpu_affinity=cpus,
                       deadline_seconds=deadline, per_process_cpu_limit=cpu,
                       address_space_bytes=space, whole_supervisor_alarm_seconds=deadline+600,
                       finalization_seconds=900, final_receipt_seconds=5,
                       external_supervisor_AS_soft=512*MIB,
                       cpu_limit_is_aggregate=False, source_closure=packaged_closure_pin,
                       hydrated_closure=closure_pin, global_host_budget=16*GIB,
                       provider=provider, physical_domain=64*GIB,
                       persistent_device=4*GIB, persistent_host=2*GIB,
                       compiler_host_per_generation=8*GIB, compiler_max_admitted_generations=3,
                       module_opaque_device=8*GIB, known_module_limit=GIB,
                       retained_module_host_limit=2*GIB, finite_catalogue_limit=1024,
                       diagnostic_host_reserved=32*MIB)
        if args.mode in MODES:
            receipt['planned_work'] = {
                'mode':args.mode,
                'gemma_calls': PLANNED_CALLS[args.mode],
                'qwen_calls':0,'genuine_original_producers':True,'reference_operands':False}
        (output/'prepared-invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if args.mode in MODES:
            sampler = Sampler()
            _LIVE['sampler'] = sampler
            sampler.one(-1)
        with guard:
            child = subprocess.Popen(argv, cwd=source, env=env,
                                     stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True,
                                     pass_fds=() if binary is None else
                                     (binary.file.fileno(),) + tuple(fd for fd,_ in data_lease.bindings()))
            _LIVE['child'] = child
            receipt['pid'] = child.pid
            if sampler is not None:
                sampler.begin(child.pid)
            log = owner.Log(child.stdout, output/'execution.log')
            _LIVE['log'] = log
        guard.rethrow()
        receipt['observed_exit'] = log.observe(child, deadline)
    except BaseException as error:
        receipt['errors'].append(repr(error))
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                held_objects = held_data + held_cache + held_binary
                # Mandatory original closes: sampler join, group TERM/two-KILL/reap/absence,
                # raw stdout fallback and bounded EOF. Record-only managed signals.
                with guard:
                    if sampler is not None:
                        try:
                            sampler.finish_before_reap(child.pid if child is not None else -1)
                        except BaseException as error:
                            receipt['errors'].append('sampling: '+repr(error))
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
                    if child is not None and sampler is not None:
                        try:
                            cleanup = receipt.get('foreground_cleanup', {})
                            require(cleanup.get('leader_reaped') and cleanup.get('group_absent'),
                                    'post-retirement observation requires original owner closure')
                            receipt['post_retirement_observation'] = after_retirement()
                        except BaseException as error:
                            receipt['errors'].append('post retirement: '+repr(error))
                    if log is not None:
                        try:
                            log.finish()
                        except BaseException as error:
                            receipt['errors'].append('EOF: '+repr(error))
                        receipt['log'] = {'retained': log.retained, 'total': log.total,
                                          'eof': log.eof, 'closed': log.closed}
                guard.note()
                final_end = min(time.monotonic(), work_end) + 900
                full_after = True
                try:
                    for item in held_objects:
                        try:
                            if aborted:
                                full_after = False
                                item.identity()
                                receipt['aborted_full_SHA_after_skipped'] = True
                            else:
                                alarm_until(final_end)
                                item.after()
                        except BaseException as error:
                            full_after = False
                            receipt['errors'].append('held: '+repr(error))
                    if data_lease is not None:
                        try:
                            if not aborted:
                                alarm_until(final_end)
                                data_lease.authority_after()
                        except BaseException as error:
                            full_after = False
                            receipt['errors'].append('original data authority: '+repr(error))
                except BaseException as error:
                    # A signal in a gap between hashes still reaches every close.
                    full_after = False
                    receipt['errors'].append('held: '+repr(error))
                finally:
                    # Outer finally: every original held data/cache/binary FD and the
                    # joint lease close independently, even after a hash/authority failure.
                    with guard:
                        for item in held_objects:
                            try:
                                item.close()
                            except BaseException as error:
                                receipt['errors'].append('held close: '+repr(error))
                        if data_lease is not None:
                            try:
                                data_lease.close()
                            except BaseException as error:
                                receipt['errors'].append('lease close: '+repr(error))
                    guard.note()
                try:
                    if closure is not None and packaged_closure is not None and control is not None and not aborted:
                        alarm_until(final_end)
                        require(pin(base/'source-closure.json') == packaged_closure_pin,
                                'packaged source closure authority changed')
                        require(pin(base/'closure.json') == closure_pin,
                                'hydrated source closure authority changed')
                        require(pin(base/'control.json') == control_pin, 'controller authority changed')
                        check_closure(base, packaged_closure)
                        check_closure(base, control)
                        check_closure(base, closure)
                        check_source(base)
                        receipt['source_after'] = 'UNCHANGED'
                        receipt['control_after'] = control_pin
                except BaseException as error:
                    receipt['errors'].append('verification: '+repr(error))
                try:
                    if data is not None and not aborted and full_after:
                        alarm_until(final_end)
                        require(pin(base/'qwen/data-closure.json') == q.DATA_PROOF
                                and pin(base/'gemma/data-closure.json') == g.DATA_PROOF,
                                'both data authorities unchanged')
                        receipt['data_after'] = 'UNCHANGED_HELD_FULL_SHA'
                        if held_cache:
                            receipt['cache_manifest_and_import_parent_after'] = 'UNCHANGED_HELD_FULL_SHA'
                            receipt['reuse_authentication'] = {
                                'source_closure':packaged_closure_pin,'original_data':data_pin,
                                'same_original_cache_objects':True,'transfer':False,
                                'import':False,'cache_lock_writes':True}
                except BaseException as error:
                    receipt['errors'].append('verification: '+repr(error))
                try:
                    if sampler is not None:
                        receipt['sample_observations'] = sampler.report(output)
                        require(receipt['sample_observations']['readable_vram_samples'] > 1,
                                'actual readable VRAM coverage')
                except BaseException as error:
                    receipt['errors'].append('verification: '+repr(error))
                try:
                    if child is not None and receipt.get('observed_exit') == 0 and not receipt['errors'] \
                            and receipt.get('source_after') == 'UNCHANGED':
                        if args.mode == 'build':
                            artifacts = {}
                            for line in (output/'execution.log').read_text().splitlines():
                                try:
                                    event = json.loads(line)
                                except ValueError:
                                    continue
                                name = event.get('target', {}).get('name')
                                if (event.get('reason') == 'compiler-artifact'
                                        and name in ARTIFACTS
                                        and event.get('target', {}).get('kind') == ['example']
                                        and event.get('executable')):
                                    path = Path(event['executable'])
                                    require(path.is_relative_to(base/'target')
                                            and path == base/'target'/ARTIFACTS[name],
                                            'declared two-artifact path: '+str(name))
                                    require(name not in artifacts, 'no duplicate build artifact: '+str(name))
                                    artifacts[name] = {'path':str(path),'proof':pin(path)}
                            require(set(artifacts) == set(ARTIFACTS) and len(artifacts) == 2,
                                    'exactly two actual reachable examples')
                            with (base/BINARY_RECORD).open('x') as file:
                                json.dump({'artifacts': artifacts,
                                           'source_closure': packaged_closure_pin,
                                           'hydrated_closure': closure_pin}, file, indent=2)
                                file.write('\n')
                            with (base/BUILD_RECEIPT).open('x') as file:
                                json.dump({'schema': 'zg.gemma-ple-source21-build-receipt.v1',
                                           'mode': 'build', 'status': 'CLOSED_PASS_SCOPED', 'errors': [],
                                           'observed_exit': receipt['observed_exit'],
                                           'source_after': receipt['source_after'],
                                           'source_closure': packaged_closure_pin,
                                           'hydrated_closure': closure_pin,
                                           'artifact_count': 2, 'artifacts': artifacts,
                                           'cargo_argv_sha256': cargo_argv_sha256(), 'cwd': str(source),
                                           'address_space_bytes': space, 'deadline_seconds': deadline},
                                          file, indent=2)
                                file.write('\n')
                            receipt['build_artifacts'] = artifacts
                            receipt['build_artifact_count'] = len(artifacts)
                            receipt['build_receipt'] = pin(base/BUILD_RECEIPT)
                        else:
                            receipt['semantic'] = semantic(args.mode, output, base)
                except BaseException as error:
                    receipt['errors'].append('verification: '+repr(error))
            except BaseException as error:
                # A signal in a gap between phases never skips the failure receipt.
                receipt['errors'].append('cleanup: '+repr(error))
            receipt['seconds'] = time.monotonic()-started
            receipt['status'] = ('CLOSED_PASS_SCOPED'
                                 if not receipt['errors'] and receipt.get('observed_exit') == 0
                                 else 'FAILED_CLOSED')
            # Bounded receipt write on its own deadline; a failed or retried write
            # can never upgrade the recorded failure to a pass.
            try:
                alarm_until(time.monotonic()+5)
                (output/'invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
            except BaseException as error:
                receipt['errors'].append('receipt write: '+repr(error))
                receipt['status'] = 'FAILED_CLOSED'
                try:
                    alarm_until(time.monotonic()+5)
                    (output/'invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
                except BaseException as error:
                    receipt['errors'].append('receipt rewrite: '+repr(error))
        finally:
            guard.restore()
    print(json.dumps({k: v for k, v in receipt.items() if k != 'argv'}, indent=2), flush=True)
    return 0 if receipt['status'] == 'CLOSED_PASS_SCOPED' else 1


if __name__ == '__main__':
    sys.exit(main())
