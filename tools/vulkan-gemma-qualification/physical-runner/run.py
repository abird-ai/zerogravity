#!/usr/bin/env python3
"""One bounded local Vulkan Gemma source/build/lifecycle process.

This is the fresh Vulkan derivative of the accepted source20 x2 supervisor.  It
does not contact a remote host: every child is local, one process per case, and
the original 70 held data descriptors and exact cache-lock roots are preserved.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
import resource
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runner'))
from sigterm_cancellation import SigtermCancellation

from pins import Held, alarm_until, check_closure, parsed, pin, relative, require
from joint_data import JointDataLease, Q_CACHE, G_CACHE, enclosed, enclosed_build

# Accepted original canonical-cache namespace. G_CACHE is
# <accepted-base>/xdg-cache/abird/zerogravity/models; only that XDG parent is
# reused, while Cargo inputs come from the fresh copied registry closure.
CANONICAL_XDG = G_CACHE.parents[2]
from modes import check_source
import gemma_data as g

PYTHON = '/nix/store/jy00n3jp5xj8s40n1fbldmkcr5n7avxc-python3-3.13.15/bin/python3'
HIP = '/nix/store/q4xzdr7070dwbcfh07f7d85zr3m9kpkr-clr-7.2.4'
CARGO = '/nix/store/4skfxmk429bn3cb81q1nv0jm36idnijv-cargo-1.95.0/bin'
RUST = '/nix/store/khz08fcmpq1rp5lhpkzvqv8icnfykv79-rustc-1.95.0/bin'
CLANG = '/nix/store/7jq50q3v4nim9dc24sl2qpdk9pfh6jqj-hipClang/bin'
CC = '/nix/store/79mr0jw3qccq7hhf1hh62knxd88dwazc-gcc-wrapper-15.3.0/bin'
PKG = '/nix/store/pc08q4ikl78wsnr8xqb5hgyw2fxz2v2v-pkg-config-wrapper-0.29.2/bin'
LIBC = '/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib'
VULKAN = '/nix/store/vjdqsn0bz2llabfqa746zfgk4kmswba5-vulkan-loader-1.4.357.0/lib'
LOADER = LIBC + '/ld-linux-x86-64.so.2'
LIBRARY_PATH = ':'.join((LIBC, VULKAN, HIP + '/lib'))
OWNER_SHA = '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'
GIB = 1024 ** 3
BASE = 'resume-all-five-vulkan-20261025-1'
MODES = ('resident', 'resident-unknown', 'resident-row-unknown',
         'session-known', 'session-release-fault',
         'snapshot-known', 'snapshot-release-fault', 'partial-native',
         'peer-vulkan-then-hip', 'peer-hip-then-vulkan')
PEER_MODES = {'peer-vulkan-then-hip': 'vulkan-then-hip',
              'peer-hip-then-vulkan': 'hip-then-vulkan'}
LIMITS = {'build': (3600, 3600, 32 * GIB)}
LIMITS.update({mode: (7200, 7200, 64 * GIB) for mode in ('resident', 'resident-unknown', 'resident-row-unknown')})
LIMITS.update({mode: (1800, 1800, 64 * GIB) for mode in MODES if mode not in LIMITS})
CARGO_ARGV = [CARGO + '/cargo', 'build', '-p', 'zerogravity',
              '--example', 'gemma4_vulkan_matrix',
              '--example', 'gemma4_vulkan_final_owner',
              '--example', 'gemma4_vulkan_same_pci_peer', '--release',
              '--features', 'gemma4-vulkan-peer-qualification,zerogravity-gpu/hardware-abort-probe',
              '--offline', '--locked', '--message-format=json']
ARTIFACTS = ('release/examples/gemma4_vulkan_matrix',
             'release/examples/gemma4_vulkan_final_owner',
             'release/examples/gemma4_vulkan_same_pci_peer')
BUILD_RECEIPT = 'build-receipt.json'
BINDING = 'BINDING.json'
REGISTRY_CLOSURE = 'registry-closure.json'
PORTABLE_BASE = Path('/home/pvl/.cache/abird/zerogravity/qualification/portable-service-20261023-1')
PORTABLE_SHA = '42ac6bf6f82e61adfe1cff7c565076361f222c51792f02df09337ff1779a791a'
REGISTRY_MEMBER_LIMIT = 64 * 1024**2
FINAL_MODES = ('session-known', 'session-release-fault',
               'snapshot-known', 'snapshot-release-fault', 'partial-native')


def cargo_argv_sha256():
    return hashlib.sha256(json.dumps(CARGO_ARGV, separators=(',', ':')).encode()).hexdigest()


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
                fields = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()
                if int(fields[2]) != group:
                    continue
                members += 1
                for line in (entry / 'status').read_text().splitlines():
                    if line.startswith('VmRSS:'):
                        rss += int(line.split()[1]) * 1024
                    elif line.startswith('VmHWM:'):
                        hwm += int(line.split()[1]) * 1024
            except (OSError, ValueError, IndexError):
                pass
        try:
            used = int(Path('/sys/bus/pci/devices/0000:c6:00.0/mem_info_vram_used').read_text())
        except (OSError, ValueError):
            used = None
        self.rows.append({'t': time.monotonic(), 'global_vram': used,
                          'owned_group_rss': rss, 'sum_member_hwm': hwm,
                          'members': members,
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
        require(not self.errors, 'sampler failed: ' + repr(self.errors))

    def report(self, output):
        require(self.rows, 'actual sampling required')
        (output / 'samples.json').write_text(json.dumps(self.rows, separators=(',', ':')) + '\n')
        used = [v['global_vram'] for v in self.rows if v['global_vram'] is not None]
        gaps = [b['t'] - a['t'] for a, b in zip(self.rows, self.rows[1:])]
        return {'count': len(self.rows), 'readable_vram_samples': len(used),
                'period_requested_ms': 20, 'first_t': self.rows[0]['t'],
                'last_t': self.rows[-1]['t'], 'max_gap_seconds': max(gaps, default=0),
                'global_vram_first': used[0] if used else None,
                'global_vram_last': used[-1] if used else None,
                'global_vram_max': max(used, default=0),
                'global_vram_sampled_delta': max(used) - used[0] if used else None,
                'owned_group_rss_max': max(v['owned_group_rss'] for v in self.rows),
                'sum_member_hwm_max': max(v['sum_member_hwm'] for v in self.rows),
                'background_shared': True, 'isolated_peak': False,
                'worst_case_bound': False, 'sum_hwm_is_simultaneous_peak': False,
                'joined_before_leader_reap': True,
                'last_sample_is_after_retirement': False,
                'post_retirement_is_separate_observation': True}


def owner_module(source):
    helper = source / 'tools/qwen27b-full-window-reference/owned_container_v4.py'
    require(pin(helper)['sha256'] == OWNER_SHA, 'unchanged original local group/EOF owner')
    spec = importlib.util.spec_from_file_location('original_owned', helper)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner


def _proof(proof):
    require(type(proof) is dict and set(proof) == {'bytes', 'sha256'}
            and type(proof['bytes']) is int and 0 < proof['bytes'] <= 2 * GIB
            and len(proof['sha256']) == 64 and proof['sha256'] == proof['sha256'].lower(),
            'bounded reviewed artifact pin')


def registry_authority(value):
    require(isinstance(value, dict) and set(value) == {'base', 'closure'}
            and value['base'] == str(PORTABLE_BASE)
            and isinstance(value['closure'], dict)
            and set(value['closure']) == {'bytes', 'sha256'}
            and type(value['closure']['bytes']) is int
            and 0 < value['closure']['bytes'] <= 16 * 1024**2
            and value['closure']['sha256'] == PORTABLE_SHA,
            'exact original portable registry authority join')


def validate_registry(base, manifest_pin, closure_pin, lock_pin, full=False):
    value, value_pin = parsed(base / REGISTRY_CLOSURE)
    require(value.get('schema') == 'zg.vulkan-gemma-registry-closure.v1'
            and value.get('source_manifest') == manifest_pin
            and value.get('source_closure') == closure_pin
            and value.get('cargo_lock') == lock_pin,
            'exact new Vulkan Gemma registry closure joins')
    records = value.get('records')
    require(isinstance(records, dict) and records, 'registry closure records')
    for rel, proof in records.items():
        relative(rel)
        require(rel.startswith('cargo/registry/'),
                'registry member inside the owned Cargo home')
        require(isinstance(proof, dict) and set(proof) == {'bytes', 'sha256'}
                and type(proof['bytes']) is int
                and 0 <= proof['bytes'] <= REGISTRY_MEMBER_LIMIT
                and len(proof['sha256']) == 64
                and proof['sha256'] == proof['sha256'].lower(),
                'bounded registry member proof: '+rel)
    require(type(value.get('files')) is int and value['files'] == len(records)
            and type(value.get('bytes')) is int
            and value['bytes'] == sum(proof['bytes'] for proof in records.values()),
            'derived registry member count and bytes')
    registry_authority(value.get('portable_authority'))
    if full:
        for rel, proof in sorted(records.items()):
            require(pin(base / relative(rel), limit=REGISTRY_MEMBER_LIMIT) == proof,
                    'registry member changed: '+rel)
    return value, value_pin


def validate_binding(value, closure_pin):
    require(set(value) == {'schema', 'source_base', 'source_closure',
                           'registry_closure', 'binaries',
                           'build_receipt', 'provenance'}
            and value['schema'] == 'zg.vulkan-gemma-build-binding.v1'
            and value['source_base'] == str(BASE)
            and value['source_closure'] == closure_pin,
            'reviewed new Vulkan Gemma build binding only')
    _proof(value['build_receipt'])
    _proof(value['provenance'])
    _proof(value['registry_closure'])
    require(set(value['binaries']) == {'matrix', 'final', 'peer'},
            'all three exact new binaries')
    for proof in value['binaries'].values():
        _proof(proof)
    return value


def bind_build(base, closure_pin):
    binding, binding_pin = parsed(base / BINDING)
    validate_binding(binding, closure_pin)
    receipt, receipt_pin = parsed(base / BUILD_RECEIPT)
    require(receipt_pin == binding['build_receipt'],
            'supplied new Vulkan Gemma build receipt pin differs')
    require(receipt.get('schema') == 'zg.vulkan-gemma-build-receipt.v1'
            and receipt['mode'] == 'build' and receipt['status'] == 'CLOSED_PASS_SCOPED'
            and receipt['errors'] == [] and receipt['observed_exit'] == 0
            and receipt['source_after'] == 'UNCHANGED'
            and receipt['source_closure'] == closure_pin
            and receipt.get('registry_closure') == binding['registry_closure']
            and receipt.get('registry_after') == 'UNCHANGED_FULL_SHA'
            and receipt['artifact_count'] == 3
            and receipt['cargo_argv_sha256'] == cargo_argv_sha256()
            and receipt['cwd'] == str(base / 'source'),
            'actual CLOSED new Vulkan Gemma build receipt')
    _proof(receipt['registry_closure'])
    require(pin(base / REGISTRY_CLOSURE) == receipt['registry_closure'],
            'registry closure authority differs')
    binaries = {}
    try:
        for role, artifact in zip(('matrix', 'final', 'peer'), ARTIFACTS):
            path = base / 'target' / artifact
            require(receipt['artifacts'][role] == {'path': str(path), 'proof': binding['binaries'][role]},
                    'build artifact/receipt binding differs')
            binaries[role] = Held(path, binding['binaries'][role], limit=2 * GIB)
    except BaseException:
        for item in binaries.values():
            item.close()
        raise
    return binaries, {'build_receipt': receipt_pin, 'binding': binding_pin,
                      'cargo_argv_sha256': cargo_argv_sha256(),
                      'registry_closure': receipt['registry_closure']}


class ManagedSignals:
    """Record managed signals only across owned child acquisition and cleanup.

    TERM/INT/HUP use the original checkpoint/observe seam. ALRM raises outside
    a critical ownership scope so alarm_until() keeps its absolute deadline;
    inside one it records a pending alarm which is checked after the child is
    assigned or after mandatory cleanup. Restoration is still unconditional.
    """
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


def write_receipt(path, receipt, seconds):
    """One bounded receipt attempt; the timer is cancelled on every path."""
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        path.write_text(json.dumps(receipt, indent=2) + '\n')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


_LIVE = {}


def _emergency_close():
    """Outer safety net: close/reap every owned object and restore handlers."""
    guard = _LIVE.get('guard')
    child = _LIVE.get('child')
    log = _LIVE.get('log')
    owner = _LIVE.get('owner')
    sampler = _LIVE.get('sampler')
    data_lease = _LIVE.get('data_lease')
    binaries = list(_LIVE.get('binaries', {}).values())
    binary = _LIVE.get('binary')
    held = list(_LIVE.get('held_data', [])) + list(_LIVE.get('held_cache', []))
    if binary is not None:
        held.append(binary)
    receipt = _LIVE.get('receipt') or {'errors': []}
    try:
        if guard is not None:
            with guard.critical():
                signal.setitimer(signal.ITIMER_REAL, 0)
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
                for item in held + binaries:
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
    parser.add_argument('--mode', choices=('build',) + MODES, required=True)
    parser.add_argument('--control-sha', required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 64 * GIB))
    base, output = args.base, args.output
    require(base.is_absolute()
            and base.parent == Path('/home/pvl/.cache/abird/zerogravity/qualification')
            and base.name == BASE
            and output.parent == Path('/home/pvl/.local/state/abird/zerogravity') / BASE
            and not output.exists(), 'fresh exact owned Vulkan Gemma namespace/output')
    output.mkdir(parents=True, mode=0o700)
    source = base / 'source'
    receipt = {'mode': args.mode, 'status': 'UNQUALIFIED', 'errors': [],
               'scope': 'actual Vulkan Gemma E2B source/build/healthy/unknown/'
                        'partial-native/final-session+snapshot/learned-peer lifecycles',
               'policy_is_provisional': True,
               'model_close_is_global_zero': False,
               'observations': 'shared VRAM/background and sampled RSS; unsampled peaks remain'}
    child = log = binary = sampler = None
    binaries = {}
    held_data = []
    held_cache = []
    data_lease = None
    started = time.monotonic()
    closure = closure_pin = data = data_pin = None
    source_info = registry = registry_pin = registry_lock = None
    guard = ManagedSignals()
    _LIVE['guard'] = guard
    _LIVE['receipt'] = receipt
    guard.__enter__()
    def note_guard(stage):
        if guard.cancel.received:
            receipt['errors'].append(stage + ': ' + repr(guard.exception()))
        pending = guard.take_pending_alarm()
        if pending is not None:
            receipt['errors'].append(stage + ': ' + repr(pending))
    work_end = started + LIMITS[args.mode][0] + 600
    alarm_until(work_end)
    owner = None
    try:
        guard.check()
        closure, closure_pin = parsed(base / 'source-closure.json')
        check_closure(base / 'source', closure)
        control, control_pin = parsed(base / 'control.json')
        require(control_pin['sha256'] == args.control_sha, 'exact separately approved controller')
        check_closure(base, control)
        source_info = check_source(base)
        receipt.update(control=control_pin)
        owner = owner_module(source)
        require(CANONICAL_XDG.is_dir(), 'canonical Gemma cache identity')
        registry = registry_pin = registry_lock = None
        if args.mode == 'build':
            registry_lock = pin(base / 'source/Cargo.lock')
            require(closure['records'].get('Cargo.lock') == registry_lock,
                    'current Cargo.lock closure join')
            registry, registry_pin = validate_registry(
                base, source_info['source_manifest'], closure_pin,
                registry_lock, full=True)
            receipt.update(registry_closure=registry_pin,
                           registry={'files': registry['files'],
                                     'bytes': registry['bytes'],
                                     'packages': registry.get('packages'),
                                     'source_manifest': registry['source_manifest'],
                                     'source_closure': registry['source_closure'],
                                     'cargo_lock': registry['cargo_lock'],
                                     'portable_authority': registry['portable_authority']})
        env = os.environ.copy()
        env.update(HIP_PATH=HIP, ROCM_PATH=HIP,
                   CARGO_HOME=str(base / 'cargo'), CARGO_TARGET_DIR=str(base / 'target'),
                   CARGO_NET_OFFLINE='true', CARGO_BUILD_JOBS='2',
                   RAYON_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
                   TMPDIR=str(base / 'tmp'), XDG_CACHE_HOME=str(CANONICAL_XDG),
                   XDG_DATA_HOME=str(base / 'xdg-data'), XDG_STATE_HOME=str(base / 'xdg-state'),
                   XDG_CONFIG_HOME=str(base / 'xdg-config'),
                   MESA_SHADER_CACHE_DIR=str(base / 'mesa'),
                   AMD_COMGR_CACHE_DIR=str(base / 'comgr'), LD_PRELOAD='', LD_AUDIT='',
                   VK_LOADER_DEBUG='', PATH=':'.join((HIP + '/bin', CLANG, CARGO, RUST,
                                                     CC, PKG, '/run/current-system/sw/bin')))
        (base / 'tmp').mkdir(exist_ok=True)
        cpus = sorted(os.sched_getaffinity(0))[:4]
        require(len(cpus) == 4, 'four actual allowed CPU threads')
        os.sched_setaffinity(0, cpus)
        deadline, cpu, space = LIMITS[args.mode]
        if args.mode == 'build':
            command = CARGO_ARGV
            isolated = [PYTHON, '-B', str(source / 'scripts/native-gpu.py'),
                        '--', *enclosed_build(base, source, command)]
        else:
            data_lease = JointDataLease(base)
            _LIVE['data_lease'] = data_lease
            held_data = data_lease.held
            held_cache = data_lease.cache
            _LIVE['held_data'] = held_data
            _LIVE['held_cache'] = held_cache
            data_pin = {'qwen': __import__('qwen_data').DATA_PROOF,
                        'gemma': g.DATA_PROOF}
            data = {'records': data_lease.records}
            receipt.update(read_only_original_data_roots=[
                               str(__import__('qwen_data').DATA_ROOT), str(g.DATA_ROOT)],
                           original_data_authorities=data_pin,
                           network_payload_transfer=False,
                           original_read_only_FD_bindings=True,
                           cache_writable_subtrees=[str(Q_CACHE / 'locks'), str(G_CACHE / 'locks')],
                           cache_reuse_read_only=False,
                           genuine_original_imports_revalidated=True)
            binaries, binding_pins = bind_build(base, closure_pin)
            _LIVE['binaries'] = binaries
            if args.mode in ('resident', 'resident-unknown', 'resident-row-unknown'):
                binary = binaries['matrix']
            elif args.mode in PEER_MODES:
                binary = binaries['peer']
            else:
                binary = binaries['final']
            _LIVE['binary'] = binary
            receipt.update(binding=binding_pins, binary_held_fd=binary.file.fileno(),
                           data_closure=data_pin, held_data_files=len(held_data),
                           loader={'path': LOADER, **pin(Path(LOADER).resolve(strict=True))},
                           library_path=LIBRARY_PATH)
            executable = '/tmp/zg-vulkan-gemma-held-binary'
            if args.mode in ('resident', 'resident-unknown', 'resident-row-unknown'):
                command = [LOADER, '--library-path', LIBRARY_PATH, executable,
                           str(g.DATA_ROOT / 'expected/invocation.json'),
                           str(output / (args.mode + '.json')), args.mode]
            elif args.mode in PEER_MODES:
                command = [LOADER, '--library-path', LIBRARY_PATH, executable,
                           PEER_MODES[args.mode], str(output / (args.mode + '.json'))]
            else:
                command = [LOADER, '--library-path', LIBRARY_PATH, executable,
                           args.mode, str(output / (args.mode + '.json'))]
            isolated = [PYTHON, '-B', str(source / 'scripts/native-gpu.py'), '--',
                        *enclosed(base, source, binary, data_lease, command, True, output)]
        argv = ['/run/current-system/sw/bin/prlimit', '--as=' + str(space),
                '--cpu=' + str(cpu), '--nofile=1024', '--', *isolated]
        receipt.update(argv=argv, cwd=str(source), actual_cpu_affinity=cpus,
                       deadline_seconds=deadline, per_process_cpu_limit=cpu,
                       address_space_bytes=space,
                       whole_supervisor_alarm_seconds=deadline + 600,
                       finalization_seconds=900,
                       external_supervisor_AS_soft=512 * 1024**2,
                       source_closure=closure_pin,
                       global_host_budget=16 * GIB, provider=20 * GIB,
                       physical_domain=64 * GIB, persistent_device=4 * GIB,
                       persistent_host=2 * GIB, compiler_host_per_generation=8 * GIB,
                       module_opaque_device=8 * GIB, known_module_limit=GIB,
                       retained_module_host_limit=2 * GIB, finite_catalogue_limit=1024)
        (output / 'prepared-invocation.json').write_text(json.dumps(receipt, indent=2) + '\n')
        sampler = Sampler()
        _LIVE['sampler'] = sampler
        sampler.one(-1)
        guard.check()
        with guard.critical():
            child = subprocess.Popen(argv, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True,
                                     pass_fds=() if binary is None else
                                     (binary.file.fileno(),) + tuple(fd for fd, _ in data_lease.bindings()))
            _LIVE['child'] = child
            receipt['pid'] = child.pid
            log = owner.Log(child.stdout, output / 'execution.log')
            _LIVE['log'] = log
        guard.check()
        sampler.begin(child.pid)
        receipt['observed_exit'] = guard.observe(log, child, deadline)
    except BaseException as error:
        receipt['errors'].append(repr(error))
    finally:
        with guard.critical():
            signal.setitimer(signal.ITIMER_REAL, 0)
            if sampler is not None:
                try:
                    sampler.finish_before_reap(child.pid if child is not None else -1)
                except BaseException as error:
                    receipt['errors'].append('sampling: ' + repr(error))
            if child is not None and owner is not None:
                # Same-child cleanup attempt precedes the original call; never replayed.
                _LIVE['cleanup_attempted'] = True
                try:
                    owner.cleanup_group(child, receipt)
                    _LIVE['reaped'] = True
                except BaseException as error:
                    receipt['errors'].append('owner: ' + repr(error))
        note_guard('sampling')
        note_guard('owner')
        final_end = min(time.monotonic(), work_end) + 900
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
                note_guard('EOF')
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
        full_after = True
        held_after = {id(item): item for item in
                      held_data + held_cache + list(binaries.values())}
        close_objs = {id(item): item for item in held_after.values()}
        if data_lease is not None:
            # Gather directly from the lease so an ALRM between lease creation
            # and the held_data/held_cache aliases cannot omit any source FD.
            for lease in data_lease.leases:
                for item in lease.held:
                    close_objs[id(item)] = item
                for name in ('authority', 'received'):
                    item = getattr(lease, name, None)
                    if item is not None:
                        close_objs[id(item)] = item
            for item in data_lease.cache:
                close_objs[id(item)] = item
        close_objs = list(close_objs.values())
        try:
            try:
                for item in held_after.values():
                    try:
                        alarm_until(final_end)
                        item.after()
                    except BaseException as error:
                        full_after = False
                        receipt['errors'].append('held: ' + repr(error))
                    finally:
                        with guard.critical():
                            try:
                                item.close()
                            except BaseException as error:
                                receipt['errors'].append('held close: ' + repr(error))
                if data_lease is not None:
                    try:
                        alarm_until(final_end)
                        data_lease.authority_after()
                    except BaseException as error:
                        full_after = False
                        receipt['errors'].append('original data authority: ' + repr(error))
                    finally:
                        with guard.critical():
                            try:
                                data_lease.close()
                            except BaseException as error:
                                receipt['errors'].append('data close: ' + repr(error))
            finally:
                # Unconditional original-descriptor closure: an ALRM between
                # items, a failed hash or one close error must never skip any
                # remaining data/cache/binary/lease/source FD. Each close is
                # independently attempted under the ownership scope.
                with guard.critical():
                    for item in close_objs:
                        try:
                            item.close()
                        except BaseException as error:
                            receipt['errors'].append('held close: ' + repr(error))
                    if data_lease is not None:
                        try:
                            data_lease.close()
                        except BaseException as error:
                            receipt['errors'].append('data close: ' + repr(error))
        except BaseException as error:
            full_after = False
            receipt['errors'].append('held cleanup: ' + repr(error))
        try:
            if closure is not None and control is not None:
                alarm_until(final_end)
                require(pin(base / 'source-closure.json') == closure_pin,
                        'source closure authority changed')
                require(pin(base / 'control.json') == control_pin,
                        'controller authority changed')
                check_closure(base, control)
                check_closure(base / 'source', closure)
                check_source(base)
                receipt['source_after'] = 'UNCHANGED'
                receipt['control_after'] = control_pin
                if args.mode == 'build':
                    require(source_info is not None and registry_pin is not None
                            and registry_lock is not None,
                            'actual copied registry authentication preceded build')
                    require(pin(base / 'source/Cargo.lock') == registry_lock,
                            'current Cargo.lock changed')
                    _after, after_pin = validate_registry(
                        base, source_info['source_manifest'], closure_pin,
                        registry_lock, full=True)
                    require(after_pin == registry_pin,
                            'registry closure authority changed')
                    receipt['registry_after'] = 'UNCHANGED_FULL_SHA'
            if data is not None and full_after:
                require(pin(base / 'qwen/data-closure.json') == data_pin['qwen']
                        and pin(base / 'gemma/data-closure.json') == data_pin['gemma'],
                        'both data authorities unchanged')
                receipt['data_after'] = 'UNCHANGED_HELD_FULL_SHA'
                if held_cache:
                    receipt['cache_manifest_and_import_parent_after'] = 'UNCHANGED_HELD_FULL_SHA'
                    receipt['reuse_authentication'] = {
                        'source_closure': closure_pin, 'original_data': data_pin,
                        'same_original_cache_objects': True, 'transfer': False,
                        'import': False, 'cache_lock_writes': True}
            if sampler is not None:
                receipt['sample_observations'] = sampler.report(output)
                require(receipt['sample_observations']['readable_vram_samples'] > 1,
                        'actual readable VRAM coverage')
            if child is not None and receipt.get('observed_exit') == 0 \
                    and not receipt['errors'] and receipt.get('source_after') == 'UNCHANGED':
                if args.mode == 'build':
                    found = {}
                    for line in (output / 'execution.log').read_text().splitlines():
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if event.get('reason') == 'compiler-artifact' \
                                and event.get('target', {}).get('kind') == ['example'] \
                                and event.get('executable'):
                            found[event['target']['name']] = Path(event['executable'])
                    require(set(found) == {'gemma4_vulkan_matrix',
                                           'gemma4_vulkan_final_owner',
                                           'gemma4_vulkan_same_pci_peer'},
                            'three actual reachable Vulkan examples')
                    artifacts = {}
                    for role, artifact in zip(('matrix', 'final', 'peer'), ARTIFACTS):
                        path = found[{'matrix': 'gemma4_vulkan_matrix',
                                      'final': 'gemma4_vulkan_final_owner',
                                      'peer': 'gemma4_vulkan_same_pci_peer'}[role]]
                        require(path.is_relative_to(base / 'target')
                                and path == base / 'target' / artifact,
                                'one actual reachable example')
                        artifacts[role] = {'path': str(path), 'proof': pin(path)}
                    record = {'schema': 'zg.vulkan-gemma-build-receipt.v1',
                              'mode': 'build', 'status': 'CLOSED_PASS_SCOPED',
                              'errors': [], 'observed_exit': receipt['observed_exit'],
                              'source_after': receipt['source_after'],
                              'source_closure': closure_pin, 'artifact_count': 3,
                              'registry_closure': registry_pin,
                              'registry_after': receipt['registry_after'],
                              'registry': receipt['registry'],
                              'cargo_argv_sha256': cargo_argv_sha256(),
                              'cwd': str(source), 'address_space_bytes': space,
                              'deadline_seconds': deadline, 'artifacts': artifacts}
                    with (base / BUILD_RECEIPT).open('x') as file:
                        json.dump(record, file, indent=2)
                        file.write('\n')
                    receipt['build_receipt'] = pin(base / BUILD_RECEIPT)
                    receipt['artifacts'] = artifacts
                    binding = {'schema': 'zg.vulkan-gemma-build-binding.v1',
                               'source_base': BASE,
                               'source_closure': closure_pin,
                               'registry_closure': registry_pin,
                               'binaries': {role: artifacts[role]['proof']
                                            for role in ('matrix', 'final', 'peer')},
                               'build_receipt': receipt['build_receipt'],
                               'provenance': closure_pin}
                    with (base / BINDING).open('x') as file:
                        json.dump(binding, file, indent=2)
                        file.write('\n')
                    receipt['build_binding'] = pin(base / BINDING)
                else:
                    result, result_pin = parsed(output / (args.mode + '.json'))
                    receipt['report'] = result_pin
                    receipt['report_status'] = result.get('status')
        except BaseException as error:
            receipt['errors'].append('verification: ' + repr(error))
        note_guard('finalization')
        receipt['seconds'] = time.monotonic() - started
        receipt['status'] = 'CLOSED_PASS_SCOPED' if not receipt['errors'] \
            and receipt.get('observed_exit') == 0 else 'FAILED_CLOSED'
        try:
            try:
                write_receipt(output / 'invocation.json', receipt, 5)
            except BaseException as error:
                # An expired receipt deadline is an honest failure; the one
                # optional retry has its own bounded deadline and can never
                # turn the original failure into a pass.
                receipt['errors'].append('receipt: ' + repr(error))
                receipt['status'] = 'FAILED_CLOSED'
                try:
                    write_receipt(output / 'invocation.json', receipt, 5)
                except BaseException as retry:
                    receipt['errors'].append('receipt retry: ' + repr(retry))
        finally:
            # Required owner/EOF/data closure is durable before managed signal
            # handlers return to their previous dispositions.
            with guard.critical():
                signal.setitimer(signal.ITIMER_REAL, 0)
            guard.__exit__(None, None, None)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'argv'}, indent=2), flush=True)
    signal.setitimer(signal.ITIMER_REAL, 0)
    return 0 if receipt['status'] == 'CLOSED_PASS_SCOPED' else 1


if __name__ == '__main__':
    sys.exit(main())
