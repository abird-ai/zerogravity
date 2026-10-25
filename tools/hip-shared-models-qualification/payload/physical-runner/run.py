#!/usr/bin/env python3
"""One bounded H build, finite shared lifecycle, or injected source-page event-probe process."""
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
from modes import check_source
from fixture_semantic import MODES, PENDING_MODES, validate_fixture

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
LIMITS = {'build': (3600,3600,32*GIB), 'known-qwen-reload':(7200,7200,64*GIB),
          'qwen-final-reader-unknown':(1800,1800,64*GIB), 'gemma-feedback-unknown':(1800,1800,64*GIB),
          'source-page-query-unknown':(1800,1800,64*GIB),
          'source-page-wait-unknown':(1800,1800,64*GIB),
          'source-page-destroy-unknown':(1800,1800,64*GIB)}
FINAL_BASE = 'resume-all-five-hip-20261025-2'
# Never written by this unit; only asserted to be distinct from its own base.
PREDECESSOR_BASES = ('hip-multiset-source12-20261024-4', 'hip-multiset-controls-v5-20261024-1',
                     'hip-multiset-controls-v6-20261024-1', 'hip-multiset-controls-v7-20261024-1',
                     'hip-multiset-source20-20261024-2',
                     'resume-all-five-hip-20261025-1')
assert FINAL_BASE not in PREDECESSOR_BASES
CARGO_ARGV = [CARGO+'/cargo', 'build', '-p', 'zerogravity', '--example', 'hip_shared_models',
              '--release', '--features', 'qwen27b-hip-qualification,gemma4-hip-qualification,zerogravity-gpu/hardware-abort-probe',
              '--offline', '--locked', '--message-format=json']
ARTIFACT = 'release/examples/hip_shared_models'
BUILD_RECEIPT = 'build-receipt.json'
BINARY_RECORD = 'model-binary.json'

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
                pass  # Process may finish between stat and status.
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
    require('REFUSED_UNQUALIFIED' not in log, 'actual shared fixture refused')
    result, result_pin = parsed(output/'model-report.json')
    return {'report':result_pin, **validate_fixture(mode, result, log)}


def after_retirement():
    """PCI/global observations only; never rediscover an owner from a reaped PID."""
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
                                'rss': rss, 'device_fds': fds, 'ownership': 'unattributed current visible process'})
        except (OSError, ValueError, IndexError):
            continue
    return {'t': time.monotonic(), 'phase': 'after_original_group_absent_and_leader_reaped',
            'pci': '0000:c6:00.0', 'global_vram': used, 'device_nodes': nodes,
            'visible_current_device_holders': holders, 'scanned_pids': scanned,
            'scan_truncated': truncated, 'permission_or_race_omissions_possible': True,
            'original_group_owned_rss': 0, 'rss_zero_reason': 'original group absence, no recycled-PID lookup',
            'shared_background': True, 'isolated_restoration_or_context_bytes_proven': False}



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
    """Outer safety net: close/reap every owned object and restore handlers."""
    guard = _LIVE.get('guard')
    child = _LIVE.get('child')
    log = _LIVE.get('log')
    owner = _LIVE.get('owner')
    sampler = _LIVE.get('sampler')
    data_lease = _LIVE.get('data_lease')
    binary = _LIVE.get('binary')
    held = list(_LIVE.get('held_data', [])) + list(_LIVE.get('held_cache', []))
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

def bind_build(base, closure_pin, hydrated_closure_pin, binding, receipt_pin):
    """Bind the ACTUAL new source20 build; absent/mismatched pins refuse early."""
    require(receipt_pin is not None and binding is not None, 'actual new source20 build binding required')
    build_receipt, build_receipt_pin = parsed(base/BUILD_RECEIPT)
    require(build_receipt_pin == receipt_pin, 'supplied new source20 build receipt pin differs')
    require(build_receipt.get('schema') == 'zg.hip-source20-build-receipt.v1'
            and build_receipt['mode'] == 'build' and build_receipt['status'] == 'CLOSED_PASS_SCOPED'
            and build_receipt['errors'] == [] and build_receipt['observed_exit'] == 0
            and build_receipt['source_after'] == 'UNCHANGED'
            and build_receipt['source_closure'] == closure_pin
            and build_receipt['hydrated_closure'] == hydrated_closure_pin
            and build_receipt['artifact_count'] == 1
            and build_receipt['cargo_argv_sha256'] == cargo_argv_sha256()
            and build_receipt['cwd'] == str(base/'source')
            and build_receipt['binary'] == {'path': str(base/'target'/ARTIFACT), 'proof': binding},
            'actual CLOSED new source20 build receipt and artifact binding')
    record, record_pin = parsed(base/BINARY_RECORD)
    require(record == {'path': str(base/'target'/ARTIFACT), 'proof': binding,
                       'source_closure': closure_pin,
                       'hydrated_closure': hydrated_closure_pin}, 'new build artifact record')
    binary = Held(base/'target'/ARTIFACT, binding, limit=2*GIB)
    return binary, {'build_receipt': build_receipt_pin, 'binary_record': record_pin,
                    'binary': binding, 'cargo_argv_sha256': cargo_argv_sha256(),
                    'hydrated_closure': hydrated_closure_pin}



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
    parser.add_argument('--binary-sha')
    parser.add_argument('--binary-bytes', type=int)
    parser.add_argument('--build-receipt-sha')
    parser.add_argument('--build-receipt-bytes', type=int)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, 64*GIB))
    base, output = args.base, args.output
    require(base.is_absolute() and base.parent == Path('/home/pvl/.cache/abird/zerogravity/qualification')
            and base.name == FINAL_BASE and output.parent == Path('/home/pvl/.local/state/abird/zerogravity')/FINAL_BASE
            and not output.exists(), 'fresh exact owned H namespace/output')
    require((args.binary_sha is None) == (args.binary_bytes is None)
            and (args.build_receipt_sha is None) == (args.build_receipt_bytes is None)
            and (args.binary_sha is None) == (args.build_receipt_sha is None),
            'build artifact/receipt binding is both bytes and SHA or absent')
    binding = None if args.binary_sha is None else {'bytes': args.binary_bytes, 'sha256': args.binary_sha}
    binding_receipt = None if args.build_receipt_sha is None else {'bytes': args.build_receipt_bytes, 'sha256': args.build_receipt_sha}
    output.mkdir(parents=True, mode=0o700)
    source = base/'source'
    receipt = {'mode': args.mode, 'status': 'UNQUALIFIED', 'errors': [],
               'scope': ('actual same-actor Qwen full64 and Gemma resident full35 on the complete '
                         'H source closure; alternating exact producers, independent roots, selected '
                         'known retirement/reload, preeffect compiler limit, global unknown, and '
                         'bounded source-page query/wait/destroy injected-policy-unknown probes'),
               'bootstrap': 'Client precedes original context partition; any setup failure retires process',
               'model_close_is_global_zero': False, 'policy_is_provisional': True,
               'observations': 'shared VRAM/background and sampled RSS; unsampled peaks remain'}
    child = log = binary = sampler = None
    held_data = []
    held_cache = []
    data_lease = None
    started = time.monotonic()
    closure = packaged_closure = closure_pin = packaged_closure_pin = data = data_pin = None
    aborted = False
    def mark_aborted():
        nonlocal aborted
        aborted = True
    guard = ManagedSignals(receipt, 'owned supervisor', mark_aborted)
    _LIVE['guard'] = guard
    _LIVE['receipt'] = receipt
    guard.install()
    receipt_end = None
    # Includes pre/post source/binary/data scans; child has its own shorter wall.
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
                   TMPDIR=str(base/'tmp'), XDG_CACHE_HOME=str(base/'xdg-cache'),
                   XDG_DATA_HOME=str(base/'xdg-data'), XDG_STATE_HOME=str(base/'xdg-state'),
                   XDG_CONFIG_HOME=str(base/'xdg-config'), MESA_SHADER_CACHE_DIR=str(base/'mesa'),
                   AMD_COMGR_CACHE_DIR=str(base/'comgr'), LD_PRELOAD='', LD_AUDIT='',
                   PATH=':'.join((HIP+'/bin', CLANG, CARGO, RUST, CC, PKG, '/run/current-system/sw/bin')))
        (base/'tmp').mkdir(exist_ok=True)
        cpus = sorted(os.sched_getaffinity(0))[:4]
        require(len(cpus) == 4, 'four actual allowed CPU threads')
        os.sched_setaffinity(0, cpus)
        deadline, cpu, space = LIMITS[args.mode]
        if args.mode == 'build':
            require(binding is None and binding_receipt is None,
                    'no predecessor or reused artifact/receipt pins for a fresh build')
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
            binary, binding_pins = bind_build(base, packaged_closure_pin, closure_pin, binding, binding_receipt)
            _LIVE['binary'] = binary
            receipt.update(binary=binding_pins['binary'], binary_record=binding_pins['binary_record'],
                           build_binding={'build_receipt': binding_pins['build_receipt'],
                                          'cargo_argv_sha256': binding_pins['cargo_argv_sha256']},
                           binary_held_fd=binary.file.fileno(),
                           data_closure=data_pin, held_data_files=len(held_data),
                           loader={'path': LOADER, **pin(Path(LOADER).resolve(strict=True))},
                           library_path=LIBRARY_PATH)
            executable = '/tmp/zg-hip-joint-held-binary'
            command = [LOADER, '--library-path', LIBRARY_PATH, executable]
            command += [args.mode, str(Q_CACHE), str(G_CACHE), str(q.DATA_ROOT/'expected'),
                        str(g.DATA_ROOT/'expected/invocation.json'), str(output/'model-report.json')]
        if args.mode == 'build':
            isolated = [PYTHON, '-B', str(source/'scripts/native-gpu.py'), '--', *enclosed_build(base, source, command)]
        else:
            isolated = [PYTHON, '-B', str(source/'scripts/native-gpu.py'), '--',
                        *enclosed(base, source, binary, data_lease, command, True, output)]
        argv = ['/run/current-system/sw/bin/prlimit', '--as='+str(space), '--cpu='+str(cpu),
                '--nofile=1024', '--', *isolated]
        receipt.update(argv=argv, cwd=str(source), actual_cpu_affinity=cpus,
                       deadline_seconds=deadline, per_process_cpu_limit=cpu, address_space_bytes=space,
                       whole_supervisor_alarm_seconds=deadline+600, finalization_seconds=900, final_receipt_seconds=5,
                       external_supervisor_AS_soft=512*1024**2,
                       cpu_limit_is_aggregate=False, source_closure=packaged_closure_pin,
                       hydrated_closure=closure_pin,
                       global_host_budget=32*GIB, provider=32*GIB, physical_domain=64*GIB,
                       persistent_device=4*GIB, persistent_host=2*GIB, compiler_host_per_generation=8*GIB, compiler_max_admitted_generations=3,
                       module_opaque_device=8*GIB, known_module_limit=GIB,
                       retained_module_host_limit=2*GIB, finite_catalogue_limit=1024)
        if args.mode in MODES:
            qwen_calls = 3 if args.mode == 'known-qwen-reload' else 0 if args.mode in PENDING_MODES else 1
            # Pending modes have no planned Gemma zero: the actual fixture report
            # must preserve whatever healthy Gemma setup work was actually needed.
            gemma_calls = 4 if args.mode == 'known-qwen-reload' else None if args.mode in PENDING_MODES else 1
            receipt['planned_work'] = {'mode':args.mode,'qwen_calls':qwen_calls,
                                       'gemma_calls':gemma_calls,
                                       'genuine_original_producers':True,'reference_operands':False}
        (output/'prepared-invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if args.mode in MODES:
            sampler = Sampler()
            _LIVE['sampler'] = sampler
            sampler.one(-1)  # Separate actual global background sample before launch.
        with guard:
            child = subprocess.Popen(argv, cwd=source, env=env,
                                     stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
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
            # Original bounded cleanup cannot be interrupted by the phase alarm.
            signal.setitimer(signal.ITIMER_REAL, 0)
            held_objects = held_data + held_cache + ([binary] if binary is not None else [])
            # Mandatory original closes: sampler join, group TERM/two-KILL/reap/absence,
            # raw stdout fallback and bounded EOF. Record-only managed signals.
            with guard:
                if sampler is not None:
                    try:
                        sampler.finish_before_reap(child.pid if child is not None else -1)
                    except BaseException as error:
                        receipt['errors'].append('sampling: '+repr(error))
                if child is not None:
                    _LIVE['cleanup_attempted'] = True
                    try:
                        owner.cleanup_group(child, receipt)
                        _LIVE['reaped'] = True
                    except BaseException as error:
                        receipt['errors'].append('owner: '+repr(error))
                    if log is None:
                        try:
                            child.stdout.close()
                        except BaseException as error:
                            receipt['errors'].append('raw stdout: '+repr(error))
                    if sampler is not None:
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
                    receipt['log'] = {'retained': log.retained, 'total': log.total, 'eof': log.eof, 'closed': log.closed}
                # Independently close the raw pipe even if EOF/finish failed.
                if child is not None and child.stdout is not None and not child.stdout.closed:
                    try:
                        child.stdout.close()
                    except BaseException as error:
                        receipt['errors'].append('raw stdout: '+repr(error))
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
                # Every original held data/cache/binary FD and the joint lease close
                # independently, even after a hash/authority failure.
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
                    require(pin(base/'source-closure.json') == packaged_closure_pin, 'packaged source closure authority changed')
                    require(pin(base/'closure.json') == closure_pin, 'hydrated source closure authority changed')
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
                            and pin(base/'gemma/data-closure.json') == g.DATA_PROOF, 'both data authorities unchanged')
                    receipt['data_after'] = 'UNCHANGED_HELD_FULL_SHA'
                    if held_cache:
                        receipt['cache_manifest_and_import_parent_after'] = 'UNCHANGED_HELD_FULL_SHA'
                        receipt['reuse_authentication'] = {'source_closure':packaged_closure_pin,'original_data':data_pin,
                            'same_original_cache_objects':True,'transfer':False,'import':False,'cache_lock_writes':True}
            except BaseException as error:
                receipt['errors'].append('verification: '+repr(error))
            try:
                if sampler is not None:
                    receipt['sample_observations'] = sampler.report(output)
                    require(receipt['sample_observations']['readable_vram_samples'] > 1, 'actual readable VRAM coverage')
            except BaseException as error:
                receipt['errors'].append('verification: '+repr(error))
            try:
                if child is not None and receipt.get('observed_exit') == 0 and not receipt['errors'] \
                        and receipt.get('source_after') == 'UNCHANGED':
                    if args.mode == 'build':
                        artifacts = []
                        for line in (output/'execution.log').read_text().splitlines():
                            try:
                                event = json.loads(line)
                            except ValueError:
                                continue
                            if (event.get('reason') == 'compiler-artifact'
                                    and event.get('target', {}).get('name') == 'hip_shared_models'
                                    and event.get('target', {}).get('kind') == ['example'] and event.get('executable')):
                                artifacts.append(Path(event['executable']))
                        require(len(artifacts) == 1 and artifacts[0].is_relative_to(base/'target')
                                and artifacts[0] == base/'target'/ARTIFACT, 'one actual reachable example')
                        proof = pin(artifacts[0])
                        with (base/BINARY_RECORD).open('x') as file:
                            json.dump({'path': str(artifacts[0]), 'proof': proof,
                                       'source_closure': packaged_closure_pin,
                                       'hydrated_closure': closure_pin}, file, indent=2)
                            file.write('\n')
                        with (base/BUILD_RECEIPT).open('x') as file:
                            json.dump({'schema': 'zg.hip-source20-build-receipt.v1', 'mode': 'build',
                                       'status': 'CLOSED_PASS_SCOPED', 'errors': [],
                                       'observed_exit': receipt['observed_exit'],
                                       'source_after': receipt['source_after'],
                                       'source_closure': packaged_closure_pin, 'hydrated_closure': closure_pin,
                                       'artifact_count': 1,
                                       'cargo_argv_sha256': cargo_argv_sha256(), 'cwd': str(source),
                                       'address_space_bytes': space, 'deadline_seconds': deadline,
                                       'binary': {'path': str(artifacts[0]), 'proof': proof}}, file, indent=2)
                            file.write('\n')
                        receipt['build_artifact'] = {'path': str(artifacts[0]), 'proof': proof,
                                                     'source_closure': packaged_closure_pin,
                                                     'hydrated_closure': closure_pin}
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
        # One absolute receipt budget covers the attempt, optional failure
        # rewrite and the final stdout; a failed or retried write can never
        # upgrade the recorded failure to a pass.
        receipt_end = time.monotonic() + 5
        try:
            alarm_until(receipt_end)
            (output/'invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
        except BaseException as error:
            receipt['errors'].append('receipt write: '+repr(error))
            receipt['status'] = 'FAILED_CLOSED'
            try:
                alarm_until(receipt_end)
                (output/'invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
            except BaseException as error:
                receipt['errors'].append('receipt rewrite: '+repr(error))
                receipt['status'] = 'FAILED_CLOSED'
        finally:
            try:
                # Final stdout shares the one five-second receipt deadline.
                # Handlers and the timer stay installed until print/flush returns or fails.
                if receipt_end is None:
                    receipt_end = time.monotonic()
                alarm_until(receipt_end)
                print(json.dumps({k: v for k, v in receipt.items() if k != 'argv'}, indent=2), flush=True)
            except BaseException as error:
                receipt['errors'].append('stdout: '+repr(error))
                receipt['status'] = 'FAILED_CLOSED'
            finally:
                guard.restore()
    return 0 if receipt['status'] == 'CLOSED_PASS_SCOPED' and not receipt['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
