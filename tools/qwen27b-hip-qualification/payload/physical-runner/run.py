#!/usr/bin/env python3
"""One bounded qwen27b natural x2 build or Q-only natural two-position process."""
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
from joint_data import JointDataLease, Q_CACHE, G_CACHE, Q_IMPORT_BASE, q, g, enclosed, enclosed_build
from modes import check_source
import natural_data
import natural_semantic
from natural_semantic import FAULT_MODES

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
# One build receipt plus one Q-only natural diagnostic per output namespace. A
# healthy nine-call diagnostic gets the full bound; shorter known/fault cases a
# matching shorter one. These are phase budgets, never acceptance evidence.
# (worker wall, supervisor wall, process address space). The 64 GiB model AS is
# the original distinct process bound; the Q-only 16 GiB provider + 16 GiB host
# is the physical-domain reservation and never a substitute address space.
LIMITS = {'build': (3600, 3600, 32*GIB)}
for _mode in natural_semantic.MODES:
    LIMITS[_mode] = (7200, 7200, 64*GIB) if _mode == 'healthy' else (1800, 1800, 64*GIB)
# Original fixed allowances: +600 s whole-work, +900 s finalization/receipt.
WORK_ALLOWANCE = 600
FINALIZATION_ALLOWANCE = 900
PLANNED_CALLS = {'healthy': 9, 'cached-unknown': 3}
for _mode in natural_semantic.MODES:
    PLANNED_CALLS.setdefault(_mode, 1)
FINAL_BASE = 'hip-multiset-natural-x2-20261024-1'
# Never written by this unit; only asserted to be distinct from its own base.
PREDECESSOR_BASES = ('hip-multiset-source20-20261024-2', 'hip-multiset-source20-20261024-1',
                     'hip-multiset-source12-20261024-4', 'hip-multiset-controls-v5-20261024-1')
assert FINAL_BASE not in PREDECESSOR_BASES
CARGO_ARGV = [CARGO+'/cargo', 'build', '-p', 'zerogravity', '--example', 'qwen27b_hip_natural',
              '--release', '--features', 'qwen27b-hip-qualification,zerogravity-gpu/hardware-abort-probe',
              '--offline', '--locked', '--message-format=json']
ARTIFACT = 'release/examples/qwen27b_hip_natural'
BUILD_RECEIPT = 'build-receipt.json'
BINARY_RECORD = 'model-binary.json'

def cargo_argv_sha256():
    return hashlib.sha256(json.dumps(CARGO_ARGV,separators=(',',':')).encode()).hexdigest()


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


def check_pending(state):
    while state['pending']:
        sig = state['pending'].pop(0)
        if sig == signal.SIGALRM:
            raise TimeoutError('owned supervisor deadline '+str(sig))
        raise InterruptedError('owned supervisor signal '+str(sig))


def close_each(objects, errors, state, label):
    """Close every distinct held object once; one failure never skips another."""
    state['critical'] = True
    with Hold(state):
        seen = set()
        for item in objects:
            if item is None or id(item) in seen:
                continue
            seen.add(id(item))
            try:
                item.close()
            except BaseException as error:
                errors.append(label+': '+repr(error))


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


def child_env(base, mode, env):
    """Child-only environment: a natural run opens the same original
    qualification cache its held import parents prove, never the unit's
    build-only xdg-cache, and the enclosure never overrides this value."""
    env = dict(env)
    if mode != 'build':
        # Pure derivation from the held import root; the joint data lease is what
        # actually proves both qualification caches exist and stay unbounded.
        root = Q_CACHE.parents[2]
        require(root != Path(base)/'xdg-cache' and root.name == 'xdg-cache'
                and root.is_absolute(),
                'exact original qualification cache derivation')
        env['XDG_CACHE_HOME'] = str(root)
    return env


def semantic(mode, output, base):
    log = (output/'execution.log').read_text()
    require('REFUSED_UNQUALIFIED' not in log, 'actual natural fixture refused')
    result, result_pin = parsed(output/'model-report.json')
    validated = natural_semantic.validate_natural(mode, result, log)
    # Serialize the bounded report flag this remote receipt actually validates,
    # instead of a top-level claim the invocation never sets.
    return {'report':result_pin,
            'trained_native_natural_serving_or_performance_qualified':
                result['trained_native_natural_serving_or_performance_qualified'],
            **validated}


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


def bind_build(base, closure_pin, hydrated_closure_pin, binding, receipt_pin):
    """Bind the ACTUAL new source20 build; absent/mismatched pins refuse early."""
    require(receipt_pin is not None and binding is not None, 'actual new source20 build binding required')
    build_receipt, build_receipt_pin = parsed(base/BUILD_RECEIPT)
    require(build_receipt_pin == receipt_pin, 'supplied new source20 build receipt pin differs')
    require(build_receipt.get('schema') == 'zg.hip-source20-natural-x2-build-receipt.v1'
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


def natural_command(mode, output, base, natural):
    """Exact five-operand natural command; extent/SHA come from external authority."""
    receipt_bytes, receipt_sha = natural.external_receipt_pin()
    return [LOADER, '--library-path', LIBRARY_PATH, '/tmp/zg-hip-natural-held-binary',
            mode, str(output/'model-report.json'), str(base/'natural-reference'),
            str(receipt_bytes), receipt_sha]


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
    sampler = _LIVE.get('sampler')
    data_lease = _LIVE.get('data_lease')
    natural = _LIVE.get('natural')
    binary = _LIVE.get('binary')
    held = list(_LIVE.get('held_data', [])) + list(_LIVE.get('held_cache', []))
    if binary is not None:
        held.append(binary)
    receipt = _LIVE.get('receipt') or {'errors': []}
    try:
        if state is not None:
            state['critical'] = True
            with Hold(state):
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
                if natural is not None:
                    for item in ([natural.receipt] if natural.receipt is not None else []) + natural.held:
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
            and not output.exists(), 'fresh exact owned two-source namespace/output')
    require((args.binary_sha is None) == (args.binary_bytes is None)
            and (args.build_receipt_sha is None) == (args.build_receipt_bytes is None)
            and (args.binary_sha is None) == (args.build_receipt_sha is None),
            'build artifact/receipt binding is both bytes and SHA or absent')
    binding = None if args.binary_sha is None else {'bytes': args.binary_bytes, 'sha256': args.binary_sha}
    binding_receipt = None if args.build_receipt_sha is None else {'bytes': args.build_receipt_bytes, 'sha256': args.build_receipt_sha}
    output.mkdir(parents=True, mode=0o700)
    source = base/'source'
    receipt = {'mode': args.mode, 'status': 'UNQUALIFIED', 'errors': [],
               'scope': 'actual same-actor Qwen full64 and Gemma resident full35 on source20; alternating exact producers, independent roots, selected known retirement/reload, preeffect compiler limit and globalunknown' ,
               'bootstrap': 'Client precedes original context partition; any setup failure retires process',
               'model_close_is_global_zero': False, 'policy_is_provisional': True,
               'observations': 'shared VRAM/background and sampled RSS; unsampled peaks remain'}
    child = log = binary = sampler = None
    held_data = []
    held_cache = []
    natural = None
    data_lease = None
    started = time.monotonic()
    closure = packaged_closure = closure_pin = packaged_closure_pin = data = data_pin = None
    aborted = False
    receipt_written = False
    state = {'critical': False, 'depth': 0, 'pending': [], 'previous': {}}
    _LIVE['state'] = state
    _LIVE['receipt'] = receipt
    def interrupted(sig, _frame):
        nonlocal aborted
        aborted = True
        if state['critical']:
            state['pending'].append(sig)
            return
        if sig == signal.SIGALRM:
            raise TimeoutError('owned supervisor deadline '+str(sig))
        raise InterruptedError('owned supervisor signal '+str(sig))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        state['previous'][sig] = signal.signal(sig, interrupted)
    # Includes pre/post source/binary/data scans; child has its own shorter wall.
    alarm_extra = WORK_ALLOWANCE
    work_end = started + LIMITS[args.mode][0] + alarm_extra
    owner = None
    try:
        alarm_until(work_end)
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
        # The natural diagnostic opens the same original qualification cache the
        # held import parents prove; the unit's own xdg-cache stays build-only and
        # the enclosure never overrides this child value.
        env = child_env(base, args.mode, env)
        cpus = sorted(os.sched_getaffinity(0))[:4]
        require(len(cpus) == 4, 'four actual allowed CPU threads')
        os.sched_setaffinity(0, cpus)
        deadline, cpu, space = LIMITS[args.mode]
        if args.mode == 'build':
            require(binding is None and binding_receipt is None,
                    'no predecessor or reused artifact/receipt pins for a fresh build')
            command = CARGO_ARGV
        else:
            # External natural authority first: absent/awaiting pins refuse every
            # natural mode before any data FD, transport or build binding.
            authority_value, authority_pin = natural_data.load_authority(base)
            natural = natural_data.NaturalReferenceLease(base/'natural-reference')
            _LIVE['natural'] = natural
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
                           cache_reuse_read_only=True, genuine_original_imports_revalidated=True,
                           natural_reference_root=str(base/'natural-reference'),
                           natural_authority=authority_pin,
                           natural_authority_status=authority_value['status'],
                           held_natural_reference_files=len(natural.bindings()),
                           natural_inputs=natural.inputs,
                           natural_feedback_qualified=False, teacher_forcing_accepted=False,
                           qwen_only=True, gemma_executed=False)
            binary, binding_pins = bind_build(base, packaged_closure_pin, closure_pin, binding, binding_receipt)
            _LIVE['binary'] = binary
            receipt.update(binary=binding_pins['binary'], binary_record=binding_pins['binary_record'],
                           build_binding={'build_receipt': binding_pins['build_receipt'],
                                          'cargo_argv_sha256': binding_pins['cargo_argv_sha256']},
                           binary_held_fd=binary.file.fileno(),
                           data_closure=data_pin, held_data_files=len(held_data),
                           loader={'path': LOADER, **pin(Path(LOADER).resolve(strict=True))},
                           library_path=LIBRARY_PATH)
            receipt_bytes, receipt_sha = natural.external_receipt_pin()
            receipt.update(natural_receipt_pin={'bytes':receipt_bytes,'sha256':receipt_sha})
            command = natural_command(args.mode, output, base, natural)
        if args.mode == 'build':
            isolated = [PYTHON, '-B', str(source/'scripts/native-gpu.py'), '--', *enclosed_build(base, source, command)]
        else:
            isolated = [PYTHON, '-B', str(source/'scripts/native-gpu.py'), '--',
                        *enclosed(base, source, binary, data_lease, command, True, output,
                              natural=natural)]
        argv = ['/run/current-system/sw/bin/prlimit', '--as='+str(space), '--cpu='+str(cpu),
                '--nofile=1024', '--', *isolated]
        receipt.update(argv=argv, cwd=str(source), actual_cpu_affinity=cpus,
                       deadline_seconds=deadline, per_process_cpu_limit=cpu, address_space_bytes=space,
                       whole_supervisor_alarm_seconds=deadline+alarm_extra,
                       finalization_seconds=FINALIZATION_ALLOWANCE, final_receipt_seconds=5,
                       external_supervisor_AS_soft=512*1024**2,
                       cpu_limit_is_aggregate=False, source_closure=packaged_closure_pin,
                       hydrated_closure=closure_pin,
                       global_host_budget=32*GIB,
                       provider=32*GIB,
                       physical_domain=64*GIB,
                       persistent_device=4*GIB, persistent_host=2*GIB,
                       compiler_host_per_generation=8*GIB, compiler_max_admitted_generations=3,
                       module_opaque_device=8*GIB, known_module_limit=GIB,
                       retained_module_host_limit=2*GIB, finite_catalogue_limit=1024,
                       qwen_only=args.mode != 'build', shared_two_source_budget=False)
        if args.mode != 'build':
            receipt['planned_work'] = {'mode':args.mode, 'qwen_calls':PLANNED_CALLS[args.mode],
                                       'gemma_calls':0, 'natural_positions':2,
                                       'genuine_original_producers':True,
                                       'reference_operands':False, 'held_natural_reference_fds':5}
        (output/'prepared-invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if args.mode != 'build':
            sampler = Sampler()
            _LIVE['sampler'] = sampler
            sampler.one(-1)  # Separate actual global background sample before launch.
        with Ownership(state):
            child = subprocess.Popen(argv, cwd=source, env=env,
                                     stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=() if binary is None else
                                     (binary.file.fileno(),) + tuple(fd for fd,_ in data_lease.bindings())
                                     + tuple(fd for fd,_ in natural.bindings()))
            _LIVE['child'] = child
            receipt['pid'] = child.pid
            if sampler is not None:
                sampler.begin(child.pid)
            log = owner.Log(child.stdout, output/'execution.log')
            _LIVE['log'] = log
        receipt['observed_exit'] = log.observe(child, deadline)
    except BaseException as error:
        receipt['errors'].append(repr(error))
    finally:
        # Original bounded cleanup: record managed signals, close every owned
        # object independently, then report any pending signal after cleanup.
        state['critical'] = True
        with Hold(state):
            # Sampler owns only reads. Join it while the original group leader is
            # still unreaped; original owner then performs TERM/two KILL/reap/EOF.
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
            elif child is not None and child.stdout is not None:
                try:
                    child.stdout.close()
                except BaseException as error:
                    receipt['errors'].append('stdout: '+repr(error))
        try:
            check_pending(state)
        except BaseException as error:
            receipt['errors'].append('deadline: '+repr(error))
        final_end = min(time.monotonic(), work_end) + FINALIZATION_ALLOWANCE
        full_after = True
        # Original data authority revalidation needs the held FDs still open.
        if data_lease is not None:
            try:
                if not aborted:
                    alarm_until(final_end)
                    data_lease.authority_after()
            except BaseException as error:
                full_after = False
                receipt['errors'].append('original data authority: '+repr(error))
        try:
            for item in held_data + held_cache + ([binary] if binary is not None else []):
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
                finally:
                    with Hold(state):
                        try:
                            item.close()
                        except BaseException as error:
                            receipt['errors'].append('held close: '+repr(error))
        except BaseException as error:
            full_after = False
            receipt['errors'].append('held verification: '+repr(error))
        finally:
            close_each(held_data + held_cache + ([binary] if binary is not None else []),
                       receipt['errors'], state, 'held close')
        if natural is not None:
            # One owner: verify the five held reference FDs, then close them once.
            try:
                if not aborted:
                    alarm_until(final_end)
                    natural.after()
            except BaseException as error:
                full_after = False
                receipt['errors'].append('natural reference: '+repr(error))
            finally:
                close_each(([natural.receipt] if natural.receipt is not None else []) + natural.held,
                           receipt['errors'], state, 'natural close')
        if data_lease is not None:
            lease_items = list(getattr(data_lease, 'cache', []))
            for lease in getattr(data_lease, 'leases', []):
                lease_items.extend(lease.held)
                lease_items.append(getattr(lease, 'authority', None))
                lease_items.append(getattr(lease, 'received', None))
            close_each(lease_items, receipt['errors'], state, 'data close')
        try:
            check_pending(state)
        except BaseException as error:
            receipt['errors'].append('deadline: '+repr(error))
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
            if natural is not None and not aborted and full_after:
                receipt['natural_after'] = 'UNCHANGED_HELD_FULL_SHA'
            if data is not None and not aborted and full_after:
                alarm_until(final_end)
                require(pin(base/'qwen/data-closure.json') == q.DATA_PROOF
                        and pin(base/'gemma/data-closure.json') == g.DATA_PROOF, 'both data authorities unchanged')
                receipt['data_after'] = 'UNCHANGED_HELD_FULL_SHA'
                if held_cache:
                    receipt['cache_manifest_and_import_parent_after'] = 'UNCHANGED_HELD_FULL_SHA'
                    receipt['reuse_authentication'] = {'source_closure':packaged_closure_pin,'original_data':data_pin,
                        'same_original_cache_objects':True,'transfer':False,'import':False,'cache_lock_writes':True}
            if sampler is not None:
                receipt['sample_observations'] = sampler.report(output)
                require(receipt['sample_observations']['readable_vram_samples'] > 1, 'actual readable VRAM coverage')
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
                                and event.get('target', {}).get('name') == 'qwen27b_hip_natural'
                                and event.get('target', {}).get('kind') == ['example'] and event.get('executable')):
                            artifacts.append(Path(event['executable']))
                    require(len(artifacts) == 1 and artifacts[0].is_relative_to(base/'target')
                            and artifacts[0] == base/'target'/ARTIFACT, 'one actual reachable example')
                    proof = pin(artifacts[0])
                    # GPU-free authentic fixture serialization from the built
                    # example itself; the grouped control consumes these bytes.
                    fixtures = base/'source-fixtures.json'
                    require(not fixtures.exists(), 'fresh source fixture path')
                    done = subprocess.run(
                        [LOADER, '--library-path', LIBRARY_PATH, str(artifacts[0]),
                         '--source-fixtures', str(fixtures)],
                        cwd=source, env=env, capture_output=True, text=True,
                        timeout=60, start_new_session=True)
                    require(done.returncode == 0 and fixtures.is_file(),
                            'actual example source fixture serialization: '+done.stderr[-2000:])
                    fixture_pin = pin(fixtures)
                    with (base/BINARY_RECORD).open('x') as file:
                        json.dump({'path': str(artifacts[0]), 'proof': proof,
                                   'source_closure': packaged_closure_pin,
                                   'hydrated_closure': closure_pin}, file, indent=2)
                        file.write('\n')
                    with (base/BUILD_RECEIPT).open('x') as file:
                        json.dump({'schema': 'zg.hip-source20-natural-x2-build-receipt.v1', 'mode': 'build',
                                   'status': 'CLOSED_PASS_SCOPED', 'errors': [],
                                   'observed_exit': receipt['observed_exit'],
                                   'source_after': receipt['source_after'],
                                   'source_closure': packaged_closure_pin, 'hydrated_closure': closure_pin,
                                   'artifact_count': 1,
                                   'cargo_argv_sha256': cargo_argv_sha256(), 'cwd': str(source),
                                   'address_space_bytes': space, 'deadline_seconds': deadline,
                                   'source_fixtures': fixture_pin,
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
        receipt['seconds'] = time.monotonic()-started
        receipt['status'] = 'CLOSED_PASS_SCOPED' if not receipt['errors'] and receipt.get('observed_exit') == 0 else 'FAILED_CLOSED'
        try:
            signal.setitimer(signal.ITIMER_REAL, 5)
            (output/'invocation.json').write_text(json.dumps(receipt, indent=2)+'\n')
            receipt_written = True
        except BaseException as error:
            receipt['errors'].append('receipt: '+repr(error))
            receipt['status'] = 'FAILED_CLOSED'
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, previous in state['previous'].items():
                try:
                    signal.signal(sig, previous)
                except BaseException:
                    pass
    print(json.dumps({k: v for k, v in receipt.items() if k != 'argv'}, indent=2), flush=True)
    return 0 if receipt_written and receipt['status'] == 'CLOSED_PASS_SCOPED' else 1


if __name__ == '__main__':
    sys.exit(main())
