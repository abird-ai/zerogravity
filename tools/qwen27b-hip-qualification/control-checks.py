#!/usr/bin/env python3
"""GPU-free grouped control for the qwen27b natural x2 unit: real local owner
stages, the five-FD natural reference lease, the natural report contract, the
explicit parent-frozen source manifest, the Q-only 16/16 budget and negative controls.

No cargo, no SSH, no model, no GPU, no network, no reference producer.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time

CACHE_SHA = '42ac6bf6f82e61adfe1cff7c565076361f222c51792f02df09337ff1779a791a'
OWNER_SHA = '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'
CANCEL_SHA = '00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128'
GIB = 1024**3
RAW_BYTES = 934776928
EXPECTED_RS = 'crates/zerogravity/examples/qwen27b_hip_matrix/expected.rs'
MUTATION_PATH = 'crates/zerogravity/examples/qwen27b_hip_natural.rs'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def owner_cleanup_case(owner):
    """Real TERM-honoring child: two authenticated KILLs precede ONE reap."""
    child = subprocess.Popen(['sh', '-c', 'trap "exit 0" TERM; while :; do sleep 0.05; done'],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             start_new_session=True)
    # Avoid the fresh-exec race (TERM before the trap exists => -15); production
    # children always run for a long time before any owned cleanup.
    time.sleep(0.3)
    receipt = {'pid': child.pid}
    owner.cleanup_group(child, receipt)
    return receipt


def receipt_value(image='sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe',
                  scope_teacher=False, teachers=False):
    scope = {'original_full_text_forward': True, 'all_851_text_source_roles': True,
             'full_hf_conv4': True, 'mixed_source_parameters_preserved': True,
             'teacher_forcing_only': scope_teacher, 'natural_feedback_qualified': not scope_teacher,
             'rust_numerical_parity_qualified': False,
             'state_root_native_serving_performance_qualified': False,
             'all_bf16_loader_equivalence_qualified': False, 'mkl_exp_producer_qualified': False}
    value = {'schema': 'zg.actual-original27b-full-window-natural-reference.v1',
             'status': 'ACTUAL_ORIGINAL27B_FULL_WINDOW_NATURAL_REFERENCE_ONLY',
             'independent_original_natural_trajectory': True, 'natural_inputs': [[11, 220], [11, 220]],
             'seed': 11, 'snapshot': '98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e',
             'repository': 'Qwen/Qwen3.8-27B', 'revision': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
             'parameter_mode': 'raw-source-assign-no-cast',
             'config_sha256': '191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab',
             'index_sha256': '77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df',
             'scope': scope, 'calls': 2, 'original_repetitions': 2, 'text_roles': 851, 'layers': 64,
             'cache_domains': 112, 'full_hf_conv_width': 4, 'records': 1040, 'raw_bytes': RAW_BYTES,
             'image_id': image,
             'image_digest': 'sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634',
             'original_source_and_owned_lifetime_closed': True,
             'rust_numerical_parity_qualified': False, 'mkl_exp_producer_qualified': False,
             'state_root_native_serving_performance_qualified': False,
             'package': {f'p{i}.py': {'bytes': i+1, 'sha256': f'{i:064x}'} for i in range(9)},
             'source_files': []}
    if teachers:
        value['teachers'] = [11, SELECTIONS[1][0]]
    return value


SELECTIONS = ((11, 0x3f80_0000), (220, 0x4000_0000), (221, 0x4000_0000))
# Actual 112-domain flatten metadata, mirroring the Rust constant.
DOMAIN_METADATA_BYTES = 112 * (16 + 3*8 + 8 + 2*8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-receipt', type=Path,
                        help='closed natural build receipt carrying the actual source fixtures')
    args = parser.parse_args()
    payload = args.payload.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(payload/'runner'), str(payload/'physical-runner')]
    from pins import parsed, pin, require
    import bootstrap_receipt as receipt_mod
    import launch
    import run as physical_run
    import modes as source_modes
    import natural_data
    import natural_semantic
    owner = load('control_original_owner',
                 payload/'source/tools/qwen27b-full-window-reference/owned_container_v4.py')
    results = {'schema': 'zg.hip-source20-natural-x2-control.v1',
               'status': 'UNQUALIFIED', 'checks': [], 'errors': [], 'GPU_or_model': False,
               'remote_effect': False, 'network': False, 'cargo': False}
    temporary = tempfile.TemporaryDirectory(prefix='natural-x2-control-', dir=args.output)
    temp = Path(temporary.name)

    def check(name, fn):
        try:
            fn()
            results['checks'].append(name)
        except BaseException as error:
            results['errors'].append(name+': '+repr(error))

    def expect_refusal(fn, why):
        try:
            fn()
        except BaseException:
            return
        raise AssertionError('accepted '+why)

    def expect_reason(fn, fragment):
        try:
            fn()
        except BaseException as error:
            require(fragment in str(error), 'intended refusal reason '+fragment+': '+str(error))
            return
        raise AssertionError('accepted reason '+fragment)

    def actual_fixtures():
        """Six serde report fixtures serialized by the built example itself."""
        require(args.build_receipt is not None and args.build_receipt.is_file(),
                'closed build receipt for the actual source fixtures')
        receipt = json.loads(args.build_receipt.read_text())
        path = args.build_receipt.parent/'source-fixtures.json'
        require(path.is_file() and receipt.get('source_fixtures') == pin(path),
                'pinned actual serialized source fixtures from the built example')
        return json.loads(path.read_text()), pin(path)

    def real_cleanup():
        case = owner_cleanup_case(owner)
        state = case['foreground_cleanup']
        require(state == {'pgid': case['pid'], 'term_sent': True, 'kill_sent_before_reap': True,
                          'leader_reaped': True, 'group_absent': True,
                          'kill_attempts_before_reap': 2, 'leader_reap_attempts': 1,
                          'all_nonzero_signals_before_reap': True, 'exit_code': 0,
                          'stage_errors': []}, 'real TERM/twoKILL/oneReap/group_absent')
        require(not owner.group_exists(case['pid']), 'real group absent')
        receipt = {'observed_exit': 0, 'errors': [], 'pid': case['pid'],
                   'foreground_cleanup': state,
                   'log': {'retained': 12, 'total': 12, 'eof': True, 'closed': True}}
        receipt_mod._owner(receipt)
        expect_refusal(lambda: receipt_mod._owner(dict(receipt, foreground_cleanup=dict(state, exit_code=-15))),
                       'a KILL-only observed exit as a successful child')

    def real_log_eof():
        script = ('import json,sys;'
                  'sys.stdout.write('+repr(receipt_mod.BANNER.decode())+'+json.dumps({"a":1})+"\\n");'
                  'sys.stdout.flush()')
        child = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        log = owner.Log(child.stdout, temp/'eof.log')
        code = log.observe(child, 30)
        owner.cleanup_group(child, {'pid': child.pid})
        log.finish()
        require(code == 0 and log.retained == log.total > 0 and log.eof and log.closed,
                'real bounded full-EOF drain')
        require((temp/'eof.log').read_bytes().startswith(receipt_mod.BANNER), 'child banner bytes')

    def real_archive_extraction():
        prepared = json.loads((payload.parent/'PREPARED.json').read_text())
        archive = Path(prepared['source_archive_operand'])
        require(archive.is_file(), 'prepared fresh code archive operand')
        closure, closure_pin = parsed(payload/'source-closure.json')
        scratch = temp/'real-extract'
        scratch.mkdir()
        umask = os.umask(0o022)
        try:
            with archive.open('rb') as raw:
                with tarfile.open(fileobj=raw, mode='r:') as tar:
                    seen = bootstrap_extract(tar, scratch)
            with (scratch/'.plain-probe').open('xb'):
                pass
        finally:
            os.umask(umask)
        require(seen == set(closure['records']) | {'source-closure.json'},
                'real extraction exact archive member union')
        require(stat.S_IMODE((scratch/'.plain-probe').stat().st_mode) == 0o644,
                'plain xb under umask022 would lose the authenticated mode')
        source_modes.check_source(scratch)
        require(stat.S_IMODE((scratch/'source/.gitattributes').stat().st_mode) == 0o600,
                'extracted 0600 member')
        require(stat.S_IMODE((scratch/'source/install.sh').stat().st_mode) == 0o755,
                'extracted 0755 member')
        assert closure_pin['sha256'] == pin(payload/'source-closure.json')['sha256']
        # Negative: one changed native byte must break the assembled source view.
        staged = scratch/'source'/MUTATION_PATH
        staged.write_bytes(staged.read_bytes()+b'\n')
        expect_refusal(lambda: source_modes.check_source(scratch),
                       'a mutated explicit source-manifest byte')

    def bootstrap_extract(tar, scratch):
        import bootstrap
        return bootstrap.extract_archive(tar, scratch,
                                         ('pins.py', 'bootstrap.py', 'source-payload.tar'))

    def natural_lease():
        root = temp/'natural-reference'
        root.mkdir()
        raw = root/'raw.bin'
        with raw.open('xb') as file:
            file.truncate(RAW_BYTES)
        raw_sha = hashlib.file_digest(raw.open('rb'), 'sha256').hexdigest()
        manifest = json.dumps({'schema': 'zg.original27b-natural-full-window-capture.v1',
                               'records': [], 'calls': []}).encode()
        (root/'manifest.json').write_bytes(manifest)
        (root/'invocation.json').write_bytes(b'{"status":"EXECUTED_ORIGINAL27B_NATURAL_FULL_WINDOW_REQUIRES_VALIDATION"}\n')
        (root/'image-inspect.json').write_bytes(b'[]\n')
        value = receipt_value()
        value['manifest'] = {'path': str(root/'manifest.json'), 'bytes': len(manifest),
                             'sha256': hashlib.sha256(manifest).hexdigest()}
        value['raw'] = {'path': str(raw), 'bytes': RAW_BYTES, 'sha256': raw_sha}
        for name in ('invocation.json', 'image-inspect.json'):
            data = (root/name).read_bytes()
            value['invocation' if name == 'invocation.json' else 'image_inspect'] = {
                'path': str(root/name), 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest()}
        receipt_path = root/natural_data.NATURAL_RECEIPT
        receipt_path.write_text(json.dumps(value))
        authority = {'schema': natural_data.AUTHORITY_SCHEMA,
                     'status': natural_data.ACCEPTED,
                     'source': 'tools/qwen27b-natural-full-window-reference validate --receipt',
                     'package': value['package'],
                     'receipt': {'bytes': receipt_path.stat().st_size,
                                 'sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest()},
                     'artifacts': {name: {k: value[name][k] for k in ('bytes', 'sha256')}
                                   for name in natural_data.NATURAL_PINS}}
        authority_path = root.parent/natural_data.AUTHORITY
        authority_path.write_text(json.dumps(authority))
        require(natural_data.validate_receipt(json.loads(receipt_path.read_text()),
                                              authority['package']) == [[11, SELECTIONS[1][0]],
                                                                        [11, SELECTIONS[1][0]]],
                'natural receipt declares two seed11 cohorts with one feedback input')
        lease = natural_data.NaturalReferenceLease(root)
        try:
            external_pin = lease.external_receipt_pin()
            require(external_pin == (authority['receipt']['bytes'], authority['receipt']['sha256']),
                    'external receipt extent/SHA comes from the accepted authority, not this lease')
            require(physical_run.natural_command('healthy', root, root, lease)[-2:]
                    == [str(external_pin[0]), external_pin[1]],
                    'natural command carries the external receipt operands')
            require(len(lease.bindings()) == 5
                    and len({fd for fd, _ in lease.bindings()}) == 5,
                    'receipt plus four distinct artifact FDs')
            require(os.fstat(lease.bindings()[0][0]).st_size == len(receipt_path.read_bytes()),
                    'held receipt bytes')
            lease.after()
        finally:
            lease.close()
        # Negative: the awaiting authority refuses before opening any corpus FD.
        awaiting = dict(authority, status=natural_data.AWAITING, receipt=None, artifacts=None)
        authority_path.write_text(json.dumps(awaiting))
        expect_refusal(lambda: natural_data.NaturalReferenceLease(root),
                       'an absent accepted capture authority')
        authority_path.write_text(json.dumps(authority))
        # Negative: a teacher-forced scope/teacher list is not a natural corpus.
        # Every negative starts from a deep copy of the pristine receipt and the
        # file is restored; a previously corrupted receipt must never leak into
        # the next case, and each refusal reason is asserted, not just any error.
        try:
            for change, fragment in (
                    ({'scope': receipt_value(scope_teacher=True)['scope']},
                     'natural feedback qualified, teacher forcing refused'),
                    ({'natural_inputs': [[SELECTIONS[1][0], SELECTIONS[1][0]],
                                          [11, SELECTIONS[1][0]]]},
                     'two natural cohorts start at seed11 with an original call0 argmax next input'),
                    ({'natural_inputs': [[11, 220], [11, SELECTIONS[2][0]]]},
                     'both natural cohorts declare the same call1 feedback input'),
                    ({'image_id': 'sha256:'+'0'*64},
                     'natural original producer scope/identity')):
                bad = json.loads(json.dumps(value))
                bad.update(change)
                receipt_path.write_text(json.dumps(bad))
                expect_reason(lambda: natural_data.validate_receipt(
                    json.loads(receipt_path.read_text()), authority['package']), fragment)
        finally:
            receipt_path.write_text(json.dumps(value))
        # Negative: tampered held bytes are refused by the declared pin.
        raw.write_bytes(b'x')
        expect_refusal(lambda: natural_data.NaturalReferenceLease(root), 'tampered raw bytes')
        raw.unlink()
        with raw.open('xb') as file:
            file.truncate(RAW_BYTES)
        # Negative: missing/extra corpus member is refused before any scan.
        (root/'extra.bin').write_bytes(b'x')
        expect_refusal(lambda: natural_data.NaturalReferenceLease(root), 'an extra corpus member')
        (root/'extra.bin').unlink()
        lease = natural_data.NaturalReferenceLease(root)
        lease.close()

    def authority_installer_controls():
        installer = load('authority_installer', Path(__file__).with_name('prepare.py'))
        installer.PAYLOAD_HELPERS = payload/'physical-runner'
        root = temp/'authority-install'
        root.mkdir()
        value = receipt_value()
        tool_pins = {name: {'bytes': 1, 'sha256': '0'*64}
                     for name in installer.NATURAL_RUNTIME + installer.NATURAL_EXTRA}
        accepted_package = {name: tool_pins[name] for name in installer.NATURAL_RUNTIME}
        value['package'] = accepted_package
        for name, basename in natural_data.PIN_BASENAME.items():
            data = ('installer-control-'+name).encode()
            path = root/basename
            path.write_bytes(data)
            value[name] = {'path': str(path), 'bytes': len(data),
                           'sha256': hashlib.sha256(data).hexdigest()}
        raw = json.dumps(value).encode()
        receipt = root/natural_data.NATURAL_RECEIPT
        receipt.write_bytes(raw)
        extent, sha = len(raw), hashlib.sha256(raw).hexdigest()
        _, artifacts, pin_value = installer.accepted_metadata(receipt, extent, sha, accepted_package)
        require(pin_value == {'bytes': extent, 'sha256': sha}
                and artifacts == {name: {k: value[name][k] for k in ('bytes', 'sha256')}
                                  for name in natural_data.NATURAL_PINS},
                'explicit external receipt pin installs its four declared artifact pins')
        authority = {'schema': natural_data.AUTHORITY_SCHEMA, 'status': natural_data.ACCEPTED,
                     'source': 'tools/qwen27b-natural-full-window-reference validate --receipt',
                     'package': value['package'], 'receipt': pin_value, 'artifacts': artifacts}
        (root.parent/natural_data.AUTHORITY).write_text(json.dumps(authority))
        installed, _ = natural_data.load_authority(root.parent)
        require(installed['status'] == natural_data.ACCEPTED
                and installed['receipt'] == pin_value and installed['artifacts'] == artifacts,
                'fresh accepted authority joins the exact external pin')
        (root.parent/natural_data.AUTHORITY).unlink()
        expect_reason(lambda: installer.accepted_metadata(receipt, extent-1, sha, accepted_package),
                      'external accepted receipt extent')
        expect_reason(lambda: installer.accepted_metadata(receipt, extent, '0'*64, accepted_package),
                      'external accepted receipt SHA-256')
        for change, fragment in (
                ({'package': {f'q{i}.py': {'bytes': i+1, 'sha256': f'{i:064x}'} for i in range(9)}},
                 'receipt package is the same frozen nine-file natural authority'),
                ({'schema': 'zg.wrong-natural-schema.v1'}, 'natural original producer scope/identity'),
                ({'scope': receipt_value(scope_teacher=True)['scope']},
                 'natural feedback qualified, teacher forcing refused'),
                ({'image_id': 'sha256:'+'0'*64}, 'natural original producer scope/identity'),
                ({'snapshot': '0'*64}, 'natural original producer scope/identity')):
            bad = json.loads(json.dumps(value))
            bad.update(change)
            bad_raw = json.dumps(bad).encode()
            bad_path = root/'bad-receipt.json'
            bad_path.write_bytes(bad_raw)
            expect_reason(lambda p=bad_path, b=len(bad_raw), h=hashlib.sha256(bad_raw).hexdigest():
                          installer.accepted_metadata(p, b, h, accepted_package), fragment)
        # Missing operand remains AWAITING; prepare's own default branch is the
        # existing no-receipt path. This control never writes a genuine capture.
        awaiting = installer.authority_value(tool_pins, None)
        require(awaiting['status'] == natural_data.AWAITING
                and awaiting['receipt'] is None and awaiting['artifacts'] is None,
                'missing external receipt defaults to awaiting')
        (root.parent/natural_data.AUTHORITY).write_text(json.dumps(awaiting))
        expect_reason(lambda: natural_data.load_authority(root.parent),
                      'accepted original natural capture pins absent')
        (root.parent/natural_data.AUTHORITY).unlink()

    def authority_prepare_accepted_main():
        """Exercise prepare's accepted branch with the real nine-file package."""
        installer = load('authority_installer_main', Path(__file__).with_name('prepare.py'))
        installer.PAYLOAD_HELPERS = payload/'physical-runner'
        prepared = json.loads((payload.parent/'PREPARED.json').read_text())
        source_root = Path(prepared['source_root_operand'])
        source_manifest = Path(prepared['root_source_manifest_operand'])
        require(source_root.is_dir() and source_manifest.is_file(),
                'prepare operands for the accepted-main control are live')
        tool = payload/'source'/installer.NATURAL_TOOL
        package = {name: pin(tool/name) for name in installer.NATURAL_RUNTIME}
        value = receipt_value()
        value['package'] = package
        root = temp/'accepted-main'
        root.mkdir()
        for name, basename in natural_data.PIN_BASENAME.items():
            data = ('accepted-main-'+name).encode()
            path = root/basename
            path.write_bytes(data)
            value[name] = {'path': str(path), 'bytes': len(data),
                           'sha256': hashlib.sha256(data).hexdigest()}
        raw = json.dumps(value).encode()
        receipt = root/'accepted-receipt.json'
        receipt.write_bytes(raw)
        extent, sha = len(raw), hashlib.sha256(raw).hexdigest()
        out = temp/'accepted-main-payload'
        archive = temp/'accepted-main.tar'
        command = [sys.executable, '-B', str(Path(__file__).with_name('prepare.py')),
                   '--payload', str(out), '--archive', str(archive),
                   '--source-root', str(source_root),
                   '--source-manifest', str(source_manifest),
                   '--accepted-receipt', str(receipt),
                   '--accepted-receipt-bytes', str(extent),
                   '--accepted-receipt-sha256', sha]
        completed = subprocess.run(command, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(completed.returncode == 0,
                'accepted-main prepare exit: ' + completed.stderr.decode(errors='replace')[-400:])
        authority = json.loads((out/'natural-authority.json').read_text())
        require(authority['status'] == natural_data.ACCEPTED
                and authority['receipt'] == {'bytes': extent, 'sha256': sha}
                and authority['package'] == package
                and set(authority['artifacts']) == set(natural_data.NATURAL_PINS)
                and authority['artifacts'] == {name: {k: value[name][k] for k in ('bytes', 'sha256')}
                                               for name in natural_data.NATURAL_PINS}
                and authority['external_operand'] == str(receipt),
                'prepare accepted branch installs the exact external receipt and nine-file package')
        prepared_out = json.loads((out.parent/'PREPARED.json').read_text())
        require(prepared_out['natural_authority_status'] == natural_data.ACCEPTED
                and prepared_out['natural_capture_pins_absent'] is False,
                'accepted-main PREPARED records the installed external authority')

    def natural_report_contract():
        fixtures, fixture_pin = actual_fixtures()
        require(set(fixtures) == set(natural_semantic.MODES) and fixture_pin['bytes'] > 0,
                'six actual serde source fixtures')
        for mode in natural_semantic.MODES:
            natural_semantic.validate_natural(mode, fixtures[mode], 'ok')

        def copy(mode):
            return json.loads(json.dumps(fixtures[mode]))

        def healthy(**changes):
            value = copy('healthy')
            value.update(changes)
            return value

        expect_refusal(lambda: natural_semantic.validate_natural(
            'healthy', healthy(status='REFUSED_UNQUALIFIED'), 'ok'), 'a refused report')
        for value, why in (
                (healthy(physical_reserved=32*GIB), 'shared 32GiB physical budget'),
                (healthy(host_reserved=32*GIB), 'a shared 32GiB host budget'),
                (healthy(physical_fenced=True), 'a healthy report fenced'),
                (healthy(lifecycle_cases_qualified=True),
                 'a lifecycle claim from one diagnostic report'),
                (healthy(diagnostic_inventory=dict(
                    fixtures['healthy']['diagnostic_inventory'], peak_bytes=512*1024**2)),
                 'a diagnostic inventory above the 384MiB harness'),
                (healthy(trained_native_natural_serving_or_performance_qualified=True),
                 'a serving claim from a diagnostic')):
            expect_refusal(lambda value=value: natural_semantic.validate_natural(
                'healthy', value, 'ok'), why)
        short = copy('healthy')
        short['actual']['ordinary_packet']['seed_readback_bytes'] = 12
        expect_refusal(lambda: natural_semantic.validate_natural('healthy', short, 'ok'),
                       'a 12B readback claim without the actual packet')
        cached = copy('cached-unknown')
        cached['physical_fenced'] = False
        expect_refusal(lambda: natural_semantic.validate_natural('cached-unknown', cached, 'ok'),
                       'a cached fault without the physical fence')

    def natural_budget_and_build_seams():
        require(physical_run.LIMITS['healthy'] == (7200, 7200, 64*GIB),
                'healthy nine-call natural worker 7200s/AS64GiB')
        require(all(physical_run.LIMITS[m][:2] == (1800, 1800) for m in natural_semantic.FAULT_MODES
                    | {'session-known', 'snapshot-known'}), 'shorter known/fault cases 1800s')
        require(all(physical_run.LIMITS[m][2] == 64*GIB for m in natural_semantic.MODES),
                'every natural mode keeps the original 64GiB process bound')
        require(physical_run.LIMITS['build'] == (3600, 3600, 32*GIB), 'build keeps its compiler budget')
        require(physical_run.WORK_ALLOWANCE == 600 and physical_run.FINALIZATION_ALLOWANCE == 900,
                'original +600 work / +900 finalization allowances')
        require(launch.LIMITS['healthy'] == 7200 and launch.LIMITS['build'] == 3600
                and all(launch.LIMITS[m] == 1800 for m in natural_semantic.FAULT_MODES
                        | {'session-known', 'snapshot-known'}),
                'launch wall limits match the physical-runner phases')
        planned = json.loads((payload/'physical-runner/planned-cost.json').read_text())['natural_case']
        require(planned['provider_bytes'] == 16*GIB and planned['host_budget_bytes'] == 16*GIB
                and planned['physical_domain_bytes'] == 32*GIB
                and planned['worker_address_space_bytes'] == 64*GIB
                and planned['worker_cpu_seconds'] == 7200
                and planned['supervisor_alarm_seconds'] == 7800
                and planned['finalization_seconds'] == 900
                and planned['work_allowance_seconds'] == 600,
                'planned cost separates the Q-only 16/16 domain from the 64GiB AS')
        require(physical_run.ARTIFACT == 'release/examples/qwen27b_hip_natural', 'natural example artifact')
        require('--example' in physical_run.CARGO_ARGV
                and physical_run.CARGO_ARGV[physical_run.CARGO_ARGV.index('--example')+1] == 'qwen27b_hip_natural'
                and 'gemma4-hip-qualification' not in ' '.join(physical_run.CARGO_ARGV), 'no gemma feature')
        require(physical_run.FINAL_BASE not in physical_run.PREDECESSOR_BASES, 'fresh natural base')
        require(physical_run.PLANNED_CALLS == {'healthy': 9, 'cached-unknown': 3,
                                               'session-known': 1, 'session-release-fault': 1,
                                               'snapshot-known': 1, 'snapshot-release-fault': 1},
                'planned full64 calls per actual mode')
        expected_cache = ('/home/pvl/.cache/abird/zerogravity/qualification'
                          '/qwen27b-hip-main400-v1-p1-source469-20261024-1/xdg-cache')
        for mode in natural_semantic.MODES:
            natural_cache = physical_run.child_env('/b', mode, {'XDG_CACHE_HOME': '/unit'})
            require(natural_cache['XDG_CACHE_HOME'] == str(physical_run.Q_CACHE.parents[2])
                    == expected_cache and natural_cache['XDG_CACHE_HOME'] != '/b/xdg-cache',
                    'natural child opens the actual original qualification cache: '+mode)
        require(physical_run.child_env('/b', 'build', {'XDG_CACHE_HOME': '/unit'})['XDG_CACHE_HOME'] == '/unit',
                'build keeps the unit cache')
        # Synthetic closed build: the actual bind seam accepts it, each mismatch refuses.
        base = temp/'synthetic-base'
        (base/'source').mkdir(parents=True)
        artifact = base/'target'/physical_run.ARTIFACT
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b'n')
        proof = pin(artifact)
        hydrated = {'bytes': 11, 'sha256': 'f'*64}
        packaged = {'bytes': 10, 'sha256': 'e'*64}
        receipt = {'schema': 'zg.hip-source20-natural-x2-build-receipt.v1', 'mode': 'build',
                   'status': 'CLOSED_PASS_SCOPED', 'errors': [], 'observed_exit': 0,
                   'source_after': 'UNCHANGED', 'source_closure': packaged,
                   'hydrated_closure': hydrated, 'artifact_count': 1,
                   'cargo_argv_sha256': physical_run.cargo_argv_sha256(), 'cwd': str(base/'source'),
                   'binary': {'path': str(artifact), 'proof': proof}}
        receipt_pin = None
        (base/'build-receipt.json').write_text(json.dumps(receipt))
        receipt_pin = pin(base/'build-receipt.json')
        (base/'model-binary.json').write_text(json.dumps(
            {'path': str(artifact), 'proof': proof, 'source_closure': packaged,
             'hydrated_closure': hydrated}))
        binary, pins = physical_run.bind_build(base, packaged, hydrated, proof, receipt_pin)
        try:
            require(pins['binary'] == proof and pins['cargo_argv_sha256'] == physical_run.cargo_argv_sha256(),
                    'actual natural build binding')
        finally:
            binary.close()
        bad = dict(receipt, cargo_argv_sha256='0'*64)
        (base/'build-receipt.json').write_text(json.dumps(bad))
        expect_refusal(lambda: physical_run.bind_build(base, packaged, hydrated, proof,
                                                       pin(base/'build-receipt.json')),
                       'a build receipt with a different cargo argv')
        (base/'build-receipt.json').write_text(json.dumps(dict(receipt, source_closure=hydrated)))
        expect_refusal(lambda: physical_run.bind_build(base, packaged, hydrated, proof,
                                                       pin(base/'build-receipt.json')),
                       'a build receipt with the wrong source closure')

    def natural_semantic_dispatch():
        fixtures, _fixture_pin = actual_fixtures()
        out = temp/'semantic-output'
        out.mkdir()
        (out/'execution.log').write_text('actual held natural join accepted\n')
        (out/'model-report.json').write_text(json.dumps(fixtures['healthy']))
        result = physical_run.semantic('healthy', out, temp)
        require(result['natural_positions'] == 2 and 'report' in result,
                'run.semantic dispatches the actual serialized natural fixture')
        (out/'model-report.json').write_text(json.dumps(fixtures['session-known']))
        require(physical_run.semantic('session-known', out, temp)['final_reader'] is True,
                'run.semantic dispatches the actual final-reader fixture')

    def natural_remote_receipt_join():
        """The launch-side join consumes the validated semantic dispatch object."""
        case = owner_cleanup_case(owner)
        state = case['foreground_cleanup']
        closure_pin = {'bytes': 10, 'sha256': 'e'*64}
        control_pin = {'bytes': 9, 'sha256': 'd'*64}
        hydrated = {'bytes': 11, 'sha256': 'f'*64}
        binding = {'binary': {'bytes': 3, 'sha256': 'a'*64},
                   'build_receipt': {'bytes': 4, 'sha256': 'b'*64}}
        out = temp/'semantic-join'
        out.mkdir()
        (out/'execution.log').write_text('actual held natural join accepted\n')
        fixtures, _fixture_pin = actual_fixtures()
        (out/'model-report.json').write_text(json.dumps(fixtures['session-known']))
        sem = physical_run.semantic('session-known', out, temp)
        require(sem['trained_native_natural_serving_or_performance_qualified'] is False,
                'remote semantics serialize the validated bounded report flag')
        remote = {'mode': 'session-known', 'status': 'CLOSED_PASS_SCOPED',
                  'control': control_pin, 'control_after': control_pin,
                  'source_closure': closure_pin, 'hydrated_closure': hydrated,
                  'source_after': 'UNCHANGED', 'observed_exit': 0, 'errors': [],
                  'pid': case['pid'], 'foreground_cleanup': state,
                  'log': {'retained': 12, 'total': 12, 'eof': True, 'closed': True},
                  'binary': binding['binary'], 'build_binding': binding,
                  'held_data_files': 70, 'held_natural_reference_files': 5,
                  'qwen_only': True, 'gemma_executed': False,
                  'natural_authority_status': natural_data.ACCEPTED,
                  'natural_feedback_qualified': False, 'teacher_forcing_accepted': False,
                  'data_after': 'UNCHANGED_HELD_FULL_SHA',
                  'natural_after': 'UNCHANGED_HELD_FULL_SHA', 'semantic': sem}
        path = temp/'remote-invocation.json'
        path.write_text(json.dumps(remote))
        require(launch.remote_receipt('session-known', path, closure_pin, control_pin,
                                      binding)['semantic']['mode'] == 'session-known',
                'launch joins the validated remote natural semantics')
        flag = 'trained_native_natural_serving_or_performance_qualified'
        for change, why in (({'semantic': dict(sem, **{flag: True})},
                             'remote semantics claiming serving/performance'),
                            ({'semantic': {k: v for k, v in sem.items() if k != flag}},
                             'remote semantics without the bounded flag'),
                            ({'natural_authority_status': natural_data.AWAITING},
                             'a remote run without an accepted natural capture')):
            path.write_text(json.dumps(dict(remote, **change)))
            expect_refusal(lambda: launch.remote_receipt('session-known', path, closure_pin,
                                                         control_pin, binding), why)
        path.write_text(json.dumps(remote))

    def declared_package_split():
        rust = (payload/'source/crates/zerogravity/examples/qwen27b_hip_matrix/expected.rs')
        text = rust.read_text()
        def table(name):
            body = text[text.index('const '+name):]
            body = body[:body.index('];')]
            import re
            rows = re.findall(r'"([^"]+)"\s*,\s*([\d_]+)\s*,\s*"([0-9a-f]{64})"', body)
            require(len(rows) == 9, name+' pins')
            return {n: (int(b.replace('_', '')), s) for n, b, s in rows}
        natural, reference = table('NATURAL_PACKAGE'), table('REFERENCE_PACKAGE')
        require(natural != reference, 'natural and teacher-forcing packages are separate authorities')
        require(natural['owned_container_v4.py'] == reference['owned_container_v4.py']
                and natural['owned_container_v4.py'][1] == OWNER_SHA, 'same single original owner helper')
        require(natural['source-pin.json'] == reference['source-pin.json']
                and natural['source_owner.py'] == reference['source_owner.py']
                and natural['archive.py'] == reference['archive.py'], 'unchanged original helper bytes')
        prepared = json.loads((payload.parent/'PREPARED.json').read_text())
        frozen = prepared['natural_tool_package_pins']
        require(set(frozen) == set(natural) | {'README.md', 'test_contract.py'}
                and prepared['natural_tool_package_files'] == 11,
                'frozen natural producer package: nine runtime + tests/docs')
        union_prefix = 'tools/qwen27b-natural-full-window-reference/'
        for name, (size, sha) in natural.items():
            path = payload/'source'/union_prefix/name
            data = path.read_bytes()
            require((len(data), hashlib.sha256(data).hexdigest()) == (size, sha)
                    and frozen[name] == {'bytes': size, 'sha256': sha}, 'natural pin '+name)
        authority = json.loads((payload/'natural-authority.json').read_text())
        require(authority['schema'] == 'zg.natural-reference-authority.v1'
                and authority['status'] == 'AWAITING_ACCEPTED_ORIGINAL_CPU_CAPTURE'
                and authority['receipt'] is None and authority['artifacts'] is None
                and authority['package'] == {name: {'bytes': size, 'sha256': sha}
                                             for name, (size, sha) in natural.items()},
                'external natural authority holds the frozen package and no capture pins')
        expect_refusal(lambda: natural_data.load_authority(payload),
                       'an authority without an accepted original CPU capture')
        # The frozen tool's own tests run from the same source union, against
        # the frozen bytes only; no mutable natural-tool reach-back.
        tool = payload/'source'/'tools/qwen27b-natural-full-window-reference'
        done = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'test_contract'],
                              cwd=tool, capture_output=True, text=True,
                              env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                                   'ZG_EXPECTED_RS': str(payload/'source'/EXPECTED_RS),
                                   'ZG_REFERENCE_TOOL': str(payload/'source/tools/'
                                                           'qwen27b-full-window-reference')})
        require(done.returncode == 0 and 'skipped' not in done.stderr
                and 'Ran 11 tests' in done.stderr,
                'frozen natural tool tests against frozen bytes: '+done.stderr[-2000:])

    def prepared_explicit_source_manifest():
        prepared = json.loads((payload.parent/'PREPARED.json').read_text())
        require(prepared['status'] == 'QWEN27B_NATURAL_X2_UNIT_PREPARED_REMOTE_HELD'
                and prepared['base'] == physical_run.FINAL_BASE
                and 'hip-multiset-source20-20261024-2' in prepared['predecessor_bases']
                and prepared['source_files'] == prepared['native_source_files'] + prepared['natural_source_files']
                and prepared['natural_source_files'] == prepared['natural_tool_package_files'] == 11
                and prepared['source_union_is_native_plus_natural'] is True
                and prepared['source_union_derived_count'] == prepared['source_files']
                and prepared['natural_capture_pins_absent'] is True
                and prepared['natural_tool_package_is_separate_authority'] is True
                and prepared['fresh_source_manifest_used'] is True
                and prepared['source_union_basis'] == 'fresh_complete_native_manifest',
                'prepared unit: derived native+natural union from one explicit manifest')
        source_manifest = Path(prepared['root_source_manifest_operand'])
        require(source_manifest.is_file()
                and pin(source_manifest) == prepared['root_source_manifest']
                and prepared['root_source_manifest_operand'].endswith('final-native-manifest.json')
                and prepared['source_root_operand'].endswith('fresh-source')
                and prepared['source_root_operand'] != prepared['source_root'],
                'explicit complete native manifest and source-root operands')
        assembled, assembled_pin = parsed(payload/'assembled-source.json')
        union = assembled['files']
        native = [k for k in union if not k.startswith('tools/qwen27b-natural-full-window-reference/')]
        natural = [k for k in union if k.startswith('tools/qwen27b-natural-full-window-reference/')]
        require(assembled_pin == prepared['source_manifest']
                and len(union) == assembled['source_count'] == prepared['source_files']
                and len(native) == prepared['native_source_files'] == assembled['native_source_count']
                and len(union) == len(native) + len(natural)
                and len(natural) == prepared['natural_source_files'] == 11
                and assembled['root_source_manifest'] == prepared['root_source_manifest']
                and assembled['implementation_qualified'] is False,
                'assembled source manifest derives union and root pin mechanically')
        closure, closure_pin = parsed(payload/'source-closure.json')
        require(closure_pin == prepared['source_closure']
                and closure['source_manifest'] == assembled_pin
                and closure['root_source_manifest'] == prepared['root_source_manifest']
                and closure['parent_frozen_source_manifest'] == prepared['root_source_manifest']
                and closure['source_count'] == prepared['source_files']
                and closure['native_source_count'] == prepared['native_source_files']
                and closure['natural_source_count'] == 11,
                'closure binds the explicit source manifest and derived counts')
        # No data/model/cache payload: nothing large, no corpus member.
        for path in payload.rglob('*'):
            if path.is_file() and 'source' not in path.relative_to(payload).parts[:1]:
                require(path.stat().st_size <= 40*1024**2, 'bounded payload member '+str(path))
                require(path.suffix not in ('.safetensors', '.gguf'), 'no model weights')
                require(path.name != 'raw.bin', 'no natural corpus bytes in the controller payload')

    def control_invariants():
        require(pin(payload/'source/tools/qwen27b-full-window-reference/owned_container_v4.py')['sha256'] == OWNER_SHA,
                'exact original owner helper')
        require(pin(payload/'runner/sigterm_cancellation.py')['sha256'] == CANCEL_SHA,
                'exact original cancellation guard')
        owner_helper = payload/'source/tools/qwen27b-natural-full-window-reference/owned_container_v4.py'
        require(owner_helper.is_file() and pin(owner_helper)['sha256'] == OWNER_SHA,
                'single original owner helper in the frozen natural union (no split package)')
        for name in ('natural_data.py', 'natural_semantic.py'):
            require((payload/'physical-runner'/name).is_file(), name+' present')
        for stale in ('runner/checks.py', 'runner/environment.py', 'runner/modes.py',
                      'runner/run.py', 'runner/source-expressions.py'):
            require(not (payload/stale).exists(), 'no stale predecessor controller: '+stale)

    check('real-local-TERM-twoKILL-before-oneReap-group-absent', real_cleanup)
    check('real-child-log-full-EOF', real_log_eof)
    check('real-archive-extraction-derived-native+natural-bytes-modes-under-umask022', real_archive_extraction)
    check('natural-reference-five-fd-lease-and-refusals', natural_lease)
    check('natural-authority-installer-external-pin-and-refusals', authority_installer_controls)
    check('natural-authority-installer-prepare-accepted-main', authority_prepare_accepted_main)
    check('natural-report-contract-and-16-16-budget-refusals', natural_report_contract)
    check('natural-Q-only-budget-and-build-binding-refusals', natural_budget_and_build_seams)
    check('run-semantic-dispatches-the-natural-report-contract', natural_semantic_dispatch)
    check('natural-remote-receipt-join', natural_remote_receipt_join)
    check('natural-and-teacher-forcing-package-authorities-are-separate', declared_package_split)
    check('prepared-explicit-source-manifest-derived-native-union-natural11', prepared_explicit_source_manifest)
    check('controller-payload-invariants', control_invariants)
    results['status'] = 'PASS_GPU_FREE_GROUPED_CONTROL' if not results['errors'] else 'FAIL_GPU_FREE_GROUPED_CONTROL'
    (args.output/'CONTROL-RESULT.json').write_text(json.dumps(results, indent=2)+'\n')
    print('CONTROL-CHECKS '+results['status'])
    for error in results['errors']:
        print(error)
    temporary.cleanup()
    return 0 if results['status'] == 'PASS_GPU_FREE_GROUPED_CONTROL' else 1


if __name__ == '__main__':
    raise SystemExit(main())
