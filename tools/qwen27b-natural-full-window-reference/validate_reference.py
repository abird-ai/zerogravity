"""Fresh ROOT-only full original producer validation. Never a Rust parity grant."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from contract import (HERE, IMAGE, IMAGE_DIGEST, MODEL_SHA, CACHE_SHA, TORCH_GIT, KINDS,
                      require, metadata, write, package, specs, roles, scope, extent,
                      finite, fingerprint, fd_sha, unique, source_header, role_view,
                      record_specs, record_metadata, argmax_bf16)
from run_reference import source_open, source_after, owned_helper, environment, argv


class Held:
    """Streaming held FD. Metadata snapshots and accepted raw pin never remint."""
    def __init__(self, path, cap):
        self.path = Path(path)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            m = os.fstat(self.fd)
            self.before, self.bytes = fingerprint(m), m.st_size
            require(stat.S_ISREG(m.st_mode) and 0 < self.bytes <= cap, 'bounded held artifact')
            self.sha = fd_sha(self.fd, 0, self.bytes)
            self.verify()
        except BaseException:
            os.close(self.fd)
            raise
    def __enter__(self):
        return self
    def __exit__(self, *_):
        os.close(self.fd)
    def read(self, offset, count):
        require(type(offset) is int and type(count) is int and 0 <= offset < self.bytes
                and 0 < count <= 4*1024**2 and offset+count <= self.bytes, 'bounded record read')
        raw = os.pread(self.fd, count, offset)
        require(len(raw) == count and fingerprint(os.fstat(self.fd)) == self.before, 'same complete held record')
        return raw
    def json(self):
        require(self.bytes <= 16*1024**2, 'bounded metadata snapshot')
        raw = os.pread(self.fd, self.bytes, 0)
        require(len(raw) == self.bytes and hashlib.sha256(raw).hexdigest() == self.sha, 'same validated metadata snapshot')
        return json.loads(raw, object_pairs_hook=unique)
    def verify(self):
        require(fingerprint(os.fstat(self.fd)) == self.before
                and fingerprint(os.stat(self.path, follow_symlinks=False)) == self.before, 'same artifact FD/path before scan')
        require(fd_sha(self.fd, 0, self.bytes) == self.sha and os.pread(self.fd, 1, self.bytes) == b''
                and fingerprint(os.fstat(self.fd)) == self.before
                and fingerprint(os.stat(self.path, follow_symlinks=False)) == self.before, 'same whole artifact SHA/EOF after scan')
    def pin(self):
        self.verify()
        return {'path': str(self.path), 'bytes': self.bytes, 'sha256': self.sha}


def validate_records(rows, raw, plan, calls):
    inputs = natural_inputs(calls)
    expected = list(record_specs())
    require(len(rows) == len(expected) == 1040 and raw.bytes == plan['expected_raw_bytes'], 'complete full64 cohort')
    offset, seen = 0, {}
    for row, spec in zip(rows, expected):
        repeat, call, layer, label, shape, dtype = spec
        n = record_metadata(row, spec, offset, inputs[repeat][call])
        b = raw.read(offset, n)
        require(hashlib.sha256(b).hexdigest() == row['sha256'] and finite(b, dtype), 'complete finite original record')
        key = (repeat, call, layer, label)
        require(key not in seen, 'unique actual record')
        if dtype == 'I64':
            require(struct.unpack('<qqq', b) == (call,)*3, 'actual original canonical positions')
        if repeat:
            previous = seen[(0, call, layer, label)]
            require(previous['bytes'] == n and raw.read(previous['offset'], n) == b, 'independent complete forward repeats exact')
        if call == 1 and label.startswith('before.'):
            previous = seen[(repeat, 0, layer, 'after.'+label.split('.', 1)[1])]
            require(previous['bytes'] == n and raw.read(previous['offset'], n) == b, 'all112 actual prior domains carried into cached call')
        seen[key] = row
        offset += n
    require(offset == raw.bytes == 934776928, 'full raw coverage/no gaps/no trailing data')
    selections, logits_records = {}, {}
    for repeat in range(2):
        for call in range(2):
            index = next(i for i, s in enumerate(expected)
                         if s[0] == repeat and s[1] == call and s[3] == 'logits')
            spec, row = expected[index], rows[index]
            logits_records[(repeat, call)] = row['sha256']
            selected, bits = argmax_bf16(raw.read(row['offset'], extent(spec[4], spec[5])))
            call_row = calls[repeat*2+call]
            require(selected == call_row['greedy_id_diagnostic_only']
                    and bits == call_row['selected_max_bits'],
                    'independent integer fullV selection/bit join with actual original call')
            selections[(repeat, call)] = (selected, bits)
        require(calls[repeat*2+1]['teacher'] == selections[(repeat, 0)][0],
                'actual natural call1 input equals independent call0 original argmax')
    join_call_records(calls, logits_records)
    return seen, selections


def join_call_records(calls, logits_records):
    """Each call fullV/predecessor pin is the actual corresponding raw record."""
    require(set(logits_records) == {(r, c) for r in range(2) for c in range(2)},
            'every cohort call has one actual fullV record')
    for repeat in range(2):
        for call in range(2):
            row = calls[repeat*2+call]
            require(row['logits_record_sha256'] == logits_records[(repeat, call)],
                    'call fullV identity is the actual corresponding record')
            if call == 0:
                require(row['predecessor_logits_sha256'] is None,
                        'seed call has no natural predecessor')
            else:
                require(row['predecessor_logits_sha256'] == logits_records[(repeat, 0)],
                        'call1 predecessor joins this cohort actual call0 record')
    return True


def natural_inputs(calls):
    """Declared per-cohort inputs: seed11 then that cohort's own call0 argmax."""
    require(len(calls) == 4, 'two independent original natural cohorts')
    inputs = []
    for repeat in range(2):
        cohort = calls[repeat*2:repeat*2+2]
        require(cohort[0]['teacher'] == 11, 'natural call0 is the original seed11')
        inputs.append([11, cohort[1]['teacher']])
    return inputs


def validate_calls(calls):
    require(len(calls) == 4, 'two independent original two-call cache cohorts')
    groups = ['embedding']+['layer.'+str(i) for i in range(64)]+['final', 'head']
    for row, (repeat, call) in zip(calls, [(r, c) for r in range(2) for c in range(2)]):
        require(all(type(row[k]) is int for k in ('repeat', 'call', 'position', 'teacher',
                                                  'cache_frontier', 'greedy_id_diagnostic_only',
                                                  'selected_max_bits'))
                and row['repeat'] == repeat and row['call'] == row['position'] == call
                and row['cache_frontier'] == call+1
                and row['original_module_order'] == groups
                and 0 <= row['greedy_id_diagnostic_only'] < 248320
                and 0 <= row['selected_max_bits'] <= 0xffffffff
                and row['selected_max_bits'] & 0xffff == 0
                and (row['selected_max_bits'] >> 23) & 0xff != 0xff
                and row['natural_feedback_used'] is (call == 1), 'actual original natural forward order/frontier/input')
        require(isinstance(row['logits_record_sha256'], str) and len(row['logits_record_sha256']) == 64
                and all(c in '0123456789abcdef' for c in row['logits_record_sha256']),
                'actual original fullV record identity shape')
        if call == 0:
            require(row['predecessor_logits_sha256'] is None, 'seed call has no natural predecessor')
        else:
            require(row['predecessor_logits_sha256'] == calls[repeat*2]['logits_record_sha256'],
                    'call1 feedback predecessor is this cohort original call0 fullV')
    for call in range(2):
        left, right = calls[call], calls[2+call]
        require(left['greedy_id_diagnostic_only'] == right['greedy_id_diagnostic_only']
                and left['logits_record_sha256'] == right['logits_record_sha256'],
                'both original repetitions are byte-identical natural calls')
    declared = natural_inputs(calls)
    for repeat in range(2):
        require(calls[repeat*2+1]['teacher'] == calls[repeat*2]['greedy_id_diagnostic_only'],
                'actual call1 input equals this cohort original call0 selection')
    for row, (repeat, call) in zip(calls, [(r, c) for r in range(2) for c in range(2)]):
        for moment in ('before', 'after'):
            flags = row[moment]
            require(len(flags) == 64, 'every original cache layer initialized disposition')
            initialized = moment == 'after' or call == 1
            for i, (flag, kind) in enumerate(zip(flags, KINDS)):
                require(type(flag['layer']) is int and flag['layer'] == i and flag['kind'] == kind, 'genuine source cache kind')
                names = ('conv', 'recurrent', 'previous') if kind == 'linear_attention' else ('key', 'value')
                require(all(flag[n] is initialized for n in names), 'real original initialization flags; no synthetic initial tensors')
    return declared


def validate(output, receipt):
    with ExitStack() as held:
        identity = package()
        plan, pins = metadata(HERE/'inputs.json'), metadata(HERE/'source-pin.json')
        files = specs(plan, pins)
        require(output.is_absolute() and output.is_dir() and receipt.is_absolute()
                and not receipt.exists() and receipt.parent.is_dir(), 'retired output/fresh receipt')
        receipt = receipt.parent.resolve()/receipt.name
        for old in (HERE, output, Path(plan['source_directory'])):
            require(receipt != old and old not in receipt.parents and receipt not in old.parents, 'receipt disjoint from source/package/capture')
        inv_file = held.enter_context(Held(output/'invocation.json', 16*1024**2))
        report_file = held.enter_context(Held(output/'captured/manifest.json', 16*1024**2))
        inv, report = inv_file.json(), report_file.json()
        require(inv['status'] == 'EXECUTED_ORIGINAL27B_NATURAL_FULL_WINDOW_REQUIRES_VALIDATION'
                and inv['package'] == identity and inv['source_before'] == inv['source_after']
                and inv['source_FDs_closed'] is True and inv['source_FD_cleanup_errors'] == []
                and inv['cleanup_errors'] == [] and inv['exit_code'] == 0
                and inv['owned_container_absent_after_reap'] is True, 'healthy original owned lifetime/source')
        g, log = inv['foreground_cleanup'], inv['log_cleanup']
        require(g['leader_reaped'] is g['group_absent'] is g['all_nonzero_signals_before_reap'] is True
                and g['stage_errors'] == [] and g['kill_attempts_before_reap'] == 2
                and log['pipe_eof'] is log['all_closed'] is True
                and log['total_bytes'] == log['retained_bytes'] <= 4*1024**2, 'TERM/KILL before reap, group absence and full EOF')
        cpus, nonce = inv['expected_cpus'], inv['owned_nonce']
        require(len(cpus) == len(set(cpus)) == 4 and all(type(c) is int and c >= 0 for c in cpus)
                and len(nonce) == 32 and all(c in '0123456789abcdef' for c in nonce)
                and inv['owned_name'] == 'zg-original27-natural-full-window-'+nonce, 'exact original CPU/name/nonce owner')
        require(inv['argv'] == argv(owned_helper(), output, plan, environment(cpus, identity), inv['owned_name'], nonce),
                'actual immutable image/env/source/package mounts/resource/entrypoint')
        inspect_file = held.enter_context(Held(output/'image-inspect.json', 1024**2))
        inspected, = inspect_file.json()
        require(inspected['Id'] == IMAGE.removeprefix('sha256:') and inspected['Digest'] == IMAGE_DIGEST
                and inv['image_id'] == IMAGE and inv['image_digest'] == IMAGE_DIGEST, 'original immutable runtime')
        require(report['schema'] == 'zg.original27b-natural-full-window-capture.v1'
                and report['status'] == 'CAPTURED_ORIGINAL27B_NATURAL_FULL_WINDOW_REQUIRES_VALIDATION'
                and report['package'] == identity and report['scope'] == scope()
                and report['parameter_mode'] == 'raw-source-assign-no-cast'
                and all(report[k] == plan[k] for k in ('snapshot', 'repository', 'revision')), 'actual raw source producer scope')
        require(report['torch'] == '2.13.0+cu129' and report['torch_git'] == TORCH_GIT
                and report['modeling_sha256'] == MODEL_SHA and report['cache_utils_sha256'] == CACHE_SHA
                and report['affinity'] == cpus and report['threads'] == 4 and report['interop_threads'] == 1
                and report['original_full_forward'] is True and report['cache_domains'] == 112
                and report['full_hf_conv_width'] == 4 and report['all_851_source_parameters_raw'] is True
                and report['all_original_source_and_mapped_bytes_unchanged'] is True
                and report['source_owner_retained_until_process_exit'] is True, 'actual original code/math/source owners')
        require(report['raw']['file'] == 'raw.bin', 'contained fixed raw filename')
        raw_file = held.enter_context(Held(output/'captured/raw.bin', plan['archive_cap']))
        require(raw_file.bytes == report['raw']['bytes'] and raw_file.sha == report['raw']['sha256'], 'whole held raw identity')
        declared = validate_calls(report['calls'])
        require(report['natural_inputs'] == declared, 'declared natural cohort inputs join actual calls')
        validate_records(report['records'], raw_file, plan, report['calls'])
        require(set(p.name for p in (output/'captured').iterdir()) == {'raw.bin', 'manifest.json'}, 'closed capture file cohort')
        fds, source_rows = source_open(plan, pins)
        try:
            require(source_rows == inv['source_before'], 'fresh whole source identities agree with actual original run')
            views, readsets, counts = report['source_views'], report['source_readsets'], report['source_counts']
            required = roles()
            names = {r['name'] for r in required}
            require(set(views) == set(readsets) == set(counts) == names and set(counts.values()) == {4}, 'complete851 raw source roles in every genuine call')
            index = metadata(Path(plan['source_directory'])/'model.safetensors.index.json')['weight_map']
            headers = {}
            for name in sorted({index[n] for n in names}):
                headers[name] = source_header(fds[name], files[name])
                require(report['source_FDs'][name] == fingerprint(os.fstat(fds[name])), 'same worker original source FD')
            totals = {}
            for role in required:
                name, shard = role['name'], index[role['name']]
                start, header = headers[shard]
                view = role_view(start, header, role, files[shard]['bytes'])
                require(view == views[name] and fd_sha(fds[shard], view['absolute'], view['bytes']) == readsets[name],
                        'independent complete raw parameter header/bytes/role/codec join')
                totals[role['group']] = totals.get(role['group'], 0)+view['mapping_bytes']
            expected_events = [(r, c, group) for r in range(2) for c in range(2)
                for group in ['embedding']+['layer.'+str(i) for i in range(64)]+['final', 'head']]
            events = report['source_events']
            require(len(events) == len(expected_events) == 268 and report['maximum_mapped_bytes'] == max(totals.values()) <= plan['mapping_cap'], 'actual one-group source mapping bound')
            for event, (repeat, call, group) in zip(events, expected_events):
                require(event['repeat'] == repeat and event['call'] == event['position'] == call
                        and event['teacher'] == declared[repeat][call] and event['group'] == group
                        and event['mapping_bytes'] == totals[group] and event['source_bound'] is True
                        and event['roles'] == sum(r['group'] == group for r in required), 'all actual original module/source events')
            source_after(fds, source_rows, plan['source_directory'])
        finally:
            for fd in fds.values():
                os.close(fd)
        require(package() == identity, 'unchanged producer/validator source after long scans')
        # Recheck the same held snapshots after source IO; never pin replacement paths.
        result = {'schema': 'zg.actual-original27b-full-window-natural-reference.v1',
            'status': 'ACTUAL_ORIGINAL27B_FULL_WINDOW_NATURAL_REFERENCE_ONLY',
            'independent_original_natural_trajectory': True,
            'natural_inputs': declared, 'seed': plan['seed'],
            **{k: plan[k] for k in ('snapshot', 'repository', 'revision', 'config_sha256', 'index_sha256', 'parameter_mode')},
            'scope': scope(), 'calls': 2, 'original_repetitions': 2, 'text_roles': 851, 'layers': 64,
            'cache_domains': 112, 'full_hf_conv_width': 4, 'records': 1040, 'raw_bytes': raw_file.bytes,
            'image_id': IMAGE, 'image_digest': IMAGE_DIGEST,
            'original_source_and_owned_lifetime_closed': True, 'rust_numerical_parity_qualified': False,
            'mkl_exp_producer_qualified': False, 'state_root_native_serving_performance_qualified': False,
            'manifest': report_file.pin(), 'raw': raw_file.pin(), 'invocation': inv_file.pin(),
            'image_inspect': inspect_file.pin(), 'package': identity, 'source_files': source_rows}
        write(receipt, result)
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args()
    validate(args.output, args.receipt)
