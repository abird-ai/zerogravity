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
                      record_specs, record_metadata, prefix_bytes)
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


def validate_records(rows, raw, plan):
    expected = list(record_specs())
    require(len(rows) == len(expected) == 96 and raw.bytes == prefix_bytes(), 'complete full64 cohort')
    offset, seen = 0, {}
    for row, spec in zip(rows, expected):
        repeat, call, teacher, layer, label, shape, dtype = spec
        n = record_metadata(row, spec, offset)
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
    require(offset == raw.bytes == prefix_bytes(), 'full raw coverage/no gaps/no trailing data')
    return seen


def validate_calls(calls):
    require(len(calls) == 2, 'two independent initial prefix caches')
    groups = ['embedding']+['layer.'+str(i) for i in range(10)]
    for repeat,row in enumerate(calls):
        require(row['repeat'] == repeat and row['call'] == row['position'] == 0
                and row['teacher'] == 11 and row['cache_frontier'] == 1
                and row['original_module_order'] == groups
                and row['stopped_after_source_unload'] is True and row['full_forward_returned'] is False,
                'genuine prefix order/frontier and stop after source unload')
        for moment in ('before','after'):
            require(len(row[moment]) == 64, 'all original cache owners retained')
            for i,(flag,kind) in enumerate(zip(row[moment],KINDS)):
                require(flag['layer'] == i and flag['kind'] == kind, 'actual source cache kind')
                initialized = moment == 'after' and i <= 9
                names = ('conv','recurrent','previous') if kind == 'linear_attention' else ('key','value')
                require(all(flag[n] is initialized for n in names), 'only original prefix state initialized')


def join_full_reference(held, report, raw):
    pin = metadata(HERE/'full-reference-pin.json')
    accepted_file = held.enter_context(Held(Path(pin['path']), 1024**2))
    require(accepted_file.bytes == pin['bytes'] and accepted_file.sha == pin['sha256'], 'unchanged accepted full64 anchor')
    accepted = accepted_file.json()
    require(accepted['schema'] == 'zg.actual-original27b-full-window-reference.v1'
            and accepted['status'] == 'ACTUAL_ORIGINAL27B_FULL_WINDOW_REFERENCE_ONLY'
            and accepted['text_roles'] == 851 and accepted['cache_domains'] == 112
            and accepted['rust_numerical_parity_qualified'] is False,
            'original-only full64 anchor')
    full_manifest = held.enter_context(Held(Path(accepted['manifest']['path']),16*1024**2))
    full_raw = held.enter_context(Held(Path(accepted['raw']['path']),1024**3))
    require(all(owner.bytes == accepted[name]['bytes'] and owner.sha == accepted[name]['sha256']
                for name,owner in [('manifest',full_manifest),('raw',full_raw)]), 'same whole full64 manifest/raw')
    old = full_manifest.json()
    require(old['status'] == 'CAPTURED_ORIGINAL27B_FULL_WINDOW_REQUIRES_VALIDATION', 'healthy original full traversal')
    indexed = {(r['repeat'],r['call'],r['layer'],r['label']):r for r in old['records']}
    joins = 0
    for row in report['records']:
        label = row['label']
        if label == 'layer9.input':
            key = (row['repeat'],0,8,'layer_output')
        elif label.startswith('layer9.'):
            continue
        else:
            key = (row['repeat'],0,row['layer'],label)
        original = indexed[key]
        require(original['bytes'] == row['bytes'] and original['dtype'] == row['dtype']
                and original['shape'] == row['shape']
                and original['sha256'] == row['sha256']
                and full_raw.read(original['offset'],original['bytes']) == raw.read(row['offset'],row['bytes']),
                'prefix producer/state same bytes as original complete computed trajectory')
        joins += 1
    require(joins == 66, 'all33 overlapping embedding/position/layer/input/state records per repeat')
    return {'accepted':accepted_file.pin(),'manifest':full_manifest.pin(),'raw':full_raw.pin(),
            'overlapping_records':joins,'all_overlapping_full_records_exact':True}


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
        require(inv['status'] == 'EXECUTED_ORIGINAL27B_LAYER9_PREFIX_REQUIRES_VALIDATION'
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
                and inv['owned_name'] == 'zg-original27-layer9-prefix-'+nonce, 'exact original CPU/name/nonce owner')
        require(inv['argv'] == argv(owned_helper(), output, plan, environment(cpus, identity), inv['owned_name'], nonce),
                'actual immutable image/env/source/package mounts/resource/entrypoint')
        inspect_file = held.enter_context(Held(output/'image-inspect.json', 1024**2))
        inspected, = inspect_file.json()
        require(inspected['Id'] == IMAGE.removeprefix('sha256:') and inspected['Digest'] == IMAGE_DIGEST
                and inv['image_id'] == IMAGE and inv['image_digest'] == IMAGE_DIGEST, 'original immutable runtime')
        require(report['schema'] == 'zg.original27b-layer9-prefix-capture.v1'
                and report['status'] == 'CAPTURED_ORIGINAL27B_LAYER9_PREFIX_REQUIRES_VALIDATION'
                and report['package'] == identity and report['scope'] == scope()
                and report['parameter_mode'] == 'raw-source-assign-no-cast'
                and all(report[k] == plan[k] for k in ('snapshot', 'repository', 'revision')), 'actual raw source producer scope')
        require(report['torch'] == '2.13.0+cu129' and report['torch_git'] == TORCH_GIT
                and report['modeling_sha256'] == MODEL_SHA and report['cache_utils_sha256'] == CACHE_SHA
                and report['affinity'] == cpus and report['threads'] == 4 and report['interop_threads'] == 1
                and report['original_full_forward'] is False and report['cache_domains'] == 18 and report['used_source_roles'] == 135
                and report['full_hf_conv_width'] == 4 and report['all_851_source_headers_validated'] is True
                and report['all_original_source_and_mapped_bytes_unchanged'] is True
                and report['source_owner_retained_until_process_exit'] is True, 'actual original code/math/source owners')
        require(report['raw']['file'] == 'raw.bin', 'contained fixed raw filename')
        raw_file = held.enter_context(Held(output/'captured/raw.bin', plan['archive_cap']))
        require(raw_file.bytes == report['raw']['bytes'] and raw_file.sha == report['raw']['sha256'], 'whole held raw identity')
        validate_calls(report['calls'])
        validate_records(report['records'], raw_file, plan)
        full_join = join_full_reference(held, report, raw_file)
        require(set(p.name for p in (output/'captured').iterdir()) == {'raw.bin', 'manifest.json'}, 'closed capture file cohort')
        fds, source_rows = source_open(plan, pins)
        try:
            require(source_rows == inv['source_before'], 'fresh whole source identities agree with actual original run')
            views, readsets, counts = report['source_views'], report['source_readsets'], report['source_counts']
            required = roles()
            names = {r['name'] for r in required}
            used = {r['name'] for r in required if r['group'] == 'embedding' or r['layer'] is not None and r['layer'] <= 9}
            require(set(views) == names and len(used) == 135 and set(readsets) == set(counts) == used
                    and set(counts.values()) == {2}, 'all851 headers and exact135 used original prefix roles')
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
                require(view == views[name] and (name not in used or fd_sha(fds[shard], view['absolute'], view['bytes']) == readsets[name]),
                        'independent complete raw parameter header/bytes/role/codec join')
                totals[role['group']] = totals.get(role['group'], 0)+view['mapping_bytes']
            expected_events = [(r, c, group) for r in range(2) for c in [0]
                for group in ['embedding']+['layer.'+str(i) for i in range(10)]]
            events = report['source_events']
            require(len(events) == len(expected_events) == 22 and report['maximum_mapped_bytes'] == max(totals[g] for _,_,g in expected_events) <= plan['mapping_cap'], 'actual one-group source mapping bound')
            for event, (repeat, call, group) in zip(events, expected_events):
                require(event['repeat'] == repeat and event['call'] == event['position'] == call
                        and event['teacher'] == (11, 353)[call] and event['group'] == group
                        and event['mapping_bytes'] == totals[group] and event['source_bound'] is True
                        and event['roles'] == sum(r['group'] == group for r in required), 'all actual original module/source events')
            source_after(fds, source_rows, plan['source_directory'])
        finally:
            for fd in fds.values():
                os.close(fd)
        require(package() == identity, 'unchanged producer/validator source after long scans')
        # Recheck the same held snapshots after source IO; never pin replacement paths.
        result = {'schema': 'zg.actual-original27b-layer9-prefix-reference.v1',
            'status': 'ACTUAL_ORIGINAL27B_LAYER9_PREFIX_REFERENCE_ONLY',
            **{k: plan[k] for k in ('snapshot', 'repository', 'revision', 'config_sha256', 'index_sha256', 'parameter_mode')},
            'scope': scope(), 'calls': 1, 'original_repetitions': 2, 'text_roles': 135, 'layers': 10,
            'original_object_layers':64, 'cache_domains': 18, 'full_hf_conv_width': 4, 'records': 96, 'full_reference_join':full_join, 'raw_bytes': raw_file.bytes,
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
