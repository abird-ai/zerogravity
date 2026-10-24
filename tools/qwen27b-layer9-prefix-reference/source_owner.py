"""All text roles held from original FDs; at most one mapped module group."""
import mmap
import os
from pathlib import Path
import stat
from contract import (require, read, metadata, specs, roles, require_config,
                      fingerprint, fd_sha, source_header, role_view)


class SourceOwner:
    def __init__(self, plan, pins):
        self.plan = plan
        self.files = specs(plan, pins)
        self.config = metadata('/source/config.json', pin=self.files['config.json'])['text_config']
        require_config(self.config)
        self.index = metadata('/source/model.safetensors.index.json', pin=self.files['model.safetensors.index.json'])['weight_map']
        self.roles = roles()
        names = {r['name'] for r in self.roles}
        actual = {n for n in self.index if n.startswith('model.language_model.') or n == 'lm_head.weight'}
        require(names == actual and len(self.index) == 1199, 'complete851 original text namespace, vision excluded')
        self.fds, self.before, self.headers, self.views = {}, {}, {}, {}
        self.readsets, self.counts, self.events = {}, {}, []
        self.active = None
        self.max_mapping_bytes = 0
        for name in sorted({self.index[n] for n in names}):
            require(Path(name).name == name and name in self.files, 'contained original shard')
            fd = os.open(Path('/source')/name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            self.fds[name] = fd
            m = os.fstat(fd)
            require(stat.S_ISREG(m.st_mode) and m.st_size == self.files[name]['bytes'], 'exact source extent')
            self.before[name] = fingerprint(m)
            require(fd_sha(fd, 0, m.st_size) == self.files[name]['sha256']
                    and fingerprint(os.fstat(fd)) == self.before[name], 'whole original source before any math')
            self.headers[name] = source_header(fd, self.files[name])
        require(len(self.fds) == 18, 'all18 genuine text source shards')
        for role in self.roles:
            shard = self.index[role['name']]
            start, header = self.headers[shard]
            self.views[role['name']] = role_view(start, header, role, self.files[shard]['bytes'])

    def bind(self, group, module, torch, tag):
        # Keep versioned source Parameters; the original caller's math remains
        # in inference_mode. Inference-created Tensors have no version counter.
        with torch.inference_mode(False):
            require(self.active is None, 'exclusive bounded source group')
            selected = [r for r in self.roles if r['group'] == group]
            require(selected and all(p.is_meta for p in module.parameters()), 'unloaded original module before binding')
            total = sum(self.views[r['name']]['mapping_bytes'] for r in selected)
            require(total <= self.plan['mapping_cap'], 'source group admission before first mapping')
            self.max_mapping_bytes = max(self.max_mapping_bytes, total)
            active = {'group': group, 'maps': {}, 'values': {}, 'versions': {}, 'module': module, 'rows': selected, 'tag': tag.copy()}
            self.active = active
            for role in selected:
                name = role['name']
                view = self.views[name]
                fd = self.fds[self.index[name]]
                require(fingerprint(os.fstat(fd)) == self.before[self.index[name]], 'same source FD before each mapping')
                mapping = mmap.mmap(fd, view['mapping_bytes'], access=mmap.ACCESS_COPY, offset=view['aligned'])
                active['maps'][name] = mapping
                dtype = torch.bfloat16 if view['dtype'] == 'BF16' else torch.float32
                key = role['local'] if role['layer'] is not None else 'weight'
                value = torch.frombuffer(mapping, dtype=dtype, count=view['bytes']//value_width(view['dtype']),
                                         offset=view['view_offset']).reshape(role['shape'])
                require(value.is_contiguous() and value.device.type == 'cpu', 'direct raw original source view')
                active['values'][key] = value
                actual = fd_sha(fd, view['absolute'], view['bytes'])
                if name in self.readsets:
                    require(self.readsets[name] == actual, 'same source role bytes across genuine calls')
                self.readsets[name] = actual
                self.counts[name] = self.counts.get(name, 0)+1
            module.load_state_dict(active['values'], strict=True, assign=True)
            current = dict(module.named_parameters())
            require(set(current) == set(active['values']), 'no missing/random parameter in original module')
            for key, value in active['values'].items():
                p = current[key]
                require(p.data_ptr() == value.data_ptr() and p.dtype == value.dtype and p.shape == value.shape,
                        'no source casts/copies or reassigned roles')
                active['versions'][key] = p._version
            self.events.append({**tag, 'group': group, 'mapping_bytes': total, 'roles': len(selected), 'source_bound': True})

    def release(self, group, module):
        active = self.active
        require(active is not None and active['group'] == group and active['module'] is module,
                'release only exact current source group')
        current = dict(module.named_parameters())
        for key, value in active['values'].items():
            require(current[key].data_ptr() == value.data_ptr() and current[key]._version == active['versions'][key],
                    'actual parameter owner/version unchanged through public forward')
        for role in active['rows']:
            name = role['name']
            view = self.views[name]
            mapped = memoryview(active['maps'][name])
            h = __import__('hashlib').sha256()
            try:
                for i in range(0, view['bytes'], self.plan['source_scan_bytes']):
                    h.update(mapped[view['view_offset']+i:view['view_offset']+min(view['bytes'], i+self.plan['source_scan_bytes'])])
            finally:
                mapped.release()
            require(h.hexdigest() == self.readsets[name], 'complete used source mapping unchanged')
        # CPU public forward has returned. Remove every learned Tensor owner
        # before unmapping; original full TextModel and its loop stay intact.
        module.to_empty(device='meta')
        current.clear()
        active['values'].clear()
        del value
        # The loop-local Tensor from bind() died before this callback.
        for mapping in active['maps'].values():
            mapping.close()
        active['maps'].clear()
        active['module'] = None
        self.active = None

    def finish(self):
        require(self.active is None and len(self.readsets) == 135
                and set(self.counts.values()) == {2}, 'all135 prefix roles used in both independent original traversals')
        for name, fd in self.fds.items():
            require(fingerprint(os.fstat(fd)) == self.before[name]
                    and fd_sha(fd, 0, self.files[name]['bytes']) == self.files[name]['sha256']
                    and os.pread(fd, 1, self.files[name]['bytes']) == b''
                    and fingerprint(os.fstat(fd)) == self.before[name]
                    and fingerprint(os.stat(Path('/source')/name, follow_symlinks=False)) == self.before[name],
                    'same whole source/EOF/current path after all original math')


def value_width(dtype):
    require(dtype in ('BF16', 'F32'), 'closed raw learned source codec')
    return 2 if dtype == 'BF16' else 4
