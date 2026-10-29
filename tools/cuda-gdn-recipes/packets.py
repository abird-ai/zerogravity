"""Actual selected GDN formal/packet joins and explicit control coverage."""
import bisect
import re
import struct

from common import require, sha


def active_arguments(kernel, launch):
    indices = []
    for key in kernel['constants']:
        match = re.fullmatch(r'\((\d+),\)', key)
        require(match is not None, 'actual constexpr argument index')
        index = int(match[1])
        require(index < len(kernel['arg_names']) and index not in indices, 'unique bounded constexpr index')
        indices.append(index)
    require(len(launch['arguments']) == len(kernel['arg_names']), 'whole actual original argument cohort')
    return [(name, launch['arguments'][i], kernel['signature'][name])
            for i, name in enumerate(kernel['arg_names']) if i not in indices]


def join_packet(kernel, launch, marker, actual, driver, arena):
    require(actual['call'] == marker['call'] == launch['call'] and actual['entry'] == kernel['name']
            and actual['module_sha256'] == marker['cubin_sha256']
            and actual['grid'] == launch['grid'] + [1] * (3 - len(launch['grid']))
            and actual['block'] == [kernel['metadata']['num_warps'] * 32, 1, 1]
            and actual['dynamic_shared'] == kernel['metadata']['shared']
            and actual['stream'] == launch['stream'], 'same original GDN native image/entry/launch/stream')
    event = driver['events'][actual['event']]
    require(event['sequence'] == actual['event'] and event['scope'] == actual['call']
            and event['function'] == actual['function'] and event['complete'] is True
            and event['decoded'] is True and event['status'] == 0 and event['launch'] is True
            and event['grid'] == actual['grid'] and event['block'] == actual['block']
            and event['shared'] == actual['dynamic_shared'] and event['stream'] == actual['stream']
            and event['attributes'] == actual['attributes'] == 0, 'complete actual selected native event')
    function = next((row for row in driver['functions'] if row['index'] == actual['function']), None)
    module = next((row for row in driver['modules'] if row['index'] == actual['module']), None)
    require(function is not None and module is not None and function['module'] == module['index']
            and module['sha256'] == marker['cubin_sha256'] and module['bytes'] == marker['cubin_bytes']
            and module['ready'] is True and function['context'] == module['context'] == event['context']
            and function['name'] == function['callback_name'] == kernel['name']
            and function['reflected'] is True and function['prelaunch_bound'] is True
            and function['prelaunch_binding_kind'] in (1, 2)
            and function['prelaunch_binding_order'] < event['order'],
            'same complete prelaunch-bound actual GDN whole image/function')
    packet = actual['parameters']
    require(type(packet) is dict and event['args'] == [packet['arena_offset'], packet['bytes']],
            'actual complete packed argument arena extent')
    offset, count = packet['arena_offset'], packet['bytes']
    require(type(offset) is type(count) is int and offset >= 0 and 0 < count <= 4096
            and offset + count <= arena['bytes'], 'bounded actual packed formals')
    import os
    raw = os.pread(arena['fd'], count, offset)
    require(len(raw) == count and sha(raw) == packet['sha256'], 'actual captured complete packet digest')
    fields = packet['fields']
    require(marker['declared_parameters'] == function['params'] == [[f['offset'], f['bytes']] for f in fields],
            'same declared/actual reflected complete parameter layout')
    end = 0
    for field in fields:
        at, size = field['offset'], field['bytes']
        require(type(at) is type(size) is int and at >= end and 0 < size <= 16 and at + size <= count
                and field['hex'] == raw[at:at + size].hex(), 'each complete actual formal byte slice')
        end = at + size
    require(end == count == function['extent'] and len(fields) == len(active_arguments(kernel, launch)) + 2
            and all(f['bytes'] == 8 and int(f['hex'], 16) == 0 for f in fields[-2:]),
            'whole actual argument and hidden-scratch ABI')
    for (name, value, kind), field in zip(active_arguments(kernel, launch), fields):
        if kind.startswith('*'):
            require(field['bytes'] == 8 and (value is None or isinstance(value, dict)), 'actual pointer formal')
            expected = 0 if value is None else value['pointer']
            require(int.from_bytes(bytes.fromhex(field['hex']), 'little') == expected,
                    'actual original pointer bits: ' + name)
        else:
            require(kind in ('i32', 'i64', 'fp32') and type(value) in (int, float, bool), 'closed actual scalar')
            size = 8 if kind == 'i64' else 4
            require(field['bytes'] == size, 'actual original scalar width')
            expected = struct.pack('<f', value) if kind == 'fp32' else (int(value) % (1 << (8 * size))).to_bytes(size, 'little')
            require(bytes.fromhex(field['hex']) == expected, 'actual original scalar bits: ' + name)
    return dict(event=actual['event'], call=actual['call'], kernel_hash=kernel['hash'],
                module_sha256=module['sha256'], context=event['context'], stream=actual['stream'],
                packet_sha256=packet['sha256'], fields=fields)


def select_native(native, marker):
    launches = native['launches']
    ids = [row['event'] for row in launches]
    require(ids == sorted(set(ids)), 'ordered unique actual native launch events')
    index = bisect.bisect_left(ids, marker['first_event'])
    require(index < len(launches), 'actual native launch after selected marker')
    return launches[index]


def controls(marker, launch, kernel, stage, rows):
    values = marker['controls']
    expected = ({'cu_seqlens', 'chunk_offsets' if stage == 'H' else 'chunk_indices'}
                if launch['call'] == 0 and stage in ('Cumsum', 'Kkt', 'Solve', 'Recompute', 'H', 'Output') else set())
    require(set(values) == expected, 'exact original captured GDN consumed-control cohort')
    views = dict(zip(kernel['arg_names'], launch['arguments']))
    result = []
    for name, value in values.items():
        view = value['view']
        require(name in views and views[name] == view and view['device'] == 'cuda:0'
                and view['dtype'] in ('torch.int32', 'torch.int64'), 'same native-bound captured control view')
        raw = bytes.fromhex(value['hex'])
        width, kind = (4, 'i') if view['dtype'] == 'torch.int32' else (8, 'q')
        require(len(raw) == view['logical_bytes'] and 0 < len(raw) <= 32 and len(raw) % width == 0,
                'complete actual captured integral control bytes')
        actual = list(struct.unpack('<' + kind * (len(raw) // width), raw))
        chunks = (rows + 63) // 64
        wanted = ([0, rows] if name == 'cu_seqlens' else [0, chunks] if name == 'chunk_offsets'
                  else [v for chunk in range(chunks) for v in (0, chunk)])
        require(actual == wanted and (name != 'chunk_indices' or view['shape'] == [chunks, 2]),
                'actual source-owned GDN control contents')
        result.append(dict(kind='ACTUAL_CAPTURED_BYTES', name=name, values=actual, view=view,
                           hex=value['hex'], native_pointer_join_proven=True))
    return result
