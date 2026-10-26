"""Append captured recipes to immutable baseline literals and price their real owners."""
import copy
import hashlib
import json
import re
from literals import declaration, factories, require

GENERATED = 'crates/zerogravity-gpu/src/cuda/native/text_native/generated.rs'
RECORDS = 'crates/zerogravity-gpu/vendor/cubecl-cuda/src/owned_command/cubin/text_policy/records.rs'
COHORT = 'crates/zerogravity-gpu/src/cuda/native/cohort_native/'
CAPTURES = {'qwen08': '39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d',
            'qwen2b': '0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0'}
ALIASES = {'Canonical47CutlassBf16Sm80': 'FixedPromptCutlassBf16Sm80',
           'ShortHfCutlassBf16Sm80': 'TextCutlassBf16Sm80',
           'ShortHfFaCausalCombineSm80': 'TextFaCausalCombineSm80',
           'ShortHfFaSplitCombineSm80': 'TextFaSplitCombineSm80'}


def without_type(value):
    return {key: child for key, child in value.items() if key != 'type'}


def enum(value, family):
    require(type(value) is str and value.startswith(family + '::'), 'exact existing enum family')
    return value.split('::')[1]


def neutral(packet):
    packet = copy.deepcopy(packet)
    packet['image'] = ALIASES.get(packet['image'], packet['image'])
    return packet


def packet_value(value, images, activation_sha):
    require(set(value) == {'type', 'image', 'roles', 'manifest', 'clears'}, 'complete baseline packet literal')
    image = value['image']
    image = None if image is None else enum(image['Some'], 'ProjectionImage')
    by_kind = {ALIASES.get(row['kind'], row['kind']): sha for sha, row in images.items()}
    require(image is None or image in by_kind, 'baseline actual whole-image mapping')
    manifest = without_type(value['manifest'])
    manifest['attributes'] = enum(manifest['attributes'], 'CubinLaunchAttributes')
    manifest['bindings'] = [dict(without_type(row), io=enum(row['io'], 'CubinIo')) for row in manifest['bindings']]
    manifest['parameters'] = [{'offset': row['offset'], 'bytes': len(row['template']),
                               'hex': bytes(row['template']).hex()} for row in manifest['parameters']]
    manifest['fixups'] = [dict(without_type(row), io=enum(row['io'], 'CubinIo')) for row in manifest['fixups']]
    return {'image': image, 'image_sha256': activation_sha if image is None else by_kind[image],
            'roles': [enum(role, 'Role') for role in value['roles']], 'manifest': manifest,
            'clears': [[enum(role, 'Role'), offset, size] for role, offset, size in value['clears']]}


def fingerprint(text, prefix, cohort):
    parts = [(name, row['raw'][row['raw'].index('{'):]) for name, row in
             sorted(factories(text, prefix).items(), key=lambda item: int(item[0].rsplit('_', 1)[1]))]
    upper = prefix.upper()
    parts += [(upper + '_SITES', declaration(text, upper + '_SITES')['raw']),
              (upper + '_PRIVATE', declaration(text, upper + '_PRIVATE')['raw']), ('cohort', cohort)]
    digest = hashlib.sha256(b'zerogravity.cuda-text-native-capture.v1\0')
    for label, value in parts:
        normalized = re.sub(r'\s+', '', re.sub(r'//[^\n]*', '', value))
        digest.update(label.encode() + b'\0' + normalized.encode() + b'\0')
    return digest.hexdigest()


def baseline(held, root, ancestor, images):
    text = held.content(root / GENERATED).decode()
    records = held.content(root / RECORDS).decode()
    result = {'text': text, 'records': records, 'profiles': {}}
    for prefix, name, b4 in (('qwen08', 'QWEN08', 'generated.rs'), ('qwen2b', 'QWEN2B', 'generated_2b.rs')):
        cohort = held.content(root / COHORT / b4).decode()
        require(fingerprint(text, prefix, cohort) == CAPTURES[prefix], 'exact genuine baseline source recipe identity')
        original = factories(text, prefix)
        packets = {fn: packet_value(row['value'], images, ancestor.ACT_SHA) for fn, row in original.items()}
        unique = {}
        for fn, packet in packets.items():
            key = (packet['image_sha256'], ancestor.abi_digest(packet['manifest']))
            require(key not in unique, 'original full-image/full-ABI packet factories unique')
            unique[key] = packet
        rows = declaration(text, name + '_SITES')['value']
        sites = []
        seen = set()
        for row in rows:
            require(set(row) == {'type', 'prompt_tokens', 'call', 'node', 'packets', 'copy_output'}
                    and row['type'] == 'SiteRecipe' and row['prompt_tokens'] in (47, 57, 62, 67, 72)
                    and type(row['call']) is type(row['node']) is int and 0 <= row['call'] < 8,
                    'complete existing source/call/site recipe')
            key = (row['prompt_tokens'], row['call'], row['node'])
            require(key not in seen, 'unique existing source recipe site')
            seen.add(key)
            require(1 <= len(row['packets']) <= 3 and all(fn in packets for fn in row['packets']),
                    'complete existing packet factory sequence')
            sites.append({'workload': row['prompt_tokens'], 'call': row['call'], 'node': row['node'],
                          'operation': 'existing', 'packets': [packets[fn] for fn in row['packets']],
                          'copy_output': row['copy_output']})
        require(len(sites) == 6720 and len({s['workload'] for s in sites}) == 5, 'unchanged genuine five-prompt baseline')
        cohort_rules = declaration(records, name)['value']
        require(cohort_rules['calls'] == 40 and len(cohort_rules['rules']) == cohort_rules['limits'][2],
                'actual unchanged baseline native rule/call capacity')
        # The original accounting charges one site header for every Head too.
        accounted = sites + [{'operation': 'head', 'packets': []} for _ in range(40)]
        charge = ancestor.metadata_bound(accounted, unique)
        expected_charge = 39845888 if prefix == 'qwen08' else 34603008
        require(charge == expected_charge, 'original full per-site + unique client/actor metadata accounting')
        result['profiles'][prefix] = {'packets': packets, 'factories': original, 'sites': sites,
            'unique': unique, 'rules': cohort_rules, 'charge': charge, 'cohort': cohort}
    return result


def shape_key(site):
    return (site['workload'], site['call'], site['node'])


def head_proof(sites, original_hex, ancestor, original_rules):
    heads = [site for site in sites if site['operation'] == 'head']
    require(len(heads) == len({site['workload'] for site in sites}) * 8, 'every actual Head observed')
    seen = set()
    proofs = []
    for head in heads:
        key = (head['workload'], head['call'])
        require(key not in seen and len(head['packets']) == 1, 'one complete source Head per actual call')
        seen.add(key)
        packet = head['packets'][0]
        full = packet['manifest']
        raw = bytearray(160)
        for field in full['parameters']:
            require(field['offset'] + field['bytes'] <= 160, 'complete original Head packet extent')
            raw[field['offset']:field['offset'] + field['bytes']] = bytes.fromhex(field['hex'])
        require(raw.hex() == original_hex, 'all actual new Head words equal the genuine original source Head')
        streamed = copy.deepcopy(full)
        rows = 1024**2 // (1024 * 2)
        streamed['grid'] = [rows // 8, 1, 1]
        streamed['bindings'][0]['bytes'] = 1024**2
        streamed['bindings'][1]['bytes'] = 1024 * 2
        scalar = bytearray.fromhex(streamed['parameters'][0]['hex'])
        scalar[68:72] = rows.to_bytes(4, 'little')
        streamed['parameters'][0]['hex'] = scalar.hex()
        for fixup in streamed['fixups']:
            fixup['extent'] = 1024**2 if fixup['binding'] == 0 else 2048 if fixup['binding'] == 1 else rows * 2
        abi = ancestor.abi_digest(streamed)
        matches = [rule for rule in original_rules if bytes(rule['image']).hex() == packet['image_sha256']
                   and bytes(rule['abi']).hex() == abi and rule['copy'] is False]
        require(len(matches) == 1, 'actual same fullimage/complete streamed Head ABI unchanged')
        proofs.append({'prompt_tokens': key[0], 'call': key[1], 'image_sha256': packet['image_sha256'],
                       'streamed_abi_sha256': abi})
    return proofs


def packet_text(name, packet, ancestor):
    packet = neutral(packet)
    image = 'None' if packet['image'] is None else 'Some(ProjectionImage::' + packet['image'] + ')'
    return ('#[inline(never)]\nfn ' + name + '() -> Packet {Packet{image:' + image + ',roles:vec!['
            + ','.join('Role::' + role for role in packet['roles']) + '],manifest:'
            + ancestor.rust_manifest(packet['manifest']) + ',clears:vec!['
            + ','.join('(Role::%s,%d,%d)' % tuple(clear) for clear in packet['clears']) + ']}}\n')


def append(held, base, added, ancestor, original_head):
    require(bool(added) and {s['workload'] for s in added} <= {48, 49, 50}, 'actual new finite capture recipes only')
    state = base['profiles']['qwen08']
    old_sites = state['sites']
    seen = {shape_key(site) for site in old_sites}
    for site in added:
        require(shape_key(site) not in seen, 'no duplicate or replaced original source site')
        seen.add(shape_key(site))
    prompts = sorted({site['workload'] for site in added})
    require(len(added) == len(prompts) * 8 * 169, 'complete every new source/call/site cohort')
    heads = head_proof(added, original_head, ancestor, state['rules']['rules'])
    unique = dict(state['unique'])
    references = {key: fn for fn, packet in state['packets'].items()
                  for key in [(packet['image_sha256'], ancestor.abi_digest(packet['manifest']))]}
    next_id = max(int(fn.rsplit('_', 1)[1]) for fn in references.values()) + 1
    factories_new, rows, rules_added = [], [], []
    existing_rules = {(bytes(row['image']).hex(), bytes(row['abi']).hex()): row for row in state['rules']['rules']}
    original_rule_keys = set(existing_rules)
    for site in added:
        if site['operation'] == 'head':
            continue
        names = []
        for original in site['packets']:
            packet = neutral(original)
            key = (packet['image_sha256'], ancestor.abi_digest(packet['manifest']))
            semantic = {key: packet[key] for key in ('image', 'image_sha256', 'roles', 'manifest', 'clears')}
            if key in unique:
                prior = unique[key]
                require({field: prior[field] for field in semantic} == semantic,
                        'identical image/ABI cannot change semantic roles or physical clear policy')
            else:
                unique[key] = semantic
                name = 'qwen08_packet_' + str(next_id)
                next_id += 1
                references[key] = name
                factories_new.append(packet_text(name, semantic, ancestor))
            names.append(references[key])
            copy_required = site['copy_output'] is not None
            if key in existing_rules:
                if copy_required and not existing_rules[key]['copy']:
                    require(key not in original_rule_keys,
                            'existing same ABI copy permission cannot be silently widened')
                    existing_rules[key]['copy'] = True
            else:
                row = {'type': 'Rule', 'image': list(bytes.fromhex(key[0])),
                       'abi': list(bytes.fromhex(key[1])), 'copy': copy_required}
                existing_rules[key] = row
                rules_added.append(row)
        rows.append('    SiteRecipe{prompt_tokens:%d,call:%d,node:%d,packets:&[%s],copy_output:%s},\n' %
                    (site['workload'], site['call'], site['node'], ','.join(names),
                     'None' if site['copy_output'] is None else 'Some(%d)' % site['copy_output']))
    complete_sites = old_sites + [{'operation': 'head', 'packets': []} for _ in range(40)] + added
    charge = ancestor.metadata_bound(complete_sites, unique)
    private = sorted({(role, binding['bytes']) for packet in unique.values()
                      for role, binding in zip(packet['roles'], packet['manifest']['bindings'])
                      if role in ('Lse', 'PrivateO', 'PartialO', 'PartialLse')},
                     key=lambda item: (('Lse', 'PrivateO', 'PartialO', 'PartialLse').index(item[0]), item[1]))
    text = base['text']
    table = declaration(text, 'QWEN08_SITES')
    at = table['value_end'] - 1
    text = text[:at] + ''.join(rows) + text[at:]
    anchor = text.index('static QWEN08_SITES:')
    text = text[:anchor] + ''.join(factories_new) + text[anchor:]
    old_guard = '!matches!(prompt_tokens, 47 | 57 | 62 | 67 | 72)'
    require(text.count(old_guard) == 1, 'exact existing finite materializer guard')
    text = text.replace(old_guard, '!matches!(prompt_tokens, ' + ' | '.join(map(str, sorted({47, 57, 62, 67, 72, *prompts}))) + ')')
    a = text.index('pub(crate) fn metadata_bytes(')
    b = text.index('pub(crate) fn sites(', a)
    segment = text[a:b]
    require(segment.count('NativeCatalogue::Source1024 => 39845888,') == 1, 'exact existing source metadata charge')
    text = text[:a] + segment.replace('NativeCatalogue::Source1024 => 39845888,',
                                    'NativeCatalogue::Source1024 => %d,' % charge) + text[b:]
    part = declaration(text, 'QWEN08_PRIVATE')
    replacement = 'const QWEN08_PRIVATE: &[(Role, usize)] = &[' + ','.join('(Role::%s,%d)' % item for item in private) + '];'
    text = text[:part['start']] + replacement + text[part['end']:]
    records = base['records']
    part = declaration(records, 'QWEN08')
    existing = part['raw']
    calls = 40 + len(prompts) * 8
    images = {key[0] for key in unique}
    functions = {(packet['image_sha256'], packet['manifest']['entry']) for packet in unique.values()}
    # The historical Head uses its existing GEMV image; its exact streamed ABI
    # was proved above. Retain its separately counted entry from the old union.
    old_functions = {(packet['image_sha256'], packet['manifest']['entry']) for packet in state['unique'].values()}
    head_keys = {(packet['image_sha256'], packet['manifest']['entry'])
                 for site in added if site['operation'] == 'head' for packet in site['packets']}
    old_images = {packet['image_sha256'] for packet in state['unique'].values()}
    require(len(old_images | {key[0] for key in head_keys}) == state['rules']['limits'][0]
            and len(old_functions | head_keys) == state['rules']['limits'][1],
            'actual unchanged historical image/function union includes proved Head')
    limits = (len(images | {key[0] for key in head_keys}), len(functions | head_keys), len(existing_rules))
    updated = re.sub(r'calls:\s*40,', 'calls: %d,' % calls, existing, count=1)
    updated, count = re.subn(r'limits:\s*\(17,\s*27,\s*164\)', 'limits: (%d,%d,%d)' % limits, updated, count=1)
    require(count == 1, 'exact genuine original module/function/rule capacities')
    where = updated.rfind(']')
    rule_text = ''.join('Rule{image:%s,abi:%s,copy:%s},' %
                        (ancestor.rust_array(row['image']), ancestor.rust_array(row['abi']), str(row['copy']).lower())
                        for row in rules_added)
    updated = updated[:where] + rule_text + updated[where:]
    records = records[:part['start']] + updated + records[part['end']:]
    # Existing factories, individual site rows and2B payloads remain exact raw
    # source bytes; only the .8B table tail/private union/accounting grows.
    for prefix in ('qwen08', 'qwen2b'):
        after = factories(text, prefix)
        for fn, original in base['profiles'][prefix]['factories'].items():
            require(after[fn]['raw'] == original['raw'], 'old packet factory bytes unchanged')
    require(declaration(text, 'QWEN2B_SITES')['raw'] == declaration(base['text'], 'QWEN2B_SITES')['raw']
            and declaration(text, 'QWEN2B_PRIVATE')['raw'] == declaration(base['text'], 'QWEN2B_PRIVATE')['raw']
            and declaration(records, 'QWEN2B')['raw'] == declaration(base['records'], 'QWEN2B')['raw'],
            'all original2B site/private/ABI recipe bytes unchanged')
    expected_old_rows = declaration(base['text'], 'QWEN08_SITES')['raw']
    require(declaration(text, 'QWEN08_SITES')['raw'].startswith(expected_old_rows[:-2]),
            'old .8B site rows preserved in original order')
    fingerprints = {prefix: fingerprint(text, prefix, base['profiles'][prefix]['cohort']) for prefix in ('qwen08', 'qwen2b')}
    require(fingerprints['qwen2b'] == CAPTURES['qwen2b'], 'original2B capture identity unchanged')
    return text, records, {'added_prompt_tokens': prompts, 'semantic_calls': calls,
        'non_head_sites': len(old_sites) + len(rows), 'added_factories': len(factories_new),
        'native_limits': list(limits), 'metadata_bytes': charge, 'private_layout': private,
        'capture_fingerprints': fingerprints, 'head_equality': heads,
        'old_packet_factories_unchanged': True, 'old_site_rows_unchanged': True, 'old_2b_unchanged': True}
