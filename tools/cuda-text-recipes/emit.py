"""Explicit accepted-capture → candidate CUDA text recipes; no production writes."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

from ancestry import accepted_capture, authenticate, bound_ancestor, load_ancestor, load_reference
from catalogue import append, baseline
from literals import require

HERE = Path(__file__).resolve().parent


def classify_joined_site(item, fields, ancestor, images, discovered, driver, modules, symbol):
    """Current passing source/packet facts may extend only their own entry cohort."""
    current = dict(discovered, images=dict(discovered['images']))
    functions = {row['index']: row for row in driver['functions']}
    pending = []
    require(len(fields) == len(item['launches']) > 0, 'every actual selected launch has complete packet fields')
    for index, (launch, packet_fields) in enumerate(zip(item['launches'], fields)):
        digest, entry = launch['image_sha256'], launch['entry']
        if digest in discovered['images'] and entry in discovered['images'][digest]['entries']:
            continue
        require(digest in images and digest in discovered['images']
                and images[digest]['bytes'] == launch['image_bytes'],
                'unchanged finite complete-image classifier identity')
        event = driver['events'][launch['event']]
        function = functions[event['function']]
        require(function['name'] == function['callback_name'] == entry
                and function['reflected'] is True and function['prelaunch_bound'] is True
                and function['prelaunch_binding_kind'] in (1, 2)
                and function['prelaunch_binding_order'] < event['order']
                and function['params'] == launch['params']
                and function['extent'] == launch['extent']
                and [[offset, len(raw)] for offset, raw in packet_fields] == launch['params'],
                'actual prelaunch full reflected ABI and every complete held packet field')
        require(symbol(modules[digest], entry), 'actual entry exported by the same held whole CUDA ELF image')
        inherited = current['images'][digest]
        current['images'][digest] = dict(inherited, entries=[*inherited['entries'], entry])
        pending.append({'source_key': item['key'], 'event': launch['event'], 'image_sha256': digest,
            'image_bytes': launch['image_bytes'], 'entry': entry, 'reflected_parameters': launch['params'],
            'extent': launch['extent'], 'argument_packet_sha256': launch['parameters']['sha256'],
            'whole_elf_symbol_verified': True, 'ptx_authority': 'none; actual precompiled whole CUDA ELF',
            'packet_index': index})
    # No cohort fact is published until the unchanged classifier proves all
    # scalar words, semantic pointers, complete ranges and source dependencies.
    built = ancestor.classify_site(item, fields, images, current, driver)
    for fact in pending:
        packet = built['packets'][fact.pop('packet_index')]
        require(packet['image_sha256'] == fact['image_sha256']
                and packet['manifest']['entry'] == fact['entry']
                and packet['scalar_bytes_preserved'] is True,
                'classified actual whole-image entry retains complete scalar bytes')
        fact['complete_abi_sha256'] = ancestor.abi_digest(packet['manifest'])
    return built, pending


def classify_capture(root, prompt_tokens, trajectory, held, stack, ancestor, images, discovered, symbol):
    observed = root / 'observed'
    sites = held.document(observed / 'native-sites.json')
    metadata = held.document(observed / 'native-metadata-validation.json')
    driver = held.document(observed / 'driver/capture.json')
    prepared = held.document(observed / 'prepared.json')
    all42 = held.document(observed / 'all42/report.json')
    result = held.document(observed / 'driver-result.json', limit=1024**2)
    ancestor.native_anchor(trajectory, metadata)
    require(metadata['complete_selected_argument_fields'] is True and metadata['mode'] == 'capture'
            and result['collector_status'] == 0 and result['argument_fields_complete'] is True,
            'healthy actual complete argument capture only; discovery/failed packets refuse')
    controls = ancestor.source_controls(all42, '0.8b', prompt_tokens)
    with bound_ancestor(ancestor):
        joined = ancestor.source_sites.join('0.8b', prompt_tokens, sites, metadata, driver, prepared)
        arena_class = ancestor.module_aliases['parameters'].ArgumentArena
        arena = stack.enter_context(arena_class(observed / 'driver/data.bin', driver['data_bytes'], metadata['parameter_arena_sha256']))
        built, cohort = [], []
        modules = {}
        functions = {item['index']: item for item in driver['functions']}
        for item in joined:
            if item['key']['operation'] == 'head':
                embeddings = [site for site in sites['records'] if site['call'] == item['key']['call']
                              and site['label'] == 'language_model.model.embed_tokens']
                weights = [row for row in prepared['source_roles'] if row['parameter'] == 'language_model.model.embed_tokens.weight']
                audits = [row for row in prepared['parameters_before'] if row['parameter'] == 'language_model.model.embed_tokens.weight']
                require(len(embeddings) == len(weights) == len(audits) == 1
                        and prepared['aliases'].get('language_model.lm_head.weight') == 'language_model.model.embed_tokens.weight'
                        and weights[0]['source'] == audits[0]['source'] == 'model.language_model.embed_tokens.weight'
                        and weights[0]['source_header']['dtype'] == 'BF16'
                        and audits[0]['source_words_sha256'] == audits[0]['loaded_words_sha256']
                        and audits[0]['cast_is_explicit_observation'] is False,
                        'actual unchanged tied BF16 source Head/embedding words/header')
                item = dict(item)
                item['source_site'] = dict(item['source_site'], weight=embeddings[0]['weight'])
            fields = []
            for launch in item['launches']:
                _, packet_fields = arena.packet(launch)
                fields.append(packet_fields)
                event = driver['events'][launch['event']]
                module = functions[event['function']]['module']
                held.hold(observed / 'driver' / ('module-%04d.cubin' % module),
                          expected_sha=launch['image_sha256'], expected_bytes=launch['image_bytes'], limit=32 * 1024**2)
                if (launch['image_sha256'] not in modules and launch['entry'] not in
                        discovered['images'].get(launch['image_sha256'], {}).get('entries', [])):
                    modules[launch['image_sha256']] = held.content(observed / 'driver' / ('module-%04d.cubin' % module))
            try:
                classified, facts = classify_joined_site(item, fields, ancestor, images, discovered, driver, modules, symbol)
                built.append(classified)
                cohort.extend(facts)
            except ValueError as error:
                raise ValueError('source0.8B/prompt%d/call%d/layer%d/%s: %s' %
                                 (prompt_tokens, item['key']['call'], item['key']['layer'], item['key']['operation'], error)) from error
        arena.revalidate()
    require(len(built) == 8 * 169, 'every actual source operation/eight-call recipe present')
    return built, controls, arena, cohort


def validate_paths(args):
    for name in ('ancestor_root', 'baseline_root', 'reference_tools', 'output'):
        require(getattr(args, name).is_absolute(), 'explicit absolute source/output paths')
    require(not args.output.exists(), 'fresh candidate output; preserve failures and previous candidates')
    output = args.output.resolve()
    require(not output.is_relative_to((args.baseline_root / 'crates').resolve()), 'candidate output cannot write production source')
    for source in (args.ancestor_root, args.reference_tools):
        source = source.resolve()
        require(not output.is_relative_to(source) and not source.is_relative_to(output),
                'candidate output disjoint from immutable source packages')
    require(1 <= len(args.capture_reference) <= 3, 'explicit finite actual captured additions')
    paths = []
    for path, digest in args.capture_reference:
        path = Path(path)
        require(path.is_absolute() and len(digest) == 64 and all(char in '0123456789abcdef' for char in digest),
                'explicit actual reference receipt path and SHA256')
        parent = path.parent.resolve()
        require(not output.is_relative_to(parent) and not parent.is_relative_to(output),
                'candidate output disjoint from actual capture input')
        paths.append(path)
    require(len(paths) == len(set(paths)), 'no duplicate capture receipt')


def write(path, raw):
    with path.open('xb') as stream:
        os.chmod(path, 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main(args):
    validate_paths(args)
    reference = load_reference(args.reference_tools)
    with ExitStack() as stack:
        held = stack.enter_context(reference.HeldFiles())
        own_sources = [path for path in HERE.iterdir() if path.is_file() and path.suffix in ('.py', '.json')]
        for path in own_sources:
            held.hold(path)
        authenticate(held, args.reference_tools, 'reference-pins.json')
        authenticate(held, args.baseline_root, 'baseline-pins.json')
        ancestor = load_ancestor(held, args.ancestor_root)
        images = held.document(args.ancestor_root / 'image-kinds.json')
        discovered = held.document(args.ancestor_root / 'discovery-inventory.json')
        proof = held.document(args.ancestor_root / 'selected-section-comparison.json')
        # Retain only the original exact-resource section comparisons. These
        # permit known entries in whole images, not new images or native math.
        for row in proof['comparisons']:
            if row['selected_code_resources_and_reference_targets_equal'] is not True:
                continue
            for digest in (row['left'], row['right']):
                require(digest in discovered['images'] and digest in images, 'original finite complete-image cohort')
                if row['entry'] not in discovered['images'][digest]['entries']:
                    discovered['images'][digest]['entries'].append(row['entry'])
        source = baseline(held, args.baseline_root, ancestor, images)
        added, anchors, controls, arenas, discoveries, cohorts = [], [], {}, [], [], []
        seen_prompts = set()
        for result_path, digest in args.capture_reference:
            root, prompt_tokens, trajectory, discovery = accepted_capture(
                held, stack, Path(result_path), digest, reference, args.reference_tools)
            require(prompt_tokens not in seen_prompts, 'one accepted actual source/request per prompt addition')
            seen_prompts.add(prompt_tokens)
            sites, control, arena, cohort = classify_capture(root, prompt_tokens, trajectory, held, stack, ancestor,
                                                           images, discovered, reference.symbol)
            added.extend(sites)
            controls[str(prompt_tokens)] = control
            arenas.append(arena)
            discoveries.append(discovery)
            cohorts.append({'capture_reference': str(result_path), 'capture_sha256': digest,
                            'prompt_tokens': prompt_tokens, 'actual_classified_entry_facts': cohort})
            anchors.append({'path': str(result_path), 'sha256': digest, 'prompt_tokens': prompt_tokens,
                            'same_request_discovery': str(discovery.root), 'held_discovery_inputs': discovery.pins()})
        added.sort(key=lambda site: (site['workload'], site['call'], site['node']))
        original_heads = held.document(args.ancestor_root / 'regenerated/streamed-head-packets.json', limit=1024**2)
        generated, records, summary = append(held, source, added, ancestor, original_heads['0.8b'])
        for discovery in discoveries:
            discovery.check()
        for arena in arenas:
            arena.revalidate()
        held.check()
        # No output directory exists until every source/data/ABI check passes.
        args.output.mkdir(mode=0o700)
        outputs = {'generated.rs': generated.encode(), 'records.rs': records.encode()}
        for name, raw in outputs.items():
            write(args.output / name, raw)
        for discovery in discoveries:
            discovery.check()
        for arena in arenas:
            arena.revalidate()
        held.check()
        report = {'schema': 'zerogravity.cuda-text-recipes.candidate.v1',
            'status': 'CANDIDATE_SOURCE_ONLY_REQUIRES_INTEGRATION_AND_PHYSICAL_VALIDATION',
            'ancestor': str(args.ancestor_root), 'baseline': str(args.baseline_root),
            'actual_capture_references': anchors, 'summary': summary, 'source_controls': controls,
            'actual_capture_entry_cohorts': cohorts,
            'inherited_abi_classifier_bytes_unchanged': True,
            'source_membership_change': {'model': 'Qwen/Qwen3.5-0.8B', 'added_prompt_tokens': sorted(seen_prompts),
                                         'other_source_prompt_membership_unchanged': True},
            'held_inputs': held.pins() + [arena.pin() for arena in arenas],
            'outputs': {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} for name, raw in outputs.items()},
            'production_files_written': False, 'admission_changed': False,
            'native_recipes_qualified': False, 'model_execution_qualified': False, 'performance_qualified': False,
            'integration_required': ['bind independently recomputed capture fingerprint on model/GPU sides',
                'actual per-source prompt admission and complete GDN source/control/ABI proof',
                'derive vendor base/cohort call counts and all physical/metadata/image-section costs',
                'source/state/fullV/natural execution, cancellation/physical/serving checks and final gate']}
        write(args.output / 'candidate-source.json', (json.dumps(report, sort_keys=True, indent=2) + '\n').encode())
        held.check()
    return summary


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ('ancestor-root', 'baseline-root', 'reference-tools', 'output'):
        value.add_argument('--' + name, required=True, type=Path)
    value.add_argument('--capture-reference', action='append', nargs=2, metavar=('RESULT_JSON', 'SHA256'), required=True)
    return value


if __name__ == '__main__':
    main(parser().parse_args())
