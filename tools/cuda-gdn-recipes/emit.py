"""Extract authenticated actual GDN qualification fixtures; no runtime integration."""
import argparse
from contextlib import ExitStack
from pathlib import Path

from capture import extract
from common import json_bytes, require, sha, write
from inputs import accepted_capture, authenticate, reference_reader, renderer
from tuples import baseline, catalogue

HERE = Path(__file__).resolve().parent


def validate_paths(args):
    names = ('baseline_root', 'reference_tools', 'renderer', 'original_tuples', 'output')
    require(all(getattr(args, n).is_absolute() for n in names), 'explicit absolute source/output paths')
    require(not args.output.exists() and not args.output.is_symlink(), 'fresh qualification fixture output')
    output = args.output.resolve()
    sources = [args.reference_tools, args.renderer, args.original_tuples, HERE,
               args.baseline_root / 'crates', args.baseline_root / 'tools']
    require(1 <= len(args.capture_reference) <= 3, 'one to three distinct actual fresh captured additions')
    paths = []
    for path, digest in args.capture_reference:
        path = Path(path)
        require(path.is_absolute() and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                'explicit actual reference receipt/SHA256')
        sources.append(path.parent)
        paths.append(path.resolve())
    require(len(paths) == len(set(paths)), 'no duplicate actual capture receipt')
    for source in sources:
        source = source.resolve()
        require(not output.is_relative_to(source) and not source.is_relative_to(output),
                'output disjoint from immutable source/capture/production paths')


def main(args):
    validate_paths(args)
    with reference_reader(args.reference_tools) as reference, ExitStack() as stack:
        held = stack.enter_context(reference.HeldFiles())
        for path in sorted(HERE.iterdir()):
            if path.is_file() and path.suffix in ('.py', '.json', '.md'):
                held.hold(path)
        authenticate(held, args.reference_tools, 'reference-pins.json')
        original = baseline(held, args.baseline_root, args.original_tuples)
        render = renderer(held, args.renderer)
        outputs, captures, anchors, discoveries = {}, [], [], []
        seen = set()
        for path, digest in args.capture_reference:
            root, rows, trajectory, discovery = accepted_capture(
                held, stack, Path(path), digest, reference, args.reference_tools)
            require(rows not in seen, 'one actual trajectory per prompt addition')
            seen.add(rows)
            sources, record, facts = extract(held, root, rows, render, original, args.baseline_root)
            require(not set(outputs) & set(sources), 'unique actual source output names')
            outputs.update(sources)
            outputs['qwen08_%d/recipe.json' % rows] = json_bytes(record)
            outputs['qwen08_%d/actual-capture.json' % rows] = json_bytes(facts)
            captures.append(record)
            anchors.append(dict(path=str(path), sha256=digest, prompt_tokens=rows,
                accepted_trajectory_sha256=sha(held.content(root / 'trajectory-validation.json')),
                same_request_discovery=str(discovery.root), held_discovery_inputs=discovery.pins()))
            discoveries.append(discovery)
        captures.sort(key=lambda row: row['input_tokens'])
        # This aggregate is qualification fixture data, never a production selector
        # or learned-parameter identity. Original tuples and old hashes stay intact.
        expanded = original[:6] + captures + original[6:]
        outputs['preserved-original.json'] = json_bytes(original)
        outputs['qualification-catalogue.json'] = json_bytes(dict(
            records=expanded, original96_subset_preserved=True,
            original_recipe_count=12, original_tuple_count=96,
            added_recipe_count=len(captures), added_tuple_count=9 * len(captures),
            original_catalogue_sha256={p: catalogue(original, p) for p in ('Qwen08', 'Qwen2B')},
            fixture_catalogue_sha256={p: catalogue(expanded, p) for p in ('Qwen08', 'Qwen2B')},
            production_identity=False))
        for discovery in discoveries:
            discovery.check()
        held.check()
        report = dict(schema='zerogravity.cuda-gdn-recipes.qualification-candidate.v1',
            status='AUTHENTICATED_SOURCE_FIXTURES_ONLY',
            baseline=str(args.baseline_root), renderer=str(args.renderer),
            actual_capture_references=anchors, prompt_tokens=sorted(seen),
            missing_requested_prompt_tokens=sorted({48, 49, 50} - seen),
            all_original96_tuples_and12_recipes_unchanged=True,
            original_2b_fixture_catalogue_unchanged=True,
            complete_requested_source_set=seen == {48, 49, 50},
            outputs={name: dict(bytes=len(raw), sha256=sha(raw)) for name, raw in outputs.items()},
            held_inputs=held.pins(),
            production_files_written=False, production_dispatch_generated=False,
            admission_changed=False, native_recipes_qualified=False,
            model_execution_qualified=False, performance_qualified=False,
            integration_permitted=False,
            unresolved=['Complete native-bound consumed control contents/producer proof is absent.',
                'Production must select by operator/shape/numerics/ABI contract, independent of fixture matrices/catalogue aggregates.',
                'Actual native source/state/fullV/natural, cancellation, physical ownership and serving qualification remain separate.'],
            source_identity_note='Whole Rust source changes create a new identity even when preserved CUDA tuples agree; '
                                 'fixture collection confers no old-binary or model accuracy claim.')
        args.output.mkdir(mode=0o700)
        for name, raw in outputs.items():
            path = args.output / name
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            write(path, raw)
        for discovery in discoveries:
            discovery.check()
        held.check()
        write(args.output / 'candidate-source.json', json_bytes(report))
        held.check()
    return dict(status=report['status'], prompt_tokens=sorted(seen), new_sources=9 * len(seen),
                complete_requested_source_set=report['complete_requested_source_set'])


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-root', 'reference-tools', 'renderer', 'original-tuples', 'output'):
        value.add_argument('--' + name, required=True, type=Path)
    value.add_argument('--capture-reference', action='append', nargs=2, metavar=('RESULT_JSON', 'SHA256'), required=True)
    return value


if __name__ == '__main__':
    print(json_bytes(main(parser().parse_args())).decode(), end='')
