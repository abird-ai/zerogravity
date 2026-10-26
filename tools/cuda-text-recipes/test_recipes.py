"""Pure source/grammar/refusal checks; fixtures never stand in for model captures."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import os

from ancestry import accepted_capture, authenticate, bound_ancestor, load_ancestor, load_reference
from catalogue import CAPTURES, baseline, fingerprint, packet_text, packet_value
from emit import classify_joined_site, validate_paths
from literals import Literal, balanced, declaration, factories


class LiteralTests(unittest.TestCase):
    def test_nested_literal_preserves_every_scalar_byte_and_role(self):
        source = 'Packet{image:None,roles:vec![Role::X],manifest:CubinKernelManifest{entry:"entry{one}".into(),parameters:vec![CubinParameter{offset:0,template:[0,255,17].to_vec()}]},clears:vec![]}'
        parsed = Literal(source).parse()
        self.assertEqual(parsed['roles'], ['Role::X'])
        self.assertEqual(parsed['manifest']['parameters'][0]['template'], [0, 255, 17])
        self.assertEqual(parsed['manifest']['entry'], 'entry{one}')
        self.assertEqual(balanced(source, source.index('{'))[1], len(source))

    def test_code_unknown_struct_duplicates_and_trailing_content_refuse(self):
        for source in ('std::process::Command::new("x")', 'unsafe{run()}', 'Packet{image:None,image:None}',
                       'Unobserved{bytes:1}', '[1,2].unknown()', '[1,2];run()', '[1,2', 'vec![guess()]'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                Literal(source).parse()

    def test_declaration_selects_exact_member_and_rejects_ambiguous_names(self):
        text = 'static A: &[SiteRecipe] = &[SiteRecipe{prompt_tokens:47}];\nconst AB: &[usize] = &[72];'
        self.assertEqual(declaration(text, 'A')['value'][0]['prompt_tokens'], 47)
        with self.assertRaises(ValueError):
            declaration(text + '\n' + text, 'A')


class InputTests(unittest.TestCase):
    def arguments(self):
        return SimpleNamespace(ancestor_root=Path('/retained/ancestor'), baseline_root=Path('/retained/baseline'),
            reference_tools=Path('/retained/reference'), output=Path('/fresh-candidate-output-fixture'),
            capture_reference=[('/actual/reference-result.json', 'a' * 64)])

    def test_outputs_are_explicit_fresh_and_outside_production_or_inputs(self):
        args = self.arguments()
        validate_paths(args)
        for path in (Path('/retained/baseline/crates/candidate'), Path('/retained/ancestor/candidate'),
                     Path('/retained/reference/candidate'), Path('/actual/candidate'), Path('relative')):
            changed = copy.copy(args)
            changed.output = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_paths(changed)
        with tempfile.TemporaryDirectory() as output:
            args.output = Path(output)
            with self.assertRaises(ValueError):
                validate_paths(args)
            root = Path(output)
            (root / 'baseline/crates').mkdir(parents=True)
            (root / 'alias').symlink_to(root / 'baseline/crates', target_is_directory=True)
            args.baseline_root = root / 'baseline'
            args.output = root / 'alias/candidate'
            with self.assertRaises(ValueError):
                validate_paths(args)

    def test_missing_duplicate_and_unanchored_capture_inputs_refuse(self):
        for inputs in ([], [('/actual/reference-result.json', 'invented')],
                       [('/actual/reference-result.json', 'a' * 64)] * 2,
                       [('relative/reference-result.json', 'a' * 64)]):
            args = self.arguments()
            args.capture_reference = inputs
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                validate_paths(args)

    def test_discovery_and_failed_receipts_never_reach_recipe_generation(self):
        for result in ({'schema': 'zerogravity.qwen35-cuda-reference.result.v1', 'mode': 'discover', 'ancestry_stage': 'discover'},
                       {'schema': 'zerogravity.qwen35-cuda-reference.result.v1', 'mode': 'capture', 'ancestry_stage': 'capture',
                        'original_trajectory_validation': 'FAIL'}):
            held = SimpleNamespace(document=lambda *args, **kwargs: result)
            with self.subTest(result=result), self.assertRaises(ValueError):
                accepted_capture(held, None, Path('/actual/reference-result.json'), 'a' * 64, None, None)


class CohortRefusalTests(unittest.TestCase):
    def fixture(self):
        # Refusal-only grammar fixture; never a passing source/model receipt.
        digest = 'a' * 64
        launch = {'event': 0, 'image_sha256': digest, 'image_bytes': 8, 'entry': 'actual_entry',
                  'params': [[0, 8]], 'extent': 8, 'parameters': {'sha256': 'b' * 64}}
        function = {'index': 0, 'name': 'actual_entry', 'callback_name': 'actual_entry',
                    'reflected': True, 'prelaunch_bound': True, 'prelaunch_binding_kind': 1,
                    'prelaunch_binding_order': 1, 'params': [[0, 8]], 'extent': 8}
        return ({'key': {'prompt_rows': 48}, 'launches': [launch]}, [[(0, bytes(8))]],
                {digest: {'bytes': 8}}, {'images': {digest: {'entries': ['old_entry']}}},
                {'functions': [function], 'events': [{'function': 0, 'order': 2}]}, {digest: b'fixture'})

    def test_missing_image_symbol_reflection_or_packet_fields_refuse(self):
        for fault in ('image', 'bytes', 'symbol', 'reflected', 'prelaunch', 'formals', 'extent', 'fields', 'missing'):
            item, fields, images, discovered, driver, modules = self.fixture()
            untouched = copy.deepcopy(discovered)
            function = driver['functions'][0]
            if fault == 'image': images.clear()
            if fault == 'bytes': images['a' * 64]['bytes'] = 7
            if fault == 'reflected': function['reflected'] = False
            if fault == 'prelaunch': function['prelaunch_bound'] = False
            if fault == 'formals': function['params'] = [[0, 4]]
            if fault == 'extent': function['extent'] = 4
            if fault == 'fields': fields = [[(0, bytes(7))]]
            if fault == 'missing': fields = []
            ancestor = SimpleNamespace(classify_site=lambda *_: self.fail('unproved cohort reached classifier'))
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                classify_joined_site(item, fields, ancestor, images, discovered, driver, modules,
                                     lambda *_: fault != 'symbol')
            self.assertEqual(discovered, untouched)

    def test_actual_entry_proof_cannot_waive_original_source_abi_classifier(self):
        item, fields, images, discovered, driver, modules = self.fixture()
        untouched = copy.deepcopy(discovered)
        def refusal(*_):
            raise ValueError('original semantic pointer/scalar/range refusal')
        with self.assertRaisesRegex(ValueError, 'original semantic'):
            classify_joined_site(item, fields, SimpleNamespace(classify_site=refusal),
                                 images, discovered, driver, modules, lambda *_: True)
        self.assertEqual(discovered, untouched)


@unittest.skipUnless(os.environ.get('ZG_RECIPE_ANCESTOR') and os.environ.get('ZG_RECIPE_BASELINE')
                     and os.environ.get('ZG_RECIPE_REFERENCE'), 'set explicit authenticated pure source paths')
class OriginalSourceTests(unittest.TestCase):
    def test_actual_baseline_fingerprints_owner_costs_and_packet_completeness(self):
        ancestor_root = Path(os.environ['ZG_RECIPE_ANCESTOR'])
        baseline_root = Path(os.environ['ZG_RECIPE_BASELINE'])
        reference_root = Path(os.environ['ZG_RECIPE_REFERENCE'])
        reference = load_reference(reference_root)
        with reference.HeldFiles() as held:
            authenticate(held, reference_root, 'reference-pins.json')
            authenticate(held, baseline_root, 'baseline-pins.json')
            ancestor = load_ancestor(held, ancestor_root)
            with bound_ancestor(ancestor):
                images = held.document(ancestor_root / 'image-kinds.json')
                source = baseline(held, baseline_root, ancestor, images)
                for prefix, expected_charge in (('qwen08', 39845888), ('qwen2b', 34603008)):
                    profile = source['profiles'][prefix]
                    self.assertEqual(len(profile['sites']), 6720)
                    self.assertEqual(profile['charge'], expected_charge)
                    self.assertEqual(fingerprint(source['text'], prefix, profile['cohort']), CAPTURES[prefix])
                    # Round-trip an actual unchanged baseline packet, not a
                    # substituted new prompt or claimed capture output.
                    packet = next(iter(profile['packets'].values()))
                    rendered = packet_text(prefix + '_packet_999', packet, ancestor)
                    parsed = factories(rendered, prefix)[prefix + '_packet_999']['value']
                    self.assertEqual(packet_value(parsed, images, ancestor.ACT_SHA), packet)
                arena_type = ancestor.module_aliases['parameters'].ArgumentArena
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'actual-packet-fixture'
                    path.write_bytes(bytes(8))
                    with arena_type(path, 8, hashlib.sha256(bytes(8)).hexdigest()) as arena:
                        with self.assertRaises(ValueError):
                            arena.packet({'parameters': None})
                        packet = {'parameters': {'arena_offset': 0, 'bytes': 8,
                            'sha256': hashlib.sha256(bytes(8)).hexdigest(), 'fields': []}, 'extent': 8, 'params': [[0, 8]]}
                        with self.assertRaises(ValueError):
                            arena.packet(packet)
                        packet['parameters']['fields'] = [{'offset': 0, 'bytes': 8, 'hex': '01' + '00' * 7}]
                        with self.assertRaises(ValueError):
                            arena.packet(packet)
            held.check()


if __name__ == '__main__':
    unittest.main()
