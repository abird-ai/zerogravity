"""Pure failed-layout protocol mutations; no collector/provider/device execution."""
import ast
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from contract import read_json, require, sha
from discovery import HeldFiles
from late_layout import ABSENT, late_facts, matching_discovery


@unittest.skipUnless(os.environ.get('ZG_CAPTURE_SOURCE'), 'set retained capture source for original strict guards')
class LateLayoutTests(unittest.TestCase):
    def test_cross_request_layout_reuse_never_changes_the_failed_parent_numerical_target(self):
        original = {'prompt_tokens': 47, 'prompt_token_ids': [1, 2], 'generation': {'max_new_tokens': 8}}
        current = {'prompt_tokens': 48, 'prompt_token_ids': [1, 2, 3], 'generation': {'max_new_tokens': 8}}
        parent = SimpleNamespace(root=Path('/parent47'), result={'tokenized_request': original})
        future = SimpleNamespace(root=Path('/future48'), result={'tokenized_request': current})
        self.assertIs(matching_discovery(original, [future, parent]), parent)
        self.assertEqual(future.result['tokenized_request'], current)
        for held in ([future], [parent, parent], []):
            with self.assertRaises(ValueError):
                matching_discovery(original, held)
        changed = copy.deepcopy(original)
        changed['generation']['max_new_tokens'] = 7
        with self.assertRaises(ValueError):
            matching_discovery(changed, [parent, future])

    def guards(self):
        root = Path(os.environ['ZG_CAPTURE_SOURCE'])
        source = root / 'package/validate_native.py'
        pin = read_json(Path(__file__).parent / 'capture-pins.json')['package/validate_native.py']
        self.assertEqual(sha(source), pin['sha256'])
        functions = [node for node in ast.parse(source.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in ('validate_scopes', 'validate_pointer_observations')]
        namespace = {'require': require}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<original guards pure fixture>', 'exec'), namespace)
        return namespace

    def fixture(self):
        def event(index, api, order, scope=0, launch=False, function=0):
            return {'sequence': index, 'api': api, 'order': order, 'exit_order': order + 1,
                    'scope': scope, 'tid': 1, 'complete': True, 'decoded': True, 'status': 0,
                    'context': 9, 'launch': launch, 'function': function, 'layout': 0,
                    'args': [8, 8], 'data': [0, 5], 'values': [0] * 24,
                    'grid': [1, 1, 1], 'block': [32, 1, 1], 'shared': 0, 'attributes': 0}
        load = event(0, 'cuModuleLoadData', 1)
        load['exit_order'] = 3
        load['values'][16] = 456
        getter = event(1, 'cuModuleGetFunction', 4)
        getter['values'][1], getter['values'][3] = 456, 123
        events = [load, getter]
        for call in range(8):
            launch = event(len(events), 'cuLaunchKernel', 6 + call * 2, call, True, 0 if call == 0 else 1)
            if call == 0:
                launch['args'], launch['layout'] = [ABSENT, 0], ABSENT
            events.append(launch)
        scopes = []
        for call in range(8):
            rows = 47 if call == 0 else 1
            scopes.append({'sequence': call, 'begun': True, 'closed': True, 'call': call, 'role': 0,
                'tid': 1, 'A': 1, 'A_bytes': rows * 4, 'B': 2, 'B_bytes': rows * 24,
                'C': 3, 'C_bytes': 496640, 'workspace': 0, 'workspace_bytes': 0,
                'first': 0 if call == 0 else call + 2, 'last': call + 3})
        module = {'index': 0, 'context': 9, 'ready': True, 'load_order': 2,
                  'unload_order': 0, 'sha256': 'a' * 64}
        function = {'index': 0, 'module': 0, 'reflected': True, 'prelaunch_bound': True,
            'prelaunch_binding_kind': 1, 'prelaunch_binding_order': 5, 'handle': 123,
            'native_module': 456, 'context': 9, 'extent': 8, 'params': [[0, 8]],
            'callback_name': 'late', 'name': 'late'}
        known = dict(function, index=1, handle=124, callback_name='known', name='known')
        counts = {name: 0 for name in ('candidates', 'duplicates', 'queries', 'registered', 'unregistered',
                  'query_failed', 'unsupported_memory', 'overflow', 'range_failed', 'foreign_context', 'invalid_range')}
        driver = {'schema': 'zg.vllm.blas-driver.v1', 'mode': 2, 'failures': 64, 'replay_qualified': False,
            'contexts': [{'handle': 9}], 'events': events, 'startup_events': [], 'scopes': scopes,
            'functions': [function, known], 'modules': [module], 'pointers': [],
            'pointer_observations': dict(counts, policy='registered-memory-only-v1', capacity=8192)}
        catalog = {'layouts': [{'callback_name': 'known', 'name': 'known', 'module_sha256': 'a' * 64, 'params': [[0, 8]]}]}
        return driver, catalog

    def test_only_actual_absent_layout_is_derived_and_packets_remain_missing(self):
        driver, catalog = self.fixture()
        with tempfile.TemporaryDirectory() as directory, HeldFiles() as held:
            path = Path(directory) / 'arena'
            path.write_bytes(b'late\0' + bytes(11))
            layout, missing = late_facts(driver, catalog, held.hold(path), self.guards(), 47)
            self.assertEqual(layout['layouts'][0]['name'], 'late')
            self.assertEqual(len(missing), 1)
            self.assertEqual(driver['events'][2]['args'], [ABSENT, 0])
            self.assertEqual(len(catalog['layouts']), 1)

    def test_failure_bits_events_lifetimes_reflection_and_present_packets_are_strict(self):
        original, catalog = self.fixture()
        mutations = (
            lambda d: d.update(failures=64 | 4),
            lambda d: d['events'][2].update(decoded=False),
            lambda d: d['events'][0].update(complete=False),
            lambda d: d['events'][2].update(status=1),
            lambda d: d['scopes'][7].update(closed=False),
            lambda d: d['functions'][0].update(reflected=False),
            lambda d: d['functions'][0].update(params=[[0, 8], [4, 4]]),
            lambda d: d['functions'][0].update(prelaunch_binding_order=99),
            lambda d: d['functions'][0].update(native_module=457),
            lambda d: d['events'][1]['values'].__setitem__(3, 999),
            lambda d: d['events'][1].update(data=[0, 4]),
            lambda d: d['modules'][0].update(load_order=4),
            lambda d: d['modules'].append(dict(d['modules'][0], index=2)),
            lambda d: d['modules'][0].update(unload_order=6),
            lambda d: d['events'][2].update(args=[ABSENT, 8]),
            lambda d: d['events'][2].update(layout=0),
            lambda d: d['events'][3].update(args=[20, 8]),
            lambda d: d['pointer_observations'].update(overflow=1),
        )
        with tempfile.TemporaryDirectory() as directory, HeldFiles() as held:
            path = Path(directory) / 'arena'
            path.write_bytes(b'late\0' + bytes(11))
            for index, mutate in enumerate(mutations):
                driver = copy.deepcopy(original)
                mutate(driver)
                with self.subTest(mutation=index), self.assertRaises(ValueError):
                    late_facts(driver, catalog, held.hold(path), self.guards(), 47)
            changed = copy.deepcopy(catalog)
            changed['layouts'].append({'callback_name': 'late', 'name': 'late', 'module_sha256': 'a' * 64, 'params': [[0, 8]]})
            with self.assertRaises(ValueError):
                late_facts(original, changed, held.hold(path), self.guards(), 47)

    def test_public_library_materialization_requires_its_original_live_owner(self):
        driver, catalog = self.fixture()
        for event in driver['events']:
            event['order'] += 4
            event['exit_order'] += 4
        driver['modules'][0]['load_order'] += 4
        for function in driver['functions']:
            function['prelaunch_binding_order'] += 4
        driver['events'][0]['api'] = 'cuLibraryGetModule'
        driver['events'][0]['values'][1] = 999
        origin = copy.deepcopy(driver['events'][0])
        origin.update(api='cuLibraryLoadData', order=1, exit_order=2)
        origin['values'][16] = 999
        driver['startup_events'] = [origin]
        with tempfile.TemporaryDirectory() as directory, HeldFiles() as held:
            path = Path(directory) / 'arena'
            path.write_bytes(b'late\0' + bytes(11))
            self.assertEqual(len(late_facts(driver, catalog, held.hold(path), self.guards(), 47)[1]), 1)
            for mutation in ('wrong_owner', 'unloaded', 'duplicate_owner'):
                changed = copy.deepcopy(driver)
                if mutation == 'wrong_owner':
                    changed['startup_events'][0]['values'][16] = 1000
                elif mutation == 'duplicate_owner':
                    changed['startup_events'].append(copy.deepcopy(origin))
                else:
                    unload = copy.deepcopy(origin)
                    unload.update(api='cuLibraryUnload', order=8, exit_order=9)
                    unload['values'][0] = 999
                    changed['startup_events'].append(unload)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    late_facts(changed, catalog, held.hold(path), self.guards(), 47)


if __name__ == '__main__':
    unittest.main()
