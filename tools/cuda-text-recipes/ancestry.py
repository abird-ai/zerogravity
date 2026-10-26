"""Authenticate pure ancestor code and accepted capture receipts before deriving data."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import types

from literals import require

HERE = Path(__file__).resolve().parent


def load_module(path, name, raw):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


@contextmanager
def aliases():
    names = ('contract', 'discovery', 'sites', 'geometry', 'parameters', 'classify')
    previous = {name: sys.modules.get(name) for name in names}
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def manifest(name):
    value = json.loads((HERE / name).read_text())
    require(type(value) is dict and bool(value), 'fixed source ancestry manifest')
    return value


def authenticate(held, root, name):
    for relative, pin in manifest(name).items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'relative owned source pin')
        held.hold(root / path, expected_sha=pin['sha256'], expected_bytes=pin['bytes'])


def load_reference(root):
    # Authenticate bytes before the first import. Reuse the reviewed bounded
    # reader only after pinning its source with a separate local bootstrap.
    import hashlib
    import os
    import stat
    sources, opened = {}, []
    def stamp(value):
        return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    try:
        for relative, pin in manifest('reference-pins.json').items():
            path = root / relative
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            opened.append((path, fd, stamp(os.fstat(fd))))
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and before.st_size == pin['bytes'], 'reference source size')
            raw = os.pread(fd, before.st_size + 1, 0)
            require(len(raw) == before.st_size and hashlib.sha256(raw).hexdigest() == pin['sha256']
                    and stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()), 'exact reviewed reference source bytes')
            sources[relative] = raw
        with aliases():
            load_module(root / 'contract.py', 'contract', sources['contract.py'])
            reference = load_module(root / 'discovery.py', 'discovery', sources['discovery.py'])
            # Reuse the authenticated pure whole-ELF symbol guard, never an SDK.
            reference.symbol = load_module(root / 'late_layout.py', '_zg_recipe_elf', sources['late_layout.py']).symbol
        for path, fd, before in opened:
            require(before == stamp(os.fstat(fd)) == stamp(path.lstat()), 'reference source unchanged through exact-byte import')
    finally:
        for _, fd, _ in opened:
            os.close(fd)
    return reference


def load_ancestor(held, root):
    authenticate(held, root, 'ancestor-pins.json')
    with aliases():
        def imported(name, alias):
            path = root / 'package' / name
            return load_module(path, alias, held.content(path))
        sites = imported('sites.py', 'sites')
        # Workload membership is data. Actual addition admission below is .8B
        # fresh48/49/50 only; inherited graph/ABI/role checks remain unchanged.
        sites.PREFILLS = (47, 57, 62, 67, 72, 48, 49, 50)
        parameters = imported('parameters.py', 'parameters')
        geometry = imported('geometry.py', 'geometry')
        classify = imported('classify.py', 'classify')
        emitter = imported('generate.py', '_zg_text_recipe_ancestor')
        emitter.source_sites = sites
        emitter.module_aliases = {'sites': sites, 'parameters': parameters, 'geometry': geometry, 'classify': classify}
    return emitter


@contextmanager
def bound_ancestor(emitter):
    with aliases():
        sys.modules.update(emitter.module_aliases)
        yield


def prepared_capture(held, root, result, reference_root):
    prepared = held.document(root / 'prepared-package.json', result['prepared_package_sha256'], limit=1024**2)
    pins = manifest('reference-pins.json')
    stage = json.loads((reference_root / 'capture-pins.json').read_text())
    expected = {str(Path(name).relative_to('package')): pin for name, pin in stage.items()
                if name.startswith('package/')}
    require(set(prepared) == set(expected) | {'reference_request.py', 'tokenize_prompt.py', 'request.json'},
            'complete full-ABI prepared source cohort')
    for relative, digest in prepared.items():
        held.hold(root / 'package' / relative, expected_sha=digest)
    for relative, pin in expected.items():
        if relative not in ('matrix_contract.py', 'validate_native.py'):
            held.hold(root / 'package' / relative, expected_sha=pin['sha256'], expected_bytes=pin['bytes'])
    import hashlib
    native = held.content(root / 'package/validate_native.py').decode()
    require(native.count('prompt_rows == PROMPT_ROWS') == 1
            and hashlib.sha256(native.replace('prompt_rows == PROMPT_ROWS',
                'prompt_rows in (47,57,62,67,72)').encode()).hexdigest() == expected['validate_native.py']['sha256'],
            'only declared native prompt guard changed')
    for staged, source in (('reference_request.py', 'contract.py'), ('matrix_contract.py', 'matrix_contract.py'),
                           ('tokenize_prompt.py', 'tokenize_prompt.py')):
        held.hold(root / 'package' / staged, expected_sha=pins[source]['sha256'])
    require(held.document(root / 'package/request.json', limit=1024**2) == result['tokenized_request'],
            'same exact tokenizer-derived request in executed source package')


def accepted_capture(held, stack, result_path, digest, reference, reference_root):
    require(result_path.name == 'reference-result.json', 'explicit actual reference result receipt')
    root = result_path.parent
    result = held.document(result_path, digest, limit=1024**2)
    require(result['schema'] == 'zerogravity.qwen35-cuda-reference.result.v1'
            and result['mode'] == result['ancestry_stage'] == 'capture'
            and result['original_trajectory_validation'] == 'PASS'
            and result['eight_calls_complete'] is True and result['natural_calls'] == 8
            and all(result[name] is False for name in
                    ('native_recipes_qualified', 'model_execution_qualified', 'performance_qualified')),
            'passing complete packed reference, no native/model/performance promotion')
    import hashlib
    require(result['ancestor_manifest_sha256'] == hashlib.sha256((reference_root / 'capture-pins.json').read_bytes()).hexdigest(),
            'exact separate full-ABI collector/source ancestry')
    request = result['request']
    require(request['model'] == 'Qwen/Qwen3.5-0.8B' and request['prompt_tokens'] in (48, 49, 50)
            and request['phase'] == 'prefill' and request['cached_tokens'] == 0
            and request['request_batch_size'] == 1, 'only actual new source/prompt additions')
    comparison = held.document(root / 'discovery-comparison.json', result['discovery_comparison_sha256'])
    require(comparison['status'] == 'PASS_COMPLETE_SAME_REQUEST_ORIGINAL_BYTES',
            'actual completed same-request source/state/fullV/prior comparison')
    reference.command_complete(held.document(root / 'comparison-command.json', limit=1024**2), held, root, 'comparison')
    discovery = stack.enter_context(reference.Discovery(Path(comparison['discovery']), request))
    require(result['tokenized_request'] == discovery.result['tokenized_request'], 'same original tokenizer operands')
    prepared_capture(held, root, result, reference_root)
    # Re-run only pure held-byte comparisons. This invokes no model, provider,
    # validator subprocess or device and cannot turn failed references green.
    checked = reference.compare(discovery, root)
    for key in set(checked) | set(comparison):
        if key in ('held_discovery_inputs', 'held_capture_inputs'):
            def pins(rows):
                require(len({row['path'] for row in rows}) == len(rows), 'unique held comparison inputs')
                return {row['path']: (row['bytes'], row['sha256']) for row in rows}
            require(pins(checked[key]) == pins(comparison[key]), 'same held comparison source/byte identities')
        else:
            require(checked[key] == comparison[key], 'same complete original input/byte comparison receipt')
    for pin in comparison['held_capture_inputs']:
        path = Path(pin['path'])
        require(path.is_absolute() and path.is_relative_to(root) and '..' not in path.parts,
                'only this accepted capture owns its inputs')
        held.hold(path, expected_sha=pin['sha256'], expected_bytes=pin['bytes'])
    trajectory = held.document(root / 'trajectory-validation.json', result['trajectory_sha256'])
    require(trajectory['original']['source_receipt_sha256'] ==
            'c4c1d349743c0e0ba51ab07f769348e6b13f9896c354303c7af5e839c11dbf09',
            'same original BF16 checkpoint source')
    return root, request['prompt_tokens'], trajectory, discovery
