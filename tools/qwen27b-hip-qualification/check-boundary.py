#!/usr/bin/env python3
"""Boundary check for the one real native-gpu wrapper capture: derive the mandatory
banner from the pinned source, compare with the carried parser, and require exactly
one banner followed by one control result. No cargo, no SSH, no model/GPU."""
import argparse
import ast
import json
import sys
from pathlib import Path


def derived_banner(source):
    tree = ast.parse(source.read_text())
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print'
                and len(node.args) == 1 and isinstance(node.args[0], ast.JoinedStr)):
            values = node.args[0].values
            literals = [v for v in values if isinstance(v, ast.Constant)]
            text = ''.join(str(v.value) for v in literals)
            if 'native-gpu: ' in text and 'mandatory user/network isolation' in text:
                names = [v.value.id for v in values if isinstance(v, ast.FormattedValue)
                         and isinstance(v.value, ast.Name)]
                found.append((node.args, names))
    if len(found) != 1:
        raise SystemExit('BOUNDARY-CHECK FAIL: expected one mandatory banner print, got '+str(len(found)))
    arguments, names = found[0]
    binding = {'llvm_status': 'LLVM verification not requested'}
    if set(names) - set(binding):
        raise SystemExit('BOUNDARY-CHECK FAIL: unexpected banner variables '+repr(names))
    expression = ast.fix_missing_locations(ast.Expression(ast.Tuple(elts=list(arguments), ctx=ast.Load())))
    parts = eval(compile(expression, '<banner>', 'eval'), {'__builtins__': {}}, binding)
    return (''.join(parts)+'\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrapper-log', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.payload/'runner'))
    from bootstrap_receipt import BANNER, _one
    result = {'status': 'UNQUALIFIED', 'errors': []}
    try:
        derived = derived_banner(args.source/'scripts/native-gpu.py')
        if derived != BANNER:
            result['errors'].append('carried BANNER differs from pinned native-gpu.py literal')
        raw = args.wrapper_log.read_bytes()
        if not raw.startswith(derived):
            result['errors'].append('wrapper log does not start with the mandatory banner')
        body = raw[len(derived):] if raw.startswith(derived) else raw
        if derived in body:
            result['errors'].append('more than one mandatory banner line')
        if not body.startswith(b'CONTROL-CHECKS '):
            result['errors'].append('control result does not follow the banner exactly')
        _one(derived+b'{"a":1}\n', True)
        for variant in (derived.rstrip(b'\n')+b'{"a":1}\n', derived+derived+b'{"a":1}\n',
                        derived+b'{"a":1}\nextra'):
            try:
                _one(variant, True)
            except BaseException:
                continue
            result['errors'].append('banner/nested parser accepted a banner or trailing variant')
        control = json.loads((args.output/'CONTROL-RESULT.json').read_text())
        if control['status'] != 'PASS_GPU_FREE_GROUPED_CONTROL' or control['errors']:
            result['errors'].append('grouped control result is not PASS')
        result['banner'] = {'bytes': len(derived), 'sha256': __import__('hashlib').sha256(derived).hexdigest()}
        result['wrapper_bytes'] = len(raw)
    except BaseException as error:
        result['errors'].append(repr(error))
    result['status'] = 'BOUNDARY_CHECK_PASS' if not result['errors'] else 'BOUNDARY_CHECK_FAIL'
    (args.output/'BOUNDARY-CHECK-RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print('BOUNDARY-CHECK '+result['status'])
    for error in result['errors']:
        print(error)
    return 0 if result['status'] == 'BOUNDARY_CHECK_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
