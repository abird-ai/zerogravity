#!/usr/bin/env python3
"""Bind the ACTUAL closed qwen27b natural x2 build to its artifact and build receipt."""
import argparse
import json
from pathlib import Path
import sys

PAYLOAD = Path(__file__).resolve().parent/'payload'
sys.path[:0] = [str(PAYLOAD/'runner'), str(PAYLOAD/'physical-runner')]
from pins import parsed, pin, require
from bootstrap_receipt import _owner, _one

BASE = 'hip-multiset-natural-x2-20261024-1'
ARTIFACT = 'target/release/examples/qwen27b_hip_natural'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.build_dir.is_absolute() and args.output.is_absolute() and not args.output.exists(),
            'fresh absolute binding path')
    control, control_pin = parsed(PAYLOAD/'control.json')
    _, closure_pin = parsed(PAYLOAD/'source-closure.json')
    remote = _one((args.build_dir/'execution.log').read_bytes(), False)
    _owner(remote)
    require(remote['mode'] == 'build' and remote['status'] == 'CLOSED_PASS_SCOPED'
            and remote['control'] == control_pin and remote['control_after'] == control_pin
            and remote['source_closure'] == closure_pin
            and remote['source_after'] == 'UNCHANGED' and remote['errors'] == [],
            'actual closed qwen27b natural x2 build only')
    hydrated = remote['hydrated_closure']
    require(type(hydrated) is dict and set(hydrated) == {'bytes', 'sha256'}
            and type(hydrated['bytes']) is int and hydrated['bytes'] > closure_pin['bytes']
            and hydrated['sha256'] != closure_pin['sha256'], 'separate hydrated source closure')
    artifact = remote['build_artifact']
    require(artifact['path'].endswith('/'+ARTIFACT) and artifact['source_closure'] == closure_pin
            and artifact['hydrated_closure'] == hydrated
            and type(artifact['proof']['bytes']) is int and 0 < artifact['proof']['bytes'] <= 2*1024**3
            and len(artifact['proof']['sha256']) == 64
            and artifact['proof']['sha256'] == artifact['proof']['sha256'].lower(),
            'actual built natural example proof')
    receipt = remote['build_receipt']
    require(type(receipt['bytes']) is int and 0 < receipt['bytes'] <= 1024**2
            and len(receipt['sha256']) == 64 and remote['build_receipt']['sha256'] == receipt['sha256'],
            'actual new build receipt pin')
    provenance = {'build_invocation': pin(args.build_dir/'invocation.json'),
                  'build_stdout': pin(args.build_dir/'execution.log')}
    binding = {'schema': 'zg.hip-source20-natural-x2-build-binding.v1',
               'source_base': str(remote['build_artifact']['path']).rsplit('/target/', 1)[0],
               'source_closure': closure_pin,
               'binary': artifact['proof'], 'build_receipt': receipt,
               'provenance': provenance['build_invocation']}
    require(binding['source_base'].endswith('/'+BASE), 'actual new natural build base name')
    args.output.write_text(json.dumps(binding, indent=2)+'\n')
    print(json.dumps({'status': 'BOUND_NEW_QWEN27B_NATURAL_X2_BUILD', 'binding': pin(args.output),
                      'binary': binding['binary'], 'build_receipt': binding['build_receipt'],
                      'provenance': provenance}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
