#!/usr/bin/env python3
"""Bind the ACTUAL closed H build to its one artifact and build receipt."""
import argparse
import json
from pathlib import Path
import os
import sys

BASE = 'resume-all-five-hip-20261025-2'
ARTIFACT = 'release/examples/hip_shared_models'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path,
                        default=Path(__file__).resolve().parent/'payload')
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = args.payload
    if not (payload.is_absolute() and payload.is_dir() and not payload.is_symlink()):
        raise ValueError('absolute prepared payload directory')
    sys.path[:0] = [str(payload/'runner'), str(payload/'physical-runner')]
    sys.dont_write_bytecode = True
    from pins import parsed, pin, require
    from bootstrap_receipt import _owner, _one
    require(args.build_dir.is_absolute() and args.output.is_absolute()
            and not os.path.lexists(args.output)
            and not args.output.is_relative_to(Path(os.path.realpath(payload))),
            'fresh absolute binding path outside the payload')
    require(args.output.parent.is_dir() and not args.output.parent.is_symlink()
            and Path(os.path.realpath(args.output)) == args.output,
            'canonical symlink-free binding path')
    control, control_pin = parsed(payload/'control.json')
    manifest, manifest_pin = parsed(payload/'assembled-source.json', limit=16*1024**2)
    packaged, closure_pin = parsed(payload/'source-closure.json')
    actual = {p.relative_to(payload).as_posix(): pin(p)
              for p in sorted(payload.rglob('*')) if p.is_file()}
    require(control['source_manifest'] == manifest_pin
            and control['source_count'] == len(manifest['files'])
            and control['changed_paths'] == {}
            and set(control['records']) == set(actual)-{'control.json','source-closure.json'}
            and control['records'] == {k: actual[k] for k in control['records']}
            and packaged['source_manifest'] == manifest_pin
            and packaged['source_count'] == len(manifest['files'])
            and set(packaged['records']) == set(actual)-{'source-closure.json'}
            and packaged['records'] == {k: actual[k] for k in packaged['records']},
            'binding uses the complete prepared source manifest')
    remote = _one((args.build_dir/'execution.log').read_bytes(), False)
    _owner(remote)
    require(remote['mode'] == 'build' and remote['status'] == 'CLOSED_PASS_SCOPED'
            and remote['control'] == control_pin and remote['control_after'] == control_pin
            and remote['source_closure'] == closure_pin
            and remote['source_after'] == 'UNCHANGED' and remote['errors'] == [],
            'actual closed assembled-source build only')
    hydrated = remote['hydrated_closure']
    require(type(hydrated) is dict and set(hydrated) == {'bytes', 'sha256'}
            and type(hydrated['bytes']) is int and hydrated['bytes'] > closure_pin['bytes']
            and hydrated['sha256'] != closure_pin['sha256'], 'separate hydrated source closure')
    artifact = remote['build_artifact']
    path = Path(artifact['path'])
    require(path.name == Path(ARTIFACT).name and path.is_absolute()
            and path.parent.as_posix().endswith('/target/'+str(Path(ARTIFACT).parent))
            and artifact['source_closure'] == closure_pin
            and artifact['hydrated_closure'] == hydrated,
            'actual built artifact path/source binding')
    proof = artifact['proof']
    require(type(proof) is dict and set(proof) == {'bytes', 'sha256'}
            and type(proof['bytes']) is int and 0 < proof['bytes'] <= 2*1024**3
            and len(proof['sha256']) == 64 and proof['sha256'] == proof['sha256'].lower(),
            'actual built artifact proof')
    source_base = path.as_posix().rsplit('/target/', 1)[0]
    require(Path(source_base).name == BASE, 'actual new owned H source base')
    receipt = remote['build_receipt']
    require(type(receipt) is dict and set(receipt) == {'bytes', 'sha256'}
            and type(receipt['bytes']) is int and 0 < receipt['bytes'] <= 1024**2
            and len(receipt['sha256']) == 64 and receipt['sha256'] == receipt['sha256'].lower(),
            'actual new build receipt pin')
    provenance = pin(args.build_dir/'invocation.json')
    binding = {'schema': 'zg.hip-source20-build-binding.v1',
               'source_base': source_base,
               'source_closure': closure_pin,
               'binary': proof, 'build_receipt': receipt,
               'provenance': provenance}
    with args.output.open('x') as file:
        file.write(json.dumps(binding, indent=2)+'\n')
    print(json.dumps({'status': 'BOUND_NEW_H_BUILD', 'binding': pin(args.output),
                      'binary': proof, 'build_receipt': receipt,
                      'provenance': provenance}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
