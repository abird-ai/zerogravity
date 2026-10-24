"""Independent metadata/whole-byte validator; never loads original Torch."""
import argparse
from pathlib import Path
from contract import (IMAGE, IMAGE_DIGEST, MODEL_SHA, TORCH_GIT, digest, fixture,
                      metadata, package, parse, read, require, source_joins, write)


def validate(directory):
    invocation = metadata(directory / 'invocation.json')
    require(invocation['status'] == 'ORIGINAL_PORTABLE_BF16_REFERENCE_REQUIRES_VALIDATION'
            and invocation['exit_code'] == 0 and invocation['cleanup_errors'] == []
            and invocation['source_FDs_closed'] is True and invocation['source_before'] == invocation['source_after'],
            'actual source/original owner closure')
    require(invocation['image_id'] == IMAGE and invocation['image_digest'] == IMAGE_DIGEST
            and invocation['package'] == package(), 'same immutable original image/reference package')
    owner = invocation['foreground_cleanup']
    require(owner['all_nonzero_signals_before_reap'] is True and owner['leader_reaped'] is True
            and owner['group_absent'] is True and owner['kill_attempts_before_reap'] == 2
            and owner['stage_errors'] == [] and invocation['owned_container_absent_after_reap'] is True,
            'TERM/two KILL before reap, actual container/group absence')
    require(invocation['log_cleanup']['pipe_eof'] is True and invocation['log_cleanup']['all_closed'] is True
            and invocation['log_cleanup']['total_bytes'] == invocation['log_cleanup']['retained_bytes'] <= 1024*1024,
            'actual output EOF after all owners retired')
    from run_reference import command, environment, helper_load
    cpus = invocation['expected_cpus']
    require(len(cpus) == len(set(cpus)) == 4 and all(type(c) is int and c >= 0 for c in cpus), 'four actual CPU IDs')
    nonce = invocation['owned_nonce']
    require(len(nonce) == 32 and all(c in '0123456789abcdef' for c in nonce)
            and invocation['owned_name'] == 'zg-portable-bf16-original-' + nonce, 'exact owned nonce/name')
    require(invocation['argv'] == command(helper_load(), directory, environment(cpus), invocation['owned_name'], nonce),
            'exact original foreground/image/mount/environment/resource argv')
    inspected, = metadata(directory / 'image-inspect.json')
    require(inspected['Id'] == IMAGE.removeprefix('sha256:') and inspected['Digest'] == IMAGE_DIGEST, 'actual image inspect')
    plan = metadata(directory / 'payload/input.json')
    require(plan == invocation['plan'] and plan['rows'] == 97 and plan['token'] == 16, 'same fixed operator cohort')
    source = read(directory / 'source/source.safetensors')
    require(source == fixture(plan['width']) and digest(source) == plan['source_sha256']
            and len(source) == plan['source_bytes'], 'complete actual three-role source')
    capture = metadata(directory / 'payload/capture.json')
    require(capture['schema'] == 'zg.portable-bf16-original.v1'
            and capture['status'] == 'ORIGINAL_REFERENCE_REQUIRES_VALIDATION'
            and capture['torch'] == '2.13.0+cu129' and capture['torch_git'] == TORCH_GIT
            and capture['transformers'] == '5.17.0' and capture['modeling_sha256'] == MODEL_SHA
            and capture['cpu_capability'] == 'AVX2', 'actual pinned original operator provenance')
    require(capture['address_space_limit'] == [16*1024**3]*2 and capture['fd_limit'] == [1024,1024], 'actual original AS/FD limits')
    require(capture['affinity'] == cpus and capture['threads'] == 4 and capture['interop_threads'] == 1, 'actual CPU/thread bounds')
    require(capture['scope'] == {'synthetic_source': True, 'trained_model': False, 'portable_arithmetic_qualified': False,
                                'runtime_rust_source_in_original': False, 'state_root': False, 'serving': False, 'performance': False},
            'expected-only finite operator scope')
    require([capture[k] for k in ('source_sha256', 'width', 'rows', 'token')]
            == [plan[k] for k in ('source_sha256', 'width', 'rows', 'token')], 'original/source call join')
    raw = read(directory / 'payload/reference.bin')
    require(len(raw) == capture['bytes'] and digest(raw) == capture['sha256'], 'whole original reference bytes')
    require(parse(raw, plan['source_sha256'], plan['width']) == capture['records'], 'every materialized original stage')
    source_joins(raw, source, capture['records'], plan['width'])
    return {'status': 'ACCEPTED_EXPECTED_ONLY_SYNTHETIC_PUBLIC_OPERATOR_REFERENCE',
            'source_sha256': plan['source_sha256'], 'width': plan['width'], 'rows': 97, 'token': 16,
            'reference': {'path': str((directory / 'payload/reference.bin').resolve()), 'bytes': len(raw), 'sha256': digest(raw)},
            'invocation_sha256': digest(read(directory / 'invocation.json')),
            'trained_model': False, 'portable_arithmetic_qualified': False, 'state_root': False, 'serving': False, 'performance': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--new-accepted', type=Path, required=True)
    args = parser.parse_args()
    require(args.original.is_absolute() and args.new_accepted.is_absolute(), 'absolute input/output paths')
    write(args.new_accepted, validate(args.original))


if __name__ == '__main__':
    main()
