"""Owned offline original CPU operator capture; no import-time model/runtime load."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from contract import HERE, HELPER_SHA, IMAGE, IMAGE_DIGEST, digest, fingerprint, fixture, package, read, require, write


def helper_load():
    require(digest(read(HERE / 'owned_container_v4.py')) == HELPER_SHA, 'unchanged original lifecycle owner')
    spec = importlib.util.spec_from_file_location('portable_original_owner', HERE / 'owned_container_v4.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def environment(cpus):
    return {'CUDA_VISIBLE_DEVICES': '-1', 'NVIDIA_VISIBLE_DEVICES': 'void', 'ATEN_CPU_CAPABILITY': 'avx2',
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'USE_HUB_KERNELS': '0', 'OMP_NUM_THREADS': '4',
               'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'OMP_DYNAMIC': 'FALSE', 'MKL_DYNAMIC': 'FALSE',
               'ZG_ALLOWED_CPUS': ','.join(map(str, cpus)), 'PYTHONPATH': '/probe', 'PYTHONDONTWRITEBYTECODE': '1', 'LD_PRELOAD': '', 'LD_AUDIT': '', 'VLLM_PLUGINS': ''}


def command(helper, path, env, name, nonce):
    args = helper.PODMAN + ['run', '--name', name, '--label', helper.LABEL + '=' + nonce,
        '--cidfile', str(path / 'owned.cid'), '--rm', '--pull=never', '--network=none', '--read-only',
        '--cap-drop=all', '--security-opt=no-new-privileges', '--memory=12g', '--memory-swap=12g', '--cpus=4',
        '--pids-limit=256', '--ulimit', 'nofile=1024:1024', '--ulimit', 'fsize=1048576:1048576',
        '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g']
    for key, value in env.items():
        args += ['-e', key + '=' + value]
    return args + ['-v', str(HERE) + ':/probe:ro', '-v', str(path / 'source') + ':/source:ro',
                  '-v', str(path / 'payload') + ':/output:rw', '--entrypoint', 'python3', IMAGE, '/probe/capture.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, choices=(1024, 2048), required=True)
    parser.add_argument('--new-output', type=Path, required=True)
    args = parser.parse_args()
    path = args.new_output
    require(path.is_absolute() and not path.exists() and path.parent.is_dir(), 'fresh absolute output')
    path = path.parent.resolve() / path.name
    require(path != HERE and HERE not in path.parents and path not in HERE.parents, 'output disjoint from package')
    before_package = package()
    helper = helper_load()
    cpus = sorted(os.sched_getaffinity(0))[:4]
    require(len(cpus) == 4, 'four allowed CPUs')
    os.sched_setaffinity(0, cpus)
    path.mkdir(mode=0o700)
    source = path / 'source'
    output = path / 'payload'
    source.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    raw = fixture(args.width)
    source_path = source / 'source.safetensors'
    source_path.write_bytes(raw)
    plan = {'width': args.width, 'rows': 97, 'token': 16, 'source_sha256': digest(raw), 'source_bytes': len(raw)}
    write(output / 'input.json', plan)
    del raw
    fd = os.open(source_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    before = fingerprint(os.fstat(fd))
    require(digest(os.pread(fd, plan['source_bytes'] + 1, 0)) == plan['source_sha256'], 'held source before math')
    nonce = uuid.uuid4().hex
    name = 'zg-portable-bf16-original-' + nonce
    result = {'status': 'UNQUALIFIED', 'package': before_package, 'plan': plan, 'owned_name': name, 'owned_nonce': nonce,
              'image_id': IMAGE, 'image_digest': IMAGE_DIGEST, 'source_before': before, 'expected_cpus': cpus}
    child = log = None
    failure = None
    start = time.monotonic()

    def stop(_signum, _frame):
        raise InterruptedError('bounded original reference stopped; retire exact owner')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(signum, stop)
    signal.alarm(360)
    try:
        inspected = helper.command(helper.PODMAN + ['image', 'inspect', IMAGE], 15)
        require(inspected.returncode == 0, 'approved immutable image already present; no acquisition')
        (path / 'image-inspect.json').write_bytes(inspected.stdout)
        actual, = json.loads(inspected.stdout)
        require(actual['Id'] == IMAGE.removeprefix('sha256:') and actual['Digest'] == IMAGE_DIGEST, 'actual pinned original image')
        argv = command(helper, path, environment(cpus), name, nonce)
        result['argv'] = argv
        write(path / 'prepared-invocation.json', result)
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        result['owned_cli_pid'] = child.pid
        log = helper.Log(child.stdout, path / 'execution.log')
        result['exit_code'] = log.observe(child, 240)
    except BaseException as error:
        failure = error
    finally:
        result['cleanup_errors'] = helper.cleanup(child, name, nonce, path / 'owned.cid', result) if child is not None else []
        if log is not None:
            try:
                log.finish()
            except BaseException as error:
                result['cleanup_errors'].append('log:' + repr(error))
            result['log_cleanup'] = {'pipe_eof': log.eof, 'all_closed': log.closed, 'total_bytes': log.total, 'retained_bytes': log.retained}
        try:
            require(fingerprint(os.fstat(fd)) == before == fingerprint(os.stat(source_path, follow_symlinks=False))
                    and digest(os.pread(fd, plan['source_bytes'] + 1, 0)) == plan['source_sha256']
                    and fingerprint(os.fstat(fd)) == before, 'same held original source after owner closure')
            result['source_after'] = fingerprint(os.fstat(fd))
            require(package() == before_package, 'same reference source package after owner closure')
            require(failure is None and result.get('exit_code') == 0 and not result['cleanup_errors'], 'actual original/ownership closure')
            result['status'] = 'ORIGINAL_PORTABLE_BF16_REFERENCE_REQUIRES_VALIDATION'
        except BaseException as error:
            failure = failure or error
        finally:
            os.close(fd)
            result['source_FDs_closed'] = True
        result['elapsed_seconds_not_benchmark'] = time.monotonic() - start
        if failure is not None:
            result.update(status='FAIL_UNQUALIFIED', failure=repr(failure))
        write(path / 'invocation.json', result)
    require(result['status'] == 'ORIGINAL_PORTABLE_BF16_REFERENCE_REQUIRES_VALIDATION', 'preserved failed attempt')


if __name__ == '__main__':
    main()
