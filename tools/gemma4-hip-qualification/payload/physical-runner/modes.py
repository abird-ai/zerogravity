"""Complete fresh source manifest bytes/modes for the ROOT-assembled workspace."""
import os
import stat

from pins import parsed, pin, relative, require

MANIFEST = 'assembled-source.json'
CONTROL = 'control.json'
IGNORE = ('.git', '.agents', 'target')  # matches the manifest's excluded directory names
BASE = 'resume-all-five-gemma4-20261025-2'
REQUIRED_CONFIG = ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml',
                   'scripts/check.sh', 'scripts/native-gpu.py', 'README.md')
# Match runner/bootstrap.extract_archive: 64 MiB/member, 192 MiB complete source.
MANIFEST_LIMIT = 16*1024**2
SOURCE_MEMBER_LIMIT = 64*1024**2
SOURCE_TOTAL_LIMIT = 192*1024**2
REQUIRED_UNITS = ('crates/zerogravity-core/',
                  'crates/zerogravity-model-qwen35/',
                  'crates/zerogravity-model-gemma4/',
                  'crates/zerogravity-gpu/',
                  'crates/zerogravity/')


def check_source(base):
    manifest, manifest_pin = parsed(base/MANIFEST, limit=MANIFEST_LIMIT)
    control, control_pin = parsed(base/CONTROL)
    files = manifest.get('files')
    require(isinstance(files, dict) and files, 'complete source manifest files')
    require(type(len(files)) is int and len(files) > 0, 'derived source count')
    if 'count' in manifest:
        require(manifest['count'] == len(files), 'manifest count equals records')
    require(control['source_manifest'] == manifest_pin
            and control.get('source_count') == len(files)
            and control.get('changed_paths') == {}
            and control.get('base') == BASE,
            'controller bound to this complete source manifest')
    for prefix in REQUIRED_UNITS:
        require(any(name.startswith(prefix) for name in files),
                'complete five-lane unit closure: '+prefix)
        require(prefix+'Cargo.toml' in files,
                'required unit manifest: '+prefix+'Cargo.toml')
    for name, expected in files.items():
        require(name != '.git' and not name.startswith('.git/'), 'Git-free source staging')
        require(expected.get('kind') == 'file', 'file record: '+name)
        require(type(expected.get('bytes')) is int
                and 0 <= expected['bytes'] <= SOURCE_MEMBER_LIMIT,
                'bounded record bytes: '+name)
        require(type(expected.get('mode')) is int
                and 0 <= expected['mode'] <= 0o7777, 'record mode: '+name)
        sha = expected.get('sha256')
        require(isinstance(sha, str) and len(sha) == 64 and sha == sha.lower()
                and all(c in '0123456789abcdef' for c in sha),
                'record sha256: '+name)
    require(sum(expected['bytes'] for expected in files.values()) <= SOURCE_TOTAL_LIMIT,
            'bounded complete source bytes')
    seen = set()
    source = base/'source'
    for path in source.rglob('*'):
        rel = path.relative_to(source)
        if any(part in IGNORE for part in rel.parts):
            continue
        rel = rel.as_posix()
        require(not path.is_symlink(), 'no source symlink: '+rel)
        if path.is_dir():
            continue
        require(path.is_file(), 'regular source file: '+rel)
        seen.add(rel)
    require(seen == set(files), 'exact declared source cohort, no extras')
    for name, expected in files.items():
        path = source/relative(name)
        st = os.lstat(path)
        require(stat.S_ISREG(st.st_mode) and not path.is_symlink(),
                'regular non-symlink source: '+name)
        require(stat.S_IMODE(st.st_mode) == expected['mode'], 'source mode: '+name)
        require(pin(path) == {k: expected[k] for k in ('bytes', 'sha256')},
                'complete source bytes: '+name)
    for name in REQUIRED_CONFIG:
        require(name in files, 'required workspace/build/config input: '+name)
    return {'source_manifest': manifest_pin, 'control': control_pin,
            'source_count': len(files), 'files': len(seen)}
