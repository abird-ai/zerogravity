"""Exact fresh Vulkan Gemma source bytes AND modes from the supplied ROOT manifest."""
import stat
from pathlib import Path

from pins import parsed, pin, relative, require

MANIFEST = 'assembled-source.json'
CONTROL = 'control.json'
CLOSURE = 'source-closure.json'
MANIFEST_LIMIT = 16 * 1024**2
SOURCE_MEMBER_LIMIT = 64 * 1024**2
SOURCE_TOTAL_LIMIT = 192 * 1024**2
REQUIRED_CONFIG = ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml',
                   'scripts/native-gpu.py', 'README.md')
IGNORE = ('.git', '.agents', 'target')


def check_source(base):
    """Bind the staged tree to the exact supplied ROOT manifest and control."""
    manifest, manifest_pin = parsed(base/MANIFEST, limit=MANIFEST_LIMIT)
    control, control_pin = parsed(base/CONTROL)
    closure, closure_pin = parsed(base/CLOSURE)
    files = manifest.get('files')
    require(isinstance(files, dict) and files, 'complete supplied ROOT manifest files')
    if 'count' in manifest:
        require(manifest['count'] == len(files), 'manifest count equals records')
    require(control.get('source_manifest') == manifest_pin
            and control.get('source_count') == len(files)
            and control.get('changed_paths') == {}
            and control.get('source_closure') == closure_pin,
            'controller bound to the complete supplied ROOT manifest')
    records = closure.get('records')
    require(isinstance(records, dict), 'source closure records')
    require(closure.get('source_manifest') == manifest_pin
            and closure.get('source_count') == len(files)
            and closure.get('files') == len(files)
            and set(records) == set(files),
            'source closure exact supplied ROOT manifest union')
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
    for name in REQUIRED_CONFIG:
        require(name in files, 'required workspace/build/config input: '+name)
    source = base/'source'
    root = Path(source).resolve()
    seen = set()
    for path in sorted(source.rglob('*')):
        rel = path.relative_to(source)
        if any(part in IGNORE for part in rel.parts):
            continue
        require(not path.is_symlink(), 'no source symlink: '+rel.as_posix())
        if path.is_dir():
            continue
        require(path.is_file(), 'regular source file: '+rel.as_posix())
        require(path.resolve().is_relative_to(root),
                'source parent alias refused: '+rel.as_posix())
        seen.add(rel.as_posix())
    require(seen == set(files), 'exact declared source cohort, no extras')
    for name, expected in files.items():
        path = source/relative(name)
        require(not path.is_symlink() and path.is_file()
                and path.resolve().is_relative_to(root),
                'source regular member only: '+name)
        require(stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) == expected['mode'],
                'source mode: '+name)
        proof = pin(path)
        require(proof == {k: expected[k] for k in ('bytes', 'sha256')},
                'complete source bytes: '+name)
        require(records[name] == proof, 'source closure pin: '+name)
    return {'source_manifest': manifest_pin, 'control': control_pin,
            'source_closure': closure_pin, 'source_count': len(files),
            'files': len(seen)}
