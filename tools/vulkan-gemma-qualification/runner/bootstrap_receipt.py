"""Exact source20 wrapper boundary: one mandatory native-gpu banner + one strict JSON."""
import json
from pins import require

# Literal emitted by pinned source20/scripts/native-gpu.py without --require-llvm.
BANNER = b'native-gpu: LLVM verification not requested; mandatory user/network isolation via unshare (setup failure aborts the command)\n'


def _one(raw, banner):
    require(isinstance(raw, bytes) and 0 < len(raw) <= 4*1024**2, 'bounded UTF8 wrapper log')
    text = raw.decode('utf-8', errors='strict')
    if banner:
        expected = BANNER.decode('ascii')
        require(text.startswith(expected), 'exact original mandatory native-gpu banner')
        text = text[len(expected):]
    require(text.startswith('{'), 'exact single JSON object, no extra prefix')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON member')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('non-finite JSON constant: '+value)
    value, end = json.JSONDecoder(object_pairs_hook=pairs, parse_constant=constant).raw_decode(text)
    require(isinstance(value, dict) and not text[end:].strip(), 'one complete JSON, no trailing content')
    return value


def _owner(value):
    require(type(value['observed_exit']) is int and value['observed_exit'] == 0
            and value['errors'] == [], 'actual successful original foreground child')
    c = value['foreground_cleanup']
    require(type(value['pid']) is int and value['pid'] > 0
            and type(c['pgid']) is int and c['pgid'] == value['pid']
            and c['term_sent'] is True and c['kill_sent_before_reap'] is True
            and type(c['exit_code']) is int and c['exit_code'] == 0
            and c['leader_reaped'] is True and c['group_absent'] is True
            and type(c['kill_attempts_before_reap']) is int and c['kill_attempts_before_reap'] == 2
            and type(c['leader_reap_attempts']) is int and c['leader_reap_attempts'] == 1
            and c['all_nonzero_signals_before_reap'] is True and c['stage_errors'] == []
            and type(value['log']['retained']) is int and type(value['log']['total']) is int
            and 0 < value['log']['retained'] == value['log']['total'] <= 4*1024**2
            and value['log']['eof'] is True and value['log']['closed'] is True,
            'original group/twoKILL-before-oneReap/leader-reaped/group-absent/fullEOF closure')
    return value


def _hydrate_report(child, closure, cache_sha):
    require(child['source_count'] == closure['source_count']
            and child['source_manifest'] == closure['source_manifest']
            and child['parent_frozen_source_manifest'] == closure['parent_frozen_source_manifest']
            and child['reused_cache_closure_sha256'] == cache_sha
            and type(child['files']) is int and child['files'] >= closure['files']
            and type(child['bytes']) is int and child['bytes'] >= closure['bytes']
            and child['scope'] == closure['scope'], 'exact source20 hydration report')
    return child


def parse_hydrate(raw, receipt, closure, cache_sha):
    """Nested native-gpu child: exact one banner then one hydration report."""
    _owner(receipt)
    return _hydrate_report(_one(raw, True), closure, cache_sha)


def parse_closed_bootstrap(raw, control, archive, closure, closure_pin, cache_sha):
    remote = _one(raw, False)
    _owner(remote)
    require(remote['status'] == 'CLOSED_PASS_SCOPED'
            and remote['control'] == control and remote['control_after'] == control
            and remote['source_archive'] == archive
            and remote['source_archive_after'] == 'UNCHANGED_HELD_FULL_SHA'
            and remote['source_closure'] == closure_pin
            and remote['hydrated_closure']['sha256'] != closure_pin['sha256']
            and remote['hydrated_closure']['bytes'] > closure_pin['bytes']
            and remote['gpu_or_model_executed'] is False, 'same closed bootstrap/controller/archive/source authority')
    _hydrate_report(remote['hydrate'], closure, cache_sha)
    return remote
