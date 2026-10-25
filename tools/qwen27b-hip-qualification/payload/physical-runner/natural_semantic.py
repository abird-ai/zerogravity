"""Finite acceptance for the qwen27b natural two-position diagnostic report.

Reads only the report the runtime actually wrote. No expected value is supplied
here: the runtime compared the held natural corpus itself, and this module checks
that the claimed scopes, selections, budgets and fault disposition are consistent
with that one report. Shapes are the actual serde JSON shapes emitted by
crates/zerogravity/examples/qwen27b_hip_natural.rs.
"""
from pins import require

GIB = 1024**3
DIAGNOSTIC_HOST_BYTES = 384*1024**2
MODES = ('healthy', 'cached-unknown', 'session-known', 'session-release-fault',
         'snapshot-known', 'snapshot-release-fault')
FAULT_MODES = frozenset(('cached-unknown', 'session-release-fault', 'snapshot-release-fault'))
FINAL_MODES = frozenset(('session-known', 'session-release-fault',
                         'snapshot-known', 'snapshot-release-fault'))
SCHEMA = 'zg.qwen27b-natural-two-position-diagnostic.v2'
VOCAB = 248320


def selection(value, key):
    """Actual `(id, max_bits)` tuple serialized as a two-element array."""
    require(type(value) is list and len(value) == 2
            and type(value[0]) is int and 0 <= value[0] < VOCAB
            and type(value[1]) is int and 0 <= value[1] <= 0xffffffff,
            'actual original fullV selection '+key)
    return value


def closure(value, key):
    require(type(value) is dict and value['exact'] is True and value['mismatches'] == 0,
            'scope closure exact from held bytes '+key)
    return value


def final_reader(value, mode):
    wrapper = 'Snapshot' if mode.startswith('snapshot-') else 'Session'
    fault = mode in FAULT_MODES
    require(type(value) is dict and value['lease_bytes'] == 12
            and value['aliases_inside_wrapper'] == 1
            and value['candidate_owners_before_drop'] == 1
            and value['model_owners_before_drop'] == 1
            and value['candidate_workspace_target_aliases'] == 0
            and value['candidate_retired'] is True
            and value['release_hits'] == 1
            and value['reader_hits_before_native_retirement'] > 0
            and value['native_modules_before'] > 0
            and value['native_retirement_attempts'] > 0
            and value['wrapper'] == wrapper
            and value['fault'] is fault,
            'true typed final12B reader record joined to session/snapshot/fault mode')
    if fault:
        require(value['native_admission_retained_after'] is True
                and value['native_modules_after'] > 0
                and value['original_service_retained_after'] is True
                and value['quarantined_handles_after'] > 0,
                'fault final reader retains actual original native/service/quarantine')
    else:
        require(value['native_admission_retained_after'] is False
                and value['native_modules_after'] == 0
                and value['original_service_retained_after'] is False
                and value['quarantined_handles_after'] == 0,
                'known final reader retires actual original native/service/quarantine')
    return value


def validate_natural(mode, report, log):
    require(mode in MODES and report['mode'] == mode and report['schema'] == SCHEMA,
            'actual natural diagnostic report for this mode')
    require('QWEN27B_NATURAL_REFUSED' not in log,
            'actual held natural join accepted before model work')
    fault = mode in FAULT_MODES
    final = mode in FINAL_MODES
    require(report['status'] == 'PASS_DIAGNOSTIC_NATURAL_UNQUALIFIED',
            'actual natural diagnostic pass: '+str(report['failure']))
    require(report['failure'] is None and report['actual'] is not None,
            'actual natural result present')
    require(report['trained_native_natural_serving_or_performance_qualified'] is False
            and report['lifecycle_cases_qualified'] is False,
            'no lifecycle/serving/performance claim from one diagnostic report')
    require(report['source_close_refused'] is fault,
            'source close refused exactly on the fault modes')
    require(report['physical_reserved'] == (16*GIB if fault else 4*GIB),
            'actual physical reservation: 16GiB fault fence or 4GiB model')
    require(report['physical_fenced'] is fault, 'physical fence exactly on fault modes')
    require(report['diagnostic_host_bytes'] == DIAGNOSTIC_HOST_BYTES,
            'actual diagnostic host harness extent')
    h0 = 2*GIB + DIAGNOSTIC_HOST_BYTES
    if fault:
        # Bounded retention predicate: a fault may retain the actual original
        # native modules/service/quarantine, so HOST is unfenced, at most the
        # original 16GiB reserve and at least H0 plus the distinct control and
        # retained module-HOST extents. This is not exact fault accounting.
        retained = 64*1024**2 + 4*1024**2
        if final:
            reader = report['actual']['original_final_reader']
            retained += reader['native_host_bytes_before']
        require(report['host_fenced'] is False
                and h0 + retained <= report['host_reserved'] <= 16*GIB,
                'fault HOST bounded by original 16GiB and actual retained owners')
    else:
        require(report['host_reserved'] == h0,
                'healthy/known HOST is exactly 2GiB+384MiB')
        require(report['host_fenced'] is False, 'healthy/known host not fenced')
    inventory = report['diagnostic_inventory']
    require(type(inventory) is dict and inventory['host_bytes'] == DIAGNOSTIC_HOST_BYTES
            and 0 < inventory['peak_bytes'] <= inventory['host_bytes']
            and inventory['root_bytes'] == 159121408
            and inventory['readback_window_bytes'] == 3*1024**2
            and inventory['domain_metadata_bytes'] > 0
            and inventory['reference_bytes'] > 0
            and inventory['admission_bytes'] <= inventory['host_bytes']
            and inventory['observed_bytes'] <= inventory['host_bytes']
            and inventory['snapshot_bytes'] <= inventory['host_bytes']
            and inventory['committed_bytes'] <= inventory['host_bytes']
            and inventory['report_bytes'] <= inventory['host_bytes'],
            'actual phase-owned diagnostic inventory maxima inside the 384MiB harness')
    for key in ('mathematical_original_reference_compared', 'ordinary_full_state_compared',
                'full4_compared'):
        require(report[key] is (mode == 'healthy'), 'original comparison scope '+key)
    # Typed source retention is joined to the actual final native observation,
    # and all five original natural FD owners are rechecked before the report.
    unknown = fault or mode == 'cached-unknown'
    require(report['retention_mode'] == ('unknown' if unknown else 'known'),
            'typed source retention disposition')
    require(report['typed_final_retention_join'] is True,
            'actual final native observation agrees with the typed disposition')
    lease = report['reference_lease']
    require(type(lease) is dict and type(lease['owners']) is list
            and len(lease['owners']) == 5 and lease['independent_artifact_pins'] == 5
            and lease['retained_across_validation_and_report'] is True
            and lease['mutable_path_reopen_as_replacement'] is False
            and all(type(o) is dict and o['bytes'] > 0 and len(o['sha256']) == 64
                    and o['fd_identity']['ino'] > 0 for o in lease['owners']),
            'five retained original natural FD owners with independent pins')
    if unknown:
        require(report['retained_source_context_bytes'] > 0
                and report['retained_source_metadata_bytes'] > 0
                and type(report['retained_native_inventory']) is str
                and report['released_host_assertion'] is None,
                'unknown/fault retains the source extent and no released HOST')
    else:
        require(report['retained_source_context_bytes'] == 0
                and report['retained_source_metadata_bytes'] == 0
                and report['retained_native_inventory'] is None
                and report['released_host_assertion'] is True,
                'known disposition releases the actual source extent')
    pin = report['natural_accepted_pin']
    require(type(pin) is dict and set(pin) == {'bytes', 'sha256'}
            and type(pin['bytes']) is int and 0 < pin['bytes'] <= 1024**2
            and len(pin['sha256']) == 64 and pin['sha256'] == pin['sha256'].lower(),
            'externally supplied natural receipt pin')
    actual = report['actual']
    require(type(actual) is dict, 'actual natural result object')
    if mode == 'cached-unknown':
        require(actual['cached_unknown_root_unchanged'] is True
                and actual['published_cached_roots'] == 0
                and type(actual['cached_expected_records']) is int
                and actual['cached_expected_records'] > 0
                and type(actual['cached_expected_words']) is int
                and actual['cached_expected_words'] > 0,
                'cached fault published nothing before its actual refusal')
        cached = actual['cached_expected_closure']
        require(cached['interrupted'] is True and cached['exact_completed_prefix'] is True
                and cached['mismatches'] == 0
                and type(cached['observed']) is dict
                and cached['observed']['records'] == actual['cached_expected_records']
                and cached['observed']['float_words'] == actual['cached_expected_words']
                and cached['unavailable_float_words'] > 0,
                'cached fault closed only its actual observed prefix')
        tuples = actual['selected_output_tuples']
        require(tuples['seed_input'] == 11
                and selection(tuples['seed'], 'cached seed') == [
                    actual['first']['selected_id'], actual['first']['max_bits']]
                and tuples['next'] is None
                and tuples['seed'][0] != 11
                and tuples['distinct_from_seed_input'] is True,
                'cached fault retains the held seed input 11 join without a next output')
        return {'mode': mode, 'interrupted_prefix_only': True}
    if final:
        closure(actual['scope_closure'], 'final call0')
        selection(actual['selection'], 'first')
        tuples = actual['selected_output_tuples']
        require(tuples['seed_input'] == 11
                and selection(tuples['seed'], 'final seed') == [
                    actual['first']['selected_id'], actual['first']['max_bits']]
                and selection(tuples['next'], 'final next') == actual['selection']
                and tuples['seed'][0] != 11 and tuples['next'][0] != 11
                and tuples['distinct_from_seed_input'] is True,
                'final selected output tuples join the held seed input 11')
        final_reader(actual['original_final_reader'], mode)
        scope = actual['scope_closure']
        require(scope['declared_proposals'] == [[0, 0]],
                'final reader closes only the seed call0 scope')
        return {'mode': mode, 'final_reader': True}
    require(actual['actual_full64_calls'] == 9 and actual['source_capacity'] == 2,
            'nine actual full64 calls on a two-position capacity')
    require(actual['diagnostic_committed_state_compared'] is True
            and actual['eos_text_policy_implemented'] is False,
            'committed state compared, no text policy claim')
    packet = actual['ordinary_packet']
    require(packet['seed_readback_bytes'] == 16 and packet['next_readback_bytes'] == 16,
            'ordinary pair reads the actual 12B packet')
    seed = selection(packet['seed'], 'ordinary seed')
    nxt = selection(packet['next'], 'ordinary next')
    tuples = actual['selected_output_tuples']
    require(tuples['seed_input'] == 11
            and selection(tuples['seed'], 'ordinary seed tuple') == seed
            and selection(tuples['next'], 'ordinary next tuple') == nxt
            and tuples['seed'][0] != 11 and tuples['next'][0] != 11
            and tuples['distinct_from_seed_input'] is True,
            'held seed input 11 is distinct from both selected output tuples')
    first = selection(actual['first_selection'], 'first')
    cached = selection(actual['cached_selection'], 'cached')
    peer = selection(actual['peer_selection'], 'independent peer')
    peer_cached = selection(actual['peer_cached_selection'], 'independent peer cached')
    late = selection(actual['late_cancel']['selection'], 'late cancellation')
    replay = selection(actual['restore_replay']['selection'], 'restore replay')
    # The input token 11 belongs to the independently validated source call; the
    # packet is the selected output tuple. Join every actual selector and each
    # completed call's selected_id/max_bits, never a seed-input constant.
    full4 = actual['full4']
    require(type(full4) is dict and full4['full4'] is True
            and full4['natural_source_events_per_call'] == 67
            and type(full4['full4_scope']) is dict
            and type(full4['selections']) is list and len(full4['selections']) == 2
            and all(type(row) is list and len(row) == 2 for row in full4['selections']),
            'two cohorts x two natural positions compared after the fact')
    joined = full4['selections']
    require(joined[0][0] == joined[1][0] == seed == first == peer,
            'ordinary seed = first = peer = both full4 seeds (selected output tuple)')
    require(joined[0][1] == joined[1][1] == nxt == cached == peer_cached == late == replay,
            'ordinary next = every cached/peer/late/replay selection')
    for name, row, expected in (('first', actual['first'], first),
                                ('cached', actual['cached'], cached),
                                ('peer', actual['independent_peer'], peer),
                                ('peer_cached', actual['independent_peer_cached'], peer_cached),
                                ('late', actual['late_cancel']['scope'], late),
                                ('replay', actual['restore_replay']['scope'], replay)):
        require(type(row) is dict and row['selected_id'] == expected[0]
                and row['max_bits'] == expected[1],
                'completed call '+name+' selected_id/max_bits equals its joined tuple')
    require(full4['join']['independent_original_natural_trajectory'] is True
            and full4['join']['actual_observed'] is True
            and full4['join']['teacher_forcing_only'] is False
            and full4['join']['newcalls'] == 0,
            'full4 join reports the actual independent natural trajectory')
    scope4 = full4['full4_scope']
    require(scope4['declared_proposals'] == [[0, 0], [0, 1], [1, 0], [1, 1]]
            and scope4['float_records'] == 1036
            and scope4['float_words'] == 240896000
            and scope4['positions'] == 12
            and scope4['mismatches'] == 0 and scope4['exact'] is True,
            'full4 scope is four proposals, 1036 float records, 240896000 words, 12 positions')
    committed = actual['ordinary_committed']
    require(type(committed) is dict and committed['repeat'] == 0 and committed['call'] == 1
            and committed['domains'] == 112 and committed['words'] == 39780352
            and committed['mismatches'] == 0,
            'ordinary committed comparison is repeat0/call1/112 domains/39780352 words')
    for name in ('late_cancel', 'restore_replay'):
        scope = actual[name]['scope_closure']
        require(scope['declared_proposals'] == [[0, 1]],
                name+' closes only the call1 scope')
        closure(scope, name)
    return {'mode': mode, 'natural_positions': 2}
