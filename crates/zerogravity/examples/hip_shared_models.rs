//! Finite same-actor genuine Qwen/Gemma lifecycle; no serving/speed eligibility.
#[allow(dead_code)]
#[path = "gemma4_cpu_matrix/comparison.rs"]
mod comparison;
#[allow(dead_code)]
#[path = "gemma4_hip_matrix/expected.rs"]
mod gexpected;
#[allow(dead_code)]
#[path = "gemma4_cpu_matrix/io.rs"]
mod io;
#[allow(dead_code)]
#[path = "qwen27b_hip_matrix/expected.rs"]
mod qexpected;
use io::{Proof, need, write_new};
use serde_json::{Value, json};
use std::{
    fmt::{self, Write as _},
    future::Future,
    path::PathBuf,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    thread,
};
use zerogravity::{
    cache::{CacheOptions, CacheStore},
    gemma4_cpu::{Gemma4CpuContext, Gemma4CpuLoaded},
    json_allocation::{ValueConstruction, census_json, census_scratch_peak},
    qwen27b_gpu::{Qwen27bGpuContext, Qwen27bGpuLoaded},
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};
use zerogravity_gpu::{
    Device, Gemma4Bf16GpuCandidate, Gemma4Bf16GpuSession, GemmaTextHostOwners,
    HipSourceSetIdentity, Qwen27bSourceGpuSession, SharedHipNative, SourcePageEventProbeMode,
    SourcePageProbeParts,
};
const GIB: u64 = 1024 * 1024 * 1024;
const PROBE_TEXT_CAP: usize = 8 * 1024;
/// Static diagnostic key names and short literal values stay below this bound.
const DIAGNOSTIC_KEY_CAP: usize = 128;
fn bounded_text(args: fmt::Arguments<'_>) -> String {
    struct Fixed {
        bytes: Vec<u8>,
        truncated: bool,
    }
    impl fmt::Write for Fixed {
        fn write_str(&mut self, text: &str) -> fmt::Result {
            let room = PROBE_TEXT_CAP.saturating_sub(self.bytes.len());
            if text.len() <= room {
                self.bytes.extend_from_slice(text.as_bytes());
            } else {
                let mut end = room.min(text.len());
                while end > 0 && !text.is_char_boundary(end) {
                    end -= 1;
                }
                self.bytes.extend_from_slice(&text.as_bytes()[..end]);
                self.truncated = true;
            }
            Ok(())
        }
    }
    let mut fixed = Fixed {
        bytes: Vec::with_capacity(PROBE_TEXT_CAP + 3),
        truncated: false,
    };
    let _ = fixed.write_fmt(args);
    let mut text = String::from_utf8(fixed.bytes).expect("bounded UTF-8 input");
    if fixed.truncated {
        text.push_str("...");
    }
    text
}
/// Pinned `json_allocation` node geometry, derived from a one-member object:
/// `tree = 2*node + 1` because it has one member, one object and one key byte.
fn diagnostic_json_node_upper() -> Proof<usize> {
    let layout = br#"{"a":1}"#;
    let allocation = census_json(layout, ValueConstruction::SerdeValue)?;
    let tree = allocation.tree_heap_upper();
    let node = tree
        .checked_sub(1)
        .and_then(|n| n.checked_div(2))
        .ok_or("source-page diagnostic json node geometry")?;
    need(
        tree == 2 * node + 1,
        "source-page diagnostic json pinned node geometry",
    )?;
    Ok(node)
}
/// `(members+objects)*node + array_backing + string_capacity` for one
/// source-shaped diagnostic tree. String capacity is bounded per member key
/// and per `bounded_text` field before allocation.
fn diagnostic_tree_upper(
    members: usize,
    objects: usize,
    array_capacity: usize,
    text_fields: usize,
    node: usize,
) -> Proof<usize> {
    let nodes = members
        .checked_add(objects)
        .ok_or("source-page diagnostic json nodes")?;
    let node_bytes = nodes
        .checked_mul(node)
        .ok_or("source-page diagnostic json node bytes")?;
    let array_bytes = array_capacity
        .checked_mul(2)
        .and_then(|n| n.checked_mul(std::mem::size_of::<Value>()))
        .ok_or("source-page diagnostic json array bytes")?;
    let key_bytes = members
        .checked_mul(DIAGNOSTIC_KEY_CAP)
        .ok_or("source-page diagnostic json keys")?;
    let text_bytes = text_fields
        .checked_mul(PROBE_TEXT_CAP + 3)
        .ok_or("source-page diagnostic json text")?;
    let total = node_bytes
        .checked_add(array_bytes)
        .and_then(|n| n.checked_add(key_bytes))
        .and_then(|n| n.checked_add(text_bytes))
        .ok_or("source-page diagnostic json tree")?;
    Ok(total)
}
/// Pretty-serialized added content: escaping, scalar text, punctuation and
/// indentation from the named member/array/container/depth counts.
fn diagnostic_serialized_upper(
    members: usize,
    objects: usize,
    array_entries: usize,
    scalars: usize,
    text_fields: usize,
    depth: usize,
) -> Proof<usize> {
    let key_bytes = members
        .checked_mul(DIAGNOSTIC_KEY_CAP)
        .ok_or("source-page diagnostic serializer keys")?;
    let text_bytes = text_fields
        .checked_mul(PROBE_TEXT_CAP + 3)
        .ok_or("source-page diagnostic serializer text")?;
    let string_bytes = key_bytes
        .checked_add(text_bytes)
        .ok_or("source-page diagnostic serializer strings")?;
    let containers = objects
        .checked_add(array_entries)
        .ok_or("source-page diagnostic serializer containers")?;
    let shaped = members
        .checked_add(array_entries)
        .and_then(|n| n.checked_add(containers))
        .ok_or("source-page diagnostic serializer shape")?;
    let shaped_depth = shaped
        .checked_add(1)
        .ok_or("source-page diagnostic serializer depth")?;
    let escaped = string_bytes
        .checked_mul(6)
        .ok_or("source-page diagnostic serializer escape")?;
    let scalar_text = scalars
        .checked_mul(24)
        .ok_or("source-page diagnostic serializer scalars")?;
    let punctuation = shaped
        .checked_mul(8)
        .ok_or("source-page diagnostic serializer punctuation")?;
    let indentation = depth
        .checked_mul(shaped_depth)
        .and_then(|n| n.checked_mul(2))
        .ok_or("source-page diagnostic serializer indent")?;
    let total = escaped
        .checked_add(scalar_text)
        .and_then(|n| n.checked_add(punctuation))
        .and_then(|n| n.checked_add(indentation))
        .ok_or("source-page diagnostic serializer")?;
    Ok(total)
}
fn diagnostic_sum(values: impl IntoIterator<Item = usize>) -> Proof<usize> {
    let mut total = 0usize;
    for value in values {
        total = total
            .checked_add(value)
            .ok_or("source-page diagnostic sum")?;
    }
    Ok(total)
}
/// Source-derived diagnostic delta only. Actual probe `json!` shapes are
/// 39/7 raw and 166/41 after the ten ledger snapshots and 26 outcome fields;
/// one ledger is 8/3, `native` is 28/3 plus one two-entry array, and the added
/// event is 4/1. The coefficients bound the named simultaneous owners, not
/// retained report rows: four probe trees (accumulated receipt, two raw
/// snapshots, one replacement/clone), two native comparison trees, twenty
/// transient ledger trees and two added-event trees. Existing 128 MiB fixture
/// admission is retained unchanged; this delta is charged inside it once.
fn probe_report_bound() -> Proof<usize> {
    const PROBE_MEMBERS: usize = 166;
    const PROBE_OBJECTS: usize = 41;
    const PROBE_SCALARS: usize = 126;
    const PROBE_TEXT_FIELDS: usize = 5;
    const PROBE_MAX_DEPTH: usize = 4;
    const NATIVE_MEMBERS: usize = 28;
    const NATIVE_OBJECTS: usize = 3;
    const NATIVE_ARRAY_CAPACITY: usize = 4;
    const LEDGER_MEMBERS: usize = 8;
    const LEDGER_OBJECTS: usize = 3;
    const EVENT_MEMBERS: usize = 4;
    const EVENT_OBJECTS: usize = 1;
    const EVENT_SCALARS: usize = 4;
    const EVENT_TEXT_FIELDS: usize = 1;
    const EVENT_MAX_DEPTH: usize = 3;
    const PROBE_TREES: usize = 4;
    const NATIVE_TREES: usize = 2;
    const EXTRA_LEDGER_TREES: usize = 20;
    const EVENT_TREES: usize = 2;

    let census_scratch = census_scratch_peak(br#"{"a":1}"#.len())?;
    let node = diagnostic_json_node_upper()?;
    let probe_tree =
        diagnostic_tree_upper(PROBE_MEMBERS, PROBE_OBJECTS, 0, PROBE_TEXT_FIELDS, node)?;
    let native_tree = diagnostic_tree_upper(
        NATIVE_MEMBERS,
        NATIVE_OBJECTS,
        NATIVE_ARRAY_CAPACITY,
        0,
        node,
    )?;
    let ledger_tree = diagnostic_tree_upper(LEDGER_MEMBERS, LEDGER_OBJECTS, 0, 0, node)?;
    let event_tree =
        diagnostic_tree_upper(EVENT_MEMBERS, EVENT_OBJECTS, 4, EVENT_TEXT_FIELDS, node)?;
    let retained = diagnostic_sum([
        PROBE_TREES
            .checked_mul(probe_tree)
            .ok_or("source-page diagnostic probe trees")?,
        NATIVE_TREES
            .checked_mul(native_tree)
            .ok_or("source-page diagnostic native trees")?,
        EXTRA_LEDGER_TREES
            .checked_mul(ledger_tree)
            .ok_or("source-page diagnostic ledger trees")?,
        EVENT_TREES
            .checked_mul(event_tree)
            .ok_or("source-page diagnostic event trees")?,
    ])?;
    let probe_bytes = diagnostic_serialized_upper(
        PROBE_MEMBERS,
        PROBE_OBJECTS,
        0,
        PROBE_SCALARS,
        PROBE_TEXT_FIELDS,
        PROBE_MAX_DEPTH,
    )?;
    let event_bytes = diagnostic_serialized_upper(
        EVENT_MEMBERS,
        EVENT_OBJECTS,
        1,
        EVENT_SCALARS,
        EVENT_TEXT_FIELDS,
        EVENT_MAX_DEPTH,
    )?;
    let added_bytes = diagnostic_sum([probe_bytes, event_bytes])?;
    // Pinned Vec doubling: old+new live backing for a J-byte output.
    let serializer = added_bytes
        .checked_add(8)
        .and_then(|n| n.checked_mul(3))
        .ok_or("source-page diagnostic serializer growth")?;
    let formatter_fields = diagnostic_sum([
        PROBE_TREES
            .checked_mul(PROBE_TEXT_FIELDS)
            .ok_or("source-page diagnostic formatter probes")?,
        EVENT_TREES
            .checked_mul(EVENT_TEXT_FIELDS)
            .ok_or("source-page diagnostic formatter events")?,
    ])?;
    let formatter = formatter_fields
        .checked_mul(PROBE_TEXT_CAP + 3)
        .ok_or("source-page diagnostic formatter")?;
    diagnostic_sum([retained, serializer, formatter, census_scratch])
}
const QSNAP: &str = "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e";
const GSNAP: &str = "ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1";
struct ThreadWake(thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => thread::park(),
        }
    }
}
fn binding(root: zerogravity::session_root::RootBinding) -> Value {
    json!({"session":root.session().get(),"root":root.root(),"frontier":root.frontier()})
}
fn identity(i: HipSourceSetIdentity) -> Value {
    json!({"source":format!("{:?}",i.source),"source_service_generation":i.source_service_generation,"actor":format!("{:?}",i.actor),"set_key":i.set_key,"logical_stream":i.logical_stream})
}
fn ledger(
    arbiter: &ResourceArbiter,
    host: &PhysicalDeviceKey,
    gpu: &PhysicalDeviceKey,
) -> Proof<Value> {
    let host = arbiter.snapshot(host)?;
    let gpu = arbiter.snapshot(gpu)?;
    Ok(json!({
        "host":{"limit":host.limit,"reserved":host.reserved,"fenced":host.fenced},
        "device":{"limit":gpu.limit,"reserved":gpu.reserved,"fenced":gpu.fenced}
    }))
}
macro_rules! receipt_json {
    ($r:expr) => {{ let r=&$r; json!({
        "wrapper":r.wrapper,"fault":r.fault,"source":r.source,"lease":r.lease,
        "service_generation":r.service_generation,"lease_bytes":r.lease_bytes,
        "aliases_inside_wrapper":r.aliases_inside_wrapper,"candidate_owners_before_drop":r.candidate_owners_before_drop,
        "model_owners_before_drop":r.model_owners_before_drop,"candidate_workspace_target_aliases":r.candidate_workspace_target_aliases,
        "candidate_retired":r.candidate_retired,"release_hits":r.release_hits,
        "native_retirement_attempts":r.native_retirement_attempts,"reader_hits_before_native_retirement":r.reader_hits_before_native_retirement,
        "native_modules_before":r.native_modules_before,"native_code_bytes_before":r.native_code_bytes_before,
        "native_host_bytes_before":r.native_host_bytes_before,"native_device_bytes_before":r.native_device_bytes_before,
        "native_admission_retained_after":r.native_admission_retained_after,"native_modules_after":r.native_modules_after,
        "original_service_retained_after":r.original_service_retained_after,"quarantined_handles_after":r.quarantined_handles_after }) }};
}
fn native(shared: &SharedHipNative) -> Proof<Value> {
    let n = shared.inspect_native()?;
    Ok(
        json!({"sets":n.sets.iter().map(|s| json!({"set_key":s.set_key,"logical_stream":s.logical_stream,"catalogue_keys":s.catalogue_keys,"non_null_modules":s.non_null_modules,"admitted_modules":s.admitted_native.modules,"PT_LOAD_bytes":s.admitted_native.code_object_load_bytes,"retained_code_host_bytes":s.admitted_native.retained_host_bytes,"scoped_metadata_bytes":s.scoped_metadata_bytes,"sealed":s.sealed,"usable":s.usable,"sealed_miss_refused":s.sealed_miss_refused})).collect::<Vec<_>>(),"original_key_overlap":n.original_key_overlap,"default_stream_refused":n.default_stream_refused,"compiler_host_reserved":shared.compiler_host_reserved_bytes()?,"available_model_capacity":shared.available_model_capacity()?,"resident_bytes":shared.resident_bytes()?}),
    )
}
fn live(
    shared: &SharedHipNative,
    q: HipSourceSetIdentity,
    g: HipSourceSetIdentity,
) -> Proof<Value> {
    need(
        q.actor == g.actor
            && q.source != g.source
            && q.source_service_generation != g.source_service_generation
            && q.set_key != g.set_key
            && q.logical_stream.is_some()
            && g.logical_stream.is_some()
            && q.logical_stream != g.logical_stream,
        "same physical actor; distinct actual source service and scoped logical identity",
    )?;
    let n = native(shared)?;
    let rows = n["sets"]
        .as_array()
        .ok_or("actual native set observations")?;
    need(
        rows.len() == 2
            && n["original_key_overlap"].as_u64().is_some_and(|n| n > 0)
            && n["default_stream_refused"] == true,
        "two real sets overlap original full keys but default never routes",
    )?;
    for (id, count) in [(q, 117), (g, 184)] {
        let row = rows
            .iter()
            .find(|s| s["set_key"] == id.set_key)
            .ok_or("actual source set key")?;
        need(
            row["logical_stream"] == id.logical_stream.unwrap()
                && row["catalogue_keys"] == count
                && row["non_null_modules"] == count
                && row["admitted_modules"] == count
                && row["sealed"] == true
                && row["usable"] == true
                && row["sealed_miss_refused"] == true,
            "complete actual source-native catalogue and sealed preflight",
        )?;
    }
    Ok(n)
}
#[derive(Default)]
struct Counts {
    q_calls: usize,
    q_commits: usize,
    g_calls: usize,
    g_instrumented: usize,
    g_commits: usize,
}
fn qstep(
    s: &mut Qwen27bSourceGpuSession,
    expected: &mut qexpected::Expected,
    repeat: usize,
    call: usize,
    counts: &mut Counts,
    events: &mut Vec<Value>,
) -> Proof<()> {
    expected.begin(repeat, call)?;
    let before = s.binding();
    let pending = block_on(s.prepare_teacher(expected, None))?;
    counts.q_calls += 1;
    let comparison = expected.finish()?;
    need(
        comparison["exact"] == true,
        "actual complete Qwen full64/fullV/all112 state domains equal original",
    )?;
    need(s.binding() == before, "Qwen Ready has no publication")?;
    let output = s.commit(pending)?;
    counts.q_commits += 1;
    drop(output);
    need(
        s.binding().root() == before.root() + 1 && s.binding().frontier() == (call + 1) as u64,
        "actual Qwen teacher commit frontier",
    )?;
    events.push(json!({"kind":"qwen-completed-committed","comparison":comparison,"before":binding(before),"after":binding(s.binding())}));
    Ok(())
}
fn gstep(
    s: &mut Gemma4Bf16GpuSession,
    expected: &mut gexpected::Expected,
    normal: bool,
    counts: &mut Counts,
    events: &mut Vec<Value>,
) -> Proof<()> {
    let before = s.binding();
    expected.begin(0);
    let pending = if normal {
        block_on(s.prepare_prompt(1, None, None))?
    } else {
        block_on(s.prepare_prompt(1, Some(expected), None))?
    };
    counts.g_calls += 1;
    if !normal {
        counts.g_instrumented += 1;
    }
    let comparison = if normal {
        expected.normal_packet(0, pending.readback(), before)?;
        json!({"normal_observer_none":true,"packet_exact":true,"ordinary_chunk_bytes":pending.readback().chunk_bytes})
    } else {
        let c = expected.completed(pending.readback(), before)?;
        need(
            c["exact"] == true,
            "all35/fullV/15KV diagnostic comparisons",
        )?;
        c
    };
    need(s.binding() == before, "Gemma Ready has no publication")?;
    let call = s.commit(pending)?;
    counts.g_commits += 1;
    need(
        call.root == s.binding()
            && call.root.root() == before.root() + 1
            && call.root.frontier() == 1
            && call.selected_id == expected.cell.calls[0].selected_id,
        "actual Gemma source/root/private feedback publication",
    )?;
    if normal {
        let snapshot = s.snapshot()?;
        expected.normal_state(0, &block_on(snapshot.read_domains())?)?;
        drop(snapshot);
    }
    events.push(json!({"kind":"gemma-completed-committed","comparison":comparison,"before":binding(before),"after":binding(s.binding()),"selected_id":call.selected_id,"max_bits":call.max_bits}));
    Ok(())
}
fn new_gstep(
    g: &Arc<Gemma4Bf16GpuCandidate>,
    ids: Arc<[u32]>,
    expected: &mut gexpected::Expected,
    counts: &mut Counts,
    events: &mut Vec<Value>,
) -> Proof<()> {
    let mut s = block_on(g.session(ids))?;
    gstep(&mut s, expected, true, counts, events)?;
    drop(s);
    Ok(())
}
fn probe_matches(mode: &str, p: &Value, qi: &HipSourceSetIdentity) -> bool {
    let stage = if mode.contains("-query-") {
        "query"
    } else if mode.contains("-wait-") {
        "wait"
    } else {
        "destroy"
    };
    let Some(logical) = p["identity"]["logical_stream"].as_u64() else {
        return false;
    };
    p["schema"] == "zg.hip-source-page-event-probe.v1"
        && p["mode"] == mode
        && p["injection"]["hits"] == 1
        && p["injection"]["stage"] == stage
        && p["injection"]["result"]
            == if stage == "destroy" {
                "injected-destroy-refused"
            } else {
                "injected-policy-unknown"
            }
        && p["identity"]["source"] == format!("{:?}", qi.source)
        && p["identity"]["service"] == format!("{:?}", qi.actor)
        && p["identity"]["generation"] == qi.source_service_generation
        && p["identity"]["actor"] == format!("{:?}", qi.actor)
        && p["identity"]["set_key"] == qi.set_key
        && p["identity"]["page_allocation_id"]
            .as_u64()
            .is_some_and(|n| n > 0)
        && qi.logical_stream == Some(logical)
        && p["event"]["identity"].as_u64().is_some_and(|n| n > 0)
        && p["event"]["first_serial"].as_u64().is_some_and(|n| n > 0)
        && p["event"]["current_serial"].as_u64().is_some_and(|n| n > 0)
        && p["event"]["current_serial"].as_u64() >= p["event"]["first_serial"].as_u64()
        && p["reached"]["copy"] == true
        && p["reached"]["consumer"] == true
        && p["reached"]["record"] == true
        && p["raw"]["destroy"].is_null()
        && p["raw"]["query_error_count"] == 0
        && (p["raw"]["query_first"].is_null()
            || p["raw"]["query_first"] == 0
            || p["raw"]["query_first"] == 600)
        && (p["raw"]["query_last"].is_null()
            || p["raw"]["query_last"] == 0
            || p["raw"]["query_last"] == 600)
        && (p["raw"]["synchronize"].is_null() || p["raw"]["synchronize"] == 0)
        && p["evidence"]["prepare_error"].is_string()
        && p["evidence"]["binding_unchanged"] == true
        && p["evidence"]["native_usable_ok"] == true
        && p["ledger"]["before"]["device"]["fenced"] == false
        && p["ledger"]["fault"]["device"]["fenced"] == true
        && p["ledger"]["after_public_drop"]["device"]["fenced"] == true
        && p["ledger"]["after_public_drop"]["device"]["reserved"] == 32 * GIB
        && p["witnesses"]["pending_window_required"] == (stage != "destroy")
        && p["outcome"]["original_service_fenced"] == true
        && p["outcome"]["original_native_fenced"] == true
        && p["outcome"]["qwen_reentry_refused"] == true
        && p["outcome"]["gemma_reentry_refused"] == true
        && p["outcome"]["qwen_original_session_refused"] == true
        && p["outcome"]["qwen_original_session_unchanged"] == true
        && p["outcome"]["qwen_original_native_unchanged"] == true
        && p["outcome"]["qwen_original_probe_unchanged"] == true
        && p["outcome"]["qwen_original_ledger_unchanged"] == true
        && p["outcome"]["gemma_original_session_refused"] == true
        && p["outcome"]["observed_pending"] == false
        && p["outcome"]["qwen_reentry_pending"] == false
        && p["outcome"]["gemma_reentry_pending"] == false
        && p["outcome"]["qwen_reentry_native_unchanged"] == true
        && p["outcome"]["qwen_reentry_ledger_unchanged"] == true
        && p["outcome"]["gemma_original_native_unchanged"] == true
        && p["outcome"]["gemma_original_binding_unchanged"] == true
        && p["outcome"]["gemma_original_probe_unchanged"] == true
        && p["outcome"]["gemma_original_ledger_unchanged"] == true
        && p["outcome"]["gemma_candidate_ledger_unchanged"] == true
        && p["ledger"]["before"]["host"]["limit"] == p["ledger"]["fault"]["host"]["limit"]
        && p["ledger"]["before"]["host"]["limit"]
            == p["ledger"]["after_public_drop"]["host"]["limit"]
        && p["ledger"]["before"]["device"]["limit"] == p["ledger"]["fault"]["device"]["limit"]
        && p["ledger"]["before"]["device"]["limit"]
            == p["ledger"]["after_public_drop"]["device"]["limit"]
        && p["outcome"]["qwen_close_refused"] == true
        && p["outcome"]["gemma_close_refused"] == true
        && (stage != "destroy" || p["outcome"]["finish_failed"] == true)
        && p["outcome"]["publication"] == false
        && p["outcome"]["commit"] == false
        && match stage {
            "query" => {
                p["raw"]["query_first"].is_number()
                    && p["raw"]["query_last"].is_number()
                    && (p["raw"]["query_success_count"]
                        .as_u64()
                        .is_some_and(|n| n > 0)
                        || p["raw"]["query_not_ready_count"]
                            .as_u64()
                            .is_some_and(|n| n > 0))
                    && p["event"]["window_payload_bytes"]
                        .as_u64()
                        .is_some_and(|n| n > 0)
                    && p["event"]["source_window_charge_bytes"]
                        .as_u64()
                        .is_some_and(|n| n > 0)
                    && p["injection"]["hits"] == 1
                    && p["counters"]["selected_retire_success"] == 0
                    && p["counters"]["selected_recycle_success"] == 0
                    && p["counters"]["selected_destroy_calls"] == 0
                    && p["counters"]["selected_destroy_success"] == 0
                    && p["witnesses"]["run_owner_after_public_drop"] == true
                    && p["witnesses"]["window_after_public_drop"] == true
            }
            "wait" => {
                p["raw"]["synchronize"] == 0
                    && p["raw"]["query_first"].is_number()
                    && p["raw"]["query_last"].is_number()
                    && (p["raw"]["query_success_count"]
                        .as_u64()
                        .is_some_and(|n| n > 0)
                        || p["raw"]["query_not_ready_count"]
                            .as_u64()
                            .is_some_and(|n| n > 0))
                    && p["event"]["window_payload_bytes"]
                        .as_u64()
                        .is_some_and(|n| n > 0)
                    && p["event"]["source_window_charge_bytes"]
                        .as_u64()
                        .is_some_and(|n| n > 0)
                    && p["injection"]["retirement_hold_count"]
                        .as_u64()
                        .is_some_and(|n| n > 0)
                    && p["counters"]["selected_retire_success"] == 0
                    && p["counters"]["selected_recycle_success"] == 0
                    && p["counters"]["selected_destroy_calls"] == 0
                    && p["counters"]["selected_destroy_success"] == 0
                    && p["witnesses"]["run_owner_after_public_drop"] == true
                    && p["witnesses"]["window_after_public_drop"] == true
            }
            "destroy" => {
                p["raw"]["destroy"].is_null()
                    && p["outcome"]["finish_failed"] == true
                    && p["counters"]["selected_retire_success"]
                        .as_u64()
                        .is_some_and(|n| n > 0)
                    && p["counters"]["selected_destroy_calls"] == 0
                    && p["counters"]["selected_destroy_success"] == 0
            }
            _ => false,
        }
}
fn probe_json(parts: &SourcePageProbeParts) -> Proof<(Value, bool, bool, Option<String>)> {
    let s = parts.snapshot_state()?;
    let (run, window, witness_error) = match parts.witness_live(&s) {
        Ok((run, window)) => (run, window, None),
        Err(error) => (false, false, Some(bounded_text(format_args!("{error}")))),
    };
    let raw = json!({
        "schema":"zg.hip-source-page-event-probe.v1",
        "mode":s.mode,
        "identity":{"source":bounded_text(format_args!("{:?}",parts.source)),
            "service":bounded_text(format_args!("{:?}",s.service)),
            "generation":s.generation,
            "actor":bounded_text(format_args!("{:?}",s.actor)),
            "set_key":s.set_key,
            "logical_stream":s.logical_stream,"page_allocation_id":s.page_allocation_id},
        "reached":{"copy":s.reached_copy,"consumer":s.reached_consumer,"record":s.reached_record},
        "event":{"identity":s.event_identity,"first_serial":s.first_serial,
            "current_serial":s.current_serial,
            "window_payload_bytes":s.window_payload_bytes,
            "source_window_charge_bytes":s.source_window_charge_bytes},
        "raw":{"query_first":s.query_first,"query_last":s.query_last,
            "query_success_count":s.query_success_count,
            "query_not_ready_count":s.query_not_ready_count,
            "query_error_count":s.query_error_count,"synchronize":s.synchronize,
            "destroy":s.destroy},
        "injection":{"hits":s.injection_hits,"stage":s.injection_stage,
            "result":s.injection_result,"retirement_hold_count":s.retirement_hold_count},
        "counters":{"selected_retire_success":s.selected_retire_success,
            "selected_recycle_success":s.selected_recycle_success,
            "selected_destroy_calls":s.destroy_calls,
            "selected_destroy_success":s.destroy_success},
        "witness_error":witness_error.clone(),
    });
    Ok((raw, run, window, witness_error))
}
fn merge_probe_into(dst: &mut Value, src: &Value) {
    if let (Some(dst), Some(src)) = (dst.as_object_mut(), src.as_object()) {
        for (key, value) in src {
            dst.insert(key.clone(), value.clone());
        }
    } else {
        *dst = src.clone();
    }
}
fn run() -> Proof<()> {
    let a = std::env::args().skip(1).collect::<Vec<_>>();
    need(
        a.len() == 6,
        "usage: hip_shared_models MODE ABS_QCACHE ABS_GCACHE ABS_QEXPECTED ABS_GEXPECTED_RECEIPT ABS_FRESH_REPORT",
    )?;
    need(
        matches!(
            a[0].as_str(),
            "known-qwen-reload"
                | "qwen-final-reader-unknown"
                | "gemma-feedback-unknown"
                | "source-page-query-unknown"
                | "source-page-wait-unknown"
                | "source-page-destroy-unknown"
        ),
        "six finite shared modes only",
    )?;
    let paths = a[1..].iter().map(PathBuf::from).collect::<Vec<_>>();
    need(
        paths.iter().all(|p| p.is_absolute()) && !paths[4].exists(),
        "absolute source/cache/reference and fresh report paths",
    )?;
    let q_scope: &[(usize, usize)] = match a[0].as_str() {
        "known-qwen-reload" => &[(0, 0), (0, 1), (1, 0)],
        "qwen-final-reader-unknown" | "gemma-feedback-unknown" => &[(0, 0)],
        "source-page-query-unknown"
        | "source-page-wait-unknown"
        | "source-page-destroy-unknown" => &[(0, 0)],
        _ => unreachable!("validated finite shared modes"),
    };
    let (want_q_calls, want_q_commits, want_g_calls, want_g_instrumented, want_g_commits) =
        match a[0].as_str() {
            "known-qwen-reload" => (3, 3, 4, 1, 4),
            "qwen-final-reader-unknown" => (1, 0, 1, 1, 1),
            "gemma-feedback-unknown" => (1, 1, 1, 1, 1),
            "source-page-query-unknown" | "source-page-wait-unknown" => (0, 0, 0, 0, 0),
            "source-page-destroy-unknown" => (0, 0, 0, 0, 0),
            _ => unreachable!("validated finite shared modes"),
        };
    let wanted_events: &[(&str, usize)] = match a[0].as_str() {
        "known-qwen-reload" => &[
            ("both-original-sources-live", 1),
            ("qwen-completed-committed", 3),
            ("gemma-completed-committed", 4),
            ("known-qwen-retired", 1),
            ("actual-qwen-reloaded", 1),
            ("fourth-preparation-refused-preeffect", 1),
        ],
        "qwen-final-reader-unknown" => &[
            ("both-original-sources-live", 1),
            ("gemma-completed-committed", 1),
            ("actual-qwen-final-reader-fault", 1),
        ],
        "gemma-feedback-unknown" => &[
            ("both-original-sources-live", 1),
            ("qwen-completed-committed", 1),
            ("gemma-completed-committed", 1),
            ("actual-gemma-final-feedback-fault", 1),
        ],
        "source-page-query-unknown"
        | "source-page-wait-unknown"
        | "source-page-destroy-unknown" => &[
            ("both-original-sources-live", 1),
            ("source-page-event-probe", 1),
        ],
        _ => unreachable!("validated finite shared modes"),
    };
    let qctx = Qwen27bGpuContext::new_shared_hip_qualification()?;
    let gctx = Gemma4CpuContext::new_shared_hip_qualification()?;
    let arbiter = ResourceArbiter::process_shared();
    let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
    let gpu_key = PhysicalDeviceKey::new("pci", "0000:c6:00.0")?;
    let harness = arbiter.admit(
        &host_key,
        ResourceOwner::new("shared-HIP-fixture", "bounded-source-frame-report")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 128 * 1024 * 1024,
        }],
    )?;
    let qs = CacheStore::open_existing(CacheOptions::new(Some(paths[0].clone()), 96 * GIB))?;
    let gs = CacheStore::open_existing(CacheOptions::new(Some(paths[1].clone()), 96 * GIB))?;
    let q_snapshot = Arc::new(qs.open_snapshot(QSNAP)?);
    let g_snapshot = Arc::new(gs.open_snapshot(GSNAP)?);
    let input = gctx.prepare_text(&g_snapshot, "One", None)?;
    let g_ids: Arc<[u32]> = Arc::from(input.ids());
    need(g_ids.as_ref() == [4906], "actual managed One token")?;
    let q_source = Qwen27bGpuLoaded::open(q_snapshot.clone(), qctx.clone(), None)?;
    let g_source = Gemma4CpuLoaded::open(g_snapshot.clone(), gctx.clone(), None)?;
    let host = GemmaTextHostOwners {
        additional_recorded_calls: 1,
        additional_frame_bytes: 32 * 1024 * 1024,
    };
    let plan = q_source.shared_hip_cold_plan(&g_source, &input, &host)?;
    let cold = json!({"qwen":format!("{:?}",plan.qwen()),"gemma":format!("{:?}",plan.gemma()),"provider_upper":plan.complete_provider_upper_bytes(),"host_upper":plan.complete_host_upper_bytes()});
    let probe_bound = probe_report_bound()?;
    need(
        probe_bound <= 128 * 1024 * 1024,
        "source-derived diagnostic delta stays inside the fixed fixture admission",
    )?;
    need(
        plan.complete_host_upper_bytes()
            .checked_add(128 * 1024 * 1024)
            .and_then(|n| n.checked_add(512 * 1024 * 1024))
            .is_some_and(|n| n <= 32 * GIB as usize),
        "complete cold host plus Rust fixture and external supervisor before bootstrap",
    )?;
    let shared = SharedHipNative::bootstrap(Device::Hip(Default::default()), plan)?;
    let mut events = Vec::new();
    let mut counts = Counts::default();
    let mut oracle_scope = None;
    let mut source_page_probe = Value::Null;
    let mut probe_identity = None;
    let mut witness_fault: Option<String> = None;
    // All finite failures after bootstrap retain an actual partial report.
    let result = (|| -> Proof<()> {
        let q = block_on(q_source.load_shared_hip_qualification(shared.clone(), None))?;
        let (g, mut g_session) =
            block_on(g_source.load_shared_hip_qualification(shared.clone(), input, host, None))?;
        let qi = q.native_identity();
        probe_identity = Some(qi);
        let q_logical = qi
            .logical_stream
            .ok_or("actual Qwen source logical stream absent")?;
        let gi = g.native_identity()?;
        need(
            q.peer_native_view_refused(&g),
            "actual prepared token rejects sibling pinned view on same actor",
        )?;
        events.push(json!({"kind":"both-original-sources-live","qwen":identity(qi),"gemma":identity(gi),"native":live(&shared,qi,gi)?,"qwen_actual":format!("{:?}",q.inventory()),"gemma_actual":format!("{:?}",g.inventory())}));
        let mut qe = qexpected::Expected::new(paths[2].clone())?;
        let matrix = gexpected::Matrix::open(&paths[3])?;
        let mut ge = matrix.cell(0, g_source.source_identity())?;
        need(
            ge.cell.prompt_ids.as_slice() == g_ids.as_ref(),
            "actual tokenizer equals expected-only source sequence",
        )?;
        let mut q_session = block_on(q.session())?;
        if a[0] == "qwen-final-reader-unknown" {
            qe.begin(0, 0)?;
            let before = q_session.binding();
            let pending = block_on(q_session.prepare_teacher(&mut qe, None))?;
            counts.q_calls += 1;
            let comparison = qe.finish()?;
            need(
                comparison["exact"] == true && q_session.binding() == before,
                "real full64 Ready before fault",
            )?;
            gstep(&mut g_session, &mut ge, false, &mut counts, &mut events)?;
            drop((q_session, q));
            let receipt = pending.qualify_final_owner_drop(true)?;
            events.push(json!({"kind":"actual-qwen-final-reader-fault","comparison":comparison,"receipt":receipt_json!(receipt),"native":native(&shared)?}));
            let sibling = block_on(g.session(g_ids.clone()));
            need(
                sibling.is_err(),
                "global unknown refuses sibling new root before enqueue",
            )?;
            drop(sibling);
            need(
                q_source.clone().close(None).is_err() && g_source.clone().close(None).is_err(),
                "both authentic sources retained after shared unknown",
            )?;
            drop((g_session, g));
        } else if a[0] == "gemma-feedback-unknown" {
            qstep(&mut q_session, &mut qe, 0, 0, &mut counts, &mut events)?;
            gstep(&mut g_session, &mut ge, false, &mut counts, &mut events)?;
            drop(g);
            let receipt = g_session.qualify_final_feedback_drop(true)?;
            events.push(json!({"kind":"actual-gemma-final-feedback-fault","receipt":receipt_json!(receipt),"native":native(&shared)?}));
            let before = q_session.binding();
            let before_scope = qe.summary();
            need(
                qe.in_progress() == (0, 0),
                "no Qwen producer progress before refused declaration",
            )?;
            qe.begin(0, 1)?;
            let refused = block_on(q_session.prepare_teacher(&mut qe, None));
            need(
                refused.is_err()
                    && q_session.binding() == before
                    && qe.summary() == before_scope
                    && qe.in_progress() == (0, 0),
                "global unknown refuses actual sibling cached proposal/publication without progress",
            )?;
            drop(refused);
            need(
                q_source.clone().close(None).is_err() && g_source.clone().close(None).is_err(),
                "both authentic sources retained after shared unknown",
            )?;
            drop((q_session, q));
        } else if a[0].starts_with("source-page-") {
            let mode = match a[0].as_str() {
                "source-page-query-unknown" => SourcePageEventProbeMode::QueryUnknown,
                "source-page-wait-unknown" => SourcePageEventProbeMode::WaitUnknown,
                "source-page-destroy-unknown" => SourcePageEventProbeMode::DestroyUnknown,
                _ => unreachable!("validated source-page probe mode"),
            };
            let ledger_before = ledger(&arbiter, &host_key, &gpu_key)?;
            let parts = block_on(q_session.arm_source_page_event_probe(mode))?;
            qe.begin(0, 0)?;
            let before = q_session.binding();
            let prepare = block_on(q_session.prepare_teacher(&mut qe, None));
            let prepare_failed = prepare.is_err();
            let prepare_error = prepare
                .as_ref()
                .err()
                .map(|error| bounded_text(format_args!("{error:?}")));
            let prepare_pending = prepare.is_ok();
            if prepare_pending {
                counts.q_calls += 1;
            }
            let binding_unchanged = q_session.binding() == before;
            let finish_failed = prepare_failed
                && mode == SourcePageEventProbeMode::DestroyUnknown
                && prepare.as_ref().err().is_some_and(|error| {
                    matches!(
                        error,
                        zerogravity::Error::State(message)
                            if message.starts_with("source page finish original authority")
                    )
                });
            let service_fenced = q.physical_fenced();
            // Seed a real partial receipt immediately after prepare, before any
            // later fallible snapshot/ledger/reentry observation can discard it.
            source_page_probe = json!({
                "schema":"zg.hip-source-page-event-probe.v1",
                "mode":a[0],
                "observed_pending":prepare_pending,
                "injection":{"hits":0,"stage":"none","result":"none"},
                "evidence":{"prepare_error":prepare_error.clone(),"binding_unchanged":binding_unchanged},
                "outcome":{"original_service_fenced":service_fenced,"finish_failed":finish_failed,
                    "publication":false,"commit":false},
                "ledger":{"before":ledger_before.clone()}
            });
            let initial_probe = probe_json(&parts)?;
            if witness_fault.is_none() {
                witness_fault = initial_probe.3.clone();
            }
            merge_probe_into(&mut source_page_probe, &initial_probe.0);
            drop(initial_probe);
            drop(prepare);
            let ledger_fault = ledger(&arbiter, &host_key, &gpu_key)?;
            source_page_probe["ledger"]["fault"] = ledger_fault.clone();
            // Refuse through the ORIGINAL public session before dropping it.
            let q_before_reentry = q_session.binding();
            let q_progress_before_reentry = qe.in_progress();
            let q_probe_before_original = probe_json(&parts)?;
            if witness_fault.is_none() {
                witness_fault = q_probe_before_original.3.clone();
            }
            merge_probe_into(&mut source_page_probe, &q_probe_before_original.0);
            let q_native_before_original = native(&shared)?;
            let q_ledger_before_original = ledger(&arbiter, &host_key, &gpu_key)?;
            source_page_probe["ledger"]["qwen_original_before"] = q_ledger_before_original.clone();
            let q_original_pending = block_on(q_session.prepare_teacher(&mut qe, None));
            let q_original_pending_observed = q_original_pending.is_ok();
            if q_original_pending_observed {
                counts.q_calls += 1;
            }
            let q_original_reentry = q_original_pending.is_err();
            let q_original_reentry_unchanged = q_session.binding() == q_before_reentry
                && qe.in_progress() == q_progress_before_reentry;
            let q_probe_after_original = probe_json(&parts)?;
            if witness_fault.is_none() {
                witness_fault = q_probe_after_original.3.clone();
            }
            merge_probe_into(&mut source_page_probe, &q_probe_after_original.0);
            let q_native_after_original = native(&shared)?;
            let q_ledger_after_original = ledger(&arbiter, &host_key, &gpu_key)?;
            source_page_probe["ledger"]["qwen_original_after"] = q_ledger_after_original.clone();
            let q_probe_unchanged = q_probe_after_original.0 == q_probe_before_original.0
                && q_probe_after_original.1 == q_probe_before_original.1
                && q_probe_after_original.2 == q_probe_before_original.2;
            let q_native_unchanged = q_native_after_original == q_native_before_original;
            let q_ledger_unchanged = q_ledger_after_original == q_ledger_before_original;
            drop((
                q_probe_after_original,
                q_probe_before_original,
                q_native_after_original,
                q_native_before_original,
            ));
            drop(q_original_pending);
            drop((q_session, q));
            let (probe, run, window, probe_witness) = probe_json(&parts)?;
            if witness_fault.is_none() {
                witness_fault = probe_witness;
            }
            merge_probe_into(&mut source_page_probe, &probe);
            drop(probe);
            let q_native_before_reentry = native(&shared)?;
            let q_ledger_before_reentry = ledger(&arbiter, &host_key, &gpu_key)?;
            source_page_probe["ledger"]["qwen_reentry_before"] = q_ledger_before_reentry.clone();
            let q_reentry =
                block_on(q_source.load_shared_hip_qualification(shared.clone(), None)).is_err();
            let q_native_after_reentry = native(&shared)?;
            let q_ledger_after_reentry = ledger(&arbiter, &host_key, &gpu_key)?;
            source_page_probe["ledger"]["qwen_reentry_after"] = q_ledger_after_reentry.clone();
            let q_reentry_native_unchanged = q_native_after_reentry == q_native_before_reentry;
            let q_reentry_ledger_unchanged = q_ledger_after_reentry == q_ledger_before_reentry;
            drop((q_native_after_reentry, q_native_before_reentry));
            // Original Gemma session and candidate both refuse before submit.
            let g_binding_before_original = g_session.binding();
            let g_probe_before_original = probe_json(&parts)?;
            if witness_fault.is_none() {
                witness_fault = g_probe_before_original.3.clone();
            }
            merge_probe_into(&mut source_page_probe, &g_probe_before_original.0);
            let g_before_original = native(&shared)?;
            let g_ledger_before_original = ledger(&arbiter, &host_key, &gpu_key)?;
            source_page_probe["ledger"]["gemma_original_before"] = g_ledger_before_original.clone();
            let g_original_pending = block_on(g_session.prepare_prompt(1, None, None));
            let g_original_pending_observed = g_original_pending.is_ok();
            if g_original_pending_observed {
                counts.g_calls += 1;
            }
            let g_original_reentry = g_original_pending.is_err();
            drop(g_original_pending);
            let g_binding_after_original = g_session.binding();
            let g_probe_after_original = probe_json(&parts)?;
            if witness_fault.is_none() {
                witness_fault = g_probe_after_original.3.clone();
            }
            merge_probe_into(&mut source_page_probe, &g_probe_after_original.0);
            let g_after_original = native(&shared)?;
            let g_ledger_after_original = ledger(&arbiter, &host_key, &gpu_key)?;
            source_page_probe["ledger"]["gemma_original_after"] = g_ledger_after_original.clone();
            let g_reentry = block_on(g.session(g_ids.clone())).is_err();
            let g_ledger_after_candidate = ledger(&arbiter, &host_key, &gpu_key)?;
            source_page_probe["ledger"]["gemma_candidate_after"] = g_ledger_after_candidate.clone();
            let g_probe_unchanged = g_probe_after_original.0 == g_probe_before_original.0
                && g_probe_after_original.1 == g_probe_before_original.1
                && g_probe_after_original.2 == g_probe_before_original.2;
            let g_native_unchanged = g_after_original == g_before_original;
            let g_binding_unchanged = g_binding_after_original == g_binding_before_original;
            let g_ledger_unchanged = g_ledger_after_original == g_ledger_before_original;
            let g_candidate_ledger_unchanged = g_ledger_after_candidate == g_ledger_before_original;
            drop((
                g_probe_after_original,
                g_probe_before_original,
                g_after_original,
                g_before_original,
            ));
            let native = shared.stream_preflight_usable(q_logical);
            let native_usable_ok = native.is_ok();
            let native_fenced = matches!(native.as_ref(), Ok(false));
            let ledger_after = ledger(&arbiter, &host_key, &gpu_key)?;
            source_page_probe["ledger"]["after_public_drop"] = ledger_after.clone();
            let pending_window_required = mode != SourcePageEventProbeMode::DestroyUnknown;
            source_page_probe["witness_error"] =
                witness_fault.clone().map_or(Value::Null, Value::String);
            source_page_probe["witnesses"] = json!({"run_owner_after_public_drop":run,
                "window_after_public_drop":window,"pending_window_required":pending_window_required});
            source_page_probe["ledger"] = json!({"before":ledger_before.clone(),"fault":ledger_fault.clone(),
                "after_public_drop":ledger_after.clone(),
                "qwen_original_before":q_ledger_before_original,
                "qwen_original_after":q_ledger_after_original,
                "qwen_reentry_before":q_ledger_before_reentry,
                "qwen_reentry_after":q_ledger_after_reentry,
                "gemma_original_before":g_ledger_before_original,
                "gemma_original_after":g_ledger_after_original,
                "gemma_candidate_after":g_ledger_after_candidate});
            source_page_probe["evidence"] = json!({"prepare_error":prepare_error,
                "binding_unchanged":binding_unchanged,"native_usable_ok":native_usable_ok});
            source_page_probe["outcome"] = json!({"original_service_fenced":service_fenced,
                "original_native_fenced":native_fenced,"qwen_reentry_refused":q_reentry,
                "gemma_reentry_refused":g_reentry,"finish_failed":finish_failed,
                "qwen_original_session_refused":q_original_reentry,
                "qwen_original_session_unchanged":q_original_reentry_unchanged,
                "qwen_original_native_unchanged":q_native_unchanged,
                "qwen_original_probe_unchanged":q_probe_unchanged,
                "qwen_original_ledger_unchanged":q_ledger_unchanged,
                "qwen_reentry_native_unchanged":q_reentry_native_unchanged,
                "qwen_reentry_ledger_unchanged":q_reentry_ledger_unchanged,
                "gemma_original_session_refused":g_original_reentry,
                "gemma_candidate_refused":g_reentry,
                "gemma_original_native_unchanged":g_native_unchanged,
                "gemma_original_binding_unchanged":g_binding_unchanged,
                "gemma_original_probe_unchanged":g_probe_unchanged,
                "gemma_original_ledger_unchanged":g_ledger_unchanged,
                "gemma_candidate_ledger_unchanged":g_candidate_ledger_unchanged,
                "observed_pending":prepare_pending,
                "qwen_reentry_pending":q_original_pending_observed,
                "gemma_reentry_pending":g_original_pending_observed,
                "publication":false,"commit":false});
            events.push(json!({"kind":"source-page-event-probe","mode":a[0],
                "prepare_error":source_page_probe["evidence"]["prepare_error"],
                "binding_unchanged":binding_unchanged}));
            need(
                prepare_failed
                    && binding_unchanged
                    && service_fenced
                    && native_usable_ok
                    && native_fenced
                    && q_reentry
                    && g_reentry
                    && q_original_reentry
                    && q_original_reentry_unchanged
                    && g_original_reentry
                    && !prepare_pending
                    && !q_original_pending_observed
                    && !g_original_pending_observed
                    && q_reentry_native_unchanged
                    && q_reentry_ledger_unchanged
                    && q_native_unchanged
                    && q_probe_unchanged
                    && q_ledger_unchanged
                    && g_probe_unchanged
                    && g_ledger_unchanged
                    && g_binding_unchanged
                    && g_candidate_ledger_unchanged
                    && g_native_unchanged
                    && witness_fault.is_none()
                    && ledger_before["device"]["fenced"] == false
                    && ledger_before["host"]["limit"] == ledger_fault["host"]["limit"]
                    && ledger_before["host"]["limit"] == ledger_after["host"]["limit"]
                    && ledger_before["device"]["limit"] == ledger_fault["device"]["limit"]
                    && ledger_before["device"]["limit"] == ledger_after["device"]["limit"]
                    && ledger_after["host"]["fenced"] == false
                    && ledger_fault["device"]["fenced"] == true
                    && ledger_after["device"]["fenced"] == true
                    && ledger_after["device"]["reserved"] == 32 * GIB,
                "source-page injection refuses original prepare before Pending; original service/native fenced and ledger retained",
            )?;
            need(
                source_page_probe["mode"] == a[0]
                    && source_page_probe["injection"]["hits"] == 1
                    && source_page_probe["identity"]["logical_stream"] == q_logical
                    && source_page_probe["witnesses"]["pending_window_required"]
                        == pending_window_required
                    && source_page_probe["outcome"]["finish_failed"] == finish_failed
                    && (mode == SourcePageEventProbeMode::DestroyUnknown
                        || source_page_probe["ledger"]["after_public_drop"]["host"]["reserved"]
                            .as_u64()
                            .zip(source_page_probe["event"]["source_window_charge_bytes"].as_u64())
                            .is_some_and(|(reserved, charge)| reserved >= charge)),
                "one selected source-page event with actual logical identity and truthful finish witness",
            )?;
            if mode == SourcePageEventProbeMode::QueryUnknown {
                need(
                    source_page_probe["raw"]["query_first"].is_number()
                        && (source_page_probe["raw"]["query_success_count"]
                            .as_u64()
                            .is_some_and(|n| n > 0)
                            || source_page_probe["raw"]["query_not_ready_count"]
                                .as_u64()
                                .is_some_and(|n| n > 0))
                        && source_page_probe["injection"]["stage"] == "query",
                    "actual hipEventQuery success precedes injected query unknown",
                )?;
            } else if mode == SourcePageEventProbeMode::WaitUnknown {
                need(
                    source_page_probe["raw"]["synchronize"] == 0
                        && source_page_probe["injection"]["stage"] == "wait"
                        && source_page_probe["injection"]["retirement_hold_count"]
                            .as_u64()
                            .is_some_and(|n| n > 0),
                    "actual hipEventSynchronize success and selected hold precede injected wait unknown",
                )?;
            } else {
                need(
                    source_page_probe["raw"]["destroy"].is_null()
                        && (source_page_probe["raw"]["synchronize"].is_null()
                            || source_page_probe["raw"]["synchronize"] == 0)
                        && source_page_probe["evidence"]["prepare_error"].is_string()
                        && source_page_probe["outcome"]["finish_failed"] == true
                        && source_page_probe["injection"]["stage"] == "destroy"
                        && source_page_probe["injection"]["result"] == "injected-destroy-refused"
                        && source_page_probe["counters"]["selected_retire_success"]
                            .as_u64()
                            .is_some_and(|n| n > 0)
                        && source_page_probe["counters"]["selected_destroy_calls"] == 0
                        && source_page_probe["counters"]["selected_destroy_success"] == 0,
                    "destroy refusal inside serialized destroy before actual hipEventDestroy",
                )?;
            }
            drop((g_session, g));
        } else {
            qstep(&mut q_session, &mut qe, 0, 0, &mut counts, &mut events)?;
            gstep(&mut g_session, &mut ge, false, &mut counts, &mut events)?;
            drop(g_session);
            qstep(&mut q_session, &mut qe, 0, 1, &mut counts, &mut events)?;
            let snapshot = q_session.snapshot()?;
            drop((q_session, q));
            let receipt = snapshot.qualify_final_owner_drop(false)?;
            need(
                q_source.native_owner_count()? == 0
                    && !shared.stream_preflight_usable(qi.logical_stream.unwrap())?,
                "actual Qwen last source reader retired; stale source view refuses",
            )?;
            let after = native(&shared)?;
            need(
                after["sets"].as_array().is_some_and(|s| s.len() == 1)
                    && after["compiler_host_reserved"] == 16 * GIB
                    && after["available_model_capacity"] == 12 * GIB,
                "Qwen native/model quota retired, compiler16GiB retained, Gemma sibling still live",
            )?;
            events.push(json!({"kind":"known-qwen-retired","receipt":receipt_json!(receipt),"native":after}));
            new_gstep(&g, g_ids.clone(), &mut ge, &mut counts, &mut events)?;
            let reload = block_on(q_source.load_shared_hip_qualification(shared.clone(), None))?;
            let ri = reload.native_identity();
            need(
                ri.source == qi.source
                    && ri.source_service_generation != qi.source_service_generation
                    && ri.set_key > qi.set_key
                    && ri.logical_stream != qi.logical_stream,
                "same authentic source reload has new real service/key/logical identity",
            )?;
            events.push(json!({"kind":"actual-qwen-reloaded","identity":identity(ri),"native":live(&shared,ri,gi)?}));
            let mut rs = block_on(reload.session())?;
            qstep(&mut rs, &mut qe, 1, 0, &mut counts, &mut events)?;
            drop((rs, reload));
            need(
                q_source.native_owner_count()? == 0
                    && !shared.stream_preflight_usable(ri.logical_stream.unwrap())?,
                "reloaded Qwen readers really retire",
            )?;
            new_gstep(&g, g_ids.clone(), &mut ge, &mut counts, &mut events)?;
            let before = native(&shared)?;
            let refused = block_on(q_source.load_shared_hip_qualification(shared.clone(), None));
            need(
                matches!(&refused,Err(zerogravity::Error::Limit(why)) if why=="HIP cold cumulative compiler preparation generations exhausted"),
                "fourth admitted preparation refused before queue submission/RTC",
            )?;
            drop(refused);
            let after = native(&shared)?;
            need(
                before == after
                    && q_source.native_owner_count()? == 0
                    && after["compiler_host_reserved"] == 24 * GIB,
                "ordinary precompiler refusal has no set/ledger/compiler credit or sibling fencing",
            )?;
            events.push(json!({"kind":"fourth-preparation-refused-preeffect","native":after}));
            new_gstep(&g, g_ids.clone(), &mut ge, &mut counts, &mut events)?;
            drop(g);
            need(
                shared.inspect_native()?.sets.is_empty(),
                "all actual native sets retired",
            )?;
        }
        if !a[0].starts_with("source-page-") {
            oracle_scope = Some(qe.finish_scoped(q_scope)?);
        }
        need(
            counts.q_calls == want_q_calls
                && counts.q_commits == want_q_commits
                && counts.g_calls == want_g_calls
                && counts.g_instrumented == want_g_instrumented
                && counts.g_commits == want_g_commits,
            "mode-specific actual Qwen/Gemma calls/commits and full35 evidence scope",
        )?;
        for &(kind, expected) in wanted_events {
            need(
                events.iter().filter(|e| e["kind"] == kind).count() == expected,
                format!("mode-specific lifecycle event {kind} exactly {expected}"),
            )?;
        }
        ge.after()?;
        Ok(())
    })();
    let fault = a[0] != "known-qwen-reload";
    let qclose = q_source.clone().close(None);
    let gclose = g_source.clone().close(None);
    if !source_page_probe.is_null() {
        source_page_probe["outcome"]["qwen_close_refused"] = json!(qclose.is_err());
        source_page_probe["outcome"]["gemma_close_refused"] = json!(gclose.is_err());
    }
    let final_native = native(&shared);
    let actual_compiler_host = shared.compiler_host_reserved_bytes()?;
    drop((
        q_source, g_source, q_snapshot, g_snapshot, qs, gs, qctx, gctx, shared,
    ));
    let gpu = arbiter.snapshot(&gpu_key)?;
    let host = arbiter.snapshot(&host_key)?;
    let final_ok = if fault {
        gpu.fenced && gpu.reserved == 32 * GIB && qclose.is_err() && gclose.is_err()
    } else {
        !gpu.fenced
            && gpu.reserved == 4 * GIB
            && qclose.is_ok()
            && gclose.is_ok()
            && host.reserved == 26 * GIB + 64 * 1024 + 128 * 1024 * 1024
    };
    let probe_ok = !a[0].starts_with("source-page-")
        || (witness_fault.is_none()
            && probe_identity
                .as_ref()
                .is_some_and(|qi| probe_matches(a[0].as_str(), &source_page_probe, qi)));
    let success = result.is_ok() && final_ok && probe_ok;
    if let Some(receipt) = source_page_probe.as_object_mut()
        && let Some(error) = witness_fault.as_ref()
    {
        receipt.insert("witness_error".to_owned(), Value::String(error.clone()));
    }
    write_new(
        &paths[4],
        &json!({"schema":"zg.hip-two-source-original-lifecycle.v1","mode":a[0],"status":if success{"PASS_FINITE_SHARED_SOURCE_SCOPE"}else{"REFUSED_UNQUALIFIED"},"failure":result.as_ref().err().map(|e| bounded_text(format_args!("{e}"))),"cold":cold,"events":events,"source_page_probe":source_page_probe,"qwen_declared_scope":q_scope.iter().map(|&(repeat,call)|json!([repeat,call])).collect::<Vec<_>>(),"qwen_oracle_scope":oracle_scope,"actual_completed_full64_calls":counts.q_calls,"actual_qwen_commits":counts.q_commits,"actual_completed_full35_calls":counts.g_calls,"actual_gemma_calls":counts.g_calls,"actual_instrumented_full35_calls":counts.g_instrumented,"actual_ordinary_packet_fullkv_calls":counts.g_calls-counts.g_instrumented,"actual_gemma_commits":counts.g_commits,"qwen_source_close_refused":qclose.is_err(),"gemma_source_close_refused":gclose.is_err(),"final_native":final_native.ok(),"physical_reserved":gpu.reserved,"physical_fenced":gpu.fenced,"host_reserved":host.reserved,"host_fenced":host.fenced,"diagnostic_host":128*1024*1024,"persistent_compiler_host":actual_compiler_host,"trained_serving_speed_or_general_fit_qualified":false}),
    )?;
    drop(harness);
    result?;
    need(success, "actual final source/native/physical/host outcome")
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
