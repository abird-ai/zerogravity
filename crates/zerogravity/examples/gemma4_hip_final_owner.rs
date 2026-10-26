//! Original Gemma4 sole final feedback wrapper: mode-exact native retirement.
//! Warm commit, reset, then one instrumented all-hit commit; the wrapper that
//! holds that final packet is dropped last so reader/native order is real.
#![recursion_limit = "256"]
use serde_json::{Value, json};
use std::{
    error::Error as StdError,
    fs::OpenOptions,
    future::Future,
    io::Write,
    path::{Path, PathBuf},
    pin::pin,
    sync::Arc,
    task::{Poll, Wake, Waker},
    thread,
};
use zerogravity::{
    Error,
    cache::{CacheOptions, CacheStore},
    gemma4::execution::{Context, Model},
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter},
};
use zerogravity_gpu::{Device, TextHostOwners, TextStorage};
type Proof<T> = Result<T, Box<dyn StdError>>;
fn need(ok: bool, why: impl Into<String>) -> Proof<()> {
    if ok { Ok(()) } else { Err(why.into().into()) }
}
fn write_new(path: &Path, value: &Value) -> Proof<()> {
    let raw = serde_json::to_vec_pretty(value)?;
    need(raw.len() <= 1024 * 1024, "bounded final-owner report")?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&raw)?;
    file.write_all(b"\n")?;
    Ok(())
}
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
    let mut cx = std::task::Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => thread::park(),
        }
    }
}
fn binding(root: zerogravity::session_root::RootBinding) -> Value {
    json!({
        "session": root.session().get(),
        "root": root.root(),
        "session_epoch": root.session_epoch(),
        "frontier": root.frontier(),
        "source_bound": root.source().is_bound(),
        "source": format!("{:?}", root.source()),
    })
}
fn traffic(
    total: zerogravity_gpu::TextTraffic,
    before: zerogravity_gpu::TextTraffic,
) -> Proof<(u64, u64, u64, u64, usize, usize)> {
    Ok((
        total
            .selected_row_uploads
            .checked_sub(before.selected_row_uploads)
            .ok_or("selected row upload underflow")?,
        total
            .selected_row_reuses
            .checked_sub(before.selected_row_reuses)
            .ok_or("selected row reuse underflow")?,
        total
            .selected_row_avoided_upload_bytes
            .checked_sub(before.selected_row_avoided_upload_bytes)
            .ok_or("selected row avoided underflow")?,
        total
            .value_upload_bytes
            .checked_sub(before.value_upload_bytes)
            .ok_or("value upload underflow")?,
        total
            .producer_readback_bytes
            .checked_sub(before.producer_readback_bytes)
            .ok_or("producer readback underflow")?,
        total
            .kv_diagnostic_readback_bytes
            .checked_sub(before.kv_diagnostic_readback_bytes)
            .ok_or("KV readback underflow")?,
    ))
}
/// Actual native-endian F32 words; the caller pins 4-byte-aligned extent first.
fn words(actual: &[u8]) -> impl Iterator<Item = f32> + '_ {
    actual
        .chunks_exact(4)
        .map(|w| f32::from_ne_bytes(w.try_into().unwrap()))
}
/// One completed call's real boundary/fullV/all-domain counters; no replay.
#[derive(Default)]
struct Completed {
    rows: usize,
    frontier: usize,
    seen: Vec<bool>,
    layers: Vec<bool>,
    planes: Vec<bool>,
    boundaries: usize,
    full_v: usize,
    best: Option<(u32, u32)>,
}
impl Completed {
    fn new(rows: usize, frontier: usize) -> Self {
        Self {
            rows,
            frontier,
            seen: vec![false; rows * 70],
            layers: vec![false; 35],
            planes: vec![false; 30],
            ..Self::default()
        }
    }
    fn full35(&self) -> bool {
        self.boundaries == 70 * self.rows && self.layers.iter().all(|v| *v)
    }
}
impl zerogravity_gpu::TextObserver for Completed {
    fn completed_producer(
        &mut self,
        position: usize,
        o: &zerogravity::accelerator::program::TextObservation,
        actual: &[u8],
    ) -> zerogravity::Result<()> {
        let checked = (|| -> Proof<()> {
            need(
                !actual.is_empty() && actual.len() % 4 == 0,
                "bounded F32 frame",
            )?;
            if matches!(o.stage, "block.input" | "block.output" | "logits") {
                need(words(actual).all(f32::is_finite), "finite actual producer")?;
            }
            match o.stage {
                "logits" => {
                    need(
                        position + 1 == self.frontier
                            && actual.len() == 262144 * 4
                            && self.full_v == 0,
                        "one last-row complete fullV producer",
                    )?;
                    // Original lowest-index-first argmax with the strict `>` tie
                    // policy: equal words keep the earlier index and its bits.
                    let mut best = (0u32, f32::NEG_INFINITY);
                    for (index, value) in words(actual).enumerate() {
                        if value > best.1 {
                            best = (index as u32, value);
                        }
                    }
                    self.best = Some((best.0, best.1.to_bits()));
                    self.full_v += 1;
                }
                "block.input" | "block.output" => {
                    need(
                        o.layer < 35 && actual.len() == 1536 * 4 && position < self.frontier,
                        "complete actual block row",
                    )?;
                    let index = o.layer * 2 + usize::from(o.stage == "block.output");
                    let seen = position * 70 + index;
                    need(!self.seen[seen], "unique actual block/source row")?;
                    self.seen[seen] = true;
                    self.layers[o.layer] = true;
                    self.boundaries += 1;
                }
                _ => return Ok(()),
            }
            Ok(())
        })();
        checked.map_err(|e| Error::State(e.to_string()))
    }
    fn completed_kv(
        &mut self,
        prior: bool,
        domain: zerogravity::accelerator::program::KvDomain,
        value: bool,
        active: usize,
        actual: &[u8],
    ) -> zerogravity::Result<()> {
        let checked = (|| -> Proof<()> {
            need(
                domain.0 < 15 && active <= self.frontier,
                "actual KV domain/prefix",
            )?;
            if active == 0 {
                need(actual.is_empty(), "empty actual prior prefix")?;
                return Ok(());
            }
            let width = if domain.0 % 5 == 4 { 512 } else { 256 };
            need(
                actual.len() == active * width * 4,
                "actual canonical KV prefix extent",
            )?;
            if !prior && active == self.frontier {
                let plane = domain.0 * 2 + usize::from(value);
                need(!self.planes[plane], "unique final actual KV plane")?;
                self.planes[plane] = true;
            }
            Ok(())
        })();
        checked.map_err(|e| Error::State(e.to_string()))
    }
}
pub fn run() -> Proof<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    need(
        args.len() == 2
            && matches!(
                args[0].as_str(),
                "session-known"
                    | "session-release-fault"
                    | "snapshot-known"
                    | "snapshot-release-fault"
            ),
        "usage: gemma4_hip_final_owner \
         [session-known|session-release-fault|snapshot-known|snapshot-release-fault] FRESH_REPORT",
    )?;
    let mode = args[0].as_str();
    let fault = mode.ends_with("release-fault");
    let snapshot_mode = mode.starts_with("snapshot");
    let report = PathBuf::from(&args[1]);
    need(
        report.is_absolute() && !report.exists(),
        "absolute fresh final-owner report",
    )?;
    let context = Context::new_hip_qualification()?;
    let harness = ResourceArbiter::process_shared().admit(
        &PhysicalDeviceKey::new("host", "system-ram")?,
        zerogravity::resource_arbiter::ResourceOwner::new(
            "Gemma-final-owner",
            "bounded-report-frame",
        )?,
        &[zerogravity::resource_arbiter::AllocationRequest {
            kind: zerogravity::resource_arbiter::AllocationClass::Scratch,
            bytes: 32 * 1024 * 1024,
        }],
    )?;
    let store = CacheStore::open_existing(CacheOptions::new(None, 96 * 1024 * 1024 * 1024))?;
    let snapshot = Arc::new(
        store.open_snapshot("ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1")?,
    );
    let prompt_text = "One";
    let input = context.prepare_text(&snapshot, prompt_text, None)?;
    let prompt_ids = input.ids().to_vec();
    need(prompt_ids.len() == 1, "genuine INITIAL1 source prompt")?;
    let loaded = Model::open(snapshot.clone(), context.clone(), None)?;
    let (candidate, mut session) = block_on(loaded.load_hip_qualification(
        Device::Hip(Default::default()),
        input,
        TextHostOwners {
            additional_recorded_calls: 2,
            additional_frame_bytes: 32 * 1024 * 1024,
        },
        TextStorage::ResidentProjections,
        None,
    ))?;
    need(
        !candidate.trained_model_qualified(),
        "explicit candidate scope without model/serving permit",
    )?;
    let inventory = format!("{:?}", candidate.inventory());
    let source_frontier = session.binding().frontier();
    need(source_frontier == 0, "fresh original session frontier0")?;
    // 1. Warm complete but unobserved 35-layer traversal; publishes root1.
    let start_binding = session.binding();
    let warm_before = block_on(candidate.traffic())?;
    let pending = block_on(session.prepare_prompt(1, None, None))?;
    let warm_packet = session.commit(pending)?;
    let warm_after = block_on(candidate.traffic())?;
    let warmup = traffic(warm_after, warm_before)?;
    let warm_binding = warm_packet.root;
    need(
        warm_binding.session() == start_binding.session()
            && warm_binding.source() == start_binding.source()
            && warm_binding.session_epoch() == start_binding.session_epoch()
            && warm_binding.root() == start_binding.root() + 1
            && warm_binding.frontier() == 1
            && session.binding() == warm_binding,
        "warm commit advances only the original root of one session",
    )?;
    // 2. Reset publishes an advanced root/epoch with empty state and no packet.
    block_on(session.reset())?;
    let reset_binding = session.binding();
    need(
        reset_binding.session() == warm_binding.session()
            && reset_binding.source() == warm_binding.source()
            && reset_binding.root() == warm_binding.root() + 1
            && reset_binding.session_epoch() == warm_binding.session_epoch() + 1
            && reset_binding.frontier() == 0,
        "reset advances original root/epoch and clears published state",
    )?;
    // 3. One instrumented all-hit prompt: cached PLE row, full35/fullV/allKV.
    let hit_before = block_on(candidate.traffic())?;
    let mut observer = Completed::new(prompt_ids.len(), prompt_ids.len());
    let pending = block_on(session.prepare_prompt(1, Some(&mut observer), None))?;
    need(
        session.binding() == reset_binding,
        "hit Ready keeps the reset binding until commit",
    )?;
    need(
        observer.full35() && observer.planes.iter().all(|v| *v) && observer.full_v == 1,
        "completed all-hit call observed every layer/fullV/domain before commit",
    )?;
    let final_packet = session.commit(pending)?;
    need(
        observer.best == Some((final_packet.selected_id, final_packet.max_bits)),
        "actual fullV winner equals this committed private packet",
    )?;
    let hit_after = block_on(candidate.traffic())?;
    let final_hit = traffic(hit_after, hit_before)?;
    let final_binding = final_packet.root;
    need(
        final_binding.session() == reset_binding.session()
            && final_binding.source() == reset_binding.source()
            && final_binding.root() == reset_binding.root() + 1
            && final_binding.session_epoch() == reset_binding.session_epoch()
            && final_binding.frontier() == 1
            && session.binding() == final_binding,
        "final all-hit commit advances only the original root",
    )?;
    need(
        warmup.0 + warmup.1 == prompt_ids.len() as u64
            && warmup.3 == warmup.0 * 17_920
            && warmup.2 == warmup.1 * 17_920
            && warmup.4 == 0
            && warmup.5 == 0
            && final_hit.0 == 0
            && final_hit.1 == prompt_ids.len() as u64
            && final_hit.3 == 0
            && final_hit.2 == prompt_ids.len() as u64 * 17_920
            && final_hit.4 > 0
            && final_hit.5 > 0,
        "ordinary warm row upload then genuine cached-row all-hit diagnostic",
    )?;
    let warmup_json = json!({"uploads":warmup.0,"reuses":warmup.1,"ple_row_h2d":warmup.3,"value_upload_total":warmup.3,"avoided_h2d":warmup.2,"producer_d2h":warmup.4,"kv_diagnostic_d2h":warmup.5});
    let final_hit_json = json!({"uploads":final_hit.0,"reuses":final_hit.1,"ple_row_h2d":final_hit.3,"value_upload_total":final_hit.3,"avoided_h2d":final_hit.2,"producer_d2h":final_hit.4,"kv_diagnostic_d2h":final_hit.5});
    let transitions = json!({
        "warmup": binding(warm_binding),
        "reset": binding(reset_binding),
        "final": binding(final_binding),
    });
    // 4. The wrapper holding the final committed packet is the only candidate
    //    owner: no caller clone, no recreated authority, no extra session.
    let receipt = if snapshot_mode {
        let retained = session.snapshot()?;
        drop(session);
        drop(candidate);
        retained.qualify_final_feedback_drop(fault)?
    } else {
        drop(candidate);
        session.qualify_final_feedback_drop(fault)?
    };
    need(
        receipt.wrapper == if snapshot_mode { "Snapshot" } else { "Session" }
            && receipt.fault == fault
            && receipt.candidate_owners_before_drop == 1
            && receipt.model_owners_before_drop == if snapshot_mode { 1 } else { 2 }
            && receipt.candidate_retired
            && receipt.release_hits == 1
            && receipt.native_retirement_attempts == 1
            && receipt.reader_hits_before_native_retirement == 1
            && receipt.native_modules_before > 0
            && receipt.native_modules_after
                == if fault {
                    receipt.native_modules_before
                } else {
                    0
                }
            && receipt.native_admission_retained_after == fault
            && receipt.original_service_retained_after == fault
            && receipt.aliases_inside_wrapper == 1
            && receipt.lease_bytes == 12
            && receipt.service_generation >= 1
            && (fault || receipt.quarantined_handles_after == 0)
            && (!fault || receipt.quarantined_handles_after > 0),
        "original sole wrapper reader-before-native retirement order",
    )?;
    let close = loaded.clone().close(None);
    need(
        close.is_err() == fault,
        "original managed source close refused exactly under retained fault",
    )?;
    drop(loaded);
    drop(snapshot);
    drop(store);
    drop(context);
    let arbiter = ResourceArbiter::process_shared();
    let physical = arbiter.snapshot(&PhysicalDeviceKey::new("pci", "0000:c6:00.0")?)?;
    let host = arbiter.snapshot(&PhysicalDeviceKey::new("host", "system-ram")?)?;
    let expected_device = if fault {
        20 * 1024 * 1024 * 1024
    } else {
        4 * 1024 * 1024 * 1024
    };
    // Coarse retained HOST envelope only: 2GiB+32MiB..16GiB brackets the live
    // wrapper/native/service owners; it is not a per-owner retained-byte proof,
    // the typed final-reader/native/service identities above are the authority.
    need(
        physical.reserved == expected_device
            && physical.fenced == fault
            && !host.fenced
            && if fault {
                host.reserved >= 2 * 1024 * 1024 * 1024 + 32 * 1024 * 1024
                    && host.reserved <= 16 * 1024 * 1024 * 1024
            } else {
                host.reserved == 2 * 1024 * 1024 * 1024 + 32 * 1024 * 1024
            },
        "actual retained/retired original provider envelope at report phase",
    )?;
    write_new(
        &report,
        &json!({
            "schema":"zg.gemma4-original-hip-final-feedback-owner.v1",
            "status":"PASS_ACTUAL_FINAL_OWNER",
            "mode":mode,
            "failure":Value::Null,
            "numerical_replay_performed":false,
            "old_order_physical_negative_performed":false,
            "trained_native_natural_serving_or_performance_qualified":false,
            "source_frontier":source_frontier,
            "warmup_completed_full35_calls":1usize,
            "warmup_calls":1usize,
            "warmup_commits":1usize,
            "observed_completed_full35_calls":1usize,
            "total_completed_full35_calls":2usize,
            "actual_completed_layers":observer.layers.iter().filter(|v|**v).count(),
            "actual_completed_proposed_state_planes":observer.planes.iter().filter(|v|**v).count(),
            "actual_completed_fullV":observer.full_v,
            "commits":2usize,
            "diagnostic_host_reserved":32u64*1024*1024,
            "final_feedback_on_completed_hit":true,
            "warmup":warmup_json,
            "final_hit":final_hit_json,
            "root_transitions":transitions,
            "source_close_refused":close.is_err(),
            "source_close_error":close.as_ref().err().map(ToString::to_string),
            "physical_reserved":physical.reserved,
            "physical_fenced":physical.fenced,
            "host_reserved":host.reserved,
            "host_fenced":host.fenced,
            "inventory":inventory,
            "prompt_text":prompt_text,
            "prompt_ids":prompt_ids,
            "actual_selected":[final_packet.selected_id,final_packet.max_bits],
            "original_final_owner":{
                "wrapper":receipt.wrapper,
                "fault":receipt.fault,
                "source":receipt.source,
                "target_lease":receipt.lease,
                "target_lease_bytes":receipt.lease_bytes,
                "service_generation":receipt.service_generation,
                "aliases_inside_only_wrapper":receipt.aliases_inside_wrapper,
                "candidate_owners_before_drop":receipt.candidate_owners_before_drop,
                "model_owners_before_drop":receipt.model_owners_before_drop,
                "candidate_workspace_target_aliases":receipt.candidate_workspace_target_aliases,
                "candidate_retired":receipt.candidate_retired,
                "target_final_reader_hits":receipt.release_hits,
                "native_retirement_attempts":receipt.native_retirement_attempts,
                "reader_hits_before_native_retirement":receipt.reader_hits_before_native_retirement,
                "actual_native_modules_before":receipt.native_modules_before,
                "actual_native_PT_LOAD_bytes_before":receipt.native_code_bytes_before,
                "actual_native_host_bytes_before":receipt.native_host_bytes_before,
                "actual_native_device_bytes_before":receipt.native_device_bytes_before,
                "native_admission_retained_after":receipt.native_admission_retained_after,
                "actual_native_modules_after":receipt.native_modules_after,
                "original_service_retained_after":receipt.original_service_retained_after,
                "actual_quarantined_handles_after":receipt.quarantined_handles_after,
            },
        }),
    )?;
    // The 32 MiB report frame stays admitted through the whole run and is
    // released only after the identity report exists.
    drop(harness);
    println!(
        "GEMMA_FINAL_OWNER_PASS mode={mode} real_producer=full35 target_final_reader=1 \
         native_retirement_after_reader=true source_close_refused={fault} \
         process_retirement=true serving_performance=false"
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("GEMMA_FINAL_OWNER_REFUSED: {error}");
        std::process::exit(1);
    }
}
