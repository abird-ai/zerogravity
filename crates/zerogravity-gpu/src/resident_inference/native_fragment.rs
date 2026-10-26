//! Initial physical fragment on SAME model/session/root/plan/dispatch ownership.
//! No PreparedInference, publication, cached continuation or model route.
use super::*;
use crate::cuda::native::CubinQualificationInputs;
use crate::cuda::native::LayerMlpCubinInputs;
use crate::cuda::native::native_plan::{prepare_connected_native_images, prepare_native_images};
use crate::physical_quarantine::CompletionOwner;
use crate::resident_program::native::capacity_bound;
use crate::resident_program::source_bindings::SourceBindings;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::source_weights::{SourceOwnerWitness, VerifiedTensorSource};
use zerogravity_core::accelerator::program::{SourceCall, SourceCallMode, ValueId};

pub(in crate::resident_inference) mod continuation;
mod faults;
pub(crate) mod fixed_prompt_natural;
#[cfg(all(feature = "cuda-native", target_os = "linux"))]
mod source_readback;
pub(crate) mod text;
pub(crate) use continuation::PrefixPhaseEnqueue;
mod continuation_qualify;
pub use continuation_qualify::{
    GdnSequenceFault, GdnSequenceFaultOutcome, GdnSequenceReceipt, GdnSequenceRestoreReceipt,
    qualify_gdn_sequence, qualify_gdn_sequence_fault, qualify_gdn_sequence_restore_refusal,
};
pub use faults::{
    GdnFragmentFault, GdnFragmentFaultOutcome, qualify_gdn_fragment_fault,
    qualify_gdn_mlp_fragment_fault,
};
use faults::{Mode, Outcome};
#[derive(Clone, Debug)]
pub struct GdnFragmentReceipt {
    pub rows: usize,
    pub dispatches: usize,
    pub native_launches: usize,
    pub readback_bytes: usize,
    pub carried_domains: usize,
    pub prior_root: RootBinding,
    pub model_root_published: bool,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
}
/// Borrowed actual physical bytes, made only inside original completion checks.
pub struct GdnFragmentReadback<'a> {
    invocation: &'a SourceCall,
    raw: &'a [Bytes],
    value_count: usize,
    gdn_count: usize,
    fused_nodes: &'a [usize],
    prior: &'a [(StateSlot, std::ops::Range<usize>)],
    candidate: &'a [(StateSlot, std::ops::Range<usize>)],
}
impl GdnFragmentReadback<'_> {
    pub fn invocation(&self) -> &SourceCall {
        self.invocation
    }
    pub fn value(&self, id: ValueId) -> Result<&[u8]> {
        if id.0 as usize >= self.value_count {
            return Err(Error::State("fragment value ID out of scope".into()));
        }
        let rms = 1 + self.gdn_count;
        let index = match id.0 {
            0 => rms + 1,
            1 => rms + 2,
            v => rms + v as usize + 1,
        };
        self.raw
            .get(index)
            .map(|raw| raw.as_ref())
            .ok_or_else(|| Error::State("fragment actual value absent".into()))
    }
    /// Independently computed unrounded F32 sum; exposed only after completion.
    pub fn fused_sum_f32(&self, node_index: usize) -> Result<&[u8]> {
        let slot = self
            .fused_nodes
            .iter()
            .position(|index| *index == node_index)
            .ok_or_else(|| Error::State("fused node outside recorded recipe".into()))?;
        let start = 1 + self.gdn_count + 3 + self.value_count - 2 + slot * 4;
        self.raw
            .get(start + 1)
            .map(|raw| raw.as_ref())
            .ok_or_else(|| Error::State("actual fused sum absent".into()))
    }
    pub fn state(&self, slot: StateSlot, candidate: bool) -> Result<(&[u8], Option<&[u8]>)> {
        let records = if candidate {
            self.candidate
        } else {
            self.prior
        };
        let (_, range) = records
            .iter()
            .find(|(value, _)| *value == slot)
            .ok_or_else(|| Error::State("original fragment state absent".into()))?;
        let first = self.raw[range.start].as_ref();
        Ok((
            first,
            if range.len() == 2 {
                Some(self.raw[range.start + 1].as_ref())
            } else {
                None
            },
        ))
    }
}
fn not_cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("native fragment cancelled".into()));
    }
    Ok(())
}
fn merge_accesses(
    target: &mut Vec<(ResidentLease, DeviceAccess)>,
    more: impl IntoIterator<Item = (ResidentLease, DeviceAccess)>,
) {
    for (lease, access) in more {
        if let Some((_, old)) = target
            .iter_mut()
            .find(|(other, _)| other.id() == lease.id())
        {
            if access == DeviceAccess::Write {
                *old = access;
            }
        } else {
            target.push((lease, access));
        }
    }
}
fn domain_readbacks(
    cohort: &mut Vec<crate::resident_leases::ResidentReadback>,
    domains: &[Domain],
) -> Result<Vec<(StateSlot, std::ops::Range<usize>)>> {
    let mut keys = Vec::with_capacity(domains.len());
    for domain in domains {
        let start = cohort.len();
        for lease in &domain.leases {
            cohort.push(lease.readback(0, lease.bytes())?);
        }
        keys.push((domain.slot, start..cohort.len()));
    }
    Ok(keys)
}
/// ROOT supplies the already-approved source/images and an expected-byte check
/// invoked only AFTER genuine arithmetic/D2H inside the original completion.
pub async fn qualify_gdn_fragment<F>(
    device: Device,
    record: Arc<SourceBindings>,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<GdnFragmentReceipt>
where
    F: FnOnce(GdnFragmentReadback<'_>) -> Result<()>,
{
    if record.selection().completed_layers() != 0 {
        return Err(Error::Unsupported(
            "complete MLP fragment requires separate connected images".into(),
        ));
    }
    match qualify_inner(
        device,
        record,
        NativeFragmentImages::Projection(inputs),
        cancelled,
        validate,
        Mode::Compare,
    )
    .await?
    {
        Outcome::Compared(out) => Ok(out),
        Outcome::Fault(_) => Err(Error::State(
            "positive native fragment returned fault".into(),
        )),
    }
}
pub async fn qualify_gdn_mlp_fragment<F>(
    device: Device,
    record: Arc<SourceBindings>,
    inputs: LayerMlpCubinInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<GdnFragmentReceipt>
where
    F: FnOnce(GdnFragmentReadback<'_>) -> Result<()>,
{
    not_cancelled(cancelled)?;
    if !matches!(record.selection().completed_layers(), 1 | 3) {
        return Err(Error::Unsupported(
            "complete initial layer0/prefix0..2 requires closed record".into(),
        ));
    }
    match qualify_inner(
        device,
        record,
        NativeFragmentImages::Connected(inputs),
        cancelled,
        validate,
        Mode::Compare,
    )
    .await?
    {
        Outcome::Compared(out) => Ok(out),
        Outcome::Fault(_) => Err(Error::State("complete fragment returned fault".into())),
    }
}
enum NativeFragmentImages {
    Projection(CubinQualificationInputs),
    Connected(LayerMlpCubinInputs),
}
impl NativeFragmentImages {
    fn device_cap(&self) -> u64 {
        match self {
            Self::Projection(v) => v.device_cap,
            Self::Connected(v) => v.projections.device_cap,
        }
    }
}
async fn qualify_inner<F>(
    device: Device,
    record: Arc<SourceBindings>,
    inputs: NativeFragmentImages,
    cancelled: Option<&AtomicBool>,
    validate: F,
    mode: Mode,
) -> Result<Outcome>
where
    F: FnOnce(GdnFragmentReadback<'_>) -> Result<()>,
{
    not_cancelled(cancelled)?;
    // This existing API remains initial-only even when request-phase metadata
    // can be recorded for a separate unpublished transaction consumer.
    let source = record.source().clone();
    let invocation = record.call().clone();
    if invocation.mode() != SourceCallMode::Prefill
        || invocation.rows().base() != 0
        || !matches!(invocation.rows().len(), 1 | 64)
    {
        return Err(Error::Unsupported(
            "existing native qualifier is initial-only".into(),
        ));
    }
    validate_provider(&device)?;
    let cap = capacity_bound(&record)?;
    let device_cap = inputs.device_cap();
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner {
            model: format!("original-hf-native-{:?}", source.source_identity()),
            session: "unpublishable-initial-fragment".into(),
        },
        cap,
        device_cap,
    )?;
    let physical_key = service.physical_key();
    let initial = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&physical_key)?;
    if initial.fenced || initial.reserved != cap as u64 {
        return Err(Error::State(
            "native qualifier requires original solitary envelope".into(),
        ));
    }
    if mode.fault() == Some(GdnFragmentFault::AdmissionPressure) {
        return faults::pressure(service, GdnFragmentFault::AdmissionPressure).map(Outcome::Fault);
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: cap as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let program =
        ResidentProgram::prepare_fragment_values(service.clone(), record.clone(), cancelled)
            .await?;
    slot.finish()?;
    let states = program.budget().states.clone();
    let state_bytes = program.budget().prior_state_bytes;
    let model = Arc::new(ResidentModel {
        service: service.clone(),
        workspace: Mutex::new(Workspace {
            program,
            greedy: None,
            leases: Vec::new(),
            state_accesses: Vec::new(),
            logit_accesses: Vec::new(),
            batch: None,
        }),
        allocation: Mutex::new(()),
        states,
        state_bytes,
        capacity: 512,
        vocab: 248320,
        state_pool: std::sync::OnceLock::new(),
        native_batch: false,
        #[cfg(feature = "hardware-abort-probe")]
        abort_after_launch: std::sync::atomic::AtomicUsize::new(0),
        #[cfg(feature = "hardware-abort-probe")]
        launch_seen: std::sync::atomic::AtomicUsize::new(0),
    });
    let session = model.session().await?;
    let base = session.binding();
    if base.frontier() != 0 {
        return Err(Error::State(
            "fragment requires genuine initial all42 root".into(),
        ));
    }
    not_cancelled(cancelled)?;
    let candidate = model.allocate_state(false).await?;
    let attempt = session
        .authority
        .begin_advance(invocation.rows().len() as u64)?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: cap as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut workspace = model.workspace.lock().await;
    let body = workspace.program.native_body()?.clone();
    let mut all_domains = Vec::with_capacity(body.gdn_nodes().len());
    for (layer, nodes) in body.gdn_nodes().iter().enumerate() {
        let layer_call = body.invocation_for_layer(layer)?;
        all_domains.push(session.bind_gdn_prefix_state(
            &attempt,
            &candidate,
            workspace.program.plan(),
            &layer_call,
            if invocation.rows().len() == 1 {
                crate::cuda::gdn::GdnPhase::InitialOne
            } else {
                crate::cuda::gdn::GdnPhase::Initial64
            },
            *nodes,
        )?);
    }
    body.prepare_gdn(
        service.clone(),
        workspace.program.plan(),
        workspace.program.values(),
        all_domains,
        cancelled,
    )
    .await?;
    not_cancelled(cancelled)?;
    let mut init = service.begin_initialization(base.root())?;
    // Independent immutable source payload owner prevents an images→body cycle.
    let image_owner: CompletionOwner = record.clone();
    init.retain_completion_owner(body.completion_owner())?;
    let images = match inputs {
        NativeFragmentImages::Projection(inputs) => prepare_native_images(
            &service,
            image_owner,
            &source,
            workspace.program.plan(),
            inputs,
            &mut init,
            cancelled,
        )?,
        NativeFragmentImages::Connected(inputs) => prepare_connected_native_images(
            &service,
            image_owner,
            &source,
            workspace.program.plan(),
            inputs,
            &mut init,
            cancelled,
        )?,
    };
    body.prepare_native(
        service.clone(),
        workspace.program.plan(),
        workspace.program.values(),
        images,
        &mut init,
        cancelled,
    )?;
    drop(init.complete_sync().await?);
    let state_ids = session
        .domains
        .iter()
        .chain(&candidate)
        .flat_map(|d| d.leases.iter().map(ResidentLease::id))
        .collect::<BTreeSet<_>>();
    let owned = workspace.program.finalize_fragment_preflight(&state_ids)?;
    if workspace.program.budget().total_with_state()? > cap {
        return Err(Error::Limit(
            "actual native graph exceeds its preadmitted closed scope".into(),
        ));
    }
    workspace.leases = owned.clone();
    workspace.state_accesses = owned;
    workspace.logit_accesses = Vec::new();
    slot.finish()?;
    let bindings = bind_domains(&session.domains, &candidate, 0)?;
    let mut accesses = body.accesses(workspace.program.values())?;
    merge_accesses(
        &mut accesses,
        session
            .domains
            .iter()
            .flat_map(|d| d.leases.iter().cloned().map(|l| (l, DeviceAccess::Read))),
    );
    merge_accesses(
        &mut accesses,
        candidate
            .iter()
            .flat_map(|d| d.leases.iter().cloned().map(|l| (l, DeviceAccess::Write))),
    );
    workspace.program.validate_runtime(
        service.client(),
        RuntimeParameters {
            token: invocation.rows().ids()[0],
            position: 0,
            valid_tokens: invocation.rows().len(),
        },
        &bindings,
    )?;
    not_cancelled(cancelled)?;
    let mut cohort = body.readbacks(workspace.program.values())?;
    let prior_keys = domain_readbacks(&mut cohort, &session.domains)?;
    let candidate_keys = domain_readbacks(&mut cohort, &candidate)?;
    let weights = body.immutable_weights()?;
    let weight_start = cohort.len();
    for (lease, _) in &weights {
        cohort.push(lease.readback(0, lease.bytes())?);
    }
    let cohort = Arc::new(cohort);
    let value_count = workspace.program.values().len();
    let gdn_count = body.gdn_nodes().len();
    let fused_nodes = body.fused_nodes();
    let readback_bytes = cohort.iter().map(|r| r.bytes()).sum();
    body.require_readback_capacity(readback_bytes)?;
    body.arm_fault(mode.fault())?;
    if let Mode::Fault { probe, .. } = &mode {
        probe.arm(faults::ProbeContext {
            service: &service,
            source: &source,
            body: &body,
            readbacks: &cohort,
            accesses: &accesses,
            base,
            root_buffers: session
                .domains
                .iter()
                .chain(&candidate)
                .map(|d| d.leases.len())
                .sum(),
        })?;
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: state_bytes as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut flight = service.submit(base.root(), &accesses)?;
    flight.retain_completion_owner(body.completion_owner())?;
    flight.retain_completion_owner(cohort.clone())?;
    let submitted = workspace.program.launch_fragment(
        service.client(),
        RuntimeParameters {
            token: invocation.rows().ids()[0],
            position: 0,
            valid_tokens: invocation.rows().len(),
        },
        &bindings,
        cancelled,
    )?;
    if submitted.logits.is_some()
        || submitted.proposals.len() != 42
        || submitted
            .proposals
            .iter()
            .zip(workspace.program.plan().proposed_domains())
            .any(|((version, _), expected)| version != expected)
    {
        return Err(Error::State(
            "native fragment cannot produce model logits/root or incomplete physical state receipt"
                .into(),
        ));
    }
    let promoted = accesses
        .iter()
        .filter(|(_, access)| *access == DeviceAccess::Write)
        .map(|(lease, _)| lease.clone())
        .collect::<Vec<_>>();
    let (ready, read) = flight
        .complete_with_readback(&promoted, &cohort, |raw| {
            not_cancelled(cancelled)?;
            body.validate_readbacks(raw, value_count)?;
            for (_, range) in &prior_keys {
                for bytes in &raw[range.clone()] {
                    if bytes.iter().any(|b| *b != 0) {
                        return Err(Error::State(
                            "initial actual original root changed/nonzero".into(),
                        ));
                    }
                }
            }
            for (slot, range) in &candidate_keys {
                let touched = match *slot {
                    StateSlot::Conv(zerogravity_core::accelerator::program::ConvDomain(i))
                    | StateSlot::Recurrent(
                        zerogravity_core::accelerator::program::RecurrentDomain(i),
                    ) => i < gdn_count,
                    _ => false,
                };
                if !touched {
                    for bytes in &raw[range.clone()] {
                        if bytes.iter().any(|b| *b != 0) {
                            return Err(Error::State(
                                "untouched original40 domains not carried exactly".into(),
                            ));
                        }
                    }
                }
            }
            for (actual, (_, original)) in raw[weight_start..].iter().zip(&weights) {
                let actual: &[u8] = actual.as_ref();
                if actual != *original {
                    return Err(Error::State(
                        "actual immutable native source W changed".into(),
                    ));
                }
            }
            validate(GdnFragmentReadback {
                invocation: &invocation,
                raw,
                value_count,
                gdn_count,
                fused_nodes: &fused_nodes,
                prior: &prior_keys,
                candidate: &candidate_keys,
            })
        })
        .await?;
    drop(read);
    drop(ready);
    slot.finish()?;
    if session.binding() != base || attempt.base() != base {
        return Err(Error::State(
            "unpublished fragment changed original root".into(),
        ));
    }
    // Unload only after genuine completion, inside another original immutable-W
    // flight; unknown/error never enters a best-effort destroy/recovery branch.
    let retire_accesses = weights
        .iter()
        .map(|(l, _)| (l.clone(), DeviceAccess::Read))
        .collect::<Vec<_>>();
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: 1,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut retirement = service.submit(base.root(), &retire_accesses)?;
    retirement.retain_completion_owner(body.completion_owner())?;
    unsafe {
        body.retire_after_physical_completion()?;
    }
    let reads = weights
        .iter()
        .map(|(l, _)| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let raw = retirement
        .complete_read_only_readback(&reads, |raw| {
            for (actual, (_, expected)) in raw.iter().zip(&weights) {
                let actual: &[u8] = actual.as_ref();
                if actual != *expected {
                    return Err(Error::State(
                        "immutable W changed during native retirement".into(),
                    ));
                }
            }
            Ok(())
        })
        .await?;
    drop(raw);
    slot.finish()?;
    let dispatches = submitted.dispatches;
    let native_launches = submitted.fused_dispatches;
    drop(submitted);
    drop(reads);
    drop(retire_accesses);
    drop(weights);
    drop(cohort);
    drop(promoted);
    drop(accesses);
    drop(bindings);
    drop(body);
    drop(workspace);
    drop(candidate);
    drop(attempt);
    drop(session);
    drop(model);
    drop(source);
    let final_resident_bytes = service.resident_bytes()?;
    if final_resident_bytes != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "native fragment original resident ledger not zero".into(),
        ));
    }
    drop(service);
    let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&physical_key)?;
    if physical.fenced || physical.reserved != 0 {
        return Err(Error::State(
            "native fragment original physical envelope not retired".into(),
        ));
    }
    Ok(Outcome::Compared(GdnFragmentReceipt {
        rows: invocation.rows().len(),
        dispatches,
        native_launches,
        readback_bytes,
        carried_domains: 42 - 2 * gdn_count,
        prior_root: base,
        model_root_published: false,
        final_resident_bytes,
        final_physical_reserved_bytes: physical.reserved,
    }))
}

pub(crate) mod model_state;

mod full_model_qualify;
pub use full_model_qualify::{
    SourceDomainReadback, SourceModelFault, SourceModelFaultOutcome, SourceModelInputs,
    SourceModelReadback, SourceModelReceipt, SourceModelScopes, qualify_source_model,
    qualify_source_model_fault,
};

pub use full_model_qualify::natural_cold::{
    ColdModelSlots, PreparedNaturalCold, prepare_natural_cold,
};
pub(crate) use full_model_qualify::natural_cold::{NaturalColdSnapshot, NaturalRetirementProbe};

// Original completion-bound natural publication; existing routes stay closed.
pub(crate) mod natural_publication;
pub(crate) mod natural_reuse;
pub use natural_publication::{
    CommittedBf16Feedback, NaturalCommitted, NaturalFault, NaturalFaultExecution,
    NaturalFaultOutcome, NaturalFeedbackRequest, NaturalObserver, NaturalReceipt,
    NaturalReuseFault, NaturalReuseFaultExecution, NaturalReuseFaultOutcome, NaturalReuseReceipt,
    NaturalReuseRound, NaturalSlot, NaturalStatePool, NaturalStep, qualify_natural_fault,
    qualify_natural_reuse_fault,
};
pub use natural_reuse::{
    PreparedReusableNatural, ReusableNaturalStorage, prepare_reusable_natural_cold,
    qualify_natural_reuse_pressure,
};

mod qwen2b_component;
pub use qwen2b_component::{
    Qwen2bGdnComponentFault, Qwen2bGdnComponentFaultOutcome, Qwen2bGdnComponentReadback,
    Qwen2bGdnComponentReceipt, qualify_qwen2b_gdn_component,
    qualify_qwen2b_gdn_component_arithmetic, qualify_qwen2b_gdn_component_fault,
};

mod fixed_prompt_component;
pub use fixed_prompt_component::{
    FixedPromptGdnComponentFault, FixedPromptGdnComponentFaultOutcome,
    FixedPromptGdnComponentReadback, FixedPromptGdnComponentReceipt,
    qualify_fixed_prompt_gdn_component, qualify_fixed_prompt_gdn_component_fault,
};

#[cfg(target_os = "linux")]
mod post_attention_component;
#[cfg(target_os = "linux")]
pub use post_attention_component::{
    PostAttentionArithmetic, PostAttentionComponentFault, PostAttentionComponentFaultOutcome,
    PostAttentionComponentReadback, PostAttentionComponentReceipt,
    qualify_post_attention_component, qualify_post_attention_component_fault,
    qualify_qwen2b_post_attention_component_arithmetic,
};

#[cfg(target_os = "linux")]
mod mlp_component;
#[cfg(target_os = "linux")]
pub use mlp_component::{
    MlpComponentFault, MlpComponentFaultOutcome, MlpComponentReadback, MlpComponentReceipt,
    qualify_mlp_component, qualify_mlp_component_fault,
};

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
mod fixed_prompt_model_prefix;
#[cfg(all(feature = "cuda-native", target_os = "linux"))]
pub use fixed_prompt_model_prefix::{
    FixedPromptModelPrefixFault, FixedPromptModelPrefixFaultOutcome,
    FixedPromptModelPrefixReadback, FixedPromptModelPrefixReceipt,
    FixedPromptQkKvQualificationInputs, qualify_fixed_prompt_model_fa_prefix,
    qualify_fixed_prompt_model_fa_prefix_fault, qualify_fixed_prompt_model_kv_prefix,
    qualify_fixed_prompt_model_kv_prefix_fault, qualify_fixed_prompt_model_layer3_prefix,
    qualify_fixed_prompt_model_layer3_prefix_fault, qualify_fixed_prompt_model_prefix,
    qualify_fixed_prompt_model_prefix_fault, qualify_fixed_prompt_model_qkv_prefix,
    qualify_fixed_prompt_model_qkv_prefix_fault,
};

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
mod fixed_prompt_full_model;
#[cfg(target_os = "linux")]
mod qwen2b_mlp_component;
#[cfg(all(feature = "cuda-native", target_os = "linux"))]
pub use fixed_prompt_full_model::{
    FixedPromptFullModelFault, FixedPromptFullModelFaultOutcome, FixedPromptFullModelReadback,
    FixedPromptFullModelReceipt, qualify_fixed_prompt_full_model,
    qualify_fixed_prompt_full_model_fault,
};
#[cfg(target_os = "linux")]
pub use qwen2b_mlp_component::{
    Qwen2bMlpComponentFault, Qwen2bMlpComponentFaultOutcome, Qwen2bMlpComponentReceipt,
    Qwen2bMlpComponentSource, qualify_qwen2b_mlp_component, qualify_qwen2b_mlp_component_fault,
};

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
pub use fixed_prompt_natural::{
    FixedPromptCallReceipt, FixedPromptCloseReceipt, FixedPromptEmptySnapshot,
    FixedPromptFeedbackRequest, FixedPromptLifecycleProfile, FixedPromptLoaded,
    FixedPromptLoadedObservation, FixedPromptLoadedSnapshot, FixedPromptNaturalFault,
    FixedPromptNaturalFaultOutcome, FixedPromptNaturalReadback, FixedPromptPending,
    FixedPromptSession, fixed_prompt_natural_host_bound, prepare_fixed_prompt_natural,
    prepare_fixed_prompt_natural_with_profile,
};

#[cfg(target_os = "linux")]
pub use text::{
    TextCallReceipt, TextCloseReceipt, TextCohortFault, TextCohortFaultOutcome,
    TextCohortFaultRetention, TextCohortPreparation, TextCohortReadback, TextCohortReceipt,
    TextCohortRowReceipt, TextEmptySnapshot, TextFault, TextFaultOutcome, TextFeedbackRequest,
    TextLoaded, TextLoadedObservation, TextLoadedSnapshot, TextPending, TextPendingCohort,
    TextQualificationInputs, TextReadback, TextResourceInventory, TextSession, prepare_text,
    prepare_text_b4,
};
