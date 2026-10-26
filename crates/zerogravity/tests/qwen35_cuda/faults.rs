//! One deliberately uncertain cached call per fresh ROOT-owned process.
use super::*;
use native::{TextFault, TextFaultOutcome};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Conv,
    Recurrent,
    Fa,
    Head,
    HeadOverwrite,
}
impl Mode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "conv-unknown" => Ok(Self::Conv),
            "recurrent-cancel" => Ok(Self::Recurrent),
            "last-fa-unknown" => Ok(Self::Fa),
            "head-unknown" => Ok(Self::Head),
            "head-cancel" => Ok(Self::HeadOverwrite),
            _ => Err(fail("exact ZG_QWEN35_FAULT_POINT mode required")),
        }
    }
    fn fault(self, resident: bool) -> TextFault {
        match self {
            Self::Conv => TextFault::UnknownAfterFirstConvolution,
            Self::Recurrent => TextFault::CancelBeforeFirstRecurrent,
            Self::Fa => TextFault::UnknownAfterLastFa,
            Self::Head => TextFault::UnknownDuringHead,
            Self::HeadOverwrite if resident => TextFault::CancelAfterResidentHead,
            Self::HeadOverwrite => TextFault::CancelBeforeHeadOverwrite,
        }
    }
}
fn counters(outcome: &TextFaultOutcome, mode: Mode, family: &str, resident: bool) -> Result<()> {
    // Source-derived actual call1 packet counts: GDN native4/3 per layer,
    // MLP4 per layer, attention8/7 per layer (.8B/2B respectively). The last
    // FA cut excludes its actual Sig/Mul/Out/MLP tail8/7. Head is streamed.
    let (before_head, last_fa, windows, charge, images) = match family {
        "0.8b" => (216usize, 208usize, 485usize, 126 * MIB, 17),
        "2b" => (192, 185, 970, 129 * MIB, 18),
        _ => return Err(fail("fault test exact authenticated model family")),
    };
    let midway = windows / 2 + 1;
    let (native, convolution, copies, head) = match mode {
        Mode::Conv | Mode::Recurrent => (2, 5, 2, [0; 4]),
        Mode::Fa => (last_fa, 18 * 5, 2 + 6 * 3, [0; 4]),
        Mode::Head | Mode::HeadOverwrite if resident => {
            (before_head + 1, 18 * 5, 2 + 6 * 3, [0, 1, 0, 0])
        }
        Mode::Head => (
            before_head + midway,
            18 * 5,
            2 + 6 * 3 + midway,
            [midway, midway, 0, 0],
        ),
        Mode::HeadOverwrite => (
            before_head + midway,
            18 * 5,
            2 + 6 * 3 + midway,
            [midway, midway, 0, 0],
        ),
    };
    ensure(
        outcome.native_launches == native
            && outcome.convolution_dispatches == convolution
            && outcome.dispatch_copies == copies
            && outcome.head_counts == head,
        "actual cut-specific native/Conv/copy/Head deltas",
    )?;
    let [issued, retired, queries, waits, pending] = outcome.head_retirement_counts;
    match mode {
        Mode::Head if !resident => ensure(
            issued == midway - 1
                && retired <= issued
                && pending == issued - retired
                && pending <= 1
                && queries >= retired
                && waits <= retired,
            "unknown current window stays outside the prior bounded retirement marks",
        )?,
        Mode::HeadOverwrite if !resident => ensure(
            issued == midway
                && retired == midway
                && pending == 0
                && queries >= retired
                && waits <= retired,
            "late cancellation retires the current actual mark before refusing overwrite",
        )?,
        _ => ensure(
            outcome.head_retirement_counts == [0; 5],
            "no unissued Head retirement",
        )?,
    }
    // A clear is an actual extra command before its associated native packet;
    // it cannot exceed the submitted native count. Exact source-prefix totals
    // are additionally compared by ROOT across sibling fresh-process cuts.
    ensure(
        outcome.workspace_clears <= outcome.native_launches,
        "actual workspace clears require an actually reached native packet",
    )?;
    ensure(
        outcome.native_image_owners == images
            && outcome.native_kernel_owners > 0
            && outcome.native_charge_bytes == charge
            && outcome.image_admission_bytes > 0,
        "complete source-scoped native owner/charge cohort",
    )
}
fn retained(
    outcome: &TextFaultOutcome,
    mode: Mode,
    root: zerogravity::session_root::RootBinding,
    resident: bool,
    physical_reserved_bytes: u64,
) -> Result<()> {
    ensure(
        outcome.fault == mode.fault(resident)
            && outcome.status == "UNKNOWN_CACHED_CALL_RETAINED_NO_PUBLICATION"
            && outcome.root_before == root
            && outcome.root_after == root
            && outcome.root_before.frontier() == 47,
        "genuine Initial47 root survives uncertain cached call unchanged",
    )?;
    ensure(
        outcome.ready_completions == 0
            && outcome.readback_commands == 0
            && outcome.expected_callbacks == 0
            && outcome.published_roots == 0,
        "failed cached call has no Ready/D2H/expected callback/publication",
    )?;
    ensure(
        outcome.source_backing_retained
            && outcome.body_retained
            && outcome.cold_graph_retained
            && outcome.candidate_buffers_unpromoted
            && outcome.rms_run_retained
            && outcome.rms_epoch > 0
            && outcome.rms_private_bytes > 0
            && outcome.head_run_retained,
        "exact weak current/cold/source/Head run graph remains owned",
    )?;
    ensure(
        outcome.prepared_graph_leases >= outcome.submitted_current_leases
            && outcome.submitted_current_leases > 48
            && outcome.readback_descriptors == 27
            && outcome.quarantined_handles == outcome.submitted_current_leases
            && outcome.quarantined_completion_owners > 0,
        "actual original submitted handles and completion owners quarantined",
    )?;
    ensure(
        !outcome.service_usable
            && outcome.pending_events == 1
            && outcome.physical_fenced
            && outcome.physical_reserved_bytes == physical_reserved_bytes
            && outcome.resident_bytes > 0
            && outcome.resident_bytes <= qwen35_cuda::PROVIDER_MIB * MIB,
        "unknown service unusable; original nonzero reservations remain fenced",
    )?;
    ensure(
        outcome.head_resident_immutable_weight == resident,
        "actual initialized immutable tied source owner retained",
    )?;
    let in_head = !resident && matches!(mode, Mode::Head | Mode::HeadOverwrite);
    ensure(
        outcome.head_source_window_retained == in_head
            && (outcome.head_source_staging_bytes > 0) == in_head,
        "actual partially consumed Head source-window lifetime",
    )
}

pub(super) fn run() -> Result<()> {
    ensure(!cfg!(debug_assertions), "release cached fault only")?;
    let mode_text = std::env::var("ZG_QWEN35_FAULT_POINT")
        .map_err(|_| fail("ZG_QWEN35_FAULT_POINT required"))?;
    let mode = Mode::parse(&mode_text)?;
    let source = env("ZG_QWEN35_MODEL_DIR")?;
    let bundle = env("ZG_QWEN35_CUDA_KERNEL_DIR")?;
    let reference = env("ZG_QWEN35_REFERENCE_DIR")?;
    let prompts = env("ZG_QWEN35_PROMPT_FIXTURES")?;
    let public = oracle::json(&prompts, MIB)?;
    let ids = fixture_prompt(&public, 47)?;
    let host = FixtureHost::new(ResourceOwner::new("qwen35-cuda-physical", "cached-fault")?)?;
    let device =
        zerogravity::gpu::Device::cuda(0).map_err(|e| fail(format!("actual CUDA device: {e}")))?;
    let model = block_on(qwen35_cuda::load(
        device,
        &source,
        &bundle,
        &host,
        qwen35_cuda::PROVIDER_MIB * MIB,
        None,
    ))?;
    let family = model.source_metadata().family;
    let cell = reference.join(format!("{family}-47"));
    let loaded = model.resident().clone();
    let resident = loaded.resources().resident_head_bytes != 0;
    let facade_weak = Arc::downgrade(&model);
    let loaded_weak = Arc::downgrade(&loaded);
    let mut session = block_on(model.session(None))?;
    let observation = loaded.observation();
    let original = session.binding();
    ensure(
        original.frontier() == 0,
        "genuine fresh empty session before Initial47",
    )?;
    let mut callbacks = 0usize;
    let initial = block_on(session.infer_prompt_with(&ids, None, |read| {
        let _expected = host.arbiter.admit(
            &host.resource,
            host.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: (256 * MIB) as u64,
            }],
        )?;
        callbacks += 1;
        compare::completed(read, &cell, 0, 47, model.source_profile())
    }))?;
    let root = session.binding();
    ensure(
        callbacks == 1
            && initial.root == root
            && root.session() == original.session()
            && root.root() == original.root() + 1
            && root.frontier() == 47
            && initial.executed_nodes == 255
            && initial.consumed_tokens == 47
            && initial.native_launches > 0
            && initial.dispatches > 0,
        "real independently compared Initial47 commits once before cached cut",
    )?;
    let healthy = observation.snapshot()?;
    ensure(
        healthy.usable
            && healthy.pending_events == 0
            && healthy.provider_capacity_bytes == qwen35_cuda::PROVIDER_MIB * MIB,
        "Initial47 completed without a preceding fence",
    )?;
    let physical_key = healthy.physical_key.clone();
    let physical_before = host.arbiter.snapshot(&physical_key)?;
    let host_before = host.arbiter.snapshot(&host.resource)?;
    let expected_physical = (healthy.provider_capacity_bytes as u64)
        .checked_add(healthy.backend_device_reserved_bytes)
        .ok_or_else(|| fail("model/backend physical reservation overflow"))?;
    ensure(
        physical_before.limit == (qwen35_cuda::DEVICE_MIB * MIB) as u64
            && !physical_before.fenced
            && healthy.backend_device_reserved_bytes == (512 * MIB) as u64
            && healthy.backend_host_reserved_bytes == (384 * MIB) as u64
            && healthy.physical_reserved_bytes == expected_physical
            && physical_before.reserved == expected_physical
            && physical_before.reservations == 2
            && physical_before
                .active_models
                .iter()
                .map(String::as_str)
                .eq(["backend/cuda", "backend/physical"])
            && physical_before.per_class.len() == 1
            && physical_before
                .per_class
                .get(&AllocationClass::ProviderEnvelope)
                == Some(&expected_physical)
            && host_before.limit == fixture_host::declared_limit(&host)?
            && host_before.active_models.contains("backend/cuda")
            && host_before.active_models.contains(&host.owner.model)
            && host_before.reserved > healthy.backend_host_reserved_bytes,
        "original model/provider plus actual detached backend HOST/DEVICE owners",
    )?;
    host.require_retained(healthy.backend_host_reserved_bytes)?;
    // probe_cached requires sole Loaded ownership. No external service/source
    // observer may supply the retained-resource proof after the real error.
    drop(observation);
    drop(loaded);
    drop(model);
    ensure(
        facade_weak.upgrade().is_none(),
        "external facade source owner dropped",
    )?;
    let outcome = block_on(session.probe_cached(mode.fault(resident), None))?;
    println!("QWEN35_CUDA_CACHED_FAULT_OBSERVATION {outcome:?}");
    ensure(
        loaded_weak.upgrade().is_none(),
        "consumed session dropped the last external Loaded owner",
    )?;
    retained(&outcome, mode, root, resident, expected_physical)?;
    counters(&outcome, mode, family, resident)?;
    let physical = host.arbiter.snapshot(&physical_key)?;
    let retained_host = host.arbiter.snapshot(&host.resource)?;
    host.require_retained(healthy.backend_host_reserved_bytes)?;
    ensure(
        physical.fenced
            && physical.reserved == outcome.physical_reserved_bytes
            && physical.reservations == physical_before.reservations
            && physical.active_models == physical_before.active_models
            && physical.per_class == physical_before.per_class
            && retained_host.limit == host_before.limit
            && retained_host.active_models.contains("backend/cuda")
            && retained_host.active_models.contains(&host.owner.model)
            && retained_host.reserved > healthy.backend_host_reserved_bytes,
        "actual retained source/HOST/provider/backend owners survive caller drops",
    )?;
    println!(
        "QWEN35_CUDA_CACHED_FAULT_ACCOUNTING provider_bytes={} backend_device_bytes={} backend_host_bytes={} physical_before={physical_before:?} physical_after={physical:?} host_before={host_before:?} host_after={retained_host:?}",
        healthy.provider_capacity_bytes,
        healthy.backend_device_reserved_bytes,
        healthy.backend_host_reserved_bytes,
    );
    println!(
        "QWEN35_CUDA_CACHED_FAULT_PASS {}",
        serde_json::json!({
            "model":family,"mode":mode_text,"root":root.root(),"frontier":root.frontier(),
            "native":outcome.native_launches,"convolution":outcome.convolution_dispatches,
            "copies":outcome.dispatch_copies,"workspace_clears":outcome.workspace_clears,
            "head":outcome.head_counts,"head_retirement":outcome.head_retirement_counts,"prepared_graph_leases":outcome.prepared_graph_leases,
            "rms_run_retained":outcome.rms_run_retained,"rms_epoch":outcome.rms_epoch,
            "rms_private_bytes":outcome.rms_private_bytes,"rms_private_initialized":outcome.rms_private_initialized,
            "submitted_current_leases":outcome.submitted_current_leases,"readback_descriptors":outcome.readback_descriptors,
            "native_images":outcome.native_image_owners,"native_records":outcome.native_kernel_owners,
            "native_charge_bytes":outcome.native_charge_bytes,"resident_bytes":outcome.resident_bytes,
            "pending_events":outcome.pending_events,"quarantined_handles":outcome.quarantined_handles,
            "quarantined_completion_owners":outcome.quarantined_completion_owners,
            "physical_reserved_bytes":outcome.physical_reserved_bytes,"service_usable":outcome.service_usable,
            "head_source_staging_bytes":outcome.head_source_staging_bytes,
            "ready":outcome.ready_completions,"readbacks":outcome.readback_commands,
            "expected_callbacks":outcome.expected_callbacks,"published_roots":outcome.published_roots,
            "physical_fenced":outcome.physical_fenced,"performance_qualified":false
        })
    );
    // No close/recovery/finalzero claim: ROOT owns process/group/EOF retirement.
    Ok(())
}

#[test]
fn fault_modes_are_closed_and_distinct() {
    let modes = [
        "conv-unknown",
        "recurrent-cancel",
        "last-fa-unknown",
        "head-unknown",
        "head-cancel",
    ];
    for (i, a) in modes.iter().enumerate() {
        for (j, b) in modes.iter().enumerate() {
            assert_eq!(Mode::parse(a).unwrap() == Mode::parse(b).unwrap(), i == j);
        }
    }
    for wrong in [
        "",
        "positive",
        "unknown",
        "47",
        "head-before-first-window",
        "head-cancel ",
    ] {
        assert!(Mode::parse(wrong).is_err());
    }
}
