//! Source-accounted constructor bound; original admission still decides every allocation.
use super::*;
use crate::cuda::native::cohort_native::CohortPhases;
use crate::cuda::native::{model_values::bounded::StorageRecipe, text_native::TextNativeInventory};
use crate::cuda::rms::{FusedRmsPrepared, RmsPrepared};
use crate::resident_program::{bf16_greedy::PreparedBf16Greedy, native_full::text};
const ALIGN: usize = 512;
const ROOTS: usize = 14; // 8current+2Pending+2snapshot+2displaced by original commit/restore.
const CALLS: usize = 4; // workspace current, two Pending and its serialized preparing successor.
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| Error::Limit("short-HF whole resource sum overflow".into()))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Limit("short-HF whole resource product overflow".into()))
}
fn sum(items: impl IntoIterator<Item = usize>) -> Result<usize> {
    items.into_iter().try_fold(0, add)
}
fn aligned(bytes: usize) -> Result<usize> {
    if bytes == 0 {
        return Err(Error::State("short-HF zero physical owner".into()));
    }
    add(bytes, ALIGN - 1).map(|n| n & !(ALIGN - 1))
}
#[derive(Clone, Debug)]
pub struct TextResourceInventory {
    pub managed_allocations: usize,
    pub value_rms_isolated_aligned_bytes: usize,
    pub value_rms_shared_aligned_bytes: usize,
    pub shared_workspace_views: usize,
    pub managed_aligned_bytes: usize,
    pub native_provider_bytes: usize,
    pub provider_upper_bytes: usize,
    pub host_existing_bytes: usize,
    pub host_retained_additional_bytes: usize,
    pub host_transient_bytes: usize,
    pub host_upper_bytes: usize,
    pub source_chunk_upper: usize,
    pub root_count: usize,
    pub root_initialization_host_bytes: usize,
    pub readback_reservation_bytes: usize,
    pub resident_head_bytes: usize,
    pub rotary_source_positions: usize,
    pub rotary_retained_host_bytes: usize,
    pub rotary_uploaded_bytes: usize,
}
#[derive(Clone, Copy)]
pub(super) struct TextSourceAdmission<'a> {
    pub(super) rotary_positions: usize,
    pub(super) rotary_retained_host_bytes: usize,
    pub(super) compiler_context: Option<&'a crate::physical_service::CudaCompilerContextBudget>,
}
#[derive(Default)]
struct DeviceLedger {
    count: usize,
    bytes: usize,
}
impl DeviceLedger {
    fn extend(&mut self, layout: impl IntoIterator<Item = usize>) -> Result<()> {
        for bytes in layout {
            self.count = add(self.count, 1)?;
            self.bytes = add(self.bytes, aligned(bytes)?)?;
        }
        Ok(())
    }
}
fn require_fit(
    provider: usize,
    host: usize,
    limit: zerogravity_core::optimization::MemoryFootprint,
) -> Result<()> {
    if provider as u64 > limit.device_bytes || host as u64 > limit.host_bytes {
        return Err(Error::Limit(format!(
            "short-HF whole source inventory provider={provider} host={host} exceeds configured device={} host={} allowance",
            limit.device_bytes, limit.host_bytes
        )));
    }
    Ok(())
}
fn current_call_host(capture: TextKernelCapture) -> Result<usize> {
    let hidden = capture.hidden_width();
    let rows = capture
        .captured_prefill_rows()
        .iter()
        .copied()
        .max()
        .ok_or_else(|| Error::State("short-HF current captured prefill extent absent".into()))?;
    // Invocation binding512KiB +Body4MiB +StorageRecipe1MiB; authentic current
    // scope512KiB/state256KiB; all currentRMS bindings/actors/source wrappers.
    // GDN code/workspace/adapter/prepared wrappers:4x18x64KiB.
    // Native Programs2MiB +AttentionCurrent512KiB; positions/rows/sampler/head.
    sum([
        5 * MIB + CALL_METADATA_BYTES,
        512 * 1024,
        256 * 1024,
        149 * 64 * 1024,
        4 * 18 * 64 * 1024,
        2 * MIB + 512 * 1024,
        // Current transport wrappers share source-owned cold definitions;
        // their complete text/ABI/key owners are in the cold layouts below.
        3 * (3 * 513 * 8) + 64 * 1024,
        add(mul(rows, mul(hidden, 6)?)?, 2 * 64 * 1024)?,
        64 * 1024,
        64 * 1024,
    ])
}
impl TextResourceInventory {
    /// Cost-only inspection of the SAME source-owned cold analysis. No Client,
    /// computed rotary prefix, RuntimeParameters, root, native or Ready is made.
    pub fn inspect_b4(
        source: &VerifiedTensorSource,
        capture: TextKernelCapture,
        scopes: &[(Arc<SourceBindings>, GdnKernelRecipe)],
        inputs: &crate::cuda::native::FixedPromptMlpQualificationInputs,
        rotary_positions: usize,
        limit: zerogravity_core::optimization::MemoryFootprint,
    ) -> Result<Self> {
        Self::inspect(
            true,
            source,
            capture,
            scopes,
            inputs,
            rotary_positions,
            limit,
        )
    }
    /// Cost-only inspection of the original N1 source-owned cold analysis.
    /// The caller must bind the matching N1 native image cohort.
    pub fn inspect_n1(
        source: &VerifiedTensorSource,
        capture: TextKernelCapture,
        scopes: &[(Arc<SourceBindings>, GdnKernelRecipe)],
        inputs: &crate::cuda::native::FixedPromptMlpQualificationInputs,
        rotary_positions: usize,
        limit: zerogravity_core::optimization::MemoryFootprint,
    ) -> Result<Self> {
        Self::inspect(
            false,
            source,
            capture,
            scopes,
            inputs,
            rotary_positions,
            limit,
        )
    }
    fn inspect(
        cohort_b4: bool,
        source: &VerifiedTensorSource,
        capture: TextKernelCapture,
        scopes: &[(Arc<SourceBindings>, GdnKernelRecipe)],
        inputs: &crate::cuda::native::FixedPromptMlpQualificationInputs,
        rotary_positions: usize,
        limit: zerogravity_core::optimization::MemoryFootprint,
    ) -> Result<Self> {
        let verified = source;
        let required = crate::cuda::native::text_required_images_for_cohort(capture, cohort_b4)?;
        if inputs.projections.device_cap != 5120 * MIB as u64
            || inputs.projections.images.len() != required.len()
            || inputs
                .projections
                .images
                .iter()
                .map(|image| image.kind)
                .collect::<std::collections::BTreeSet<_>>()
                != required.into_iter().collect()
            || inputs.projections.images.iter().any(|image| {
                !image.admission.source().same_issuer(verified)
                    || image.bytes.len() != image.kind.bytes()
                    || image.bytes.capacity() > image.admission.bytes()
            })
            || inputs.activation.as_ref().is_none_or(|image| {
                !image.admission.source().same_issuer(verified)
                    || image.bytes.len() != 901_664
                    || image.bytes.capacity() > image.admission.bytes()
            })
        {
            return Err(Error::State(
                "short-HF inventory actual source/image owners and original caps".into(),
            ));
        }
        // Cost reservation only; no computed prefix or numerical authority.
        let rotary_retained_host_bytes =
            crate::cuda::native::source_rotary::text_rotary_retained_host_bytes(rotary_positions)?;
        let _rotary_metadata = verified.reserve_workspace(rotary_retained_host_bytes)?;
        let _metadata = verified.reserve_workspace(8 * MIB)?;
        let _owner_metadata = verified.reserve_workspace(2 * MIB)?;
        let (_bindings, scopes) = super::cold_source_scopes(source, capture, scopes)?;
        let (_, inventory) = Self::select_profile(
            cohort_b4,
            source,
            capture,
            &scopes,
            crate::cuda::native::text_head::HeadMode::environment_preference()?,
            limit,
            TextSourceAdmission {
                rotary_positions,
                rotary_retained_host_bytes,
                compiler_context: None,
            },
        )?;
        Ok(inventory)
    }
    pub(super) fn select_profile(
        cohort_b4: bool,
        source: &VerifiedTensorSource,
        capture: TextKernelCapture,
        scopes: &[Arc<crate::cuda::native::text_model::TextModelBinding>],
        preference: zerogravity_core::optimization::ResidencyPreference,
        limit: zerogravity_core::optimization::MemoryFootprint,
        admission: TextSourceAdmission<'_>,
    ) -> Result<(crate::cuda::native::text_head::HeadMode, Self)> {
        use crate::cuda::native::text_head::HeadMode;
        use zerogravity_core::optimization::{
            MemoryFootprint, ResidencyChoice, ResidencyCosts, select_immutable_residency,
        };
        let streamed = if cohort_b4 {
            Self::derive_profile_admitted(
                true,
                source,
                capture,
                scopes,
                HeadMode::Streamed,
                admission,
            )?
        } else {
            Self::derive_profile_admitted(
                false,
                source,
                capture,
                scopes,
                HeadMode::Streamed,
                admission,
            )?
        };
        let resident =
            HeadMode::resident_available(crate::resident_leases::TiedWeight::source_layout(
                source,
                head_member(scopes[0].record())?,
            )?)
            .then(|| {
                Self::derive_profile_admitted(
                    cohort_b4,
                    source,
                    capture,
                    scopes,
                    HeadMode::Resident,
                    admission,
                )
            })
            .transpose()?;
        let footprint = |inventory: &Self| MemoryFootprint {
            host_bytes: inventory.host_upper_bytes as u64,
            device_bytes: inventory.provider_upper_bytes as u64,
        };
        if cohort_b4 {
            let selected = resident.ok_or_else(|| {
                Error::Unsupported("B4 requires source-held full tied table".into())
            })?;
            require_fit(
                selected.provider_upper_bytes,
                selected.host_upper_bytes,
                limit,
            )?;
            return Ok((HeadMode::Resident, selected));
        }
        let choice = select_immutable_residency(
            ResidencyCosts {
                streamed: footprint(&streamed),
                resident: resident.as_ref().map(footprint),
            },
            limit,
            preference,
        )?;
        let selected = match choice {
            ResidencyChoice::Resident => {
                resident.ok_or_else(|| Error::State("selected resident provider absent".into()))?
            }
            ResidencyChoice::Streamed => streamed,
        };
        require_fit(
            selected.provider_upper_bytes,
            selected.host_upper_bytes,
            limit,
        )?;
        Ok((HeadMode::from_choice(choice), selected))
    }
    fn derive_profile_admitted(
        cohort_b4: bool,
        source: &VerifiedTensorSource,
        capture: TextKernelCapture,
        scopes: &[Arc<crate::cuda::native::text_model::TextModelBinding>],
        head_mode: crate::cuda::native::text_head::HeadMode,
        admission: TextSourceAdmission<'_>,
    ) -> Result<Self> {
        capture.require_cold_scopes(scopes.iter().map(Arc::as_ref))?;
        let storage = StorageRecipe::scope_geometry_inventory(scopes, text::limits())?;
        let native = TextNativeInventory::for_scopes(capture, scopes, false, cohort_b4, head_mode)?;
        let rotary_source_bytes = mul(admission.rotary_positions, 128)?;
        if admission.rotary_retained_host_bytes
            != crate::cuda::native::source_rotary::text_rotary_retained_host_bytes(
                admission.rotary_positions,
            )?
            || native.rotary_bytes > rotary_source_bytes
        {
            return Err(Error::Unsupported(
                "text source rotary span/cost cannot satisfy native bindings".into(),
            ));
        }
        let phases = CohortPhases::for_capture(capture);
        let roots = if cohort_b4 {
            // Preserve the wider candidate reserve for captured initial B4.
            super::cohort::DecodeCohortRootBound::derive(
                8,
                if phases.contains(true) { 8 } else { 4 },
            )?
            .total
        } else {
            ROOTS
        };
        let hidden = capture.hidden_width();
        let mut device = DeviceLedger {
            count: storage.physical_leases,
            bytes: storage.physical_aligned_bytes,
        };
        let mut retained = storage.workspace_host_bytes;
        let mut chunks = 0;
        let mut gdn_initial = 0;
        for (scope, actual) in scopes.iter().zip(&storage.geometries) {
            let rows = scope.call().rows().len();
            if actual.rows != rows
                || actual.values.logical_values != 327
                || actual.values.actor_slots != 98
                || actual.values.physical_slots != 281
                || actual.rms_private_leases != 106
                || actual.physical_leases
                    != actual.values.physical_slots + actual.rms_private_leases
                || actual.physical_aligned_bytes < actual.rms_private_aligned_bytes
                || actual.rms_private_bytes > actual.rms_private_aligned_bytes
                || actual.values.logical_slot_bytes > actual.values.aligned_slot_bytes
                || actual.values.reused_slots != 2
            {
                return Err(Error::State(
                    "short-HF actual original value/RMS allocator inventory changed".into(),
                ));
            }
            device.extend(TextGdnWorkspace::allocation_layout(
                scope.call(),
                *scope.gdn_recipe(),
            )?)?;
            device.extend(ColdTextGdn::allocation_layout(scope)?)?;
            device.extend(PreparedBf16Greedy::text_allocation_layout())?;
            device.extend([4])?; // actual cold Body status
            let (workspace_host, initial) =
                TextGdnWorkspace::host_layout(scope.call(), *scope.gdn_recipe())?;
            gdn_initial = add(gdn_initial, initial)?;
            retained = sum([
                retained,
                workspace_host,
                ColdTextGdn::host_bytes(scope)?,
                RmsPrepared::text_source_host_bytes(rows, hidden)?,
                mul(48, FusedRmsPrepared::text_source_host_bytes(rows, hidden)?)?,
                100 * 64 * 1024,
                // Original individual BF16 certificates plus cohort kernels;
                // 327 actual values and at most ceil(327/8) prepared cohorts.
                (327 + 327usize.div_ceil(8))
                    * crate::cuda::kernel::PinnedKernel::KEY_METADATA_BYTES,
                5 * MIB,
                64 * 1024,
            ])?;
            // No sharing/dedup or source-offset alignment assumed. Every small
            // gamma/row is shorter than1MiB, hence at most two actual chunks.
            chunks = add(chunks, mul(2, rows + 49)?)?;
        }
        device.extend(PendingModelWeights::allocation_layout(&scopes[0])?)?;
        device.extend(ModelGdnParameters::allocation_layout(&scopes[0])?)?;
        let root = FullModelProgram::text_root_layout(&scopes[0])?;
        for _ in 0..roots {
            device.extend(root.iter().copied())?;
        }
        let root_initialization_host_bytes = mul(3, sum(root.iter().copied())?)?;
        // Fresh status/packet owners include current, pending and retained
        // committed feedback. No initialized flag reset is inferred by reuse.
        for _ in 0..roots {
            device.extend([4, 12])?;
        }
        device.extend([8_519_680, 3 * 513 * 8, native.rotary_bytes])?;
        device.extend(native.device_layout)?;
        chunks = add(chunks, 18 * 4 * 2)?; // actual72 learned GDN parameter windows
        let readback = FullModelProgram::text_readback_reservation(scopes)?;
        retained = sum([
            retained,
            // Model/app templates are already admitted in host_existing; GPU
            // owns only the physical prepared-site cache below.
            TextModelBindings::prepared_metadata_bytes(capture)?,
            114 * 64 * 1024,
            18 * (64 + 256) * 1024,
            mul(chunks, MIB + 24)?,
            native.retained_host_bytes,
            crate::resident_leases::backing_pool::ResidentPool::metadata_bytes(root.len(), roots)?,
            readback,
            mul(CALLS, current_call_host(capture)?)?,
            2 * 64 * 1024,
        ])?;
        // Distinct original phases: coldGDNzeros; native cache initialization;
        // serialized Sessionroot initialization; source verification/diagnostic
        // callback. Source W is1MiB-at-a-time, never full tied/learned hostshadow.
        let mut transient = gdn_initial
            .max(native.initialization_host_bytes)
            .max(root_initialization_host_bytes)
            .max(256 * MIB + 64 * MIB);
        let host_existing = usize::try_from(source.workspace_reserved_bytes()?)
            .map_err(|_| Error::Limit("short-HF host snapshot usize extent".into()))?;
        if host_existing < admission.rotary_retained_host_bytes {
            return Err(Error::State(
                "text retained rotary source absent from host snapshot".into(),
            ));
        }
        if cohort_b4 {
            // Full cold cache owned zero plus pinned/fallback staging. Sum with
            // other cold peaks conservatively while CohortLibrary prepares.
            transient = add(
                transient,
                super::cohort::ColdCohort::initialization_host_bytes()?,
            )?;
            device.extend(super::cohort::DecodeCohortWorkspace::allocation_layout(
                capture, scopes,
            )?)?;
            device.extend([crate::resident_program::native_cohort::CACHE_BYTES, 4])?;
            // Exactly one current body; completed Pending retains only original
            // candidate roots and fresh packets already priced by22 roots.
            // Price only source-supported phases. Both .8B phases remain
            // conservatively summed despite exclusive execution.
            for initial in phases.iter() {
                device.extend(
                    crate::resident_program::native_cohort::FullModelCohort::allocation_layout(
                        initial, capture,
                    ),
                )?;
            }
            let body_host = phases.iter().try_fold(0usize, |peak, initial| {
                Ok::<_, Error>(peak.max(
                    crate::resident_program::native_cohort::FullModelCohort::host_bytes(
                        initial, capture,
                    )?,
                ))
            })?;
            // The N1 and B4 completion callbacks hold the SAME model.workspace
            // lock. Retain the actual N1 admission and reserve only the delta.
            let diagnostic_delta = super::cohort::READBACK_BYTES
                .checked_sub(readback)
                .ok_or_else(|| Error::State("B4 shared original readback bound".into()))?;
            retained = sum([
                retained,
                diagnostic_delta,
                super::cohort::DecodeCohortWorkspace::host_bytes()?,
                crate::resident_program::native_cohort::CohortLibrary::host_bytes(capture)?,
                body_host,
                // Four genuine current records/scope/state/provenance owners;
                // price each with the existing complete N1 bound, no Arc shortcut.
                mul(4, current_call_host(capture)?)?,
                64 * 1024, // exact cohort input issuer/join
            ])?;
        }
        // Real detached process/context admission, separate from source owner.
        let already_reserved = admission
            .compiler_context
            .map(|c| c.require_host(source))
            .transpose()?
            .unwrap_or(0) as usize;
        if host_existing < already_reserved {
            return Err(Error::State(
                "CUDA HOST context absent from original whole-domain snapshot".into(),
            ));
        }
        let retained = sum([
            retained,
            crate::physical_service::CUDA_COMPILER_HOST_BYTES as usize - already_reserved,
            crate::cuda::rms::FullModelRmsScratch::readiness_temporary_host_bytes(cohort_b4)?,
        ])?;
        let host = sum([host_existing, retained, transient])?;
        let provider = add(device.bytes, native.provider_bytes)?;
        Ok(Self {
            managed_allocations: device.count,
            value_rms_isolated_aligned_bytes: storage.isolated_physical_aligned_bytes,
            value_rms_shared_aligned_bytes: storage.physical_aligned_bytes,
            shared_workspace_views: storage.workspace_views,
            managed_aligned_bytes: device.bytes,
            native_provider_bytes: native.provider_bytes,
            provider_upper_bytes: provider,
            host_existing_bytes: host_existing,
            host_retained_additional_bytes: retained,
            host_transient_bytes: transient,
            host_upper_bytes: host,
            source_chunk_upper: chunks,
            root_count: roots,
            root_initialization_host_bytes,
            rotary_source_positions: admission.rotary_positions,
            rotary_retained_host_bytes: admission.rotary_retained_host_bytes,
            rotary_uploaded_bytes: native.rotary_bytes,
            readback_reservation_bytes: readback,
            resident_head_bytes: if head_mode.resident() {
                crate::resident_leases::TiedWeight::source_layout(
                    source,
                    head_member(scopes[0].record())?,
                )?
                .bytes()
            } else {
                0
            },
        })
    }
}
pub(super) fn root_host_admission(
    source: &VerifiedTensorSource,
    bytes: usize,
) -> Result<Vec<WindowWorkspace>> {
    let mut left = bytes;
    let mut tickets = Vec::with_capacity(bytes.div_ceil(32 * MIB));
    while left != 0 {
        let n = left.min(32 * MIB);
        tickets.push(source.reserve_workspace(n)?);
        left -= n;
    }
    Ok(tickets)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_original_cap_alignment_and_overflow_never_round_down() {
        assert_eq!(aligned(4).unwrap(), 512);
        assert_eq!(aligned(513).unwrap(), 1024);
        assert!(aligned(0).is_err());
        assert!(aligned(usize::MAX).is_err());
        assert!(add(usize::MAX, 1).is_err());
        assert!(mul(usize::MAX, 2).is_err());
        let limit = zerogravity_core::optimization::MemoryFootprint {
            host_bytes: (4096 * MIB) as u64,
            device_bytes: (4608 * MIB) as u64,
        };
        require_fit(4608 * MIB, 4096 * MIB, limit).unwrap();
        assert!(require_fit(4608 * MIB + 1, 0, limit).is_err());
        assert!(require_fit(0, 4096 * MIB + 1, limit).is_err());
    }
    #[test]
    fn cohort_complete_host_delta_uses_original_exclusive_readback_owner() -> Result<()> {
        use crate::resident_program::native_cohort::{CohortLibrary, FullModelCohort};
        let readback_delta = super::super::cohort::READBACK_BYTES - 192 * MIB;
        let geometry = super::super::cohort::DecodeCohortWorkspace::host_bytes()?;
        let capture = TextKernelCapture::bind(
            "39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d",
        )?;
        let library = CohortLibrary::host_bytes(capture)?;
        let body = FullModelCohort::host_bytes(true, capture)?
            .max(FullModelCohort::host_bytes(false, capture)?);
        let current_rows = mul(4, current_call_host(capture)?)?;
        let delta = sum([
            readback_delta,
            geometry,
            library,
            body,
            current_rows,
            64 * 1024,
            super::super::cohort::ColdCohort::initialization_host_bytes()?,
        ])?;
        // Accepted316 .8B bound is a regression anchor, not model/runtime evidence.
        let anchored = 3_608_137_747usize;
        let upper = sum([anchored, delta, 16 * MIB])?; // actual retained qualifier index
        println!(
            "B4_SOURCE_HOST_BOUND readback_delta={readback_delta} geometry={geometry} library={library} body={body} rows={current_rows} delta={delta} anchored_upper={upper}"
        );
        // Historical accepted .8B cohort regression bound (anchor + delta +
        // retained qualifier index), NOT current full-host fit and NOT the
        // current4GiB refusal; any refusal stays as diagnosis.
        require_fit(
            0,
            upper,
            zerogravity_core::optimization::MemoryFootprint {
                host_bytes: (4096 * MIB) as u64,
                device_bytes: (4096 * MIB) as u64,
            },
        )?;
        Ok(())
    }
    #[test]
    fn n1_fourteen_roots_are_distinct_from_b4_bounds() -> Result<()> {
        assert_eq!(ROOTS, 14);
        for fingerprint in [
            "39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d",
            "4d77cb6637843a57a17fc093ad1e5241ddf083bf8d9ade0c9bbaa118bc9448e2",
        ] {
            let capture = TextKernelCapture::bind(fingerprint)?;
            let phases = CohortPhases::for_capture(capture);
            let b4 = super::super::cohort::DecodeCohortRootBound::derive(
                8,
                if phases.contains(true) { 8 } else { 4 },
            )?
            .total;
            assert_eq!(
                b4,
                if capture.supports_initial_batch() {
                    22
                } else {
                    18
                }
            );
            assert_ne!(b4, ROOTS);
        }
        Ok(())
    }
    #[test]
    fn repeated_distinct_roots_and_tiny_current_packets_are_each_aligned() {
        let mut d = DeviceLedger::default();
        for _ in 0..14 {
            d.extend([4, 12]).unwrap();
        }
        assert_eq!((d.count, d.bytes), (28, 14 * 1024));
        let narrow = TextKernelCapture::bind(
            "39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d",
        )
        .unwrap();
        let wide = TextKernelCapture::bind(
            "4d77cb6637843a57a17fc093ad1e5241ddf083bf8d9ade0c9bbaa118bc9448e2",
        )
        .unwrap();
        assert!(current_call_host(wide).unwrap() > current_call_host(narrow).unwrap());
    }
}
