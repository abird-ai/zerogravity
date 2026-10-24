//! Complete logical owners and aligned device bytes; no allocation permit.
use super::*;
use zerogravity_core::accelerator::program::{
    ActivationPrecision, ProgramNumericalProfile, ValueStorageBinding, ValueStorageLimits,
    ValueStorageRetention, ValueStorageReusePolicy,
};

const RECORDED_CALL_BYTES: usize = 4 * 1024 * 1024;
pub(super) const LOWERER_METADATA_BYTES: usize = 1024 * 1024;
pub(super) const ALIGNMENT: usize = 256;

#[derive(Clone, Debug)]
pub struct GemmaTextHostOwners {
    /// Other simultaneously retained original4MiB call records.
    pub additional_recorded_calls: usize,
    /// Other concurrent D2H/expected/decoder frame allocations.
    pub additional_frame_bytes: usize,
}
#[derive(Clone, Debug)]
pub struct GemmaTextResourceInventory {
    pub unique_source_matrices: usize,
    pub complete_source_bytes: usize,
    pub source_page_bytes: usize,
    /// Reuses the existing page/control lease; no additional allocation.
    pub selected_row_cache_capacity: usize,
    pub selected_row_cache_row_bytes: usize,
    pub source_transport_host_bytes: usize,
    pub value_slot_bytes: usize,
    pub value_slots: usize,
    pub source_gamma_bytes: usize,
    pub math_scratch_bytes: usize,
    pub projection_scratch_bytes: usize,
    pub status_and_packet_bytes: usize,
    pub original_greedy_workspace_bytes: usize,
    pub single_root_payload_bytes: usize,
    pub six_root_device_bytes: usize,
    pub owned_device_bytes: usize,
    pub device_with_six_roots_bytes: usize,
    pub host_upper_bytes: usize,
    pub native_host_bytes: usize,
    pub native_device_bytes: usize,
    pub largest_binding_bytes: usize,
    pub resident_matrices: usize,
    pub resident_matrix_bytes: usize,
    pub resident_upload_windows: usize,
    pub resident_control_host_bytes: usize,
    pub root_pool_metadata_bytes: usize,
    pub root_initialization_host_bytes: usize,
    pub projection_max_rows: usize,
    pub persistent_context_device_bytes: usize,
    pub persistent_context_host_bytes: usize,
    pub complete_physical_device_bytes: usize,
}
pub(super) struct Layout {
    pub inventory: GemmaTextResourceInventory,
    pub bindings: Vec<ValueStorageBinding>,
    pub slots: Vec<usize>,
    pub gammas: Vec<usize>,
    pub math: Vec<usize>,
    pub states: Vec<StateRequirement>,
}
pub(super) fn aligned(n: usize) -> Result<usize> {
    n.checked_add(ALIGNMENT - 1)
        .map(|n| n & !(ALIGNMENT - 1))
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Limit("Gemma allocation alignment overflow".into()))
}
pub(super) fn sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |a, b| {
        a.checked_add(b)
            .ok_or_else(|| Error::Limit("Gemma lowerer owner sum overflow".into()))
    })
}
impl Layout {
    pub(super) fn derive(
        recorded: &Gemma4Bf16TextPlan,
        math_bytes: &[usize],
        host: &GemmaTextHostOwners,
        native: &GemmaTextNativeOwners,
        storage: GemmaTextStorage,
    ) -> Result<Self> {
        Self::derive_cost(
            recorded,
            math_bytes,
            host,
            storage,
            native.host_bytes(),
            native.device_bytes(),
            native.context_host_bytes(),
            native.context_device_bytes(),
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn derive_cost(
        recorded: &Gemma4Bf16TextPlan,
        math_bytes: &[usize],
        host: &GemmaTextHostOwners,
        storage: GemmaTextStorage,
        native_host: usize,
        native_device: usize,
        context_host: usize,
        context_device: usize,
    ) -> Result<Self> {
        let selection = storage::Selection::derive(recorded, storage)?;
        let p = recorded.plan();
        if p.numerical_profile() != ProgramNumericalProfile::Gemma4E2bBf16TextCandidateV1
            || p.is_operator_slice()
            || p.prefix_capacity() != 64
            || p.frontier_delta() != 1
            || p.nodes().len() > 8192
            || p.values().len() > 4096
            || p.required_domains()
                .iter()
                .copied()
                .ne((0..15)
                    .map(|n| StateSlot::Kv(zerogravity_core::accelerator::program::KvDomain(n))))
            || p.precisions()
                .iter()
                .any(|v| *v != ActivationPrecision::Bf16WidenedF32)
        {
            return Err(Error::Unsupported(
                "distinct complete E2B BF16 source contract".into(),
            ));
        }
        let physical = p
            .values()
            .iter()
            .map(|v| ResidentBufferLayout::new(ResidentPhysicalType::F32, v.capacity_elements()?))
            .collect::<Result<Vec<_>>>()?;
        let storage = ResidentBufferLayout::plan_values(
            p,
            &physical,
            ValueStorageRetention::Reuse {
                escaped: &[],
                pinned: &[],
                taps: &[],
            },
            ValueStorageLimits {
                max_nodes: 8192,
                max_values: 4096,
                max_slots: 128,
                max_slot_bytes: 1024 * 1024,
                max_total_bytes: 16 * 1024 * 1024,
                alignment: ALIGNMENT,
                reuse_policy: ValueStorageReusePolicy::ExactAlignedSize,
            },
        )?;
        storage.require_plan(p)?;
        let slots = storage
            .slots()
            .iter()
            .map(|s| s.reserved_bytes)
            .collect::<Vec<_>>();
        let gammas = p
            .constants()
            .iter()
            .map(|v| aligned(v.values.len() * 4))
            .collect::<Result<Vec<_>>>()?;
        if math_bytes.len() > 64 || math_bytes.iter().any(|n| *n == 0 || *n > 1024 * 1024) {
            return Err(Error::Limit(
                "Gemma actual bounded math scratch layout".into(),
            ));
        }
        let math = math_bytes
            .iter()
            .map(|n| aligned(*n))
            .collect::<Result<Vec<_>>>()?;
        let mut states = BTreeMap::<StateSlot, StateRequirement>::new();
        for node in p.nodes() {
            if matches!(node.operation, PlanOperation::AppendKv) {
                let effect = node
                    .state
                    .ok_or_else(|| Error::State("Gemma original append state seal".into()))?;
                let [key, value] = node.inputs.as_slice() else {
                    return Err(Error::State("Gemma original append KV arity".into()));
                };
                let slab = p.values()[key.0 as usize].capacity_elements()?;
                if !matches!(slab, 256 | 512)
                    || p.values()[value.0 as usize].capacity_elements()? != slab
                    || states
                        .insert(
                            effect.write.slot,
                            StateRequirement {
                                slot: effect.write.slot,
                                layout: StateLayout::Kv { slab, capacity: 64 },
                                bytes: 2 * 64 * slab * 4,
                            },
                        )
                        .is_some()
                {
                    return Err(Error::State(
                        "Gemma fifteen unique original KV owners".into(),
                    ));
                }
            }
        }
        if states.len() != 15 {
            return Err(Error::State(
                "Gemma complete fifteen KV domain inventory".into(),
            ));
        }
        let root = sum(states.values().map(|s| s.bytes))?;
        if root != recorded.inventory().kv_root_bytes {
            return Err(Error::State(
                "Gemma source-derived root payload differs".into(),
            ));
        }
        let projection_scratch_bytes = sum([
            aligned(12288 * 2)?,
            aligned(selection.projection_rows * 4)?,
            aligned(selection.projection_rows * 4)?,
        ])?;
        let greedy = crate::resident_greedy::workspace_bytes(262144)?;
        let status_and_packet_bytes = sum([2 * ALIGNMENT, 2048])?;
        let greedy_workspace = sum(greedy)?;
        let tables = sum(math::immutable_tables()
            .iter()
            .map(|v| aligned(v.len()))
            .collect::<Result<Vec<_>>>()?)?;
        let owned = sum([
            source_page::PAGE_BYTES,
            selection.aligned_bytes,
            storage.total_bytes(),
            sum(gammas.iter().copied())?,
            sum(math.iter().copied())?,
            projection_scratch_bytes,
            status_and_packet_bytes,
            tables,
        ])?;
        let six_roots = root
            .checked_mul(6)
            .ok_or_else(|| Error::Limit("Gemma six-root inventory".into()))?;
        // known_host prices ONE call only. Program's actual owned plan clone is
        // another retained record, plus every declared concurrent call/frame.
        let copies = host
            .additional_recorded_calls
            .checked_add(1)
            .and_then(|n| n.checked_mul(RECORDED_CALL_BYTES))
            .ok_or_else(|| Error::Limit("Gemma concurrent call records".into()))?;
        let resident_control = if selection.aligned_bytes == 0 {
            0
        } else {
            selection
                .matrices
                .len()
                .checked_mul(64 * 1024)
                .ok_or_else(|| Error::Limit("Gemma retained authentic uploader controls".into()))?
        };
        // Window owns payload+integrity chunk; Bytes+backend bounce are separate.
        // The existing family baseline already prices3 pages; this additional
        // chunk keeps source-shaped authentication scratch explicit, conservatively.
        let integrity_chunk = recorded.source().source().staging_chunk_bytes();
        let root_planes = states
            .values()
            .map(|r| match r.layout {
                StateLayout::Vector { .. } => 1usize,
                StateLayout::Kv { .. } => 2usize,
            })
            .sum();
        let root_pool_metadata =
            crate::resident_leases::backing_pool::ResidentPool::metadata_bytes(root_planes, 6)?;
        let root_initialization_host = root
            .checked_mul(3)
            .ok_or_else(|| Error::Limit("Gemma root initialization host peak".into()))?;
        let additional = sum([
            copies,
            LOWERER_METADATA_BYTES,
            64 * 1024 + 4096,
            native_host,
            context_host,
            resident_control,
            integrity_chunk,
            root_pool_metadata,
            root_initialization_host,
            host.additional_frame_bytes,
            65536, // one actual call seal; immutable plan is shared
        ])?;
        let host_upper = recorded.inventory().host_with_lowerers(additional)?;
        if host_upper as u64 > recorded.source().source().workspace_limit_bytes()? {
            return Err(Error::Limit(
                "Gemma actual lowerer host cost exceeds original source domain".into(),
            ));
        }
        let largest = slots
            .iter()
            .chain(&gammas)
            .chain(&math)
            .copied()
            .chain([
                source_page::PAGE_BYTES,
                selection.largest_binding,
                selection.projection_rows * 4,
            ])
            .max()
            .unwrap();
        Ok(Self {
            inventory: GemmaTextResourceInventory {
                unique_source_matrices: recorded.inventory().unique_source_matrices,
                complete_source_bytes: recorded.inventory().full_source_matrix_bytes,
                source_page_bytes: source_page::PAGE_BYTES,
                selected_row_cache_capacity: selection.cached_row_capacity,
                selected_row_cache_row_bytes: selection.cached_row_bytes,
                source_transport_host_bytes: sum([3 * source_page::PAGE_BYTES, integrity_chunk])?,
                value_slot_bytes: storage.total_bytes(),
                value_slots: slots.len(),
                source_gamma_bytes: sum(gammas.iter().copied())?,
                math_scratch_bytes: sum(math.iter().copied())?,
                projection_scratch_bytes,
                status_and_packet_bytes,
                original_greedy_workspace_bytes: greedy_workspace,
                single_root_payload_bytes: root,
                six_root_device_bytes: six_roots,
                owned_device_bytes: owned,
                device_with_six_roots_bytes: sum([
                    owned,
                    six_roots,
                    greedy_workspace,
                    6 * ALIGNMENT,
                    native_device,
                ])?,
                host_upper_bytes: host_upper,
                native_host_bytes: native_host,
                native_device_bytes: native_device,
                largest_binding_bytes: largest,
                resident_matrices: if selection.aligned_bytes == 0 {
                    0
                } else {
                    selection.matrices.len()
                },
                resident_matrix_bytes: selection.aligned_bytes,
                resident_upload_windows: selection.upload_windows,
                resident_control_host_bytes: resident_control,
                root_pool_metadata_bytes: root_pool_metadata,
                root_initialization_host_bytes: root_initialization_host,
                projection_max_rows: selection.projection_rows,
                persistent_context_device_bytes: context_device,
                persistent_context_host_bytes: context_host,
                complete_physical_device_bytes: sum([
                    owned,
                    six_roots,
                    greedy_workspace,
                    6 * ALIGNMENT,
                    native_device,
                    context_device,
                ])?,
            },
            bindings: storage.bindings().to_vec(),
            slots,
            gammas,
            math,
            states: states.into_values().collect(),
        })
    }
}
