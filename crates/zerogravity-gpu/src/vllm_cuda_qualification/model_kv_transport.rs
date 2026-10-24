//! Canonical original RootAttempt K/V ↔ exact native per-head paired lanes.
use super::model_cache::ModelCache;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use crate::resident_inference::native_fragment::model_state::BoundModelState;
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentReadback};
use crate::vllm_ptx_inline::PinnedKernel;
use cubecl::server::Handle;
use cubecl_runtime::kernel::BufferIOAttr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{StateEffect, VllmBf16Phase};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::OriginalWindowModelSource;

const SOURCE: &str = include_str!("model_kv_copy.cu");
const CANONICAL_BYTES: usize = 512 * 512 * 4;
const NATIVE_BYTES: usize = 5 * 2 * 544 * 512 * 2;
struct Copy {
    kernel: PinnedKernel,
    grid: [u32; 3],
    handles: Vec<Handle>,
}
pub(crate) struct ModelKvTransport {
    incoming: Option<Copy>,
    outgoing: Copy,
    binding: Arc<BoundModelState>,
    cache: ResidentLease,
    _arena: Arc<ModelCache>,
    status: ResidentLease,
    _source: Arc<OriginalWindowModelSource>,
    _metadata: WindowWorkspace,
}
fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State(
            "full24 original KV transport cancelled".into(),
        ));
    }
    Ok(())
}
fn native_index(block: usize, position: usize, column: usize, value: bool) -> Result<usize> {
    if !matches!(block, 1 | 4) || position >= 512 || column >= 512 {
        return Err(Error::Limit("closed original native KV coordinates".into()));
    }
    Ok(block * 557_056
        + (column / 256) * 512
        + position * 1024
        + usize::from(value) * 256
        + column % 256)
}
impl ModelKvTransport {
    pub(crate) fn prepare(
        source: Arc<OriginalWindowModelSource>,
        binding: Arc<BoundModelState>,
        arena: Arc<ModelCache>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        source.require_vllm08_full_source()?;
        arena.require_binding(&binding)?;
        let call = binding.call();
        let base = call.request().teachers().base();
        let end = base
            .checked_add(call.rows())
            .filter(|end| *end <= 67)
            .ok_or_else(|| Error::Unsupported("closed source rotary/KV prefix0..67".into()))?;
        let block = match (call.request().legacy_phase()?, base) {
            (VllmBf16Phase::InitialOne, 0) | (VllmBf16Phase::CachedOne, 1) => 4,
            (VllmBf16Phase::Initial64, 0)
            | (VllmBf16Phase::ContinuedTwo, 64)
            | (VllmBf16Phase::CachedOne, 66) => 1,
            _ => {
                return Err(Error::Unsupported(
                    "exact original five-phase cache page ABI".into(),
                ));
            }
        };
        Self::prepare_checked(source, binding, arena, init, cancelled, (base, end, block))
    }
    pub(crate) fn prepare_canonical47(
        cut: &Arc<super::canonical47_model_prefix::Canonical47PrefixCut>,
        binding: Arc<BoundModelState>,
        arena: Arc<ModelCache>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        if !cut.is_attention_cut() {
            return Err(Error::Unsupported(
                "canonical47 exact35-node KV transport".into(),
            ));
        }
        binding.require_canonical47_attention_kv(cut.scope())?;
        arena.require_binding(&binding)?;
        Self::prepare_checked(
            cut.scope().source().source().clone(),
            binding,
            arena,
            init,
            cancelled,
            (0, 47, 4),
        )
    }
    fn prepare_checked(
        source: Arc<OriginalWindowModelSource>,
        binding: Arc<BoundModelState>,
        arena: Arc<ModelCache>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
        (base, end, block): (usize, usize, usize),
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let cache = arena.lease().clone();
        let call = binding.call();
        let service = cache.service();
        init.require_service(service)?;
        if !binding.layer().is_attention()
            || call.source() != source.source_identity()
            || cache.source_identity() != source.source_identity()
            || cache.bytes() != NATIVE_BYTES
            || !cache.is_initialized()
            || cache.role() != BufferRole::RecurrentState
        {
            return Err(Error::State(
                "full24 original initialized paired KV arena/source".into(),
            ));
        }
        native_index(block, end - 1, 511, true)?;
        let metadata = source.source().reserve_workspace(64 * 1024)?;
        init.retain_completion_owner(source.clone())?;
        init.retain_completion_owner(binding.clone())?;
        let status = init.upload_bytes(&[0; 4], BufferRole::Scratch)?;
        let leases = binding.leases();
        if leases.iter().any(|(lease, _)| {
            lease.bytes() != CANONICAL_BYTES
                || !Arc::ptr_eq(lease.service(), service)
                || lease.id() == cache.id()
        }) {
            return Err(Error::State(
                "full24 KV original canonical input/output extents".into(),
            ));
        }
        let make = |incoming: bool| -> Result<Copy> {
            let entry = if incoming {
                "zg_vllm_full_kv_in"
            } else {
                "zg_vllm_full_kv_out"
            };
            let source: Arc<str> = format!("// cubecl.cuda.dynamic_shared_bytes=0\n#define ZG_ENTRY {entry}\n#define ZG_IN {}\n#define ZG_BASE {base}u\n#define ZG_END {end}u\n#define ZG_BLOCK {block}u\n{SOURCE}", usize::from(incoming)).into();
            use BufferIOAttr::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};
            let mut bindings = vec![
                (CANONICAL_BYTES, R),
                (CANONICAL_BYTES, R),
                (NATIVE_BYTES, if incoming { RW } else { R }),
            ];
            let mut handles = vec![leases[0].0.handle(), leases[2].0.handle(), cache.handle()];
            if !incoming {
                bindings.extend([(CANONICAL_BYTES, W), (CANONICAL_BYTES, W)]);
                handles.extend([leases[1].0.handle(), leases[3].0.handle()]);
            }
            bindings.push((4, RW));
            handles.push(status.handle());
            let bindings = Arc::from(bindings);
            let kernel = PinnedKernel::new(entry, source, 128, bindings)?;
            let count = if incoming { base * 512 } else { 512 * 512 };
            let grid = [count.div_ceil(128) as u32, 1, 1];
            kernel.validate(service.client(), grid, &handles)?;
            Ok(Copy {
                kernel,
                grid,
                handles,
            })
        };
        let incoming = if base == 0 { None } else { Some(make(true)?) };
        let outgoing = make(false)?;
        let prepared = Arc::new(Self {
            incoming,
            outgoing,
            binding,
            cache,
            status,
            _arena: arena,
            _source: source,
            _metadata: metadata,
        });
        init.retain_completion_owner(prepared.clone())?;
        Ok(prepared)
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.binding
            .leases()
            .into_iter()
            .chain([
                (self.cache.clone(), DeviceAccess::Write),
                (self.status.clone(), DeviceAccess::Write),
            ])
            .collect()
    }
    pub(crate) fn before_append(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        effect: StateEffect,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.binding.require_enqueue(self.cache.service(), permit)?;
        self.binding.require_effect(effect, false)?;
        if let Some(copy) = &self.incoming {
            copy.kernel
                .launch(self.cache.service().client(), copy.grid, &copy.handles)?;
            Ok(1)
        } else {
            Ok(0)
        }
    }
    pub(crate) fn after_append(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        effect: StateEffect,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.binding.require_enqueue(self.cache.service(), permit)?;
        self.binding.require_effect(effect, false)?;
        self.outgoing.kernel.launch(
            self.cache.service().client(),
            self.outgoing.grid,
            &self.outgoing.handles,
        )?;
        Ok(1)
    }
    pub(crate) fn status_readback(&self) -> Result<ResidentReadback> {
        self.status.readback(0, 4)
    }
    pub(crate) fn validate_status(bytes: &[u8]) -> Result<()> {
        if bytes != [0; 4] {
            return Err(Error::State(
                "full24 exact finite BF16 canonical KV transport".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_head_paired_planes_and_canonical_capacity_do_not_alias() {
        for block in [1, 4] {
            // Actual all42 step metadata, not a contiguous-shape inference.
            assert_eq!(
                native_index(block, 1, 0, false).unwrap()
                    - native_index(block, 0, 0, false).unwrap(),
                1024
            );
            assert_eq!(
                native_index(block, 0, 256, false).unwrap()
                    - native_index(block, 0, 0, false).unwrap(),
                512
            );
            let mut used = std::collections::BTreeSet::new();
            for position in 0..512 {
                for column in 0..512 {
                    for value in [false, true] {
                        let index = native_index(block, position, column, value).unwrap();
                        assert!(index < NATIVE_BYTES / 2);
                        assert!(used.insert(index));
                        assert_eq!(index / (2 * 544 * 512), block);
                    }
                }
            }
        }
        for coordinates in [(0, 0, 0), (5, 0, 0), (1, 512, 0), (1, 0, 512)] {
            assert!(native_index(coordinates.0, coordinates.1, coordinates.2, false).is_err());
        }
        assert!(ModelKvTransport::validate_status(&[0; 4]).is_ok());
        assert!(ModelKvTransport::validate_status(&1u32.to_le_bytes()).is_err());
    }
}

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
mod natural;

mod whole;

pub(crate) mod canonical_natural;
