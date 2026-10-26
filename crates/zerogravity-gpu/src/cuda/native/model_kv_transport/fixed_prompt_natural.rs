//! SAME source KV transport with root-free storage and fresh current handles.
use super::*;
use crate::cuda::native::model_cache::SharedModelCache;
use crate::resident_inference::native_fragment::fixed_prompt_natural::FixedPromptCallInput;

pub(crate) struct FixedPromptKvStorage {
    arena: Arc<SharedModelCache>,
    status: ResidentLease,
}
pub(crate) struct FixedPromptKvTransport {
    storage: Arc<FixedPromptKvStorage>,
    input: Arc<FixedPromptCallInput>,
    binding: Arc<BoundModelState>,
    incoming: Option<Copy>,
    outgoing: Copy,
    _metadata: WindowWorkspace,
}
impl FixedPromptKvStorage {
    pub(crate) fn prepare(
        arena: Arc<SharedModelCache>,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        init.require_service(arena.lease().service())?;
        let status = init.upload_bytes(&[0; 4], BufferRole::Scratch)?;
        let result = Arc::new(Self { arena, status });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn bind(
        self: &Arc<Self>,
        input: Arc<FixedPromptCallInput>,
        binding: Arc<BoundModelState>,
    ) -> Result<Arc<FixedPromptKvTransport>> {
        let (base, end) = (input.base(), input.base() + input.rows());
        if !matches!((base, input.rows()), (0, 47) | (47..=53, 1))
            || end > 54
            || binding.kv_domain().is_err()
        {
            return Err(Error::Unsupported(
                "canonical initial47/seven cached KV spans only".into(),
            ));
        }
        binding.require_fixed_prompt_published(input.scope())?;
        self.arena.require_binding(input.scope(), &binding)?;
        native_index(4, end - 1, 511, true)?;
        let metadata = input.source().reserve_workspace(64 * 1024)?;
        let service = self.arena.lease().service();
        let leases = binding.leases();
        let mut ids = std::collections::BTreeSet::new();
        for (lease, _) in &leases {
            if !Arc::ptr_eq(service, lease.service())
                || lease.bytes() != CANONICAL_BYTES
                || lease.source_identity() != input.call().source()
                || lease.service_generation() != service.generation()
                || [self.arena.lease().id(), self.status.id()].contains(&lease.id())
                || !ids.insert(lease.id())
            {
                return Err(Error::State(
                    "canonical KV exact fresh canonical handles differ".into(),
                ));
            }
        }
        let make = |incoming: bool| -> Result<Copy> {
            let entry = if incoming {
                "zg_vllm_full_kv_in"
            } else {
                "zg_vllm_full_kv_out"
            };
            let source:Arc<str>=format!("// cubecl.cuda.dynamic_shared_bytes=0\n#define ZG_ENTRY {entry}\n#define ZG_IN {}\n#define ZG_BASE {base}u\n#define ZG_END {end}u\n#define ZG_BLOCK 4u\n{SOURCE}",usize::from(incoming)).into();
            use BufferIOAttr::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};
            let mut bindings = vec![
                (CANONICAL_BYTES, R),
                (CANONICAL_BYTES, R),
                (NATIVE_BYTES, if incoming { RW } else { R }),
            ];
            let mut handles = vec![
                leases[0].0.handle(),
                leases[2].0.handle(),
                self.arena.lease().handle(),
            ];
            if !incoming {
                bindings.extend([(CANONICAL_BYTES, W), (CANONICAL_BYTES, W)]);
                handles.extend([leases[1].0.handle(), leases[3].0.handle()]);
            }
            bindings.push((4, RW));
            handles.push(self.status.handle());
            let kernel = PinnedKernel::new(entry, source, 128, Arc::from(bindings))?;
            let count = if incoming { base * 512 } else { 512 * 512 };
            let grid = [count.div_ceil(128) as u32, 1, 1];
            kernel.validate(service.client(), grid, &handles)?;
            Ok(Copy {
                kernel,
                grid,
                handles,
            })
        };
        Ok(Arc::new(FixedPromptKvTransport {
            storage: self.clone(),
            input,
            binding,
            incoming: if base == 0 { None } else { Some(make(true)?) },
            outgoing: make(false)?,
            _metadata: metadata,
        }))
    }
    pub(crate) fn status(&self) -> &ResidentLease {
        &self.status
    }
}
impl FixedPromptKvTransport {
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.binding
            .leases()
            .into_iter()
            .chain([
                (self.storage.arena.lease().clone(), DeviceAccess::Write),
                (self.storage.status.clone(), DeviceAccess::Write),
            ])
            .collect()
    }
    pub(crate) fn before(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        effect: StateEffect,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.input
            .require_permit(permit, self.storage.status.service())?;
        self.binding
            .require_enqueue(self.storage.status.service(), permit)?;
        self.binding.require_effect(effect, false)?;
        if let Some(copy) = &self.incoming {
            copy.kernel.launch(
                self.storage.status.service().client(),
                copy.grid,
                &copy.handles,
            )?;
            Ok(1)
        } else {
            Ok(0)
        }
    }
    pub(crate) fn after(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        effect: StateEffect,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.input
            .require_permit(permit, self.storage.status.service())?;
        self.binding
            .require_enqueue(self.storage.status.service(), permit)?;
        self.binding.require_effect(effect, false)?;
        self.outgoing.kernel.launch(
            self.storage.status.service().client(),
            self.outgoing.grid,
            &self.outgoing.handles,
        )?;
        Ok(1)
    }
    pub(crate) fn status_readback(&self) -> Result<ResidentReadback> {
        self.storage.status.readback(0, 4)
    }
    pub(crate) fn retained(
        &self,
        input: &Arc<FixedPromptCallInput>,
        binding: &Arc<BoundModelState>,
    ) -> Result<bool> {
        binding.require_fixed_prompt_published(input.scope())?;
        self.storage
            .arena
            .retained_binding(input.scope(), binding)?;
        Ok(Arc::ptr_eq(input, &self.input) && Arc::ptr_eq(binding, &self.binding))
    }
}
