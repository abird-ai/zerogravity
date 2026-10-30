//! Model-free diagnostics: original service plus finite compiler/payload charges.
use crate::resident_leases::ResidentService;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_core::{
    Error, Result,
    accelerator::SourceIdentity,
    resource_arbiter::{
        AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey,
        ResourceArbiter, ResourceOwner,
    },
};

pub(crate) struct CudaDiagnostic {
    pub(crate) service: Arc<ResidentService>,
    pub(crate) compiler: Arc<(GlobalResourceLease, GlobalResourceLease)>,
    _payload: GlobalResourceLease,
}
impl CudaDiagnostic {
    pub(crate) fn new(label: &'static str) -> Result<Self> {
        const MIB: usize = 1024 * 1024;
        let device =
            cubecl::Device::cuda(0).map_err(|e| Error::State(format!("{label} CUDA: {e}")))?;
        let service = ResidentService::new_pinned_cuda(
            device,
            SourceIdentity::for_new_allocation()?,
            ResourceOwner::new(label, "isolated-diagnostic")?,
            256 * MIB,
            (768 * MIB) as u64,
        )?;
        let arbiter = ResourceArbiter::process_shared();
        let host = PhysicalDeviceKey::new("host", &format!("zerogravity-{label}-diagnostic"))?;
        arbiter.register_matching(host.clone(), (512 * MIB) as u64)?;
        let reserve = |key: &PhysicalDeviceKey, bytes| {
            arbiter.admit(
                key,
                ResourceOwner::new(label, "compiler-context")?,
                &[AllocationRequest {
                    kind: AllocationClass::ProviderEnvelope,
                    bytes,
                }],
            )
        };
        let compiler = Arc::new((
            reserve(&host, cubecl::cuda::CudaCompilerContextPolicy::HOST_BYTES)?,
            reserve(
                &service.physical_key(),
                cubecl::cuda::CudaCompilerContextPolicy::DEVICE_BYTES,
            )?,
        ));
        let adopted = Arc::new(AtomicBool::new(false));
        let installed = unsafe {
            service
                .pinned_cuda_queue()?
                .install_compiler_context(compiler.clone(), adopted.clone())
        };
        // Adopted code/context outlives model retirement, until process exit.
        if adopted.load(Ordering::Acquire) || installed.is_err() {
            std::mem::forget(compiler.clone());
        }
        installed.map_err(|e| Error::State(format!("{label} compiler owner: {e}")))?;
        let payload = reserve(&host, (128 * MIB) as u64)?;
        Ok(Self {
            service,
            compiler,
            _payload: payload,
        })
    }
}
