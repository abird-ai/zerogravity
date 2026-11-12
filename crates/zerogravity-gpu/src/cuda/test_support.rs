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
        let device =
            cubecl::Device::cuda(0).map_err(|e| Error::State(format!("{label} CUDA: {e}")))?;
        Self::new_on(label, device)
    }
    /// Same finite diagnostic ownership for an explicitly selected CUDA
    /// device. Keep qualification code on one compiler/charge path rather than
    /// constructing ad-hoc clients that silently fall back to PTX.
    pub(crate) fn new_on(label: &'static str, device: cubecl::Device) -> Result<Self> {
        const MIB: usize = 1024 * 1024;
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
