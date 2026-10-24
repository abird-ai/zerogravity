//! Shared module lifetime; all native work stays on the original actor.
use super::{
    CubinKernelManifest, CubinWorkspaceClear, VerifiedCubin, VocabularyHeadSelection, failure,
};
use crate::OwnedCudaQueue;
use cubecl_common::device::ServiceId;
use cubecl_core::server::{Handle, ServerError};
use std::any::Any;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
fn identity() -> Result<u64, ServerError> {
    NEXT_ID
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
        .map_err(|_| failure("cubin native identity exhausted"))
}

fn canonical_handle(
    handle: &Handle,
    service: ServiceId,
    bytes: u64,
    alignment: u64,
) -> Result<(), ServerError> {
    let allocated = (bytes != 0 && alignment.is_power_of_two())
        .then(|| bytes.checked_add(alignment - 1))
        .flatten()
        .map(|rounded| rounded / alignment * alignment)
        .ok_or_else(|| failure("cubin original allocation alignment/extent overflow"))?;
    let start = handle.offset_start.unwrap_or(0);
    let end = handle.offset_end.unwrap_or(0);
    // Checked equivalent of Handle::size_in_used, including hostile views.
    let used = handle
        .size()
        .checked_sub(start)
        .and_then(|n| n.checked_sub(end));
    if handle.service != service
        || start != 0
        || handle.size() != allocated
        || end != allocated - bytes
        || used != Some(bytes)
    {
        return Err(failure(
            "cubin canonical original handle service/extent/padding",
        ));
    }
    Ok(())
}

mod workspace;
pub use workspace::{CubinWorkspaceView, OwnedCudaBinding};
#[cfg(test)]
mod tests;

/// Same original prepaid opaque charge; no physical pointers or Ready authority.
/// Actor retains admission and all partial modules even if this token is dropped.
pub struct OwnedCudaModuleSet {
    queue: Arc<OwnedCudaQueue>,
    owner: Arc<dyn Any + Send + Sync>,
    key: u64,
    retired: AtomicBool,
}
impl std::fmt::Debug for OwnedCudaModuleSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedCudaModuleSet")
            .field("key", &self.key)
            .field("retired", &self.retired.load(Ordering::Acquire))
            .finish()
    }
}

/// Fixed manifest record; its module/function is shared by this original set.
pub struct OwnedCudaKernel {
    set: Arc<OwnedCudaModuleSet>,
    image: Arc<VerifiedCubin>,
    manifest: Arc<CubinKernelManifest>,
    allocation_alignment: u64,
    key: u64,
    retired: AtomicBool,
}
impl std::fmt::Debug for OwnedCudaKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedCudaKernel")
            .field("entry", &self.manifest.entry)
            .field("image", &self.image)
            .field("set", &self.set)
            .field("retired", &self.retired.load(Ordering::Acquire))
            .finish()
    }
}

impl OwnedCudaQueue {
    /// # Safety
    /// Owner already retains the SAME original service's prepaid native-module
    /// charge and host-image admission, before any image/native allocation.
    /// Original initialization/flight already retains this owner. The charge
    /// is a nested reservation, never a second physical arbiter or a claim that
    /// ELF bytes measure driver memory. One active set per original actor.
    pub unsafe fn admit_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| server.admit_owned_cubin_set(key, retained))
            .map_err(|e| failure(format!("original cubin set admission actor: {e:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Closed .8B layer0 GDN+MLP+activation scope; fixed7/12/21 capacity.
    ///
    /// # Safety
    /// Original initialization already retains exactly seven sealed images and
    /// source windows before native work. The SAME composite owner retains
    /// their original host admissions and prepaid native-module charge.
    /// Only the fixed connected operators may prepare/enqueue on this set;
    /// this is not a model/profile eligibility permit or caller capacity knob.
    pub unsafe fn admit_connected_qwen08_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_connected_qwen08_cubin_set(key, retained)
            })
            .map_err(|e| failure(format!("original cubin set admission actor: {e:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Closed original47 source-produced layer0 MLP; fixed5/7/8 capacity.
    /// # Safety
    /// The armed original initialization retains the exact five sealed images,
    /// source owners and SAME prepaid native charge before any load. Every
    /// manifest must match its complete finite ABI; no profile/Ready authority.
    pub unsafe fn admit_canonical47_mlp_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_canonical47_mlp_cubin_set(key, retained)
            })
            .map_err(|error| failure(format!("original47 MLP set admission actor:{error:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Closed original2B source-produced layer0 MLP; fixed5/7/7 capacity.
    /// # Safety
    /// The armed original initialization retains the exact five sealed images,
    /// source owners and SAME prepaid native charge before any load. Every
    /// manifest must match its complete finite ABI; no profile/Ready authority.
    pub unsafe fn admit_qwen2b_mlp_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| server.admit_owned_qwen2b_mlp_cubin_set(key, retained))
            .map_err(|error| failure(format!("original2B MLP set admission actor:{error:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Closed initial47 first3 plus actual layer3 QKV; fixed6/9/10 capacity.
    /// # Safety
    /// SAME armed initialization retains all six exact image/source/admission
    /// owners and prepaid charge. No intermediate packet completion or publisher.
    pub unsafe fn admit_canonical47_qkv_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_canonical47_qkv_cubin_set(key, retained)
            })
            .map_err(|error| failure(format!("original47 QKV set admission actor:{error:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Closed initial47 first3 through actual layer3 QK/KV; fixed8/11/12 capacity.
    /// # Safety
    /// SAME armed initialization retains all eight exact image/source/admission
    /// owners and prepaid charge. No intermediate packet completion or publisher.
    pub unsafe fn admit_canonical47_qkkv_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_canonical47_qkkv_cubin_set(key, retained)
            })
            .map_err(|error| failure(format!("original47 QKV set admission actor:{error:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Closed initial47 first3 through actual layer3 FA; fixed9/12/13 capacity.
    /// # Safety
    /// SAME armed initialization retains all nine exact image/source/admission
    /// owners and prepaid charge. No intermediate packet completion or publisher.
    pub unsafe fn admit_canonical47_fa_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_canonical47_fa_cubin_set(key, retained)
            })
            .map_err(|error| failure(format!("original47 QKV set admission actor:{error:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Closed initial47 full layer3 union; fixed11/14/15.
    /// # Safety
    /// SAME original armed initializer retains every sealed image/source/charge.
    pub unsafe fn admit_canonical47_layer3_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_canonical47_layer3_cubin_set(key, retained)
            })
            .map_err(|error| failure(format!("original47 QKV set admission actor:{error:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// One complete Initial47 model set; exact12/15/16, no root publication.
    /// # Safety
    /// Same armed initialization retains complete source/images/native admission.
    pub unsafe fn admit_canonical47_full_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_canonical47_full_cubin_set(key, retained)
            })
            .map_err(|error| failure(format!("original47 full set admission actor:{error:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// One finite47+seven cached original-call set.
    /// # Safety
    /// Same source/image charge retained by the original armed initialization.
    pub unsafe fn admit_canonical47_natural_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_canonical47_natural_cubin_set(key, retained)
            })
            .map_err(|error| failure(format!("original47 full set admission actor:{error:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Exact compiled short-HF profile/call catalogue on the original actor.
    /// # Safety
    /// The original armed initialization retains all source/image admissions.
    pub unsafe fn admit_short_hf_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
        two_b: bool,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_short_hf_cubin_set(key, retained, two_b, false)
            })
            .map_err(|e| failure(format!("short-HF set admission actor:{e:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Exact original N1 plus canonical47 B4 ABI union on the SAME actor/set.
    /// # Safety
    /// Actual complete source/image/native admissions precede this call. This
    /// finite ABI policy confers no row, model, RootAttempt or math authority.
    pub unsafe fn admit_short_hf_b4_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
        two_b: bool,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_short_hf_cubin_set(key, retained, two_b, true)
            })
            .map_err(|error| failure(format!("B4 original set admission actor:{error:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Closed captured layer3 leaf only. No model/profile eligibility authority.
    /// # Safety
    /// SAME original source/host/native charge and armed initialization retain
    /// exactly nine sealed images before module load; no other set races it.
    pub unsafe fn admit_first_attention_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_first_attention_cubin_set(key, retained)
            })
            .map_err(|e| failure(format!("original attention set admission actor:{e:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Closed source-owned .8B full24 mathematical union, fixed14/22/48.
    /// No model/profile, context-length or StateRoot publication authority.
    /// # Safety
    /// Original armed initialization retains the fourteen exact sealed images,
    /// immutable source and SAME prepaid native charge before native work.
    /// Caller supplies the finite original source-weight/parameter owners;
    /// all manifests still pass the exact image/provenance/whole-ABI union.
    pub unsafe fn admit_full_qwen08_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| server.admit_owned_full_qwen08_cubin_set(key, retained))
            .map_err(|error| failure(format!("original full24 set admission actor:{error:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Closed source-produced attention gate, exact11/15/26 set.
    /// # Safety
    /// Original armed initialization retains all eleven sealed source images,
    /// source windows and the SAME prepaid native charge. No capacity upgrade.
    pub unsafe fn admit_first_attention_gate_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_first_attention_gate_cubin_set(key, retained)
            })
            .map_err(|e| failure(format!("original attention set admission actor:{e:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Original full1M source cache factory, fixed10/11/12 scope.
    /// # Safety
    /// Same original source/host/native admission and armed initialization own
    /// all ten exact images before native loads. No model/Root authority.
    pub unsafe fn admit_source_rotary_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_source_rotary_cubin_set(key, retained)
            })
            .map_err(|e| failure(format!("original rotary set actor:{e:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// Closed full-vocabulary leaf only; exactly one pinned image/function/ABI.
    /// # Safety
    /// SAME original source, host admission and prepaid native-module charge
    /// are retained in an armed original initialization before native work.
    pub unsafe fn admit_vocabulary_head_cubin_set(
        self: &Arc<Self>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<OwnedCudaModuleSet>, ServerError> {
        let key = identity()?;
        let retained = owner.clone();
        self.actor
            .submit_blocking(move |server| {
                server.admit_owned_vocabulary_head_cubin_set(key, retained)
            })
            .map_err(|e| failure(format!("original head set admission actor: {e:?}")))??;
        Ok(Arc::new(OwnedCudaModuleSet {
            queue: self.clone(),
            owner,
            key,
            retired: AtomicBool::new(false),
        }))
    }
}
impl OwnedCudaModuleSet {
    fn require_active(&self) -> Result<(), ServerError> {
        if self.retired.load(Ordering::Acquire) {
            return Err(failure("cubin module set already retired"));
        }
        Ok(())
    }
    /// # Safety
    /// Original armed initialization/flight retains this set and source owners.
    /// Manifest is the actual selected complete ABI/launch, including declared
    /// zero pointer fields/pinned scalars and every required workspace op.
    /// No provider constructor, second context or Ready is involved.
    pub unsafe fn prepare(
        self: &Arc<Self>,
        image: Arc<VerifiedCubin>,
        manifest: CubinKernelManifest,
    ) -> Result<Arc<OwnedCudaKernel>, ServerError> {
        self.require_active()?;
        image.sources.require_queue(&self.queue)?;
        image.require_admission_owner(&self.owner)?;
        manifest.validate()?;
        if manifest.binary_sm != image.binary_sm() {
            return Err(failure("cubin manifest/image architecture differs"));
        }
        let key = identity()?;
        let set = self.key;
        let stream = self.queue.stream;
        let manifest = Arc::new(manifest);
        let retained_image = image.clone();
        let retained_manifest = manifest.clone();
        let allocation_alignment = self
            .queue
            .actor
            .submit_blocking(move |server| {
                server.prepare_owned_cubin(stream, set, key, retained_image, retained_manifest)
            })
            .map_err(|e| failure(format!("original cubin prepare actor: {e:?}")))??;
        Ok(Arc::new(OwnedCudaKernel {
            set: self.clone(),
            image,
            manifest,
            allocation_alignment,
            key,
            retired: AtomicBool::new(false),
        }))
    }
    /// # Safety
    /// All original flights completed physically, every kernel record retired,
    /// and no prepare/enqueue can race. Retain original guard/owners through
    /// every unload. Any error requires original fencing/quarantine; no retry.
    pub unsafe fn retire_after_physical_completion(&self) -> Result<(), ServerError> {
        self.require_active()?;
        self.retired
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| failure("cubin set retirement raced"))?;
        let set = self.key;
        self.queue
            .actor
            .submit_blocking(move |server| server.retire_owned_cubin_set(set))
            .map_err(|e| failure(format!("original cubin set retirement actor: {e:?}")))?
    }
}
impl OwnedCudaKernel {
    fn require_active(&self) -> Result<(), ServerError> {
        self.set.require_active()?;
        if self.retired.load(Ordering::Acquire) {
            return Err(failure("cubin kernel already retired"));
        }
        self.image.sources.require_queue(&self.set.queue)
    }
    /// # Safety
    /// Original armed flight retains this kernel/set and every original lease,
    /// with exclusive writes before enqueue. Success is not completion.
    pub unsafe fn enqueue(&self, bindings: Vec<Handle>) -> Result<(), ServerError> {
        self.enqueue_prefix(bindings, None)
    }
    /// Clear a fixed workspace prefix immediately before this kernel, on the
    /// same original actor command and stream. Success is not completion.
    ///
    /// # Safety
    /// Original armed flight retains this kernel/set and every original lease,
    /// with exclusive writes. The closed source operation requires this exact
    /// prefix. Every kernel read of that binding must lie in the prefix;
    /// every other prior-content dependency remains unchanged.
    pub unsafe fn enqueue_with_workspace_clear(
        &self,
        bindings: Vec<Handle>,
        clear: CubinWorkspaceClear,
    ) -> Result<(), ServerError> {
        self.enqueue_prefix(bindings, Some(clear))
    }
    fn enqueue_prefix(
        &self,
        bindings: Vec<Handle>,
        clear: Option<CubinWorkspaceClear>,
    ) -> Result<(), ServerError> {
        self.require_active()?;
        if bindings.len() != self.manifest.bindings.len() {
            return Err(failure("cubin exact original handle count"));
        }
        if let Some(clear) = clear {
            clear.validate(&self.manifest)?;
        }
        for (handle, spec) in bindings.iter().zip(&self.manifest.bindings) {
            canonical_handle(
                handle,
                self.set.queue.service_id(),
                spec.bytes,
                self.allocation_alignment,
            )?;
        }
        let key = self.key;
        let set = self.set.key;
        let stream = self.set.queue.stream;
        self.set
            .queue
            .actor
            .submit_blocking(move |server| {
                server.enqueue_owned_cubin(stream, set, key, bindings, clear, None)
            })
            .map_err(|e| failure(format!("original cubin enqueue actor: {e:?}")))?
    }
    /// Select the original last row from a complete produced normalized tensor.
    /// Generic enqueue still refuses every offset/cropped Handle.
    /// # Safety
    /// The original armed flight retains complete W/X/Y leases, the actual
    /// normalized producer and this set. The producer ran earlier in that SAME
    /// flight; every original lease has its declared Read/Write access.
    pub unsafe fn enqueue_vocabulary_head_selected(
        &self,
        weight: Handle,
        input_full: Handle,
        output: Handle,
        selection: VocabularyHeadSelection,
    ) -> Result<(), ServerError> {
        self.require_active()?;
        super::head_policy::require(&self.image, &self.manifest)?;
        let bindings = vec![weight, input_full, output];
        for (index, handle) in bindings.iter().enumerate() {
            canonical_handle(
                handle,
                self.set.queue.service_id(),
                selection.full_bytes(index)?,
                self.allocation_alignment,
            )?;
        }
        let key = self.key;
        let set = self.set.key;
        let stream = self.set.queue.stream;
        self.set
            .queue
            .actor
            .submit_blocking(move |server| {
                server.enqueue_owned_cubin(
                    stream,
                    set,
                    key,
                    bindings,
                    None,
                    Some(super::head_policy::streamed::Selection::Original(selection)),
                )
            })
            .map_err(|e| failure(format!("original selected head enqueue actor: {e:?}")))?
    }
    /// Source-shaped complete resident head, selecting its actual last input row.
    /// # Safety
    /// Original flight owns initialized W, produced X, writable Y and this set
    /// through real completion or quarantine, with their declared accesses.
    pub unsafe fn enqueue_resident_vocabulary_head(
        &self,
        bindings: [super::OwnedCudaBinding; 3],
        shape: crate::Bf16MatrixLayout,
        input_rows: usize,
    ) -> Result<(), ServerError> {
        self.require_active()?;
        super::head_policy::resident::require(&self.image, &self.manifest, shape)?;
        let selection = super::head_policy::resident::Selection { shape, input_rows };
        for (index, binding) in bindings.iter().enumerate() {
            binding.require(
                self.set.queue.service_id(),
                selection.full_bytes(index)?,
                self.allocation_alignment,
            )?;
        }
        let handles = bindings.iter().map(|b| b.handle()).collect();
        let key = self.key;
        let set = self.set.key;
        let stream = self.set.queue.stream;
        self.set
            .queue
            .actor
            .submit_blocking(move |server| {
                let result = server.enqueue_owned_cubin(
                    stream,
                    set,
                    key,
                    handles,
                    None,
                    Some(super::head_policy::streamed::Selection::Resident(selection)),
                );
                drop(bindings);
                result
            })
            .map_err(|e| failure(format!("resident head original actor: {e:?}")))?
    }
    /// Finite source-window head. Full owner handles only; native pointer
    /// selection is validated inside this original actor.
    /// # Safety
    /// Caller has registered the real source-window owner before H2D and keeps
    /// the SAME original armed flight until every checkpoint or quarantine.
    pub unsafe fn enqueue_streamed_vocabulary_head(
        &self,
        weight: Handle,
        input: Handle,
        output: Handle,
        layout: crate::Vllm08WeightLayout,
        input_rows: usize,
        ordinal: usize,
    ) -> Result<(), ServerError> {
        self.require_active()?;
        let actual = super::head_policy::streamed::require(&self.image, &self.manifest)?;
        if actual != layout {
            return Err(failure("streamed head source packet differs"));
        }
        let selection = super::head_policy::streamed::Window {
            layout,
            input_rows,
            ordinal,
        };
        let bindings = vec![weight, input, output];
        for (index, handle) in bindings.iter().enumerate() {
            canonical_handle(
                handle,
                self.set.queue.service_id(),
                selection.full_bytes(index)?,
                self.allocation_alignment,
            )?;
        }
        let key = self.key;
        let set = self.set.key;
        let stream = self.set.queue.stream;
        self.set
            .queue
            .actor
            .submit_blocking(move |server| {
                server.enqueue_owned_cubin(
                    stream,
                    set,
                    key,
                    bindings,
                    None,
                    Some(super::head_policy::streamed::Selection::Streamed(selection)),
                )
            })
            .map_err(|e| failure(format!("streamed head original actor: {e:?}")))?
    }
    /// Original source's fixed4096B FA output copy; no context/stream leaves
    /// the actor. Both exact original allocations must already be in the same
    /// armed flight, source Read and destination Write. No completion claim.
    /// # Safety
    /// This final N1 kernel was enqueued before the copy on this original queue;
    /// caller retains self/set/images/source and both original leases through
    /// real completion or quarantine. Cancellation/error never releases them.
    pub unsafe fn copy_first_attention_output(
        &self,
        source: Handle,
        destination: Handle,
    ) -> Result<(), ServerError> {
        self.require_active()?;
        super::attention_policy::require_n1_copy(&self.image, &self.manifest)?;
        self.copy_attention_output(source, destination)
    }
    /// Exact canonical47 cached FA record; old captured-leaf copy stays closed.
    /// # Safety
    /// The original armed flight retains this kernel/set and both original
    /// leases after its actual N1 FA enqueue, through completion or quarantine.
    pub unsafe fn copy_canonical47_attention_output(
        &self,
        source: Handle,
        destination: Handle,
    ) -> Result<(), ServerError> {
        self.require_active()?;
        super::canonical47_natural_policy::require_n1_copy(&self.image, &self.manifest)?;
        self.copy_attention_output(source, destination)
    }
    /// Exact compiled cached short-HF private O to public O copy.
    /// # Safety
    /// This source-bound FA packet and both whole leases share the same original
    /// armed flight; its actual enqueue precedes this copy on this queue.
    pub unsafe fn copy_short_hf_attention_output(
        &self,
        source: Handle,
        destination: Handle,
    ) -> Result<(), ServerError> {
        self.require_active()?;
        super::short_hf_policy::require_n1_copy(&self.image, &self.manifest)?;
        self.copy_attention_output(source, destination)
    }
    fn copy_attention_output(
        &self,
        source: Handle,
        destination: Handle,
    ) -> Result<(), ServerError> {
        for h in [&source, &destination] {
            canonical_handle(
                h,
                self.set.queue.service_id(),
                4096,
                self.allocation_alignment,
            )?;
        }
        let set = self.set.key;
        let key = self.key;
        let stream = self.set.queue.stream;
        self.set
            .queue
            .actor
            .submit_blocking(move |server| {
                server.copy_owned_attention_output(stream, set, key, source, destination)
            })
            .map_err(|e| failure(format!("original attention output copy actor:{e:?}")))?
    }
    /// # Safety
    /// Every original flight using this record completed physically; no enqueue
    /// races retirement. Removes only this record, never its shared module.
    pub unsafe fn retire_after_physical_completion(&self) -> Result<(), ServerError> {
        self.require_active()?;
        self.retired
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| failure("cubin record retirement raced"))?;
        let key = self.key;
        let set = self.set.key;
        self.set
            .queue
            .actor
            .submit_blocking(move |server| server.retire_owned_cubin(set, key))
            .map_err(|e| failure(format!("original cubin record retirement actor: {e:?}")))?
    }
}
