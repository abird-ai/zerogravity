//! Explicit pinned E2B plain CPU candidate. Same ReferenceText/session engine.
//! No default trained, chat, native, HTTP or performance qualification.
mod session;
mod source;
use crate::cache::{SnapshotLease, TransferAdmission};
use crate::gemma4::text::{
    Gemma4CpuInventory, OriginalGemma4Text,
    tokenizer::{Gemma4PreparedText, Gemma4TextTokenizer, Gemma4TokenizerFiles},
};
use crate::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use crate::{Error, Result};
pub use session::{Gemma4CpuCall, Gemma4CpuSession};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

pub const PROFILE: &str = "gemma4-e2b-bf16-cpu-candidate";
pub const HOST_BYTES: u64 = 128 * 1024 * 1024;
pub const MAX_CONTEXT: usize = 64;
pub const MAX_OUTPUT: usize = 8;
const MAX_SESSIONS: usize = 2;
/// Same process-wide host arbiter, explicit fixed source/control envelope.
/// Prepare all text inputs before loading the learned source; BPE never overlaps it.
pub struct Gemma4CpuInput {
    prepared: Gemma4PreparedText,
    ticket: InputTicket,
}
struct InputTicket(Arc<Gemma4CpuContext>);
impl Drop for InputTicket {
    fn drop(&mut self) {
        self.0.inputs.fetch_sub(1, Ordering::Release);
    }
}
impl Gemma4CpuInput {
    pub fn ids(&self) -> &[u32] {
        self.prepared.ids()
    }
    /// Source-bound sequence decoding; IDs carry no state/inference authority.
    pub fn decode_output(
        &self,
        ids: &[u32],
        skip_special: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<crate::gemma4::text::tokenizer::Gemma4DecodedText> {
        self.prepared.decode_output(ids, skip_special, cancelled)
    }
}
pub struct Gemma4CpuContext {
    pub(crate) host: TransferAdmission,
    inputs: AtomicUsize,
    _control: GlobalResourceLease,
}
impl Gemma4CpuContext {
    pub fn new() -> Result<Arc<Self>> {
        Self::with_host_limit(HOST_BYTES)
    }
    #[cfg(feature = "gemma4-hip-qualification")]
    pub fn new_hip_qualification() -> Result<Arc<Self>> {
        Self::with_host_limit(zerogravity_gpu::Gemma4Bf16GpuCandidate::qualification_host_bytes())
    }
    #[cfg(all(
        feature = "gemma4-hip-qualification",
        feature = "qwen27b-hip-qualification"
    ))]
    pub fn new_shared_hip_qualification() -> Result<Arc<Self>> {
        Self::with_host_limit(zerogravity_gpu::SharedHipNative::host_bytes())
    }
    fn with_host_limit(host_bytes: u64) -> Result<Arc<Self>> {
        let host = TransferAdmission {
            arbiter: ResourceArbiter::process_shared(),
            resource: PhysicalDeviceKey::new("host", "system-ram")?,
            owner: ResourceOwner::new("Gemma4E2B", "ordinary-cpu-candidate")?,
        };
        host.arbiter
            .register_matching(host.resource.clone(), host_bytes)?;
        let control = host.arbiter.admit(
            &host.resource,
            host.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 3 * 1024 * 1024,
            }],
        )?;
        Ok(Arc::new(Self {
            host,
            inputs: AtomicUsize::new(0),
            _control: control,
        }))
    }
    pub fn reserved_bytes(&self) -> Result<u64> {
        Ok(self.host.arbiter.snapshot(&self.host.resource)?.reserved)
    }
    pub fn prepare_text(
        self: &Arc<Self>,
        snapshot: &SnapshotLease,
        text: &str,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4CpuInput> {
        self.inputs
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                if n < 8 { Some(n + 1) } else { None }
            })
            .map_err(|_| Error::Limit("Gemma source preparation retains at most8inputs".into()))?;
        let ticket = InputTicket(self.clone());
        source::require(snapshot)?;
        let file = |name| snapshot.file(name)?.try_clone().map_err(Error::from);
        let prepared = Gemma4TextTokenizer::prepare(
            Gemma4TokenizerFiles {
                tokenizer: file("tokenizer.json")?,
                tokenizer_config: file("tokenizer_config.json")?,
                generation_config: file("generation_config.json")?,
                model_config: file("config.json")?,
            },
            &self.host,
            cancelled,
        )?
        .into_plain_text(text, cancelled)?;
        Ok(Gemma4CpuInput { prepared, ticket })
    }
}
struct Life {
    sessions: usize,
    closed: bool,
}
/// Model/source authority remains in OriginalGemma4Text and SnapshotLease.
pub struct Gemma4CpuLoaded {
    pub(crate) model: Arc<OriginalGemma4Text>,
    snapshot: Arc<SnapshotLease>,
    pub(crate) context: Arc<Gemma4CpuContext>,
    inventory: Gemma4CpuInventory,
    life: Mutex<Life>,
    active: AtomicBool,
}
#[cfg(feature = "gemma4-hip-qualification")]
struct GpuSourceRetention(Arc<Gemma4CpuLoaded>);
#[cfg(feature = "gemma4-hip-qualification")]
impl Drop for GpuSourceRetention {
    fn drop(&mut self) {
        match self.0.life.lock() {
            Ok(mut life) => life.sessions -= 1,
            Err(_) => std::mem::forget(self.0.clone()),
        }
    }
}
#[cfg(feature = "gemma4-hip-qualification")]
enum GpuTarget {
    Single(zerogravity_gpu::Device, zerogravity_gpu::GemmaTextStorage),
    #[cfg(feature = "qwen27b-hip-qualification")]
    Shared(Arc<zerogravity_gpu::SharedHipNative>),
}
impl Gemma4CpuLoaded {
    pub fn open(
        snapshot: Arc<SnapshotLease>,
        context: Arc<Gemma4CpuContext>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        cancelled_check(cancelled)?;
        source::require(&snapshot)?;
        let (model, inventory) = source::load(&snapshot, &context.host, cancelled)?;
        cancelled_check(cancelled)?;
        Ok(Arc::new(Self {
            model: Arc::new(model),
            snapshot,
            context,
            inventory,
            life: Mutex::new(Life {
                sessions: 0,
                closed: false,
            }),
            active: AtomicBool::new(false),
        }))
    }
    /// Exact original SnapshotLease/source::load binding remains retained by the
    /// GPU candidate and every unresolved native owner. CPU math is not invoked.
    #[cfg(feature = "gemma4-hip-qualification")]
    pub async fn load_hip_qualification(
        self: &Arc<Self>,
        device: zerogravity_gpu::Device,
        input: Gemma4CpuInput,
        host: zerogravity_gpu::GemmaTextHostOwners,
        storage: zerogravity_gpu::GemmaTextStorage,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(
        Arc<zerogravity_gpu::Gemma4Bf16GpuCandidate>,
        zerogravity_gpu::Gemma4Bf16GpuSession,
    )> {
        self.load_native(GpuTarget::Single(device, storage), input, host, cancelled)
            .await
    }
    #[cfg(all(
        feature = "gemma4-hip-qualification",
        feature = "qwen27b-hip-qualification"
    ))]
    pub async fn load_shared_hip_qualification(
        self: &Arc<Self>,
        shared: Arc<zerogravity_gpu::SharedHipNative>,
        input: Gemma4CpuInput,
        host: zerogravity_gpu::GemmaTextHostOwners,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(
        Arc<zerogravity_gpu::Gemma4Bf16GpuCandidate>,
        zerogravity_gpu::Gemma4Bf16GpuSession,
    )> {
        self.load_native(GpuTarget::Shared(shared), input, host, cancelled)
            .await
    }
    #[cfg(all(
        feature = "gemma4-hip-qualification",
        feature = "qwen27b-hip-qualification"
    ))]
    pub(crate) fn require_shared_hip_input(&self, input: &Gemma4CpuInput) -> Result<()> {
        if !Arc::ptr_eq(&input.ticket.0, &self.context) {
            return Err(Error::State(
                "Gemma shared HIP input context differs".into(),
            ));
        }
        source::require(&self.snapshot)?;
        input.prepared.require_model(&self.model)?;
        let life = self
            .life
            .lock()
            .map_err(|_| Error::State("Gemma source lifecycle poisoned".into()))?;
        if life.closed || life.sessions != 0 {
            return Err(Error::State(
                "Gemma cold plan requires unused open source".into(),
            ));
        }
        Ok(())
    }
    #[cfg(feature = "gemma4-hip-qualification")]
    async fn load_native(
        self: &Arc<Self>,
        target: GpuTarget,
        input: Gemma4CpuInput,
        host: zerogravity_gpu::GemmaTextHostOwners,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(
        Arc<zerogravity_gpu::Gemma4Bf16GpuCandidate>,
        zerogravity_gpu::Gemma4Bf16GpuSession,
    )> {
        if !Arc::ptr_eq(&input.ticket.0, &self.context) {
            return Err(Error::State(
                "Gemma HIP input belongs to another managed context".into(),
            ));
        }
        input.prepared.require_model(&self.model)?;
        {
            let mut life = self
                .life
                .lock()
                .map_err(|_| Error::State("Gemma HIP source lifecycle".into()))?;
            if life.closed || life.sessions >= MAX_SESSIONS {
                return Err(Error::State(
                    "Gemma HIP requires retained open source capacity".into(),
                ));
            }
            life.sessions += 1;
        }
        let retained = Arc::new(GpuSourceRetention(self.clone()));
        // SAFETY: exact managed source::require/load +input/context/model joins
        // above. The private retention guard owns this SAME loaded SnapshotLease
        // and lifecycle count; candidate/native/flight/quarantine retain it.
        unsafe {
            match target {
                GpuTarget::Single(device, storage) => {
                    zerogravity_gpu::Gemma4Bf16GpuCandidate::load_hip_qualification(
                        device,
                        self.model.clone(),
                        retained,
                        Arc::from(input.ids()),
                        host,
                        storage,
                        cancelled,
                    )
                    .await
                }
                #[cfg(feature = "qwen27b-hip-qualification")]
                GpuTarget::Shared(shared) => {
                    zerogravity_gpu::Gemma4Bf16GpuCandidate::load_shared_hip_qualification(
                        shared,
                        self.model.clone(),
                        retained,
                        Arc::from(input.ids()),
                        host,
                        cancelled,
                    )
                    .await
                }
            }
        }
    }
    pub fn inventory(&self) -> &Gemma4CpuInventory {
        &self.inventory
    }
    pub fn profile(&self) -> &'static str {
        PROFILE
    }
    pub fn trained_model_qualified(&self) -> bool {
        false
    }
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.model.source_identity()
    }
    pub fn session(self: &Arc<Self>, input: Gemma4CpuInput) -> Result<Gemma4CpuSession> {
        if !Arc::ptr_eq(&input.ticket.0, &self.context) {
            return Err(Error::State(
                "Gemma input belongs to another host/source preparation context".into(),
            ));
        }
        input.prepared.require_model(&self.model)?;
        let mut life = self
            .life
            .lock()
            .map_err(|_| Error::State("Gemma lifecycle lock".into()))?;
        if life.closed || life.sessions >= MAX_SESSIONS {
            return Err(Error::Limit(
                "Gemma candidate has two sessions; source must be open".into(),
            ));
        }
        life.sessions += 1;
        drop(life);
        Gemma4CpuSession::new(self.clone(), input)
    }
    pub fn close(self: Arc<Self>, cancelled: Option<&AtomicBool>) -> Result<()> {
        cancelled_check(cancelled)?;
        {
            let mut life = self
                .life
                .lock()
                .map_err(|_| Error::State("Gemma lifecycle lock".into()))?;
            if life.sessions != 0 || self.active.load(Ordering::Acquire) {
                return Err(Error::State(
                    "Gemma close requires all sessions/completions retired".into(),
                ));
            }
            if life.closed {
                return Err(Error::State("Gemma source already closed".into()));
            }
            life.closed = true;
        }
        source::after(&self.snapshot, cancelled)
    }
    fn operation(&self) -> Result<Operation<'_>> {
        self.active
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| Error::State("Gemma candidate operation already active".into()))?;
        Ok(Operation(&self.active))
    }
}
struct Operation<'a>(&'a AtomicBool);
impl Drop for Operation<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
fn cancelled_check(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|f| f.load(Ordering::Acquire)) {
        Err(Error::State("Gemma CPU request cancelled".into()))
    } else {
        Ok(())
    }
}
