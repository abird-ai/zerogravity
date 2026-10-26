//! Unchanged paired BF16/U32 copy equations, sealed to the current source span.
use crate::{
    cuda::kernel::PinnedKernel,
    resident_inference::native_fragment::{
        PrefixPhaseEnqueue, model_state::BoundModelState, text::TextCallInput,
    },
    resident_leases::ResidentLease,
};
use cubecl::server::Handle;
use cubecl_runtime::kernel::BufferIOAttr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::program::StateEffect;
use zerogravity_core::device_events::DeviceAccess;
const SOURCE: &str = include_str!("../model_kv_copy.cu");
pub(super) const CACHE_BYTES: usize = 5 * 2 * 544 * 512 * 2;
const CANONICAL_BYTES: usize = 512 * 512 * 4;
struct CopyDefinition {
    kernel: PinnedKernel,
    grid: [u32; 3],
    incoming: bool,
}
/// Genuine prompt_tokens/call geometry only; no root, source operand or handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Geometry {
    prompt_tokens: usize,
    ordinal: usize,
    base: usize,
    rows: usize,
    end: usize,
}
impl Geometry {
    fn new(prompt_tokens: usize, ordinal: usize) -> Result<Self> {
        let prompt = prompt_tokens;
        if !matches!(prompt, 47 | 57 | 62 | 67 | 72) || ordinal >= 8 {
            return Err(Error::State(
                "text cold transport prompt_tokens/call".into(),
            ));
        }
        let base = if ordinal == 0 {
            0
        } else {
            prompt + ordinal - 1
        };
        let rows = if ordinal == 0 { prompt } else { 1 };
        let end = base + rows;
        if end > 79 {
            return Err(Error::State("text cold transport exact end".into()));
        }
        Ok(Self {
            prompt_tokens,
            ordinal,
            base,
            rows,
            end,
        })
    }
    fn require(&self, input: &TextCallInput) -> Result<()> {
        let actual = Self::new(input.prompt_tokens(), input.call_ordinal())?;
        if actual != *self
            || input.base() != self.base
            || input.rows() != self.rows
            || input.base().checked_add(input.rows()) != Some(self.end)
        {
            return Err(Error::State(
                "text current transport differs from complete cold tuple".into(),
            ));
        }
        Ok(())
    }
}
pub(super) struct Definitions {
    geometry: Geometry,
    incoming: Option<CopyDefinition>,
    outgoing: CopyDefinition,
    _metadata: Arc<WindowWorkspace>,
}
impl Definitions {
    pub(super) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<usize> {
        if let Some(incoming) = &self.incoming {
            incoming.kernel.prepare_code(queue)?;
        }
        self.outgoing.kernel.prepare_code(queue)?;
        Ok(1 + usize::from(self.incoming.is_some()))
    }
    /// Full immutable text/ABI/key owners and source construction peak.
    pub(super) fn metadata_bound(prompt_tokens: usize, ordinal: usize) -> Result<usize> {
        let g = Geometry::new(prompt_tokens, ordinal)?;
        let count = if g.base == 0 { 1 } else { 2 };
        Ok(count
            * (3 * (SOURCE.len() + 512)
                + PinnedKernel::KEY_METADATA_BYTES
                + 6 * std::mem::size_of::<(usize, BufferIOAttr)>()
                + std::mem::size_of::<CopyDefinition>()
                + 64)
            + std::mem::size_of::<Self>()
            + 32)
    }
    pub(super) fn prepare(
        prompt_tokens: usize,
        ordinal: usize,
        metadata: Arc<WindowWorkspace>,
    ) -> Result<Arc<Self>> {
        let geometry = Geometry::new(prompt_tokens, ordinal)?;
        let base = geometry.base;
        let end = geometry.end;
        let make = |incoming: bool| -> Result<CopyDefinition> {
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
                (CACHE_BYTES, if incoming { RW } else { R }),
            ];
            if !incoming {
                bindings.extend([(CANONICAL_BYTES, W), (CANONICAL_BYTES, W)]);
            }
            bindings.push((4, RW));
            let kernel = PinnedKernel::new(entry, source, 128, bindings.into())?;
            if kernel.retained_metadata_bytes()? + std::mem::size_of::<CopyDefinition>()
                > Self::metadata_bound(prompt_tokens, ordinal)?
            {
                return Err(Error::Limit(
                    "text retained cold transport source/ABI/key bound".into(),
                ));
            }
            let count = if incoming { base * 512 } else { 512 * 512 };
            Ok(CopyDefinition {
                kernel,
                grid: [count.div_ceil(128) as u32, 1, 1],
                incoming,
            })
        };
        Ok(Arc::new(Self {
            geometry,
            incoming: (base != 0).then(|| make(true)).transpose()?,
            outgoing: make(false)?,
            _metadata: metadata,
        }))
    }
}
impl CopyDefinition {
    fn bind(
        &self,
        leases: &[(ResidentLease, DeviceAccess)],
        cache: &ResidentLease,
        status: &ResidentLease,
    ) -> Result<Copy> {
        let mut handles = vec![leases[0].0.handle(), leases[2].0.handle(), cache.handle()];
        if !self.incoming {
            handles.extend([leases[1].0.handle(), leases[3].0.handle()]);
        }
        handles.push(status.handle());
        self.kernel
            .validate(cache.service().client(), self.grid, &handles)?;
        Ok(Copy {
            kernel: self.kernel.clone(),
            grid: self.grid,
            handles,
        })
    }
}
struct Copy {
    kernel: PinnedKernel,
    grid: [u32; 3],
    handles: Vec<Handle>,
}
pub(super) struct Transport {
    definitions: Arc<Definitions>,
    input: Arc<TextCallInput>,
    state: Arc<BoundModelState>,
    cache: ResidentLease,
    status: ResidentLease,
    incoming: Option<Copy>,
    outgoing: Copy,
    before: AtomicBool,
    after: AtomicBool,
}
impl Transport {
    pub(super) fn bind(
        definitions: Arc<Definitions>,
        input: Arc<TextCallInput>,
        state: Arc<BoundModelState>,
        cache: ResidentLease,
        status: ResidentLease,
    ) -> Result<Self> {
        state.require_text(input.scope())?;
        definitions.geometry.require(&input)?;
        let base = input.base();
        let end = base
            .checked_add(input.rows())
            .filter(|n| *n <= 79)
            .ok_or_else(|| Error::State("text exact transport active end".into()))?;
        if end != definitions.geometry.end {
            return Err(Error::State("text full current transport end".into()));
        }
        let leases = state.leases();
        if state.kv_domain().is_err()
            || state.call() != input.call()
            || leases.len() != 4
            || cache.bytes() != CACHE_BYTES
            || status.bytes() != 4
            || !cache.is_initialized()
            || !status.is_initialized()
            || leases.iter().any(|(l, _)| {
                l.bytes() != CANONICAL_BYTES
                    || l.id() == cache.id()
                    || l.id() == status.id()
                    || !Arc::ptr_eq(l.service(), cache.service())
                    || l.service_generation() != cache.service_generation()
            })
        {
            return Err(Error::State(
                "text transport SAME current canonical/cache/status whole owners".into(),
            ));
        }
        let incoming = definitions
            .incoming
            .as_ref()
            .map(|d| d.bind(&leases, &cache, &status))
            .transpose()?;
        let outgoing = definitions.outgoing.bind(&leases, &cache, &status)?;
        Ok(Self {
            definitions,
            input,
            state,
            cache,
            status,
            incoming,
            outgoing,
            before: AtomicBool::new(false),
            after: AtomicBool::new(false),
        })
    }
    fn require(&self, permit: &PrefixPhaseEnqueue<'_>, effect: StateEffect) -> Result<()> {
        self.definitions.geometry.require(&self.input)?;
        self.state.require_text(self.input.scope())?;
        self.state.require_enqueue(self.cache.service(), permit)?;
        self.state.require_effect(effect, false)
    }
    pub(super) fn before(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        effect: StateEffect,
    ) -> Result<usize> {
        self.require(permit, effect)?;
        if self.before.swap(true, Ordering::AcqRel) {
            return Err(Error::State("text cache input replay".into()));
        }
        if let Some(copy) = &self.incoming {
            copy.kernel
                .launch(self.cache.service().client(), copy.grid, &copy.handles)?;
            Ok(1)
        } else {
            Ok(0)
        }
    }
    pub(super) fn after(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        effect: StateEffect,
    ) -> Result<usize> {
        self.require(permit, effect)?;
        if !self.before.load(Ordering::Acquire) || self.after.swap(true, Ordering::AcqRel) {
            return Err(Error::State("text actual cache append/output order".into()));
        }
        self.outgoing.kernel.launch(
            self.cache.service().client(),
            self.outgoing.grid,
            &self.outgoing.handles,
        )?;
        Ok(1)
    }
    pub(super) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.state
            .leases()
            .into_iter()
            .chain([
                (self.cache.clone(), DeviceAccess::Write),
                (self.status.clone(), DeviceAccess::Write),
            ])
            .collect()
    }
    pub(super) fn status(&self) -> &ResidentLease {
        &self.status
    }
    pub(super) fn output_submitted(&self) -> bool {
        self.after.load(Ordering::Acquire)
    }
    pub(super) fn retained(
        &self,
        input: &Arc<TextCallInput>,
        state: &Arc<BoundModelState>,
    ) -> bool {
        Arc::ptr_eq(input, &self.input)
            && Arc::ptr_eq(state, &self.state)
            && self.cache.bytes() == CACHE_BYTES
            && self.status.bytes() == 4
            && self.cache.service_generation() == self.cache.service().generation()
            && self.cache.source_identity() == input.call().source()
    }
}
