//! Defined native ABI scratch per original attempt/layer, never a StateRoot.
use crate::resident_inference::native_fragment::model_state::BoundModelState;
use crate::resident_leases::{ResidentLease, ResidentService};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{Vllm08Layer, VllmBf16Phase};
use zerogravity_core::device_events::BufferRole;
use zerogravity_core::session_root::{RootAttempt, RootBinding};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::OriginalWindowModelSource;

const BYTES: usize = 5_570_560;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum History {
    Two,
    SixtySeven,
    Canonical47Prefix,
    Canonical47AttentionKv,
    Canonical47Whole,
}
struct ZeroSource {
    bytes: Vec<u8>,
    _source: Arc<OriginalWindowModelSource>,
    // Original bytes, client H2D copy and bounded staging/metadata admission.
    _host: WindowWorkspace,
}
pub(crate) struct ModelCache {
    lease: ResidentLease,
    source: Arc<OriginalWindowModelSource>,
    base: RootBinding,
    layer: Vllm08Layer,
    history: History,
    canonical47:
        Option<Arc<crate::vllm_cuda_qualification::canonical47_model::Canonical47ModelBinding>>,
    initial: Arc<BoundModelState>,
}
fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State(
            "full24 defined native cache initialization cancelled".into(),
        ));
    }
    Ok(())
}
impl ModelCache {
    /// Caller holds its SAME original fair preparation slot; one bounded zero
    /// window is retired before the next cache/source-W allocation. No native
    /// module is active while these ordinary initialized resources are created.
    pub(crate) async fn prepare(
        source: Arc<OriginalWindowModelSource>,
        service: Arc<ResidentService>,
        attempt: &RootAttempt,
        initial: Arc<BoundModelState>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        source.require_vllm08_full_source()?;
        let first = initial.call();
        first.validate()?;
        let history = match (
            first.request().legacy_phase()?,
            first.request().teachers().base(),
            attempt.frontier(),
        ) {
            (VllmBf16Phase::InitialOne, 0, 2) => History::Two,
            (VllmBf16Phase::Initial64, 0, 67) => History::SixtySeven,
            _ => {
                return Err(Error::Unsupported(
                    "closed original full-model native-cache request histories".into(),
                ));
            }
        };
        Self::prepare_checked(source, service, attempt, initial, history, None, cancelled).await
    }
    pub(crate) async fn prepare_canonical47_prefix(
        scope: Arc<crate::vllm_cuda_qualification::canonical47_model::Canonical47ModelBinding>,
        service: Arc<ResidentService>,
        attempt: &RootAttempt,
        initial: Arc<BoundModelState>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        initial.require_canonical47_model_prefix(&scope)?;
        if scope.kind()
            != crate::vllm_cuda_qualification::canonical47_model::Canonical47CallKind::Prefill
            || attempt.base().frontier() != 0
            || attempt.frontier() != 47
        {
            return Err(Error::Unsupported(
                "canonical47 defined cache is initial first3 only".into(),
            ));
        }
        Self::prepare_checked(
            scope.source().source().clone(),
            service,
            attempt,
            initial,
            History::Canonical47Prefix,
            Some(scope),
            cancelled,
        )
        .await
    }
    pub(crate) async fn prepare_canonical47_attention_kv(
        cut: &Arc<super::canonical47_model_prefix::Canonical47PrefixCut>,
        service: Arc<ResidentService>,
        attempt: &RootAttempt,
        initial: Arc<BoundModelState>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        if !cut.is_attention_cut() || attempt.base().frontier() != 0 || attempt.frontier() != 47 {
            return Err(Error::Unsupported(
                "canonical47 exact35-node Kv3 cache".into(),
            ));
        }
        initial.require_canonical47_attention_kv(cut.scope())?;
        Self::prepare_checked(
            cut.scope().source().source().clone(),
            service,
            attempt,
            initial,
            History::Canonical47AttentionKv,
            Some(cut.scope().clone()),
            cancelled,
        )
        .await
    }
    async fn prepare_checked(
        source: Arc<OriginalWindowModelSource>,
        service: Arc<ResidentService>,
        attempt: &RootAttempt,
        initial: Arc<BoundModelState>,
        history: History,
        canonical47: Option<
            Arc<crate::vllm_cuda_qualification::canonical47_model::Canonical47ModelBinding>,
        >,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        source.require_vllm08_full_source()?;
        let first = initial.call();
        let layer = initial.layer();
        if first.source() != source.source_identity()
            || service.source_identity() != source.source_identity()
            || attempt.base().source() != source.source_identity()
            || initial.base() != attempt.base()
            || attempt.base().frontier() != 0
            || first.layer(layer.index())? != layer
        {
            return Err(Error::State(
                "full24 native cache same source/attempt/layer".into(),
            ));
        }
        let host = source.source().reserve_workspace(BYTES * 3 + 64 * 1024)?;
        let owner = Arc::new(ZeroSource {
            bytes: vec![0; BYTES],
            _source: source.clone(),
            _host: host,
        });
        let mut init = service.begin_initialization(attempt.base().root())?;
        init.retain_completion_owner(owner.clone())?;
        let lease = init.upload_bytes(&owner.bytes, BufferRole::RecurrentState)?;
        not_cancelled(cancelled)?;
        let initialized = init.complete_sync().await?;
        if !lease.is_initialized()
            || lease.bytes() != BYTES
            || !Arc::ptr_eq(lease.service(), &service)
        {
            return Err(Error::State(
                "full24 original native cache physical initialization".into(),
            ));
        }
        // Genuine H2D has completed before releasing its full host admission.
        drop(initialized);
        drop(owner);
        not_cancelled(cancelled)?;
        Ok(Arc::new(Self {
            lease,
            source,
            base: attempt.base(),
            layer,
            history,
            canonical47,
            initial,
        }))
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    pub(crate) fn require_binding(&self, binding: &BoundModelState) -> Result<()> {
        let base = binding.call().request().teachers().base();
        let phase = if matches!(
            self.history,
            History::Canonical47Prefix
                | History::Canonical47AttentionKv
                | History::Canonical47Whole
        ) {
            None
        } else {
            Some(binding.call().request().legacy_phase()?)
        };
        let allowed = match self.history {
            History::Two => matches!(
                (phase, base),
                (Some(VllmBf16Phase::InitialOne), 0) | (Some(VllmBf16Phase::CachedOne), 1)
            ),
            History::SixtySeven => matches!(
                (phase, base),
                (Some(VllmBf16Phase::Initial64), 0)
                    | (Some(VllmBf16Phase::ContinuedTwo), 64)
                    | (Some(VllmBf16Phase::CachedOne), 66)
            ),
            History::Canonical47Whole => {
                let cut = self.initial.whole_cut()?;
                binding.require_canonical47_whole(cut)?;
                phase.is_none() && base == 0
            }
            History::Canonical47Prefix => {
                let scope = self
                    .canonical47
                    .as_ref()
                    .ok_or_else(|| Error::State("canonical47 defined cache scope absent".into()))?;
                binding.require_canonical47_model_prefix(scope)?;
                phase.is_none() && base == 0
            }
            History::Canonical47AttentionKv => {
                let scope = self.canonical47.as_ref().ok_or_else(|| {
                    Error::State("canonical47 Kv3 cache source scope absent".into())
                })?;
                binding.require_canonical47_attention_kv(scope)?;
                phase.is_none() && base == 0
            }
        };
        self.lease.service().ensure_usable()?;
        if !allowed
            || !binding.same_request(&self.initial)
            || binding.base() != self.base
            || binding.layer() != self.layer
            || binding.call().source() != self.source.source_identity()
            || !self.lease.is_initialized()
            || self.lease.bytes() != BYTES
        {
            return Err(Error::State(
                "full24 native cache cannot cross original layer/request/history".into(),
            ));
        }
        Ok(())
    }
}

mod whole;

mod natural;
pub(crate) use natural::SharedModelCache;
