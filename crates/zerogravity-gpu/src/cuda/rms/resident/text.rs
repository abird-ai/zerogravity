//! Actual source rows: SAME resident tied table or bounded verified source H2D.
// Former actor: ShortSourceRows.
use super::*;
use crate::cuda::rms::FullModelRmsScratch;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use crate::resident_leases::TiedWeight;
use cubecl::cuda::TextEmbeddingWidth;
use std::sync::atomic::{AtomicBool, Ordering};

struct PackedRows {
    bytes: Vec<u8>,
    _host: WindowWorkspace,
}
enum HostPayload {
    Cold(Arc<SourceOwner>),
    Fresh(PackedRows),
}
impl AsRef<[u8]> for HostPayload {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Cold(owner) => &owner.embedding,
            Self::Fresh(rows) => &rows.bytes,
        }
    }
}
enum Payload {
    Host(Arc<HostPayload>),
    Resident(Arc<TiedWeight>),
}
pub(super) struct EmbeddingRows {
    scope: Arc<FullModelRmsScratch>,
    ids: Arc<[u32]>,
    payload: Payload,
    submitted: AtomicBool,
    _metadata: WindowWorkspace,
}
impl EmbeddingRows {
    pub(super) fn from_cold(
        scope: Arc<FullModelRmsScratch>,
        owner: Arc<SourceOwner>,
    ) -> Result<Arc<Self>> {
        let source = scope.source().source();
        if !source.same_issuer(&owner.source)
            || scope.rows()? != owner.rows
            || scope.columns() != owner.columns
            || scope.record().call().rows().ids() != owner._teacher_ids
        {
            return Err(Error::State(
                "text cold source rows differ from actual source call".into(),
            ));
        }
        let metadata = source.reserve_workspace(64 * 1024)?;
        Ok(Arc::new(Self {
            ids: Arc::from(scope.record().call().rows().ids()),
            scope,
            payload: Payload::Host(Arc::new(HostPayload::Cold(owner))),
            submitted: AtomicBool::new(false),
            _metadata: metadata,
        }))
    }
    fn fresh(scope: Arc<FullModelRmsScratch>, tied: Option<&Arc<TiedWeight>>) -> Result<Arc<Self>> {
        let source = scope.source().source();
        let ids = scope.record().call().rows().ids();
        let columns = scope.columns().count();
        let bytes = ids.len() * columns * 2;
        let metadata = source.reserve_workspace(64 * 1024)?;
        if let Some(tied) = tied {
            require_resident(&scope, tied)?;
            return Ok(Arc::new(Self {
                ids: Arc::from(ids),
                scope,
                payload: Payload::Resident(tied.clone()),
                submitted: AtomicBool::new(false),
                _metadata: metadata,
            }));
        }
        let host = source.reserve_workspace(bytes * 3 + 64 * 1024)?;
        let matrix = source_embedding(scope.record())?;
        if matrix.rows() != 248320
            || matrix.columns() != columns
            || matrix.tensor().kind != QuantKind::Bf16
        {
            return Err(Error::State(
                "text actual embedding source geometry differs".into(),
            ));
        }
        let mut packed = Vec::with_capacity(bytes);
        for &id in ids {
            let window = source.read_rows(&matrix, id as usize..id as usize + 1)?;
            if window.as_slice().len() != columns * 2 || !finite_bf16(window.as_slice()) {
                return Err(Error::State(
                    "text complete authentic BF16 source row".into(),
                ));
            }
            packed.extend_from_slice(window.as_slice());
            // Blocking source IO retired; immutable source/FD + charged packed bytes remain.
            drop(window);
        }
        Ok(Arc::new(Self {
            ids: Arc::from(ids),
            scope,
            payload: Payload::Host(Arc::new(HostPayload::Fresh(PackedRows {
                bytes: packed,
                _host: host,
            }))),
            submitted: AtomicBool::new(false),
            _metadata: metadata,
        }))
    }
    pub(super) fn validate(&self, bytes: &[u8]) -> Result<()> {
        let width = self.scope.columns().count() * 2;
        if bytes.len() != self.ids.len() * width
            || !finite_bf16(bytes)
            || matches!(&self.payload, Payload::Host(host) if bytes != host.as_ref().as_ref())
        {
            return Err(Error::State(
                "text actual source embedding output differs".into(),
            ));
        }
        let source = self.scope.source().source();
        let matrix = source_embedding(self.scope.record())?;
        for (row, &id) in self.ids.iter().enumerate() {
            let original = source.read_rows(&matrix, id as usize..id as usize + 1)?;
            if original.as_slice() != &bytes[row * width..(row + 1) * width] {
                return Err(Error::State(
                    "text embedding differs from fresh authentic tied source row".into(),
                ));
            }
        }
        Ok(())
    }
}

fn require_resident(scope: &FullModelRmsScratch, tied: &Arc<TiedWeight>) -> Result<()> {
    let original = scope.source().source();
    let lease = tied.matrix().lease();
    scope.service().ensure_usable()?;
    if !tied.source().same_issuer(original)
        || !Arc::ptr_eq(lease.service(), scope.service())
        || lease.source_identity() != original.source_identity()
        || lease.service_generation() != scope.service().generation()
        || lease.role() != BufferRole::ImmutableWeights
        || lease.bytes() != tied.bytes()
        || !lease.is_initialized()
        || tied.copy_counts()? != (tied.windows(), tied.windows())
    {
        return Err(Error::State(
            "text embedding SAME complete resident tied source/service".into(),
        ));
    }
    let matrix = source_embedding(scope.record())?;
    tied.matrix()
        .require_matrix(original, scope.service(), &matrix)?;
    // Original completed upload seals both tied source roles; current IDs still
    // come only from this fresh immutable recorded invocation.
    let ids = scope.record().call().rows().ids();
    if !matches!(ids.len(), 1 | 47 | 57 | 62 | 67 | 72)
        || ids.len() != scope.rows()?.count()
        || ids.iter().any(|&id| id >= VOCABULARY as u32)
    {
        return Err(Error::State("text resident actual source IDs".into()));
    }
    Ok(())
}
impl RmsPrepared {
    /// No new gamma/storage/spec upload. A new actual source row owner and pooled
    /// actor are bound; only original current-root submission can enqueue them.
    pub(crate) fn bind_prompt_call(
        self: &Arc<Self>,
        scope: Arc<FullModelRmsScratch>,
        tied: Option<&Arc<TiedWeight>>,
    ) -> Result<Arc<Self>> {
        if self.natural.is_some()
            || scope.rows()? != self.owner.rows
            || scope.columns() != self.owner.columns
            || !scope.source().source().same_issuer(&self.owner.source)
            || scope.binding(1)?.parameter()
                != &self.owner._parameter.parameter().source_parameter()
        {
            return Err(Error::State(
                "text ordinary authentic source/gamma/geometry rebind".into(),
            ));
        }
        let pooled = self
            .pooled
            .as_ref()
            .ok_or_else(|| Error::State("text ordinary cold pool actor absent".into()))?
            .rebind(&scope, 1)?;
        let rows = EmbeddingRows::fresh(scope, tied)?;
        let mut accesses = self.accesses.clone();
        let embedding = self.embedding().id();
        for (lease, access) in &mut accesses {
            if lease.id() == embedding {
                *access = DeviceAccess::Write;
            }
        }
        if let Payload::Resident(tied) = &rows.payload {
            accesses.push((tied.lease().clone(), DeviceAccess::Read));
        }
        Ok(Arc::new(Self {
            service: self.service.clone(),
            leases: self.leases.clone(),
            stages: self.stages.clone(),
            accesses,
            outputs: self.outputs.clone(),
            owner: self.owner.clone(),
            pooled: Some(pooled),
            natural: None,
            text: Some(rows),
        }))
    }
    pub(crate) fn enqueue_text_embedding(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        require_not_cancelled(cancelled)?;
        self.service.ensure_usable()?;
        let rows = self
            .text
            .as_ref()
            .ok_or_else(|| Error::State("text actual source row actor absent".into()))?;
        rows.scope.require_active_run()?;
        permit.require_invocation(&self.service, rows.scope.record().call(), 0, 24)?;
        self.pooled
            .as_ref()
            .ok_or_else(|| Error::State("text source RMS actor absent".into()))?
            .require_scope(&rows.scope, 1)?;
        let destination = self.embedding();
        if destination.bytes() != rows.ids.len() * self.owner.columns.count() * 2
            || rows.ids.as_ref() != rows.scope.record().call().rows().ids()
            || rows.submitted.swap(true, Ordering::AcqRel)
        {
            return Err(Error::State(
                "text one-use complete actual source row producer".into(),
            ));
        }
        let queue = self.service.pinned_cuda_queue()?;
        match &rows.payload {
            Payload::Resident(tied) => {
                require_resident(&rows.scope, tied)?;
                // SAFETY: node0's original armed flight retains the SAME complete
                // tied source Read, exclusive output Write and actual recorded IDs.
                tied.matrix().enqueue_rows(destination, rows.ids.clone())?;
            }
            Payload::Host(host) => {
                let width = match self.owner.columns {
                    Columns::Qwen08 => TextEmbeddingWidth::H1024,
                    Columns::Qwen2b => TextEmbeddingWidth::H2048,
                };
                // SAFETY: same original submitted Write cohort retains the actual
                // source/FD/window/admission owner before bounded real H2D.
                unsafe {
                    queue.enqueue_text_source_rows_workspace(
                        destination.cubin_binding()?,
                        width,
                        rows.ids.clone(),
                        host.clone(),
                    )
                }
                .map_err(|e| Error::State(format!("text authentic source embedding H2D: {e}")))?;
                self.service.record_value_upload(destination.bytes());
            }
        }
        Ok(1)
    }
    pub(crate) fn text_embedding_accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let rows = self
            .text
            .as_ref()
            .ok_or_else(|| Error::State("text source producer absent".into()))?;
        let mut accesses = vec![(self.embedding().clone(), DeviceAccess::Write)];
        if let Payload::Resident(tied) = &rows.payload {
            accesses.push((tied.lease().clone(), DeviceAccess::Read));
        }
        Ok(accesses)
    }
    pub(crate) fn text_rms_accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        if self.text.is_none() {
            return Err(Error::State("text source RMS actor absent".into()));
        }
        Ok(self
            .leases
            .iter()
            .map(|(role, lease)| {
                (
                    lease.clone(),
                    if role.read_only() {
                        DeviceAccess::Read
                    } else {
                        DeviceAccess::Write
                    },
                )
            })
            .collect())
    }
}

impl RmsPrepared {
    pub(crate) fn text_immutable_gamma(&self) -> Result<(ResidentLease, &[u8])> {
        if self.text.is_none() {
            return Err(Error::State("text source RMS owner absent".into()));
        }
        Ok((
            self.lease(Buffer::GammaRaw).clone(),
            self.owner.gamma.as_slice(),
        ))
    }
}
