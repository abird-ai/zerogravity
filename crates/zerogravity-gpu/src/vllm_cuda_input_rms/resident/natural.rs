//! True source rows from the SAME already-verified resident tied table.
use super::*;
use crate::resident_inference::{
    ResidentSession, native_fragment::natural_publication::NaturalCallInput,
};
use crate::resident_leases::TiedWeight;
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) struct NaturalRows {
    ids: Arc<[u32]>,
    input: SourceRowInput,
    table: Arc<TiedWeight>,
    destination: ResidentLease,
    submitted: AtomicBool,
    // Original cold owner retains gamma/specs and old authenticated row windows.
    // No new mathematical input or retained source chunk is introduced here.
    _cold: Arc<RmsPrepared>,
    _host: WindowWorkspace,
}
enum SourceRowInput {
    Legacy(Arc<NaturalCallInput>),
    Canonical(
        Arc<crate::resident_inference::native_fragment::canonical47_natural::CanonicalCallInput>,
    ),
}
impl SourceRowInput {
    fn source(
        &self,
    ) -> &Arc<zerogravity_model_qwen35::original_binding::OriginalWindowModelSource> {
        match self {
            Self::Legacy(input) => input.source(),
            Self::Canonical(input) => input.source().source(),
        }
    }
    fn ids(&self) -> &[u32] {
        match self {
            Self::Legacy(input) => input.ids(),
            Self::Canonical(input) => input.ids(),
        }
    }
    fn call(&self) -> &zerogravity_core::accelerator::program::Vllm08Call {
        match self {
            Self::Legacy(input) => input.call(),
            Self::Canonical(input) => input.call(),
        }
    }
    fn require_session(&self, session: &ResidentSession) -> Result<()> {
        match self {
            Self::Legacy(input) => input.require_session(session),
            Self::Canonical(input) => input.require_session(session),
        }
    }
    fn closed_rows(&self) -> bool {
        match self {
            Self::Legacy(input) => matches!(input.ids().len(), 1 | 2 | 64),
            Self::Canonical(input) => matches!(
                (input.kind(), input.ids().len()),
                (crate::vllm_cuda_qualification::canonical47_model::Canonical47CallKind::Prefill, 47)
                | (crate::vllm_cuda_qualification::canonical47_model::Canonical47CallKind::Decode { .. }, 1)
            ),
        }
    }
}
impl RmsPrepared {
    pub(crate) fn bind_natural(
        self: &Arc<Self>,
        input: Arc<NaturalCallInput>,
        table: Arc<TiedWeight>,
    ) -> Result<Arc<Self>> {
        self.bind_source_rows(SourceRowInput::Legacy(input), table)
    }
    pub(crate) fn bind_canonical_call(
        self: &Arc<Self>,
        input: Arc<
            crate::resident_inference::native_fragment::canonical47_natural::CanonicalCallInput,
        >,
        table: Arc<TiedWeight>,
    ) -> Result<Arc<Self>> {
        self.bind_source_rows(SourceRowInput::Canonical(input), table)
    }
    fn bind_source_rows(
        self: &Arc<Self>,
        input: SourceRowInput,
        table: Arc<TiedWeight>,
    ) -> Result<Arc<Self>> {
        if self.pooled.is_some()
            || self.natural.is_some()
            || input.call().rows() != self.owner.rows.count()
            || !self.owner.source.same_issuer(input.source().source())
            || !Arc::ptr_eq(table.source(), input.source())
            || input.ids() != input.call().request().teachers().ids()
            || input.ids().len() != self.owner.rows.count()
            || !input.closed_rows()
            || input.ids().iter().any(|id| *id >= 248_320)
            || !Arc::ptr_eq(table.lease().service(), &self.service)
            || table.lease().source_identity() != input.source().source_identity()
            || table.lease().role() != BufferRole::ImmutableWeights
            || table.lease().bytes() != 508_559_360
            || !table.lease().is_initialized()
            || table.copy_counts()? != (485, 485)
        {
            return Err(Error::State(
                "natural original source/table/row/input seal differs".into(),
            ));
        }
        let matrix = input.source().matrix_descriptor(
            zerogravity_model_qwen35::original_binding::QwenMatrixRole::Embedding,
        )?;
        input.source().validate_matrix(
            zerogravity_model_qwen35::original_binding::QwenMatrixRole::Output,
            &matrix,
        )?;
        if matrix.rows != 248_320 || matrix.columns != 1024 || matrix.kind != QuantKind::Bf16 {
            return Err(Error::State(
                "natural source embedding tied geometry".into(),
            ));
        }
        let host = input.source().source().reserve_workspace(64 * 1024)?;
        let rows = Arc::new(NaturalRows {
            ids: Arc::from(input.ids()),
            input,
            table,
            destination: self.embedding().clone(),
            submitted: AtomicBool::new(false),
            _cold: self.clone(),
            _host: host,
        });
        let mut accesses = self.accesses.clone();
        for (lease, access) in &mut accesses {
            if lease.id() == rows.destination.id() {
                *access = DeviceAccess::Write;
            }
        }
        accesses.push((rows.table.lease().clone(), DeviceAccess::Read));
        Ok(Arc::new(Self {
            service: self.service.clone(),
            leases: self.leases.clone(),
            stages: self.stages.clone(),
            accesses,
            outputs: self.outputs.clone(),
            owner: self.owner.clone(),
            natural: Some(rows),
            pooled: None,
            short_hf: None,
        }))
    }
    /// Body/input/table+original Write are retained by the SAME armed flight.
    /// Each private slot is claimed once by ColdModelSlots before this wrapper.
    pub(crate) fn enqueue_embedding(
        &self,
        session: &ResidentSession,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        if self.short_hf.is_some() {
            return Err(Error::Unsupported(
                "short-HF embedding requires the actual original call permit".into(),
            ));
        }
        let Some(rows) = &self.natural else {
            return Ok(0);
        };
        if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            return Err(Error::State("natural embedding cancelled".into()));
        }
        rows.input.require_session(session)?;
        self.service.ensure_usable()?;
        if !Arc::ptr_eq(rows.destination.service(), &self.service)
            || rows.destination.role() != BufferRole::Activation
            || rows.destination.bytes() != rows.ids.len() * 2048
            || rows.destination.id() != self.embedding().id()
            || rows.ids.as_ref() != rows.input.ids()
            || !rows.table.lease().is_initialized()
            || rows.submitted.swap(true, Ordering::AcqRel)
        {
            return Err(Error::State(
                "natural source rows exclusive one-use original slot".into(),
            ));
        }
        // SAFETY: private table/call constructors seal source and IDs; original
        // submitted Read/Write cohort and owner survive real completion/unknown.
        unsafe {
            let queue = self.service.pinned_cuda_queue()?;
            match &rows.input {
                SourceRowInput::Canonical(input) if input.rows() == 47 => queue
                    .enqueue_canonical47_embedding_rows(
                        rows.table.lease().handle(),
                        rows.destination.handle(),
                        rows.ids.clone(),
                    ),
                _ => queue.enqueue_vllm08_embedding_rows(
                    rows.table.lease().handle(),
                    rows.destination.handle(),
                    rows.ids.clone(),
                ),
            }
        }
        .map_err(|error| Error::State(format!("natural original embedding D2D: {error}")))?;
        Ok(rows.ids.len())
    }
    /// Independent source check AFTER arithmetic/D2H. Each authenticated window
    /// is temporary, so retained cold windows and one new complete read are both
    /// admitted; no69-window second ownership graph is hidden in row payloads.
    pub(super) fn validate_natural_embedding(&self, bytes: &[u8]) -> Result<()> {
        let rows = self
            .natural
            .as_ref()
            .ok_or_else(|| Error::State("natural row validator without binding".into()))?;
        if bytes.len() != rows.ids.len() * 2048 || !finite_bf16(bytes) {
            return Err(Error::State(
                "natural original complete finite embedding D2H".into(),
            ));
        }
        let source = rows.input.source().source();
        let matrix = source.matrix(
            "model.language_model.embed_tokens.weight",
            MatrixAxes::OutputInput,
        )?;
        for (index, id) in rows.ids.iter().enumerate() {
            let original = source.read_rows(&matrix, *id as usize..*id as usize + 1)?;
            if original.as_slice() != &bytes[index * 2048..(index + 1) * 2048] {
                return Err(Error::State(
                    "natural resident table row differs from authentic source".into(),
                ));
            }
        }
        Ok(())
    }
}
