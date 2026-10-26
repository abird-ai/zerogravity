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
    Reference(Arc<NaturalCallInput>),
    FixedPrompt(
        Arc<crate::resident_inference::native_fragment::fixed_prompt_natural::FixedPromptCallInput>,
    ),
}
impl SourceRowInput {
    fn source(&self) -> &VerifiedTensorSource {
        match self {
            Self::Reference(input) => input.source(),
            Self::FixedPrompt(input) => input.source(),
        }
    }
    fn record(&self) -> &Arc<crate::resident_program::source_bindings::SourceBindings> {
        match self {
            Self::Reference(input) => input.record(),
            Self::FixedPrompt(input) => input.record(),
        }
    }
    fn ids(&self) -> &[u32] {
        match self {
            Self::Reference(input) => input.ids(),
            Self::FixedPrompt(input) => input.ids(),
        }
    }
    fn call(&self) -> &zerogravity_core::accelerator::program::SourceCall {
        match self {
            Self::Reference(input) => input.call(),
            Self::FixedPrompt(input) => input.call(),
        }
    }
    fn require_session(&self, session: &ResidentSession) -> Result<()> {
        match self {
            Self::Reference(input) => input.require_session(session),
            Self::FixedPrompt(input) => input.require_session(session),
        }
    }
    fn closed_rows(&self) -> bool {
        match self {
            Self::Reference(input) => matches!(input.ids().len(), 1 | 2 | 64),
            Self::FixedPrompt(input) => matches!(
                (input.kind(), input.ids().len()),
                (
                    crate::cuda::native::fixed_prompt_model::FixedPromptCallKind::Prefill,
                    47
                ) | (
                    crate::cuda::native::fixed_prompt_model::FixedPromptCallKind::Decode { .. },
                    1
                )
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
        self.bind_source_rows(SourceRowInput::Reference(input), table)
    }
    pub(crate) fn bind_fixed_prompt_call(
        self: &Arc<Self>,
        input: Arc<
            crate::resident_inference::native_fragment::fixed_prompt_natural::FixedPromptCallInput,
        >,
        table: Arc<TiedWeight>,
    ) -> Result<Arc<Self>> {
        self.bind_source_rows(SourceRowInput::FixedPrompt(input), table)
    }
    fn bind_source_rows(
        self: &Arc<Self>,
        input: SourceRowInput,
        table: Arc<TiedWeight>,
    ) -> Result<Arc<Self>> {
        if self.pooled.is_some()
            || self.natural.is_some()
            || input.call().rows().len() != self.owner.rows.count()
            || !self.owner.source.same_issuer(input.source())
            || !table.source().same_issuer(input.source())
            || input.ids() != input.call().rows().ids()
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
        let matrix = source_embedding(input.record())?;
        if matrix.rows() != 248320
            || matrix.columns() != 1024
            || matrix.tensor().kind != QuantKind::Bf16
        {
            return Err(Error::State(
                "natural actual embedding geometry differs".into(),
            ));
        }
        table
            .matrix()
            .require_matrix(input.source(), &self.service, &matrix)?;
        let host = input.source().reserve_workspace(64 * 1024)?;
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
            text: None,
        }))
    }
    /// Body/input/table+original Write are retained by the SAME armed flight.
    /// Each private slot is claimed once by ColdModelSlots before this wrapper.
    pub(crate) fn enqueue_embedding(
        &self,
        session: &ResidentSession,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        if self.text.is_some() {
            return Err(Error::Unsupported(
                "text embedding requires the actual original call permit".into(),
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
                SourceRowInput::FixedPrompt(input) if input.rows() == 47 => queue
                    .enqueue_fixed_prompt_embedding_rows(
                        rows.table.lease().handle(),
                        rows.destination.handle(),
                        rows.ids.clone(),
                    ),
                _ => queue.enqueue_qwen08_embedding_rows(
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
        let source = rows.input.source();
        let matrix = source_embedding(rows.input.record())?;
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
