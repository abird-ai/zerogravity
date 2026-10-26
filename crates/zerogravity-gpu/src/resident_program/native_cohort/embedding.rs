//! Source-issued IDs select the SAME immutable tied BF16 table, without math.
use super::super::*;
use super::values::CohortValues;
use crate::resident_inference::native_fragment::{PrefixPhaseEnqueue, text::TextCallInput};
use crate::resident_leases::{ResidentInitialization, ResidentLease, TiedWeight};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

enum Ids {
    Prompt(ResidentLease),
    Feedback([ResidentLease; 4]),
}
impl Ids {
    fn leases(&self) -> &[ResidentLease] {
        match self {
            Self::Prompt(lease) => std::slice::from_ref(lease),
            Self::Feedback(leases) => leases,
        }
    }
}
pub(super) struct CohortEmbedding {
    inputs: Box<[Arc<TextCallInput>]>,
    values: Arc<CohortValues>,
    tied: Arc<TiedWeight>,
    ids: Ids,
    output: ResidentLease,
    status: ResidentLease,
    rows: usize,
    width_words: usize,
    submitted: AtomicBool,
    _metadata: Arc<WindowWorkspace>,
}
impl CohortEmbedding {
    pub(super) fn prepare(
        inputs: &[Arc<TextCallInput>],
        values: Arc<CohortValues>,
        tied: Arc<TiedWeight>,
        status: ResidentLease,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        values.require_inputs(inputs)?;
        crate::cuda::native::cohort_native::require_input_shape(inputs)?;
        init.require_service(values.service())?;
        if inputs.len() != 4
            || status.bytes() != 4
            || status.role() != BufferRole::Scratch
            || !Arc::ptr_eq(status.service(), values.service())
        {
            return Err(Error::State("B4 source embedding rows/status".into()));
        }
        let hidden = inputs[0].scope().native_capture().hidden_width();
        let rows = inputs[0].rows() * 4;
        if !matches!((inputs[0].rows(), inputs[0].base()), (47, 0) | (1, 47..=53))
            || !matches!(rows, 4 | 188)
        {
            return Err(Error::Unsupported(
                "B4 source embedding canonical47 initial/cache".into(),
            ));
        }
        let mut ids = if rows == 188 {
            Vec::with_capacity(rows * 4)
        } else {
            Vec::new()
        };
        let mut producer = None;
        for input in inputs {
            let node = &input.record().plan().nodes()[0];
            // Same complete typed source/SSA/profile proof as the N1 path;
            // keep each row's original call/teacher issuer and record ownership.
            input
                .scope()
                .require_record(input.source(), input.record())?;
            if input.rows() != inputs[0].rows()
                || input.prompt_tokens() != 47
                || input.ids().len() != input.rows()
                || input.ids().iter().any(|id| *id >= 248320)
                || !input.source().same_issuer(tied.source())
                || node.outputs.len() != 1
                || producer.is_some_and(|id| id != node.outputs[0])
            {
                return Err(Error::State(
                    "B4 every original embedding producer/source-issued IDs".into(),
                ));
            }
            let descriptor = input.record().operator(0)?.ok_or_else(|| {
                Error::State("B4 actual embedding source descriptor absent".into())
            })?;
            if descriptor.kind()
                != zerogravity_core::accelerator::program::SourceOperatorKind::EmbeddingRows
            {
                return Err(Error::State("B4 actual embedding operator differs".into()));
            }
            super::require_tied_member(input, descriptor, &tied, values.service())?;
            producer = Some(node.outputs[0]);
            if rows == 188 {
                for id in input.ids() {
                    ids.extend_from_slice(&id.to_le_bytes());
                }
            }
        }
        let table = tied.matrix().lease();
        if !Arc::ptr_eq(table.service(), values.service())
            || !table.is_initialized()
            || table.role() != BufferRole::ImmutableWeights
            || table.bytes() != 248320 * hidden * 2
            || tied.copy_counts()? != (tied.windows(), tied.windows())
        {
            return Err(Error::State(
                "B4 SAME original complete resident tied table".into(),
            ));
        }
        tied.matrix()
            .require(tied.source(), values.service(), 248320, hidden)?;
        let output = values.lease(producer.expect("four actual rows"))?.clone();
        if output.bytes() != rows * hidden * 2 || output.role() != BufferRole::Activation {
            return Err(Error::State(
                "B4 exact physical source embedding output".into(),
            ));
        }
        for other in [table, &status] {
            if BufferRange::new(&output.handle())?.overlaps(&BufferRange::new(&other.handle())?) {
                return Err(Error::State(
                    "B4 embedding output aliases table/status".into(),
                ));
            }
        }
        let metadata = Arc::new(inputs[0].source().reserve_workspace(64 * 1024)?);
        init.retain_completion_owner(metadata.clone())?;
        init.retain_completion_owner(tied.clone())?;
        let ids = if rows == 188 {
            Ids::Prompt(init.upload_bytes(&ids, BufferRole::ImmutableWeights)?)
        } else {
            let leases = inputs
                .iter()
                .map(|input| {
                    let lease = input.feedback_lease()?.ok_or_else(|| {
                        Error::State("B4 cached embedding requires genuine feedback".into())
                    })?;
                    if !Arc::ptr_eq(lease.service(), values.service())
                        || !lease.is_initialized()
                        || lease.role() != BufferRole::Output
                        || lease.bytes() != 12
                    {
                        return Err(Error::State(
                            "B4 original completed feedback owner/extent".into(),
                        ));
                    }
                    for other in [table, &output, &status] {
                        if BufferRange::new(&lease.handle())?
                            .overlaps(&BufferRange::new(&other.handle())?)
                        {
                            return Err(Error::State(
                                "B4 feedback aliases source/output/status".into(),
                            ));
                        }
                    }
                    Ok(lease.clone())
                })
                .collect::<Result<Vec<_>>>()?;
            for (index, lease) in leases.iter().enumerate() {
                for other in &leases[..index] {
                    if BufferRange::new(&lease.handle())?
                        .overlaps(&BufferRange::new(&other.handle())?)
                    {
                        return Err(Error::State(
                            "B4 independent feedback packets overlap".into(),
                        ));
                    }
                }
            }
            Ids::Feedback(
                leases
                    .try_into()
                    .map_err(|_| Error::State("B4 four feedback owners".into()))?,
            )
        };
        let result = Arc::new(Self {
            inputs: inputs.to_vec().into_boxed_slice(),
            values,
            tied,
            ids,
            output,
            status,
            rows,
            width_words: hidden / 2,
            submitted: AtomicBool::new(false),
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(super) fn enqueue(&self, node: usize, permits: &[PrefixPhaseEnqueue<'_>]) -> Result<usize> {
        if node != 0 || permits.len() != 4 {
            return Err(Error::State(
                "B4 embedding original node/all-row permits".into(),
            ));
        }
        self.values.require_inputs(&self.inputs)?;
        for (input, permit) in self.inputs.iter().zip(permits) {
            input.require_permit(permit, self.values.service())?;
        }
        if self.submitted.swap(true, Ordering::AcqRel) {
            return Err(Error::State("B4 source embedding already consumed".into()));
        }
        let table = self.tied.matrix().lease();
        let words = self.rows * self.width_words;
        match &self.ids {
            Ids::Prompt(ids) => unsafe {
                crate::program_kernels::cohort_copy::gather_rows::launch_unchecked(
                    self.values.service().client(),
                    grid(words),
                    threads(),
                    arg(&table.handle(), table.bytes() / 4),
                    arg(&ids.handle(), self.rows),
                    arg(&self.output.handle(), words),
                    arg(&self.status.handle(), 1),
                    self.rows,
                    self.width_words,
                );
            },
            Ids::Feedback([a, b, c, d]) => unsafe {
                crate::program_kernels::cohort_copy::gather_feedback_rows::launch_unchecked(
                    self.values.service().client(),
                    grid(words),
                    threads(),
                    arg(&table.handle(), table.bytes() / 4),
                    arg(&a.handle(), 3),
                    arg(&b.handle(), 3),
                    arg(&c.handle(), 3),
                    arg(&d.handle(), 3),
                    arg(&self.output.handle(), words),
                    arg(&self.status.handle(), 1),
                    self.width_words,
                );
            },
        }
        Ok(1)
    }
    pub(super) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut accesses = vec![
            (self.tied.matrix().lease().clone(), DeviceAccess::Read),
            (self.output.clone(), DeviceAccess::Write),
            (self.status.clone(), DeviceAccess::Write),
        ];
        accesses.extend(
            self.ids
                .leases()
                .iter()
                .cloned()
                .map(|lease| (lease, DeviceAccess::Read)),
        );
        accesses
    }
    /// Node-0 census attribution: the same per-node access list the flight
    /// union carries for node 0 (tied table read, source-issued IDs read,
    /// the embedding value write and the status).
    pub(super) fn node_accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.accesses()
    }
}
