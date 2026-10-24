//! State-free typed fragment on the original program loop/service/flight.
use super::*;
use crate::resident_leases::resident_matrix::ResidentMatrix;
use crate::resident_leases::{
    ResidentInitialization, ResidentLease, ResidentReadback, ResidentService,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use zerogravity_artifacts::source_weights::{
    ParameterWindow, TensorWindow, VerifiedTensorSource, WindowWorkspace,
};
use zerogravity_core::accelerator::program::RecordedOperatorSlice;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

mod bindings;
mod kernels;
pub(crate) mod qualification;
mod rms;
#[cfg(test)]
mod tests;
use bindings::{Buffer, Shape};

pub(super) struct SourceOwner {
    _embedding: TensorWindow,
    _gamma: ParameterWindow,
    _control_and_upload: WindowWorkspace,
}
pub(crate) struct TypedBody {
    service: Arc<ResidentService>,
    source: VerifiedTensorSource,
    slice: Arc<RecordedOperatorSlice>,
    shape: Shape,
    token: u32,
    weights: Arc<ResidentMatrix>,
    leases: BTreeMap<Buffer, ResidentLease>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    _owner: Arc<SourceOwner>,
    started: AtomicBool,
    next: AtomicUsize,
    commands: AtomicUsize,
    submitted_nodes: AtomicUsize,
    fault: qualification::SliceFault,
}
pub(crate) struct PreparedInputs {
    service: Arc<ResidentService>,
    source: VerifiedTensorSource,
    slice: Arc<RecordedOperatorSlice>,
    shape: Shape,
    token: u32,
    owner: Arc<SourceOwner>,
}
pub(crate) struct PendingBody {
    service: Arc<ResidentService>,
    source: VerifiedTensorSource,
    slice: Arc<RecordedOperatorSlice>,
    shape: Shape,
    token: u32,
    pending: crate::resident_leases::resident_matrix::PendingResidentMatrix,
    leases: BTreeMap<Buffer, ResidentLease>,
    owner: Arc<SourceOwner>,
    fault: qualification::SliceFault,
}
impl PendingBody {
    pub(crate) fn bind(self) -> Result<(ResidentProgram, Arc<TypedBody>)> {
        let weights = self.pending.bind_initialized()?;
        let mut accesses = vec![(weights.lease().clone(), DeviceAccess::Read)];
        for (b, l) in &self.leases {
            accesses.push((
                l.clone(),
                if matches!(b, Buffer::Embedding | Buffer::GammaRaw | Buffer::Zero) {
                    DeviceAccess::Read
                } else {
                    DeviceAccess::Write
                },
            ));
        }
        let body = Arc::new(TypedBody {
            service: self.service,
            source: self.source,
            slice: self.slice,
            shape: self.shape,
            token: self.token,
            weights,
            leases: self.leases,
            accesses,
            _owner: self.owner,
            started: AtomicBool::new(false),
            next: AtomicUsize::new(0),
            commands: AtomicUsize::new(0),
            submitted_nodes: AtomicUsize::new(0),
            fault: self.fault,
        });
        body.validate_runtime(
            body.service.client(),
            RuntimeParameters {
                token: body.token,
                position: 0,
                valid_tokens: 1,
            },
            &[],
        )?;
        let values = [
            (Buffer::Input, body.shape.width),
            (Buffer::Rms, body.shape.width),
            (Buffer::Projection, body.shape.rows),
        ]
        .into_iter()
        .map(|(b, n)| {
            DeviceBuffer::from_lease(
                body.lease(b).clone(),
                ResidentBufferLayout::new(ResidentPhysicalType::F32, n)?,
                BufferRole::Activation,
                DeviceAccess::Write,
            )
        })
        .collect::<Result<Vec<_>>>()?;
        let mut owned_ranges = OwnedRanges::new(body.service.client().service_id());
        let mut ranges = Vec::<BufferRange>::new();
        for (l, _) in body.accesses() {
            let range = BufferRange::new(&l.handle())?;
            range.require_exact_len(l.bytes())?;
            if ranges.iter().any(|prior| prior.overlaps(&range)) {
                return Err(Error::State(
                    "portable typed overlapping logical owners".into(),
                ));
            }
            owned_ranges.insert(range.clone())?;
            ranges.push(range);
        }
        let source_bytes = body.weights.lease().bytes()
            + body.lease(Buffer::Embedding).bytes()
            + body.lease(Buffer::GammaRaw).bytes()
            + 4;
        let value_bytes = (2 * body.shape.width + body.shape.rows) * 4;
        let total = body
            .accesses()
            .iter()
            .map(|(l, _)| l.bytes())
            .sum::<usize>();
        let status_bytes = body.shape.rows * 4 + 4;
        let preflight = ResidentPreflight {
            source: body.source.source_identity(),
            source_bytes,
            value_bytes,
            workspace_bytes: total - source_bytes - value_bytes - status_bytes,
            status_bytes,
            parameter_bytes: 0,
            prior_state_bytes: 0,
            candidate_state_bytes: 0,
            states: Vec::new(),
            scratch: [0; SCRATCH_SLOTS],
        };
        let program = ResidentProgram {
            _source_owner: ResidentProgramSource::TypedWindows(body.source.clone()),
            plan: body.slice.plan().clone(),
            preflight,
            values,
            status: body.lease(Buffer::Status).handle(),
            owned_ranges,
            body: ResidentProgramBody::TypedSlice(body.clone()),
        };
        Ok((program, body))
    }
}
pub(super) fn cancel(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("portable typed slice cancelled".into()));
    }
    Ok(())
}
impl TypedBody {
    pub(crate) fn prepare_inputs(
        service: Arc<ResidentService>,
        source: VerifiedTensorSource,
        slice: Arc<RecordedOperatorSlice>,
        token: u32,
        cancelled: Option<&AtomicBool>,
    ) -> Result<PreparedInputs> {
        cancel(cancelled)?;
        service.ensure_usable()?;
        let shape = Shape::bind(&slice, &source, token)?;
        if service.source_identity() != source.source_identity() || service.client().is_capturing()
        {
            return Err(Error::State("portable typed source/service/capture".into()));
        }
        let p = service.client().properties();
        use cubecl::ir::{ElemType, FloatKind, UIntKind};
        if !p.features.supports_type(ElemType::Float(FloatKind::F64))
            || !p.features.supports_type(ElemType::Float(FloatKind::F32))
            || !p.features.supports_type(ElemType::UInt(UIntKind::U32))
            || p.hardware.max_bindings < 6
            || p.hardware.max_units_per_cube < 64
            || p.hardware.max_cube_dim.0 < 64
            || p.hardware.max_shared_memory_size < 512
            || p.hardware.max_cube_count.0 < shape.rows as u32
        {
            return Err(Error::Unsupported(
                "portable typed source f64/f32/u32/workgroup/binding capability".into(),
            ));
        }
        let layout = shape.layout();
        let bytes = layout
            .iter()
            .try_fold(shape.rows * shape.width * 2, |a, (_, b)| {
                a.checked_add(*b)
                    .ok_or_else(|| Error::Limit("typed slice allocations overflow".into()))
            })?;
        if bytes > 40 * 1024 * 1024
            || service.remaining_capacity()? < bytes
            || layout
                .iter()
                .any(|(_, n)| *n as u64 > p.memory.max_page_size)
        {
            return Err(Error::Limit(
                "portable typed whole managed layout/capacity".into(),
            ));
        }
        let control = source.reserve_workspace(8 * 1024 * 1024)?;
        let em = source.matrix(
            &shape.embedding.name,
            zerogravity_core::accelerator::program::MatrixAxes::OutputInput,
        )?;
        let gp = source.parameter(&shape.gamma.name)?;
        let embedding = match cancelled {
            Some(c) => source.read_rows_cancellable(&em, token as usize..token as usize + 1, c)?,
            None => source.read_rows(&em, token as usize..token as usize + 1)?,
        };
        let gamma = match cancelled {
            Some(c) => source.read_parameter_cancellable(&gp, 0..shape.width, c)?,
            None => source.read_parameter(&gp, 0..shape.width)?,
        };
        if embedding.row_range() != (token as usize..token as usize + 1)
            || embedding.matrix().source_identity() != source.source_identity()
            || embedding.as_slice().len() != 2 * shape.width
            || gamma.parameter().source_parameter() != shape.gamma
            || !gamma.is_complete()
            || gamma.as_slice().len() != 2 * shape.width
        {
            return Err(Error::State(
                "portable typed full original embedding/gamma window".into(),
            ));
        }
        let owner = Arc::new(SourceOwner {
            _embedding: embedding,
            _gamma: gamma,
            _control_and_upload: control,
        });
        Ok(PreparedInputs {
            service,
            source,
            slice,
            shape,
            token,
            owner,
        })
    }
    pub(crate) async fn prepare(
        inputs: PreparedInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
        fault: qualification::SliceFault,
    ) -> Result<PendingBody> {
        cancel(cancelled)?;
        let PreparedInputs {
            service,
            source,
            slice,
            shape,
            token,
            owner,
        } = inputs;
        init.require_service(&service)?;
        init.retain_completion_owner(owner.clone())?;
        let wm = source.matrix(
            &shape.weights.name,
            zerogravity_core::accelerator::program::MatrixAxes::OutputInput,
        )?;
        let pending = init
            .upload_bf16_matrix(source.clone(), wm, cancelled)
            .await?;
        // Matrix binding occurs after the caller's original initialization completes.
        // Keep the pending value in a separate constructor product.
        let mut leases = BTreeMap::new();
        for (role, n) in shape.layout() {
            let lease = match role {
                Buffer::Embedding => {
                    init.upload_bytes(owner._embedding.as_slice(), BufferRole::ImmutableWeights)?
                }
                Buffer::GammaRaw => {
                    init.upload_bytes(owner._gamma.as_slice(), BufferRole::ImmutableWeights)?
                }
                Buffer::Zero => {
                    init.upload_bytes(&0u32.to_le_bytes(), BufferRole::ImmutableWeights)?
                }
                Buffer::Input | Buffer::Rms | Buffer::Projection => {
                    init.allocate_empty(n, BufferRole::Activation)?
                }
                _ => init.allocate_empty(n, BufferRole::Scratch)?,
            };
            leases.insert(role, lease);
        }
        // PendingMatrix is retained until complete_sync by the outer preparation.
        Ok(PendingBody {
            service,
            source,
            slice,
            shape,
            token,
            pending,
            leases,
            owner,
            fault,
        })
    }
    fn lease(&self, b: Buffer) -> &ResidentLease {
        &self.leases[&b]
    }
    pub(super) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    pub(super) fn validate_runtime(
        &self,
        client: &Client,
        p: RuntimeParameters,
        bindings: &[StateBinding],
    ) -> Result<()> {
        self.service.ensure_usable()?;
        if client.service_id() != self.service.client().service_id()
            || self.service.source_identity() != self.source.source_identity()
            || p.token != self.token
            || p.position != 0
            || p.valid_tokens != 1
            || !bindings.is_empty()
        {
            return Err(Error::State(
                "portable typed exact source/current token/empty state".into(),
            ));
        }
        self.weights.require(
            &self.source,
            &self.service,
            self.shape.rows,
            self.shape.width,
        )?;
        for l in self.leases.values() {
            BufferRange::new(&l.handle())?.require_exact_len(l.bytes())?;
            if !Arc::ptr_eq(l.service(), &self.service)
                || l.source_identity() != self.source.source_identity()
                || l.service_generation() != self.service.generation()
                || l.handle().service != client.service_id()
            {
                return Err(Error::State(
                    "portable typed actual lease/source/service/generation".into(),
                ));
            }
        }
        Ok(())
    }
    pub(super) fn begin(&self) -> Result<()> {
        self.service.ensure_usable()?;
        if self.started.swap(true, Ordering::AcqRel) {
            return Err(Error::State("portable typed slice executes once".into()));
        }
        Ok(())
    }
    pub(super) fn require_finished(&self) -> Result<()> {
        if !self.started.load(Ordering::Acquire) || self.next.load(Ordering::Acquire) != 3 {
            return Err(Error::State(
                "portable typed whole source traversal incomplete".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn enqueue(
        &self,
        index: usize,
        node: &PlanNode,
        flag: Option<&AtomicBool>,
    ) -> Result<usize> {
        cancel(flag)?;
        self.service.ensure_usable()?;
        if !self.started.load(Ordering::Acquire)
            || self.next.load(Ordering::Acquire) != index
            || !self
                .slice
                .plan()
                .nodes()
                .get(index)
                .is_some_and(|n| bindings::same_node(n, node))
        {
            return Err(Error::State(
                "portable typed exact current SSA node/order".into(),
            ));
        }
        let n = match index {
            0 => self.enqueue_embedding()?,
            1 => self.enqueue_rms(flag)?,
            2 => self.enqueue_projection()?,
            _ => return Err(Error::State("portable typed source node absent".into())),
        };
        self.commands.fetch_add(n, Ordering::AcqRel);
        self.submitted_nodes.store(index + 1, Ordering::Release);
        if (index == 1 && self.fault == qualification::SliceFault::AfterRms)
            || (index == 2 && self.fault == qualification::SliceFault::AfterProjection)
        {
            return Err(Error::State(
                "deliberate portable typed after actual node submission".into(),
            ));
        }
        self.next
            .compare_exchange(index, index + 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("portable typed concurrent/replayed source node".into()))?;
        Ok(n)
    }
    fn enqueue_embedding(&self) -> Result<usize> {
        let h = self.shape.width;
        unsafe {
            kernels::widen::launch_unchecked(
                self.service.client(),
                grid(h),
                threads(),
                arg(&self.lease(Buffer::Embedding).handle(), h / 2),
                arg(&self.lease(Buffer::Input).handle(), h),
            );
        }
        Ok(1)
    }
    fn enqueue_projection(&self) -> Result<usize> {
        let h = self.shape.width;
        let r = self.shape.rows;
        unsafe {
            crate::bf16_projection_kernel::project_cpu_bf16::launch_unchecked(
                self.service.client(),
                CubeCount::Static(r as u32, 1, 1),
                CubeDim::new_1d(64),
                arg(&self.weights.lease().handle(), r * h / 2),
                arg(&self.lease(Buffer::PackedRms).handle(), h / 2),
                arg(&self.lease(Buffer::Zero).handle(), 1),
                arg(&self.lease(Buffer::Accumulation).handle(), r),
                arg(&self.lease(Buffer::Projection).handle(), r),
                arg(&self.lease(Buffer::ProjectionStatus).handle(), r),
                r,
                h,
            );
        }
        Ok(1)
    }
    pub(crate) fn readbacks(&self) -> Result<Vec<ResidentReadback>> {
        self.observations()
            .into_iter()
            .map(|(_, l)| l.readback(0, l.bytes()))
            .collect()
    }
    pub(crate) fn observations(&self) -> Vec<(&'static str, &ResidentLease)> {
        use Buffer::*;
        [
            ("status", Status),
            ("embedding", Input),
            ("raw_gamma", Gamma),
            ("runtime_gamma", RuntimeGamma),
            ("squares", Squares),
            ("sum", Sum),
            ("mean", Mean),
            ("denominator", Denominator),
            ("sqrt", Sqrt),
            ("inverse", Inverse),
            ("normalized", Normalized),
            ("weighted", Weighted),
            ("rms_bf16", Rms),
            ("rms_packed", PackedRms),
            ("projection_f32", Accumulation),
            ("projection_bf16", Projection),
            ("projection_status", ProjectionStatus),
        ]
        .into_iter()
        .map(|(name, b)| (name, self.lease(b)))
        .collect()
    }
    pub(crate) fn outputs(&self) -> Vec<ResidentLease> {
        self.accesses
            .iter()
            .filter(|(_, a)| *a == DeviceAccess::Write)
            .map(|(l, _)| l.clone())
            .collect()
    }
}

impl ResidentProgram {
    pub(crate) fn launch_typed_slice(
        &mut self,
        flight: &mut crate::resident_leases::ResidentSubmission<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<super::ResidentSubmission> {
        let ResidentProgramBody::TypedSlice(body) = &self.body else {
            return Err(Error::Unsupported("not a typed operator fragment".into()));
        };
        let body = body.clone();
        flight.require_accesses(body.accesses().iter().map(|(l, a)| (l, *a)))?;
        flight.retain_completion_owner(body.clone())?;
        self.launch_impl(
            body.service.client(),
            CurrentExecution::Single {
                parameters: RuntimeParameters {
                    token: body.token,
                    position: 0,
                    valid_tokens: 1,
                },
                states: &[],
            },
            None,
            ResidentOutput::StateOnly,
            NativeExecution::new(cancelled),
        )
    }
}
