//! Ordered original two-member W packs; no captured operand or layout retag.
use super::*;
struct Payload {
    windows: Vec<TensorWindow>,
    bytes: [Vec<u8>; 2],
    descriptors: [Vllm08Operator; 2],
    bound: BoundFullModelSource,
    // Actual source windows/packs/descriptors drop before original admission.
    _host: [WindowWorkspace; 4],
}
pub(crate) struct PendingProjectionWeights {
    payload: Arc<Payload>,
    leases: [ResidentLease; 2],
}
pub(crate) struct ProjectionWeights {
    payload: Arc<Payload>,
    leases: [ResidentLease; 2],
}
impl PendingProjectionWeights {
    pub(crate) fn upload(
        bound: BoundFullModelSource,
        record: &RecordedVllm08Model,
        service: &Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        init.require_service(service)?;
        if bound.model_profile() != FullModelProfile::Qwen2B
            || service.source_identity() != bound.source().source_identity()
        {
            return Err(Error::State(
                "original2B learned upload original source/service".into(),
            ));
        }
        // Covers complete pack + original client.write clone + metadata. Each
        // authenticated TensorWindow retains its OWN complete-chunk admission.
        let source = bound.source().source();
        // Original per-ticket bound stays32MiB. Three complete pack/copy
        // cohorts +64KiB controls fit these FOUR original tickets.
        let host = [
            source.reserve_workspace(32 * 1024 * 1024)?,
            source.reserve_workspace(32 * 1024 * 1024)?,
            source.reserve_workspace(32 * 1024 * 1024)?,
            source.reserve_workspace(512 * 1024)?,
        ];
        let operators = descriptors(record)?;
        let mut bytes = [Vec::new(), Vec::new()];
        let mut windows = Vec::with_capacity(4);
        for (index, (role, operator)) in [Role::Qkvz, Role::Ba]
            .into_iter()
            .zip(&operators)
            .enumerate()
        {
            if require_source(&bound, operator)? != role {
                return Err(Error::State("source2B ordered packed roles".into()));
            }
            bytes[index] = Vec::with_capacity(role.bytes());
            for member in operator.matrices() {
                not_cancelled(cancelled)?;
                let matrix = bound.source().source().matrix(&member.name, member.axes)?;
                if matrix.tensor().kind != zerogravity_core::quant::QuantKind::Bf16
                    || matrix.rows() != member.rows
                    || matrix.columns() != 2048
                    || matrix.source_identity() != bound.source().source_identity()
                {
                    return Err(Error::State(
                        "original2B actual learned source member".into(),
                    ));
                }
                let window = match cancelled {
                    Some(flag) => bound.source().source().read_rows_cancellable(
                        &matrix,
                        0..member.rows,
                        flag,
                    )?,
                    None => bound.source().source().read_rows(&matrix, 0..member.rows)?,
                };
                if window.row_range() != (0..member.rows)
                    || window.as_slice().len() != member.rows * 4096
                    || !super::super::finite_bf16(window.as_slice())
                {
                    return Err(Error::State(
                        "original2B complete finite packed source member".into(),
                    ));
                }
                bytes[index].extend_from_slice(window.as_slice());
                windows.push(window);
            }
            if bytes[index].len() != role.bytes() || bytes[index].capacity() != role.bytes() {
                return Err(Error::Limit("exact2B admitted packed W capacity".into()));
            }
        }
        let payload = Arc::new(Payload {
            windows,
            bytes,
            descriptors: operators,
            bound,
            _host: host,
        });
        // Actual windows/converted storage retained BEFORE either H2D.
        init.retain_completion_owner(payload.clone())?;
        let qkvz = init.upload_bytes(&payload.bytes[0], BufferRole::ImmutableWeights)?;
        not_cancelled(cancelled)?;
        let ba = init.upload_bytes(&payload.bytes[1], BufferRole::ImmutableWeights)?;
        Ok(Self {
            payload,
            leases: [qkvz, ba],
        })
    }
    pub(super) fn bind_pending_original_upload(
        self,
        init: &ResidentInitialization<'_>,
    ) -> Result<Arc<ProjectionWeights>> {
        // Genuine source payload was retained by upload() BEFORE either write.
        // Verify BOTH actual W lease IDs belong to THIS still-armed initializer.
        for lease in &self.leases {
            init.require_qwen2b_projection_upload(lease)?;
        }
        let out = Arc::new(ProjectionWeights {
            payload: self.payload,
            leases: self.leases,
        });
        out.require_source_leases(false)?;
        Ok(out)
    }
    pub(crate) fn bind_initialized(self) -> Result<Arc<ProjectionWeights>> {
        let out = Arc::new(ProjectionWeights {
            payload: self.payload,
            leases: self.leases,
        });
        out.require_initialized()?;
        Ok(out)
    }
}
impl ProjectionWeights {
    pub(crate) fn source(&self) -> &Arc<OriginalWindowModelSource> {
        self.payload.bound.source()
    }
    pub(crate) fn service(&self) -> &Arc<ResidentService> {
        self.leases[0].service()
    }
    pub(super) fn lease(&self, role: Role) -> &ResidentLease {
        &self.leases[usize::from(role == Role::Ba)]
    }
    pub(super) fn require_initialized(&self) -> Result<()> {
        self.require_source_leases(true)
    }
    fn require_source_leases(&self, initialized: bool) -> Result<()> {
        self.service().ensure_usable()?;
        if self.payload.windows.len() != 4 {
            return Err(Error::State(
                "original2B four retained learned windows".into(),
            ));
        }
        for (role, lease) in [Role::Qkvz, Role::Ba].into_iter().zip(&self.leases) {
            if initialized && !lease.is_initialized()
                || lease.bytes() != role.bytes()
                || lease.role() != BufferRole::ImmutableWeights
                || !Arc::ptr_eq(lease.service(), self.service())
                || lease.source_identity() != self.source().source_identity()
            {
                return Err(Error::State(
                    "original2B genuine initialized packed W owner".into(),
                ));
            }
        }
        Ok(())
    }
    pub(super) fn require_operator(&self, operator: &Vllm08Operator) -> Result<Role> {
        self.require_initialized()?;
        self.require_operator_source(operator)
    }
    pub(super) fn require_operator_source(&self, operator: &Vllm08Operator) -> Result<Role> {
        let role = require_source(&self.payload.bound, operator)?;
        let old = &self.payload.descriptors[usize::from(role == Role::Ba)];
        if old.kind() != operator.kind()
            || old.layer() != operator.layer()
            || old.matrices() != operator.matrices()
        {
            return Err(Error::State(
                "original2B W exact learned source/role/member order".into(),
            ));
        }
        Ok(role)
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.leases
            .iter()
            .map(|lease| (lease.clone(), DeviceAccess::Read))
            .collect()
    }
    pub(crate) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        self.leases
            .iter()
            .cloned()
            .zip(self.payload.bytes.iter().map(Vec::as_slice))
            .collect()
    }
    pub(crate) fn reads(&self) -> Result<Vec<crate::resident_leases::ResidentReadback>> {
        self.leases
            .iter()
            .map(|lease| lease.readback(0, lease.bytes()))
            .collect()
    }
    pub(crate) fn validate(&self, raw: &[cubecl::bytes::Bytes]) -> Result<()> {
        if raw.len() != 2
            || raw
                .iter()
                .zip(&self.payload.bytes)
                .any(|(actual, expected)| &actual[..] != expected.as_slice())
        {
            return Err(Error::State(
                "original2B packed W changed after real native work".into(),
            ));
        }
        Ok(())
    }
}
