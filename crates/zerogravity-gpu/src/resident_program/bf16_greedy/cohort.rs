//! Full[B,V] numeric BF16 consumer; no root, native producer or Ready grant.
use super::*;

/// Call-local. Fresh row packets may escape through genuine completion; the
/// packed buffer and partials never become committed token feedback owners.
pub(crate) struct PreparedCohortGreedy {
    source: VerifiedTensorSource,
    records: Box<[Arc<SourceBindings>]>,
    service: Arc<ResidentService>,
    logits: ResidentLease,
    status: ResidentLease,
    partials: ResidentLease,
    packed: ResidentLease,
    packets: Box<[ResidentLease]>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    _metadata: WindowWorkspace,
}
impl PreparedCohortGreedy {
    pub(crate) fn allocation_layout(width: usize) -> Result<Vec<usize>> {
        Self::managed_bytes(width)?;
        Ok([width * PARTIAL_BYTES, width * PACKET_BYTES]
            .into_iter()
            .chain(std::iter::repeat_n(PACKET_BYTES, width))
            .collect())
    }
    pub(crate) fn managed_bytes(width: usize) -> Result<usize> {
        if !matches!(width, 4 | 8) {
            return Err(Error::Unsupported(
                "BF16 original cohort4/8 consumer".into(),
            ));
        }
        width
            .checked_mul(PARTIAL_BYTES + 2 * PACKET_BYTES)
            .ok_or_else(|| Error::Limit("BF16 cohort reducer byte bound".into()))
    }
    /// Only the enclosing sealed body supplies its actual whole[B,V] producer
    /// and status. Every original logical Head record is validated separately.
    pub(crate) fn prepare(
        source: VerifiedTensorSource,
        records: &[Arc<SourceBindings>],
        logits: ResidentLease,
        status: ResidentLease,
        init: &mut ResidentInitialization<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancel)?;
        let width = records.len();
        let managed = Self::managed_bytes(width)?;
        let service = logits.service().clone();
        init.require_service(&service)?;
        for record in records {
            require_head(record)?;
            if record.call().source() != source.source_identity()
                || !source.same_issuer(record.source())
                || record.selection().output() != records[0].selection().output()
            {
                return Err(Error::State("BF16 cohort every actual source Head".into()));
            }
        }
        if service.remaining_capacity()? < managed {
            return Err(Error::Limit(
                "BF16 cohort complete reducer buffer admission".into(),
            ));
        }
        let metadata = source.reserve_workspace(METADATA_BYTES)?;
        let partials = init.allocate_empty(width * PARTIAL_BYTES, BufferRole::Scratch)?;
        let packed = init.allocate_empty(width * PACKET_BYTES, BufferRole::Scratch)?;
        let mut packets = Vec::with_capacity(width);
        for _ in 0..width {
            packets.push(init.allocate_empty(PACKET_BYTES, BufferRole::Output)?);
        }
        let mut result = Self {
            source,
            records: records.to_vec().into_boxed_slice(),
            service,
            logits,
            status,
            partials,
            packed,
            packets: packets.into_boxed_slice(),
            accesses: Vec::with_capacity(width + 4),
            _metadata: metadata,
        };
        result.validate_bindings()?;
        result.accesses.extend([
            (result.logits.clone(), DeviceAccess::Read),
            (result.status.clone(), DeviceAccess::Write),
            (result.partials.clone(), DeviceAccess::Write),
            (result.packed.clone(), DeviceAccess::Write),
        ]);
        result.accesses.extend(
            result
                .packets
                .iter()
                .cloned()
                .map(|p| (p, DeviceAccess::Write)),
        );
        let result = Arc::new(result);
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    fn validate_bindings(&self) -> Result<()> {
        self.service.ensure_usable()?;
        let width = self.records.len();
        Self::managed_bytes(width)?;
        if self.source.source_identity() != self.service.source_identity()
            || self.packets.len() != width
            || !matches!(
                self.logits.role(),
                BufferRole::Activation | BufferRole::Output
            )
            || self.status.role() != BufferRole::Scratch
            || self.partials.role() != BufferRole::Scratch
            || self.packed.role() != BufferRole::Scratch
        {
            return Err(Error::State(
                "BF16 cohort retained source/typed consumer roles".into(),
            ));
        }
        let mut ranges = Vec::with_capacity(width + 4);
        let fixed = [
            (&self.logits, width * LOGITS_BYTES),
            (&self.status, 4),
            (&self.partials, width * PARTIAL_BYTES),
            (&self.packed, width * PACKET_BYTES),
        ];
        for (lease, bytes) in fixed
            .into_iter()
            .chain(self.packets.iter().map(|p| (p, PACKET_BYTES)))
        {
            if !Arc::ptr_eq(lease.service(), &self.service) || lease.bytes() != bytes {
                return Err(Error::State(
                    "BF16 cohort exact current physical extent/service".into(),
                ));
            }
            let range = BufferRange::new(&lease.handle())?;
            range.require_exact_len(bytes)?;
            if ranges
                .iter()
                .any(|prior: &BufferRange| range.overlaps(prior))
            {
                return Err(Error::State("BF16 cohort reducer storage overlaps".into()));
            }
            ranges.push(range);
        }
        if self.packets.iter().any(|p| p.role() != BufferRole::Output) {
            return Err(Error::State(
                "BF16 cohort fresh row Output packet roles".into(),
            ));
        }
        for record in &self.records {
            if record.call().source() != self.source.source_identity() {
                return Err(Error::State("BF16 cohort row source changed".into()));
            }
        }
        Ok(())
    }
    /// After real Head node and its certificates, on the SAME original flight.
    pub(crate) fn enqueue(&self, cancel: Option<&AtomicBool>) -> Result<usize> {
        not_cancelled(cancel)?;
        self.validate_bindings()?;
        let width = self.records.len();
        // SAFETY: exact B extents/disjoint ranges checked and same-flight owner
        // retained by enclosing body before node0; packed BF16 loads only.
        unsafe {
            kernels::cohort_block_winners::launch_unchecked(
                self.service.client(),
                CubeCount::Static(((width * kernels::BLOCKS) as u32).div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(self.logits.handle(), width * kernels::VOCABULARY / 2),
                BufferArg::from_raw_parts(self.partials.handle(), width * kernels::PARTIAL_WORDS),
                BufferArg::from_raw_parts(self.status.handle(), 1),
                width,
            );
        }
        not_cancelled(cancel)?;
        // SAFETY: preceding first stage covers every row on this same stream.
        unsafe {
            kernels::cohort_finish_winners::launch_unchecked(
                self.service.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(self.partials.handle(), width * kernels::PARTIAL_WORDS),
                BufferArg::from_raw_parts(self.packed.handle(), width * 3),
                BufferArg::from_raw_parts(self.status.handle(), 1),
                width,
            );
        }
        // Preserve original-bit transport into independently owned immutable
        // per-row packets. No workspace subview escapes into feedback.
        for (row, packet) in self.packets.iter().enumerate() {
            not_cancelled(cancel)?;
            // SAFETY: exact checked dense words; integer copy, no F32 coercion.
            unsafe {
                crate::program_kernels::copy_strided_words::launch_unchecked(
                    self.service.client(),
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(self.packed.handle(), width * 3),
                    BufferArg::from_raw_parts(packet.handle(), 3),
                    BufferArg::from_raw_parts(self.status.handle(), 1),
                    (row * 3) as u32,
                    0,
                    3,
                    3,
                    1,
                    3,
                    false,
                );
            }
        }
        Ok(2 + width)
    }
    pub(crate) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    pub(crate) fn packets(&self) -> &[ResidentLease] {
        &self.packets
    }
    pub(crate) fn readbacks(&self) -> Result<Vec<ResidentReadback>> {
        let mut reads = self
            .packets
            .iter()
            .map(|p| p.readback(0, PACKET_BYTES))
            .collect::<Result<Vec<_>>>()?;
        reads.push(self.status.readback(0, 4)?);
        Ok(reads)
    }
    /// Only original trusted physical completion calls this parser.
    pub(crate) fn validate_readback(&self, raw: &[Bytes]) -> Result<Vec<(u32, u32)>> {
        if raw.len() != self.records.len() + 1 {
            return Err(Error::State(
                "BF16 cohort exact row packet/status readbacks".into(),
            ));
        }
        let status = raw.last().unwrap();
        raw[..self.records.len()]
            .iter()
            .map(|packet| parse_packet(packet, status))
            .collect()
    }
}
