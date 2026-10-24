//! Frozen actual-ID transport; original flight/completion owns publication.
use super::*;
use cubecl::prelude::*;
use layout::CONTROL;
use zerogravity_artifacts::source_weights::WindowWorkspace;

pub(crate) struct PreparedMatrixRows {
    matrix: Arc<ResidentMatrix>,
    output: ResidentLease,
    ids_buffer: ResidentLease,
    status: ResidentLease,
    ids: Arc<[u32]>,
    encoded: Bytes,
    submitted: AtomicBool,
    _control: WindowWorkspace,
}
impl ResidentMatrix {
    pub(crate) fn prepare_rows(
        self: &Arc<Self>,
        ids: Arc<[u32]>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PreparedMatrixRows>> {
        if cancelled.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err(Error::State("verified matrix rows cancelled".into()));
        }
        let service = self.lease.service();
        self.require(
            &self.source,
            service,
            self.layout.rows(),
            self.layout.row_bytes() / 2,
        )?;
        require_portable(service, self.layout)?;
        let bytes = self.layout.output_bytes(&ids)?;
        if bytes as u64 > service.client().properties().memory.max_page_size {
            return Err(Error::Unsupported(
                "BF16 output storage binding/page limit".into(),
            ));
        }
        let id_bytes = ids.len() * 4;
        // Frozen Arc payload,serialized Bytes and runtime bounce; other owned
        // metadata uses CONTROL. Caller admits output/status readback separately.
        let control = self
            .source
            .reserve_workspace(CONTROL + id_bytes + 2 * id_bytes.next_multiple_of(16))?;
        let encoded = encode_ids(&ids)?;
        let output = service.allocate_uninitialized(bytes, BufferRole::Activation)?;
        let ids_buffer = service.allocate_uninitialized(id_bytes, BufferRole::Scratch)?;
        let status = service.allocate_uninitialized(id_bytes, BufferRole::Scratch)?;
        let result = Arc::new(PreparedMatrixRows {
            matrix: self.clone(),
            output,
            ids_buffer,
            status,
            ids,
            encoded,
            submitted: AtomicBool::new(false),
            _control: control,
        });
        disjoint(&[
            result.matrix.lease(),
            &result.output,
            &result.ids_buffer,
            &result.status,
        ])?;
        Ok(result)
    }
}
impl PreparedMatrixRows {
    #[cfg(test)]
    pub(super) fn inject_invalid_first(&mut self) {
        let mut bytes = self.encoded.to_vec();
        bytes[..4].copy_from_slice(&(self.matrix.layout.rows() as u32).to_le_bytes());
        self.encoded = Bytes::from_bytes_vec(bytes).shared();
    }
    pub(crate) fn output(&self) -> &ResidentLease {
        &self.output
    }
    pub(crate) fn accesses(&self) -> [(ResidentLease, DeviceAccess); 4] {
        [
            (self.matrix.lease.clone(), DeviceAccess::Read),
            (self.output.clone(), DeviceAccess::Write),
            (self.ids_buffer.clone(), DeviceAccess::Write),
            (self.status.clone(), DeviceAccess::Write),
        ]
    }
    pub(crate) fn readbacks(&self) -> Result<[ResidentReadback; 2]> {
        Ok([
            self.output.readback(0, self.output.bytes())?,
            self.status.readback(0, self.status.bytes())?,
        ])
    }
    pub(crate) fn validate(&self, bytes: &[Bytes]) -> Result<()> {
        if bytes.len() != 2
            || bytes[0].len() != self.output.bytes()
            || bytes[1].len() != self.status.bytes()
            || bytes[1].chunks_exact(4).any(|b| b != [0, 0, 0, 0])
        {
            return Err(Error::State(
                "BF16 matrix row transport status/extent".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn enqueue(self: &Arc<Self>, flight: &mut ResidentSubmission<'_>) -> Result<usize> {
        let service = self.output.service();
        flight.require_accesses(self.accesses().iter().map(|(l, a)| (l, *a)))?;
        self.matrix.require(
            &self.matrix.source,
            service,
            self.matrix.layout.rows(),
            self.matrix.layout.row_bytes() / 2,
        )?;
        disjoint(&[
            self.matrix.lease(),
            &self.output,
            &self.ids_buffer,
            &self.status,
        ])?;
        if service.client().is_capturing() || self.submitted.swap(true, Ordering::AcqRel) {
            return Err(Error::State("BF16 rows one-use noncaptured flight".into()));
        }
        flight.retain_completion_owner(self.clone())?;
        service
            .client()
            .write(&self.ids_buffer.handle(), self.encoded.clone());
        service.record_value_upload(self.encoded.len());
        let words = self.output.bytes() / 4;
        // SAFETY: exact complete original leases,byte extents,ID domain and
        // nonaliasing were checked before command1; the armed flight retains all.
        unsafe {
            crate::program_kernels::gather_bf16_matrix_words::launch_unchecked(
                service.client(),
                CubeCount::Static((words as u32).div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(
                    self.matrix.lease.handle(),
                    self.matrix.layout.bytes() / 4,
                ),
                BufferArg::from_raw_parts(self.ids_buffer.handle(), self.ids.len()),
                BufferArg::from_raw_parts(self.output.handle(), words),
                BufferArg::from_raw_parts(self.status.handle(), self.ids.len()),
                self.matrix.layout.rows() as u32,
                (self.matrix.layout.row_bytes() / 4) as u32,
            );
        }
        Ok(2) // one actual ID H2D and one actual integer gather
    }
}
fn encode_ids(ids: &[u32]) -> Result<Bytes> {
    let bytes = ids.len() * 4;
    let mut encoded = Vec::with_capacity(bytes);
    for id in ids {
        encoded.extend_from_slice(&id.to_le_bytes());
    }
    if encoded.capacity() != bytes {
        return Err(Error::Limit("BF16 ID copy capacity".into()));
    }
    let encoded = Bytes::from_bytes_vec(encoded);
    // Check the underlying allocation before shared() reports its view length.
    if encoded.capacity() > bytes.next_multiple_of(16) {
        return Err(Error::Limit("BF16 aligned ID copy capacity".into()));
    }
    Ok(encoded.shared())
}
fn disjoint(leases: &[&ResidentLease]) -> Result<()> {
    use crate::resident_program::BufferRange;
    let service = leases
        .first()
        .ok_or_else(|| Error::State("BF16 transport empty access union".into()))?
        .service();
    for (i, a) in leases.iter().enumerate() {
        service.check_lease(a)?;
        let handle = a.handle();
        let range = BufferRange::new(&handle)?;
        range.require_exact_len(a.bytes())?;
        #[cfg(test)]
        eprintln!(
            "MATRIX_TRANSPORT_VIEW logical={} backing={} start={} tail={} range={range:?}",
            a.bytes(),
            handle.size(),
            handle.offset_start.unwrap_or(0),
            handle.offset_end.unwrap_or(0)
        );
        for b in &leases[..i] {
            if a.id() == b.id() || range.overlaps(&BufferRange::new(&b.handle())?) {
                return Err(Error::State(
                    "BF16 transport overlapping logical buffers".into(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_ids_keep_order_duplicates_bits_and_bounded_alignment() {
        for ids in [vec![7], vec![9, 0, 9], vec![u32::MAX; 65_536]] {
            let encoded = encode_ids(&ids).unwrap();
            assert_eq!(encoded.len(), ids.len() * 4);
            let actual: Vec<_> = encoded
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(actual, ids);
            let retained = encoded.clone();
            drop(encoded);
            assert_eq!(retained.len(), ids.len() * 4);
        }
    }

    #[test]
    fn exact_padded_and_disjoint_shared_slices_preserve_overlap_refusals() {
        use crate::resident_program::BufferRange;
        use cubecl_common::device::{DeviceId, ServiceId};
        use cubecl_core::environment::stream::StreamId;
        let service = ServiceId::of::<()>(DeviceId::new(0, 0));
        let base = cubecl::server::Handle::new(service, StreamId { value: 0 }, 256);
        let a = base.clone().offset_start(16).offset_end(224);
        let b = base.clone().offset_start(32).offset_end(208);
        let ar = BufferRange::new(&a).unwrap();
        let br = BufferRange::new(&b).unwrap();
        ar.require_exact_len(16).unwrap();
        br.require_exact_len(16).unwrap();
        assert!(!ar.overlaps(&br));
        assert!(ar.overlaps(&BufferRange::new(&a).unwrap()));
        assert!(
            ar.overlaps(&BufferRange::new(&base.clone().offset_start(31).offset_end(209)).unwrap())
        );
        assert!(ar.require_exact_len(15).is_err());
        assert!(ar.require_exact_len(17).is_err());
        assert!(BufferRange::new(&base.clone().offset_start(256)).is_err());
        assert!(BufferRange::new(&base.clone().offset_end(257)).is_err());
        assert!(BufferRange::new(&base.offset_start(32).offset_end(225)).is_err());
    }
}
