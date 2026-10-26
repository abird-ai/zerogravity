//! Whole-B integer layout transport around the actual source GDN pair.
//! No model arithmetic, single-row geometry grant or captured tensor input.
use super::super::*;
use super::values::CohortValues;
use crate::resident_inference::native_fragment::{PrefixPhaseEnqueue, text::TextCallInput};
use crate::resident_leases::{ResidentLease, ResidentService};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Partition {
    BeforeConvolution,
    AfterConvolution,
    BeforeRecurrent,
}
struct Copy {
    partition: Partition,
    source: ResidentLease,
    destination: ResidentLease,
    source_offset: usize,
    source_stride: usize,
    destination_stride: usize,
    rows: usize,
    width: usize,
}
impl Copy {
    fn require(&self, service: &Arc<ResidentService>) -> Result<usize> {
        let words = crate::program_kernels::checked_copy_words(
            self.rows,
            self.width,
            self.source_stride,
            self.destination_stride,
        )?;
        let source_end = (self.rows - 1)
            .checked_mul(self.source_stride)
            .and_then(|n| n.checked_add(self.width))
            .and_then(|n| n.checked_add(self.source_offset));
        let destination_end = (self.rows - 1)
            .checked_mul(self.destination_stride)
            .and_then(|n| n.checked_add(self.width));
        if !Arc::ptr_eq(self.source.service(), service)
            || !Arc::ptr_eq(self.destination.service(), service)
            || !self.source.bytes().is_multiple_of(4)
            || !self.destination.bytes().is_multiple_of(4)
            || source_end.is_none_or(|n| n > self.source.bytes() / 4)
            || destination_end.is_none_or(|n| n > self.destination.bytes() / 4)
            || BufferRange::new(&self.source.handle())?
                .overlaps(&BufferRange::new(&self.destination.handle())?)
        {
            return Err(Error::State(
                "cohort GDN bounded disjoint word transport".into(),
            ));
        }
        Ok(words)
    }
    fn enqueue(&self, service: &Arc<ResidentService>, status: &ResidentLease) -> Result<()> {
        let words = self.require(service)?;
        // SAFETY: checked complete physical views and u32 source/stride domain;
        // the enclosing SAME flight retains all leases and this status epoch.
        unsafe {
            crate::program_kernels::copy_strided_words::launch_unchecked(
                service.client(),
                grid(words),
                threads(),
                arg(&self.source.handle(), self.source.bytes() / 4),
                arg(&self.destination.handle(), self.destination.bytes() / 4),
                arg(&status.handle(), 1),
                self.source_offset as u32,
                0,
                self.source_stride as u32,
                self.destination_stride as u32,
                self.rows,
                self.width,
                false,
            );
        }
        Ok(())
    }
}

pub(super) struct CohortGdnPacking {
    values: Arc<CohortValues>,
    inputs: Box<[Arc<TextCallInput>]>,
    nodes: [usize; 2],
    a: ResidentLease,
    b: ResidentLease,
    // Cached original conv mutates a pitched QKVZ tensor in place. Retain a
    // private pitch so its mutation cannot rewrite the original SSA producer.
    pitched: Option<ResidentLease>,
    status: ResidentLease,
    copies: Box<[Copy]>,
    _metadata: WindowWorkspace,
}
impl CohortGdnPacking {
    /// Every allocation is supplied by the enclosing admitted whole-body
    /// workspace. This binder performs no hidden upload or allocation.
    pub(super) fn bind(
        values: Arc<CohortValues>,
        inputs: &[Arc<TextCallInput>],
        layer: usize,
        a: ResidentLease,
        b: ResidentLease,
        pitched: Option<ResidentLease>,
        status: ResidentLease,
    ) -> Result<Arc<Self>> {
        values.require_inputs(inputs)?;
        crate::cuda::native::cohort_native::require_input_shape(inputs)?;
        let service = values.service();
        if inputs.len() != 4 {
            return Err(Error::Unsupported(
                "GDN actual canonical47 B4 packets only".into(),
            ));
        }
        let initial = (inputs[0].rows(), inputs[0].base()) == (47, 0);
        let m = if initial { 188 } else { 4 };
        let site = inputs[0].scope().gdn_site(layer)?;
        for input in inputs {
            let actual = input.scope().gdn_site(layer)?;
            if input.prompt_tokens() != 47
                || !matches!((input.rows(), input.base()), (47, 0) | (1, 47..=53))
                || input.rows() != inputs[0].rows()
                || actual.nodes != site.nodes
                || actual.operands != site.operands
            {
                return Err(Error::State(
                    "GDN every real row source pair/frontier".into(),
                ));
            }
        }
        if [(&a, m * 16 * 2), (&b, m * 16 * 2), (&status, 4)]
            .into_iter()
            .any(|(lease, bytes)| {
                lease.bytes() != bytes
                    || !Arc::ptr_eq(lease.service(), service)
                    || lease.role() != BufferRole::Scratch
            })
            || !status.is_initialized()
        {
            return Err(Error::State(
                "GDN actual contiguous B A/B and status extents".into(),
            ));
        }
        match (&pitched, initial) {
            (None, true) => {}
            (Some(lease), false)
                if lease.bytes() == 4 * 8192 * 2
                    && Arc::ptr_eq(lease.service(), service)
                    && lease.role() == BufferRole::Scratch => {}
            _ => return Err(Error::State("GDN cached private8192 pitch only".into())),
        }
        let [qkvz, ba, conv, z, _] = site.operands;
        let qkvz = values.lease(qkvz)?.clone();
        let ba = values.lease(ba)?.clone();
        let conv = values.lease(conv)?.clone();
        let z = values.lease(z)?.clone();
        if qkvz.bytes() != m * 8192 * 2
            || ba.bytes() != m * 32 * 2
            || conv.bytes() != m * 6144 * 2
            || z.bytes() != m * 2048 * 2
        {
            return Err(Error::State(
                "GDN exact physical B source SSA extents".into(),
            ));
        }
        let mut scratch = vec![a.clone(), b.clone(), status.clone()];
        scratch.extend(pitched.iter().cloned());
        for (index, lease) in scratch.iter().enumerate() {
            let range = BufferRange::new(&lease.handle())?;
            if scratch[..index]
                .iter()
                .any(|other| BufferRange::new(&other.handle()).is_ok_and(|r| range.overlaps(&r)))
            {
                return Err(Error::State("GDN private transport aliases scratch".into()));
            }
            for value in values.buffers() {
                if range.overlaps(&BufferRange::new(&value.handle)?) {
                    return Err(Error::State(
                        "GDN private transport aliases any SSA value".into(),
                    ));
                }
            }
        }
        let mut copies = vec![
            Copy {
                partition: Partition::BeforeConvolution,
                source: qkvz.clone(),
                destination: z,
                source_offset: 3072,
                source_stride: 4096,
                destination_stride: 1024,
                rows: m,
                width: 1024,
            },
            Copy {
                partition: Partition::BeforeRecurrent,
                source: ba.clone(),
                destination: b.clone(),
                source_offset: 0,
                source_stride: 16,
                destination_stride: 8,
                rows: m,
                width: 8,
            },
            Copy {
                partition: Partition::BeforeRecurrent,
                source: ba,
                destination: a.clone(),
                source_offset: 8,
                source_stride: 16,
                destination_stride: 8,
                rows: m,
                width: 8,
            },
        ];
        if let Some(private) = &pitched {
            copies.push(Copy {
                partition: Partition::BeforeConvolution,
                source: qkvz.clone(),
                destination: private.clone(),
                source_offset: 0,
                source_stride: 4096,
                destination_stride: 4096,
                rows: 4,
                width: 4096,
            });
            copies.push(Copy {
                partition: Partition::AfterConvolution,
                source: private.clone(),
                destination: conv,
                source_offset: 0,
                source_stride: 4096,
                destination_stride: 3072,
                rows: 4,
                width: 3072,
            });
        }
        for copy in &copies {
            copy.require(service)?;
        }
        let metadata = inputs[0].source().reserve_workspace(32 * 1024)?;
        let nodes = site.nodes;
        Ok(Arc::new(Self {
            values,
            inputs: inputs.to_vec().into_boxed_slice(),
            nodes,
            a,
            b,
            pitched,
            status,
            copies: copies.into_boxed_slice(),
            _metadata: metadata,
        }))
    }
    pub(super) fn require_storage(
        &self,
        values: &Arc<CohortValues>,
        inputs: &[Arc<TextCallInput>],
        a: &ResidentLease,
        b: &ResidentLease,
        pitched: Option<&ResidentLease>,
    ) -> Result<()> {
        if !Arc::ptr_eq(values, &self.values)
            || self.a.id() != a.id()
            || self.b.id() != b.id()
            || self.inputs.len() != inputs.len()
            || self
                .inputs
                .iter()
                .zip(inputs)
                .any(|(a, b)| !Arc::ptr_eq(a, b))
            || self.pitched.as_ref().map(ResidentLease::id) != pitched.map(ResidentLease::id)
        {
            return Err(Error::State(
                "B4 GDN packing current inputs/storage owners".into(),
            ));
        }
        self.values.require_inputs(inputs)
    }
    pub(super) fn enqueue(
        &self,
        node: usize,
        partition: Partition,
        permits: &[PrefixPhaseEnqueue<'_>],
    ) -> Result<usize> {
        let expected = if partition == Partition::BeforeRecurrent {
            self.nodes[1]
        } else {
            self.nodes[0]
        };
        if node != expected || permits.len() != self.inputs.len() {
            return Err(Error::State(
                "GDN transport actual original node/all row permits".into(),
            ));
        }
        self.values.require_inputs(&self.inputs)?;
        for (input, permit) in self.inputs.iter().zip(permits) {
            input.require_permit(permit, self.values.service())?;
        }
        let mut count = 0;
        for copy in self
            .copies
            .iter()
            .filter(|copy| copy.partition == partition)
        {
            copy.enqueue(self.values.service(), &self.status)?;
            count += 1;
        }
        Ok(count)
    }
    pub(super) fn accesses(&self, partition: Partition) -> Vec<(ResidentLease, DeviceAccess)> {
        self.copies
            .iter()
            .filter(|copy| copy.partition == partition)
            .flat_map(|copy| {
                [
                    (copy.source.clone(), DeviceAccess::Read),
                    (copy.destination.clone(), DeviceAccess::Write),
                ]
            })
            .chain([(self.status.clone(), DeviceAccess::Write)])
            .collect()
    }
}
