//! Actual original layer3 Sig/Mul nodes inside the original body flight.
use super::*;
use crate::vllm_cuda_qualification::attention_cubin::canonical47_layer3_packets;

pub(super) struct Programs {
    cut: Arc<Canonical47PrefixCut>,
    values: Arc<ModelValues>,
    kernels: [Arc<OwnedCudaKernel>; 2],
    counts: [AtomicUsize; 2],
    _metadata: Arc<WindowWorkspace>,
}
impl Programs {
    pub(super) fn prepare(
        cut: Arc<Canonical47PrefixCut>,
        values: Arc<ModelValues>,
        images: &Arc<PreparedImages>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        cut.require_layer3_tail()?;
        cut.scope().require_values(&values)?;
        init.require_service(values.service())?;
        let metadata = Arc::new(
            cut.scope()
                .source()
                .source()
                .source()
                .reserve_workspace(128 * 1024)?,
        );
        init.retain_completion_owner(metadata.clone())?;
        init.retain_completion_owner(cut.clone())?;
        init.retain_completion_owner(values.clone())?;
        let mut kernels = Vec::with_capacity(2);
        for (kind, manifest) in canonical47_layer3_packets::packets()? {
            not_cancelled(cancelled)?;
            let image = images.image(kind)?;
            // SAFETY: same original set/init; source-owned real producer leases retained.
            let kernel = unsafe { images.set().prepare(image, manifest) }.map_err(native_error)?;
            init.retain_completion_owner(kernel.clone())?;
            kernels.push(kernel);
        }
        let kernels: [_; 2] = kernels
            .try_into()
            .map_err(|_| Error::State("canonical47 exactly two gate records".into()))?;
        let result = Arc::new(Self {
            cut,
            values,
            kernels,
            counts: [AtomicUsize::new(0), AtomicUsize::new(0)],
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(super) fn accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        [44u32, 45, 46, 47]
            .into_iter()
            .map(|id| {
                Ok((
                    self.values.lease(ValueId(id))?.clone(),
                    if id >= 46 {
                        DeviceAccess::Write
                    } else {
                        DeviceAccess::Read
                    },
                ))
            })
            .collect()
    }
    pub(super) fn enqueue(
        &self,
        index: usize,
        cancelled: Option<&AtomicBool>,
        fault: usize,
    ) -> Result<usize> {
        self.cut.require_layer3_tail()?;
        self.cut.scope().require_values(&self.values)?;
        if !matches!(index, 36 | 37) || !matches!(fault, 0 | 11 | 12) {
            return Err(Error::State(
                "canonical47 source gate cursor/fault differs".into(),
            ));
        }
        let which = usize::from(index == 37);
        let expected = if which == 0 { [0, 0] } else { [1, 0] };
        if self.counts() != expected || !self.outputs_unpromoted()? {
            return Err(Error::State(
                "canonical47 gate original SSA/order/unpromoted output differs".into(),
            ));
        }
        let late = AtomicBool::new(index == 37 && fault == 12);
        not_cancelled(if late.load(Ordering::Acquire) {
            Some(&late)
        } else {
            cancelled
        })?;
        let ids: &[u32] = if which == 0 { &[46, 44] } else { &[47, 45, 46] };
        let handles = ids
            .iter()
            .map(|id| self.values.lease(ValueId(*id)).map(ResidentLease::handle))
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: original whole-body armed flight owns all actual input/output accesses.
        unsafe { self.kernels[which].enqueue(handles) }.map_err(native_error)?;
        self.counts[which].fetch_add(1, Ordering::AcqRel);
        if !self.outputs_unpromoted()? {
            return Err(Error::State(
                "canonical47 Sig/Mul output prematurely promoted".into(),
            ));
        }
        if index == 36 && fault == 11 {
            return Err(Error::State(
                "deliberate canonical47 unknown after actual Sigmoid".into(),
            ));
        }
        Ok(1)
    }
    pub(super) fn outputs_unpromoted(&self) -> Result<bool> {
        Ok(!self.values.lease(ValueId(46))?.is_initialized()
            && !self.values.lease(ValueId(47))?.is_initialized())
    }
    pub(super) fn counts(&self) -> [usize; 2] {
        self.counts.each_ref().map(|v| v.load(Ordering::Acquire))
    }
    pub(super) fn kernels(&self) -> &[Arc<OwnedCudaKernel>; 2] {
        &self.kernels
    }
    pub(super) unsafe fn retire(&self) -> Result<()> {
        for k in &self.kernels {
            unsafe { k.retire_after_physical_completion() }.map_err(native_error)?;
        }
        Ok(())
    }
}
