//! Cold owner graph and one original native module/record union.
use super::super::{
    model_values::ModelValues,
    model_weights::ModelWeights,
    projection_cubin::{not_cancelled, resident::PreparedImages},
    text_head::StreamedHead,
};
use super::TextKernelCapture;
use super::{
    TextNativeGeometry, TextNativePreparation, TextPrograms, packets::Catalogue, programs::insert,
};
use crate::{
    cuda::rms::{FullModelRmsScratch, FusedRmsPrepared},
    resident_inference::native_fragment::text::{TextCallInput, TextCallState},
    resident_leases::{ResidentInitialization, ResidentLease},
    resident_program::text_positions::text::TextPositions,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::{
    Error, Result,
    device_events::{BufferRole, DeviceAccess},
};

pub(crate) struct TextNativeStorage {
    source: VerifiedTensorSource,
    capture: TextKernelCapture,
    geometries: Vec<TextNativeGeometry>,
    pub(super) weights: Arc<ModelWeights>,
    pub(super) workspace: ResidentLease,
    pub(super) images: Arc<PreparedImages>,
    pub(super) pool: super::pool::Pool,
    pub(super) attention: Arc<super::attention::Storage>,
    pub(super) head: Arc<StreamedHead>,
    retired: AtomicBool,
    _metadata: Arc<Vec<WindowWorkspace>>,
}

pub(super) fn admit_host_cohort(
    source: &VerifiedTensorSource,
    bytes: usize,
) -> Result<Vec<WindowWorkspace>> {
    const CHUNK: usize = 32 * 1024 * 1024;
    if bytes == 0 || bytes > 4 * CHUNK {
        return Err(Error::Limit("text metadata/cache host cohort bound".into()));
    }
    let mut tickets = Vec::with_capacity(bytes.div_ceil(CHUNK));
    for start in (0..bytes).step_by(CHUNK) {
        tickets.push(source.reserve_workspace((bytes - start).min(CHUNK))?);
    }
    Ok(tickets)
}
impl TextNativeStorage {
    pub(crate) fn capture(&self) -> TextKernelCapture {
        self.capture
    }
    pub(crate) fn prepare_finite_code(
        &self,
        queue: &cubecl::cuda::OwnedCudaQueue,
    ) -> Result<usize> {
        self.attention.prepare_finite_code(queue)
    }
    pub(crate) fn prepare(
        preparation: TextNativePreparation,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let TextNativePreparation {
            source,
            capture,
            geometries,
            weights,
            workspace,
            rotary,
            positions,
            inputs,
            audit_head,
            cohort_b4,
            head_mode,
            tied_head,
        } = preparation;
        if geometries.len() != 6 || geometries.capacity() > 6 {
            return Err(Error::State(
                "text exact five prefill plus one row storage geometries".into(),
            ));
        }
        let mut rows = BTreeSet::new();
        for g in &geometries {
            g.scope.require_values(&g.values)?;
            weights.require_text(&g.scope)?;
            init.require_service(g.values.service())?;
            if !rows.insert(g.scope.call().rows().len())
                || g.scope.native_capture() != capture
                || !g.scope.source().same_issuer(&source)
                || !Arc::ptr_eq(g.rms_scope.record(), g.scope.record())
                || !Arc::ptr_eq(g.values.service(), workspace.service())
            {
                return Err(Error::State(
                    "text SAME source/capture/actual geometry/value/RMS owners".into(),
                ));
            }
        }
        if rows != BTreeSet::from([1, 47, 57, 62, 67, 72])
            || workspace.bytes() != 8_519_680
            || workspace.role() != BufferRole::Scratch
            || workspace.source_identity() != source.source_identity()
            || workspace.service_generation() != workspace.service().generation()
            || !weights.source().same_issuer(&source)
        {
            return Err(Error::State(
                "text native source/workspace/geometry cohort differs".into(),
            ));
        }
        // Requiring actual compiled packets precedes all new allocation/admission.
        // Six managed geometries never collapse the forty current-call packet keys.
        let scopes = geometries
            .iter()
            .map(|g| g.scope.clone())
            .collect::<Vec<_>>();
        let inventory = super::TextNativeInventory::for_scopes(
            capture, &scopes, audit_head, cohort_b4, head_mode,
        )?;
        let bytes = inventory.catalogue_bytes;
        let metadata = Arc::new(admit_host_cohort(&source, bytes)?);
        let catalogue = Catalogue::load(capture, &geometries)?;
        init.retain_completion_owner(metadata.clone())?;
        init.retain_completion_owner(weights.clone())?;
        init.retain_completion_owner(rotary.clone())?;
        init.retain_completion_owner(positions.clone())?;
        for g in &geometries {
            init.retain_completion_owner(g.scope.clone())?;
            init.retain_completion_owner(g.values.clone())?;
            init.retain_completion_owner(g.rms_scope.clone())?;
            init.retain_completion_owner(g.final_rms.clone())?;
        }
        let attention = super::attention::Storage::prepare(
            super::attention::Preparation {
                source: source.clone(),
                geometries: &geometries,
                rotary,
                positions,
                catalogue: &catalogue,
                service: workspace.service().clone(),
                private_layout: &inventory.private_layout,
            },
            init,
            cancelled,
        )?;
        init.retain_completion_owner(attention.clone())?;
        let images = super::images::prepare(
            &source,
            capture,
            &geometries,
            inputs,
            cohort_b4,
            init,
            cancelled,
        )?;
        let pool = super::pool::Pool::prepare(catalogue, images.clone(), init, cancelled)?;
        // One genuine Cold Head W slot/kernel for all six shapes; bind_call
        // supplies the current source-selected final normalized row.
        let g = &geometries[0];
        let head = StreamedHead::prepare(
            g.scope.clone(),
            g.values.clone(),
            g.rms_scope.clone(),
            g.final_rms.clone(),
            images.clone(),
            init,
            cancelled,
            audit_head,
            head_mode,
            tied_head,
        )?;
        let result = Arc::new(Self {
            source,
            capture,
            geometries,
            weights,
            workspace,
            images,
            pool,
            attention,
            head,
            retired: AtomicBool::new(false),
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn bind(
        self: &Arc<Self>,
        input: Arc<TextCallInput>,
        state: Arc<TextCallState>,
        values: Arc<ModelValues>,
        rms_scope: Arc<FullModelRmsScratch>,
        final_rms: Arc<FusedRmsPrepared>,
        positions: Arc<TextPositions>,
    ) -> Result<Arc<TextPrograms>> {
        TextPrograms::bind(
            self.clone(),
            input,
            state,
            values,
            rms_scope,
            final_rms,
            positions,
        )
    }
    pub(super) fn require_call(
        &self,
        input: &TextCallInput,
        values: &Arc<ModelValues>,
    ) -> Result<()> {
        if self.retired.load(Ordering::Acquire)
            || !self.source.same_issuer(input.source())
            || self.capture != input.scope().native_capture()
            || !self
                .geometries
                .iter()
                .any(|g| g.scope.call().rows().len() == input.rows())
            || !Arc::ptr_eq(self.workspace.service(), values.service())
            || self.workspace.service_generation() != values.service().generation()
        {
            return Err(Error::State(
                "text native current call/source/storage retired or foreign".into(),
            ));
        }
        input.scope().require_values(values)?;
        self.images.require_service(values.service())
    }
    pub(crate) fn head(&self) -> &Arc<StreamedHead> {
        &self.head
    }
    /// Same source/image/module context for separately qualified B4 records.
    pub(crate) fn cohort_workspace(&self) -> &ResidentLease {
        &self.workspace
    }
    pub(crate) fn cohort_images(&self) -> &Arc<PreparedImages> {
        &self.images
    }
    pub(crate) fn cohort_qk_parameters(
        &self,
        inputs: &[Arc<TextCallInput>],
        layer: usize,
    ) -> Result<[ResidentLease; 2]> {
        self.attention.cohort_parameters(inputs, layer)
    }
    pub(crate) fn weights(&self) -> &Arc<ModelWeights> {
        &self.weights
    }
    pub(crate) fn immutable_controls(&self) -> Vec<(ResidentLease, &[u8])> {
        self.attention.immutable_controls()
    }
    pub(crate) fn storage_accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut all = BTreeMap::new();
        for (l, a) in self
            .weights
            .accesses()
            .into_iter()
            .chain(self.attention.storage_accesses())
            .chain(self.head.accesses())
        {
            insert(&mut all, l, a);
        }
        insert(&mut all, self.workspace.clone(), DeviceAccess::Write);
        for g in &self.geometries {
            for (l, a) in g.values.accesses() {
                insert(&mut all, l, a);
            }
        }
        all.into_values().collect()
    }
    pub(crate) fn retained_source_graph(&self) -> Result<bool> {
        // Ownership only; never call require_text/ensure_usable after taint.
        if !self.source.same_issuer(self.weights.source())
            || !self
                .attention
                .retained_source(&self.source, self.workspace.service())?
            || !self.head.retained_graph()?
            || !Arc::ptr_eq(self.head.images(), &self.images)
        {
            return Ok(false);
        }
        for g in &self.geometries {
            g.scope.require_values(&g.values)?;
            if !g.scope.source().same_issuer(&self.source)
                || !Arc::ptr_eq(g.rms_scope.record(), g.scope.record())
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub(crate) fn retention(&self) -> Result<(usize, usize, usize, usize)> {
        let mut kernels = self.pool.kernels();
        kernels.push(self.head.kernel().clone());
        let witness = self.images.retention_witness(&kernels)?;
        let actual = witness.inspect()?;
        Ok((
            actual.image_owners,
            actual.kernel_owners,
            actual.native_charge_bytes,
            actual.image_admission_bytes,
        ))
    }
    pub(crate) unsafe fn retire_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        if self.retired.swap(true, Ordering::AcqRel) {
            return Err(cubecl::server::ServerError::graph_state(
                "text native union already retired",
            ));
        }
        // SAFETY: caller retains this exact complete graph on the original
        // physically completed retirement flight. Partial unload stays quarantined.
        unsafe { self.head.retire_after_physical_completion() }?;
        unsafe { self.pool.retire_after_physical_completion() }?;
        unsafe { self.images.set().retire_after_physical_completion() }
    }
}
