//! Authenticated contiguous BF16 residency; transport only, never model permission.
use super::*;
pub(in crate::resident_leases) mod layout;
pub(crate) mod qualification;
mod rows;
mod source;
#[cfg(any(
    feature = "gemma-bf16-qualification",
    feature = "qwen-source-qualification"
))]
pub(crate) mod view;
pub(crate) use source::Bf16MatrixSource;
pub(in crate::resident_leases) mod upload;
use layout::Geometry;
pub(crate) use upload::PendingResidentMatrix;

fn require_portable(service: &Arc<ResidentService>, geometry: Geometry) -> Result<()> {
    service.ensure_usable()?;
    if !matches!(
        service.provider,
        crate::backend::Provider::Hip | crate::backend::Provider::Vulkan
    ) {
        return Err(Error::Unsupported(
            "packed matrix transport explicitly selects HIP/Vulkan".into(),
        ));
    }
    let p = service.client().properties();
    geometry.portable_limits(p.memory.max_page_size)?;
    if !p
        .features
        .supports_type(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U32))
        || p.hardware.max_bindings < 4
        || p.hardware.max_units_per_cube < 32
        || p.hardware.max_cube_dim.0 < 32
        || p.hardware.max_cube_dim.1 == 0
        || p.hardware.max_cube_dim.2 == 0
        || p.hardware.max_cube_count.0 < (layout::WINDOW as u32 / 4).div_ceil(32)
        || p.hardware.max_cube_count.1 == 0
        || p.hardware.max_cube_count.2 == 0
    {
        return Err(Error::Unsupported(
            "BF16 integer transport binding/grid/u32 capability".into(),
        ));
    }
    Ok(())
}
use zerogravity_artifacts::source_weights::{VerifiedMatrix, VerifiedTensorSource};
use zerogravity_core::{accelerator::program::MatrixAxes, quant::QuantKind};

pub(crate) struct ResidentMatrix {
    lease: ResidentLease,
    source: VerifiedTensorSource,
    matrix: VerifiedMatrix,
    layout: Geometry,
    // The verified uploader owns actual coverage/checkpoints and source admission.
    _upload: crate::physical_quarantine::CompletionOwner,
}
impl ResidentMatrix {
    /// Private verified-uploader boundary, after complete source coverage AND
    /// original initialization. An initialized handle alone is not a source proof.
    pub(super) fn from_completed_upload(
        lease: ResidentLease,
        source: VerifiedTensorSource,
        matrix: VerifiedMatrix,
        upload: crate::physical_quarantine::CompletionOwner,
    ) -> Result<Arc<Self>> {
        let layout = Self::authenticate(&source, &matrix)?;
        source.require_matrix_binding(&matrix)?;
        let authenticated = matrix;
        let result = Arc::new(Self {
            lease,
            source,
            matrix: authenticated,
            layout,
            _upload: upload,
        });
        result.require(
            &result.source,
            result.lease.service(),
            result.layout.rows(),
            result.layout.row_bytes() / 2,
        )?;
        Ok(result)
    }
    pub(in crate::resident_leases) fn authenticate(
        source: &VerifiedTensorSource,
        matrix: &VerifiedMatrix,
    ) -> Result<Geometry> {
        source.require_matrix_binding(matrix)?;
        if matrix.source_identity() != source.source_identity()
            || matrix.tensor().kind != QuantKind::Bf16
            || matrix.axes() != MatrixAxes::OutputInput
        {
            return Err(Error::State(
                "resident matrix authenticated BF16 source layout".into(),
            ));
        }
        Geometry::new(matrix.rows(), matrix.columns())
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    /// Metadata only: valid even after the physical service is fenced.
    #[cfg_attr(not(feature = "cuda-native-qualification"), allow(dead_code))]
    pub(crate) fn matches_matrix(&self, matrix: &VerifiedMatrix) -> bool {
        self.matrix.same_binding(matrix)
    }
    /// Resident bytes must be this exact original source operand, not another
    /// tensor or expert plane with equal shape on the same source/service.
    #[cfg_attr(not(feature = "cuda-native-qualification"), allow(dead_code))]
    pub(crate) fn require_matrix(
        &self,
        source: &VerifiedTensorSource,
        service: &Arc<ResidentService>,
        matrix: &VerifiedMatrix,
    ) -> Result<()> {
        if !self.matches_matrix(matrix) {
            return Err(Error::State(
                "resident matrix exact original member/range differs".into(),
            ));
        }
        self.require(source, service, matrix.rows(), matrix.columns())
    }
    pub(crate) fn require(
        &self,
        source: &VerifiedTensorSource,
        service: &Arc<ResidentService>,
        rows: usize,
        columns: usize,
    ) -> Result<()> {
        service.ensure_usable()?;
        service.check_lease(&self.lease)?;
        if !self.source.same_issuer(source)
            || self.matrix.source_identity() != service.source_identity()
            || self.matrix.tensor().kind != QuantKind::Bf16
            || self.matrix.axes() != MatrixAxes::OutputInput
            || self.layout.rows() != rows
            || self.layout.row_bytes() != columns.checked_mul(2).unwrap_or(0)
            || self.lease.bytes() != self.layout.bytes()
            || self.lease.role() != BufferRole::ImmutableWeights
            || !self.lease.is_initialized()
        {
            return Err(Error::State(
                "resident matrix original owner/service/generation/geometry".into(),
            ));
        }
        Ok(())
    }
    /// Caller has authenticated the actual invocation/IDs and registered this
    /// complete Read plus destination Write on its original armed flight.
    #[cfg(feature = "cuda-native-qualification")]
    pub(crate) fn enqueue_rows(&self, destination: &ResidentLease, ids: Arc<[u32]>) -> Result<()> {
        let service = self.lease.service();
        self.require(
            &self.source,
            service,
            self.matrix.rows(),
            self.matrix.columns(),
        )?;
        service.check_lease(destination)?;
        if destination.id() == self.lease.id()
            || destination.role() != BufferRole::Activation
            || ids.len().checked_mul(self.layout.row_bytes()) != Some(destination.bytes())
        {
            return Err(Error::State(
                "resident matrix exclusive complete row destination".into(),
            ));
        }
        // SAFETY: actual producer seals and original flight remain caller-owned;
        // native transport rechecks all IDs, full ranges, stream/service and taint.
        unsafe {
            service
                .pinned_cuda_queue()?
                .enqueue_resident_bf16_rows_workspace(
                    self.lease.handle(),
                    destination.cubin_binding()?,
                    self.layout.cuda()?,
                    ids,
                )
        }
        .map_err(|e| Error::State(format!("resident matrix D2D rows: {e}")))
    }
}
