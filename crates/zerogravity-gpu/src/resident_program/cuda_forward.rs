//! Production CUDA-forward physical components.
//!
//! This module deliberately owns no model loop and no StateRoot publication.
//! It composes the existing pinned ResidentService/PhysicalQueue/flight owner;
//! every submitted native object is retained by that original flight until
//! actual completion or quarantine.
#![allow(dead_code)] // Private unselected implementation until the full body lands.

use crate::cuda::bf16_gemm::{Bf16GemmInputLayout, Bf16GemmShape};
use crate::resident_leases::{
    ResidentInitialization, ResidentLease, ResidentService, ResidentSubmission,
    forward_projection_upload::BoundSourceMatrices,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
mod arithmetic;
pub(crate) use arithmetic::PrefillProjectionClasses;
use arithmetic::ProjectionUse;
mod gemm;
use gemm::CudaForwardGemm;
#[cfg(test)]
pub(crate) mod blas_reference;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
mod attention;
pub(super) mod body;
mod cohort;
mod cohort_values;
mod cold;
pub(crate) use cohort::CudaForwardCohort;
pub(crate) use cohort_values::CudaForwardCohortValues;
mod execute;
mod geometry;
mod grouped_gdn;
mod layout;
pub(crate) mod rotary;

use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        NumericalContract, PlanOperation, SourceOperator, SourceOperatorKind,
        SourceOperatorTemplate, SourceRowSelection, ValueStorageFormat,
    },
    device_events::{BufferRole, DeviceAccess},
    model::ModelRegion,
    provider::{ExecutionMode, PreparedExecutable, PreparedRegion, ProviderKind, SelectedMode},
    resource_arbiter::{AllocationClass, AllocationRequest},
};

#[cfg(test)]
fn server_error(error: impl std::fmt::Display) -> Error {
    Error::State(format!("CUDA forward cuBLAS owner: {error}"))
}

fn exact_view(
    lease: &ResidentLease,
    offset: usize,
    bytes: usize,
    role: &str,
) -> Result<cubecl::server::Handle> {
    lease
        .view(offset, bytes)
        .map_err(|_| Error::Limit(format!("CUDA forward {role} view exceeds lease")))
}

/// One source-derived full projection. Cold ownership binds the call-free
/// template; request-local row geometry never owns or reloads weights.
pub(crate) struct CudaForwardProjection {
    region: PreparedRegion<VerifiedTensorSource>,
    weights: Arc<BoundSourceMatrices>,
    arithmetic: ProjectionUse,
}
impl PreparedExecutable for CudaForwardProjection {
    type Source = VerifiedTensorSource;
    async fn source(&self) -> Result<Self::Source> {
        Ok(self.region.model().source().clone())
    }
    async fn program(&self) -> Result<Arc<zerogravity_core::semantic::ForwardPlan>> {
        Ok(self.region.model().program().clone())
    }
    fn prepared_region(&self) -> Option<&PreparedRegion<Self::Source>> {
        Some(&self.region)
    }
}
impl CudaForwardProjection {
    pub(super) fn lowering(
        region: ModelRegion<VerifiedTensorSource>,
    ) -> Result<PreparedRegion<VerifiedTensorSource>> {
        let max_rows = Self::projection_template(&region)?.max_rows();
        PreparedRegion::new(
            region,
            SelectedMode {
                provider: ProviderKind::Cuda,
                mode: ExecutionMode::Native,
            },
            1..=max_rows,
            &[ValueStorageFormat::Bf16],
            &[ValueStorageFormat::Bf16],
        )
    }

    fn projection_template(
        region: &ModelRegion<VerifiedTensorSource>,
    ) -> Result<&SourceOperatorTemplate> {
        let [node] = region.region().nodes() else {
            return Err(Error::Unsupported(
                "CUDA projection requires one closed operation".into(),
            ));
        };
        let PlanOperation::SourceOperatorTemplate {
            descriptor,
            state: None,
        } = &node.operation
        else {
            return Err(Error::State(
                "CUDA projection region lacks source template".into(),
            ));
        };
        if descriptor.numerical_contract() != NumericalContract::CudaSm86Bf16ReferenceV1
            || !matches!(
                descriptor.kind(),
                SourceOperatorKind::Projection { selected_row: None }
            )
            || descriptor.source() != region.source().source_identity()
            || descriptor.matrices().is_empty()
            || !descriptor.parameters().is_empty()
        {
            return Err(Error::Unsupported(
                "CUDA projection region numerical/source contract differs".into(),
            ));
        }
        Ok(descriptor)
    }

    pub(crate) async fn prepare(
        service: &Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        region: ModelRegion<VerifiedTensorSource>,
        arithmetic: ProjectionUse,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        let region = Self::lowering(region)?;
        let template = Self::projection_template(region.model())?;
        let weights = init
            .upload_forward_projection(
                service,
                region.model().source().clone(),
                template.matrices(),
                cancelled,
            )
            .await?;
        Self::prepare_with_weights(region, weights, arithmetic)
    }

    pub(crate) fn prepare_with_weights(
        region: PreparedRegion<VerifiedTensorSource>,
        weights: Arc<BoundSourceMatrices>,
        arithmetic: ProjectionUse,
    ) -> Result<Arc<Self>> {
        let template = Self::projection_template(region.model())?;
        if region.selected()
            != (SelectedMode {
                provider: ProviderKind::Cuda,
                mode: ExecutionMode::Native,
            })
            || region.supported_rows() != (1..=template.max_rows())
            || region.capacity().input(0)?.row_bytes() != weights.columns() * 2
            || region.capacity().output(0)?.row_bytes() != weights.rows() * 2
            || !weights.same_source(region.model().source())
        {
            return Err(Error::State(
                "CUDA projection region selected layout/source differs".into(),
            ));
        }
        weights.require_members(template.matrices())?;
        let [input] = region.model().region().inputs() else {
            return Err(Error::State(
                "CUDA projection region input arity differs".into(),
            ));
        };
        let [output] = region.model().region().outputs() else {
            return Err(Error::State(
                "CUDA projection region output arity differs".into(),
            ));
        };
        if input.width != weights.columns()
            || output.width != weights.rows()
            || input.rows != SourceRowSelection::AllActive
            || output.rows != template.row_selection()
        {
            return Err(Error::State(
                "CUDA projection region packed geometry differs".into(),
            ));
        }
        Ok(Arc::new(Self {
            region,
            weights,
            arithmetic,
        }))
    }

    fn template(&self) -> &SourceOperatorTemplate {
        // Construction already checked this immutable original node.
        let PlanOperation::SourceOperatorTemplate { descriptor, .. } =
            &self.region.model().region().nodes()[0].operation
        else {
            unreachable!()
        };
        descriptor
    }

    fn operation(&self, descriptor: &SourceOperator) -> Result<(Bf16GemmShape, usize)> {
        let template = self.template();
        descriptor.validate()?;
        let rows = descriptor.invocation().tokens().len();
        let bound = self.region.shape(rows)?;
        let SourceOperatorKind::Projection { selected_row } = descriptor.kind() else {
            return Err(Error::State(
                "CUDA forward projection runtime descriptor kind differs".into(),
            ));
        };
        let expected_selected = match template.row_selection() {
            SourceRowSelection::AllActive => None,
            SourceRowSelection::LastActive => Some(
                rows.checked_sub(1)
                    .ok_or_else(|| Error::State("CUDA forward projection empty call".into()))?,
            ),
        };
        if descriptor.numerical_contract() != template.numerical_contract()
            || descriptor.invocation().source() != template.source()
            || descriptor.invocation().tokens().vocabulary() != template.vocabulary()
            || descriptor.matrices() != template.matrices()
            || descriptor.parameters() != template.parameters()
            || selected_row != expected_selected
        {
            return Err(Error::State(
                "CUDA forward projection runtime/template semantics differ".into(),
            ));
        }
        self.weights.require_members(descriptor.matrices())?;
        let input_layout = bound.input(0)?;
        let output_layout = bound.output(0)?;
        let active_rows = output_layout.rows();
        let operation =
            Bf16GemmShape::new(self.weights.rows(), active_rows, self.weights.columns())?;
        if descriptor.input_elements()?.as_slice() != [input_layout.byte_len()? / 2]
            || descriptor.output_elements()?.as_slice() != [output_layout.byte_len()? / 2]
        {
            return Err(Error::State(
                "CUDA forward projection operator/packed geometry differs".into(),
            ));
        }
        let input_offset = selected_row
            .unwrap_or(0)
            .checked_mul(input_layout.row_stride_bytes())
            .ok_or_else(|| Error::Limit("CUDA forward projection input offset overflow".into()))?;
        Ok((operation, input_offset))
    }

    pub(crate) fn output_bytes(&self, descriptor: &SourceOperator) -> Result<usize> {
        Ok(self.operation(descriptor)?.0.output_bytes())
    }

    pub(crate) fn accesses(
        &self,
        descriptor: &SourceOperator,
        input: &ResidentLease,
        output: &ResidentLease,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let (operation, input_offset) = self.operation(descriptor)?;
        let input_bytes = operation.input_bytes();
        if input_offset
            .checked_add(input_bytes)
            .is_none_or(|end| end > input.bytes())
            || output.bytes() < operation.output_bytes()
        {
            return Err(Error::State(
                "CUDA forward projection input/output lease extent differs".into(),
            ));
        }
        let mut accesses = vec![
            (input.clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
        ];
        if let Some(weights) = self.weights.lease() {
            accesses.push((weights.clone(), DeviceAccess::Read));
        }
        Ok(accesses)
    }

    #[allow(clippy::too_many_arguments)] // Projection/source/status ownership stays explicit.
    async fn enqueue(
        &self,
        descriptor: &SourceOperator,
        gemm: &CudaForwardGemm,
        flight: &mut ResidentSubmission<'_>,
        input: &ResidentLease,
        output: &ResidentLease,
        status: &ResidentLease,
        page: Option<&Arc<crate::resident_leases::source_page::ResidentSourcePage>>,
    ) -> Result<usize> {
        let (operation, input_offset) = self.operation(descriptor)?;
        // The reduction is selected for the full projection, never for a transport page.
        let reduction = self.arithmetic.reduction(operation);
        if let Some(weights) = self.weights.lease() {
            gemm.enqueue(
                flight,
                operation,
                reduction,
                weights,
                input,
                input_offset,
                output,
                0,
                status,
            )?;
            return Ok(1);
        }
        let page = page
            .ok_or_else(|| Error::State("CUDA forward streamed projection has no page".into()))?;
        flight.require_accesses([
            (input, DeviceAccess::Read),
            (output, DeviceAccess::Write),
            (status, DeviceAccess::Write),
        ])?;
        let input = exact_view(
            input,
            input_offset,
            operation.input_bytes(),
            "streamed GEMM input",
        )?;
        let mut base = 0;
        let mut launches = 0;
        for matrix in self.weights.matrices() {
            let geometry = crate::resident_leases::source_page::PageGeometry::new(
                matrix.rows(),
                matrix.columns(),
            )?;
            for first in (0..geometry.rows).step_by(geometry.rows_per_page) {
                let rows = first..first + (geometry.rows - first).min(geometry.rows_per_page);
                let shape = Bf16GemmShape::new(rows.len(), operation.rows, operation.inner)?;
                let offset = (base + rows.start) * 2;
                let extent = crate::cuda::bf16_gemm::output_span(
                    shape.columns,
                    shape.rows,
                    operation.columns,
                )?;
                let target = exact_view(output, offset, extent, "streamed GEMM output")?;
                page.with_rows(flight, matrix, rows, None, |view| {
                    if view.rows().len() != shape.columns
                        || view.matrix().columns() != shape.inner
                        || view.bytes() != shape.weight_bytes()
                    {
                        return Err(Error::State("CUDA streamed GEMM page geometry".into()));
                    }
                    gemm.enqueue_page(
                        shape,
                        reduction,
                        view.handle().clone(),
                        input.clone(),
                        target,
                        operation.columns,
                        status.handle(),
                    )?;
                    Ok(vec![output.clone()])
                })
                .await?;
                launches += 1;
            }
            base += matrix.rows();
        }
        Ok(launches)
    }

    /// Batched independent decode rows for a last-active projection. Every row
    /// is logically the last row of its own session, so the shared GEMM emits
    /// one output row per session rather than selecting one row from the band.
    #[allow(clippy::too_many_arguments)] // Cohort projection/status ownership stays explicit.
    async fn enqueue_decode_rows(
        &self,
        rows: usize,
        gemm: &CudaForwardGemm,
        flight: &mut ResidentSubmission<'_>,
        input: &ResidentLease,
        input_layout: Bf16GemmInputLayout,
        output: &ResidentLease,
        status: &ResidentLease,
        page: Option<&Arc<crate::resident_leases::source_page::ResidentSourcePage>>,
    ) -> Result<usize> {
        let template = self.template();
        if rows == 0
            || rows > template.max_rows()
            || template.row_selection() != SourceRowSelection::LastActive
            || !matches!(template.kind(), SourceOperatorKind::Projection { .. })
        {
            return Err(Error::State(
                "CUDA forward decode projection row geometry differs".into(),
            ));
        }
        self.weights.require_members(template.matrices())?;
        let operation = Bf16GemmShape::new(self.weights.rows(), rows, self.weights.columns())?;
        if input.bytes() < operation.input_bytes() || output.bytes() < operation.output_bytes() {
            return Err(Error::State(
                "CUDA forward decode projection input/output extent differs".into(),
            ));
        }
        let reduction = self.arithmetic.reduction(operation);
        if let Some(weights) = self.weights.lease() {
            gemm.enqueue_with_input_layout(
                flight,
                operation,
                reduction,
                input_layout,
                weights,
                input,
                0,
                output,
                0,
                status,
            )?;
            return Ok(1);
        }
        let page = page.ok_or_else(|| {
            Error::State("CUDA forward streamed decode projection has no page".into())
        })?;
        flight.require_accesses([
            (input, DeviceAccess::Read),
            (output, DeviceAccess::Write),
            (status, DeviceAccess::Write),
        ])?;
        let input = exact_view(input, 0, operation.input_bytes(), "decode GEMM input")?;
        let mut base = 0;
        let mut launches = 0;
        for matrix in self.weights.matrices() {
            let geometry = crate::resident_leases::source_page::PageGeometry::new(
                matrix.rows(),
                matrix.columns(),
            )?;
            for first in (0..geometry.rows).step_by(geometry.rows_per_page) {
                let source_rows =
                    first..first + (geometry.rows - first).min(geometry.rows_per_page);
                let shape = Bf16GemmShape::new(source_rows.len(), rows, operation.inner)?;
                let offset = (base + source_rows.start) * 2;
                let extent = crate::cuda::bf16_gemm::output_span(
                    shape.columns,
                    shape.rows,
                    operation.columns,
                )?;
                let target = exact_view(output, offset, extent, "streamed decode GEMM output")?;
                page.with_rows(flight, matrix, source_rows, None, |view| {
                    gemm.enqueue_page_with_input_layout(
                        shape,
                        reduction,
                        input_layout,
                        view.handle().clone(),
                        input.clone(),
                        target,
                        operation.columns,
                        status.handle(),
                    )?;
                    Ok(vec![output.clone()])
                })
                .await?;
                launches += 1;
            }
            base += matrix.rows();
        }
        Ok(launches)
    }
}

pub(crate) struct CudaForwardAdmission {
    layout: Arc<layout::CudaForwardLayout>,
    roots: usize,
    state_reserve_bytes: usize,
}
impl CudaForwardAdmission {
    pub(crate) fn analyze(
        plan: Arc<zerogravity_core::accelerator::program::ForwardPlan>,
    ) -> Result<Self> {
        let layout = layout::CudaForwardLayout::analyze(plan)?;
        let state_reserve_bytes = layout
            .preflight()
            .prior_state_bytes
            .checked_add(layout.preflight().candidate_state_bytes)
            .ok_or_else(|| Error::Limit("CUDA forward state reserve overflow".into()))?;
        Ok(Self {
            layout,
            roots: 2,
            state_reserve_bytes,
        })
    }
    pub(crate) fn with_roots(mut self, roots: usize) -> Result<Self> {
        if roots < 2 {
            return Err(Error::Limit(
                "CUDA forward requires prior/candidate roots".into(),
            ));
        }
        self.roots = roots;
        self.state_reserve_bytes = self
            .layout
            .preflight()
            .prior_state_bytes
            .checked_add(self.layout.preflight().candidate_state_bytes)
            .and_then(|bytes| {
                self.layout
                    .preflight()
                    .prior_state_bytes
                    .checked_mul(roots - 2)
                    .and_then(|extra| bytes.checked_add(extra))
            })
            .ok_or_else(|| Error::Limit("CUDA forward root inventory overflow".into()))?;
        self.total_bytes()?;
        Ok(self)
    }
    pub(crate) fn with_state_reserve(
        mut self,
        roots: usize,
        state_reserve_bytes: usize,
    ) -> Result<Self> {
        if roots < 2 || state_reserve_bytes == 0 {
            return Err(Error::Limit(
                "CUDA forward requires positive state reserve and roots".into(),
            ));
        }
        self.roots = roots;
        self.state_reserve_bytes = state_reserve_bytes;
        self.total_bytes()?;
        Ok(self)
    }
    pub(crate) fn fit(mut self, provider_capacity: usize) -> Result<Self> {
        if self.total_bytes()? > provider_capacity {
            Arc::make_mut(&mut self.layout).stream_weights()?;
        }
        let total = self.total_bytes()?;
        if total > provider_capacity {
            return Err(Error::resource(
                "CUDA provider memory / VRAM budget",
                format!("CUDA forward workspace/{}-root admission", self.roots),
                total as u64,
                provider_capacity as u64,
            ));
        }
        Ok(self)
    }
    pub(crate) fn total_bytes(&self) -> Result<usize> {
        let preflight = self.layout.preflight();
        let state_pair = preflight
            .prior_state_bytes
            .checked_add(preflight.candidate_state_bytes)
            .ok_or_else(|| Error::Limit("CUDA forward state pair overflow".into()))?;
        self.layout
            .total_admission_bytes()?
            .checked_sub(state_pair)
            .and_then(|bytes| bytes.checked_add(self.state_reserve_bytes))
            .ok_or_else(|| Error::Limit("CUDA forward complete root inventory overflow".into()))
    }
    pub(crate) fn source(&self) -> zerogravity_core::accelerator::SourceIdentity {
        self.layout.plan().source()
    }
    pub(crate) fn capacity(&self) -> usize {
        self.layout.plan().prefix_capacity()
    }
    pub(crate) fn vocabulary(&self) -> usize {
        self.layout.geometry().vocabulary
    }
    pub(crate) fn preflight(&self) -> &crate::resident_program::ResidentPreflight {
        self.layout.preflight()
    }
    pub(crate) fn cohort_value_bytes(&self, cohort_width: usize) -> Result<usize> {
        CudaForwardCohortValues::required_bytes(&self.layout, cohort_width)
    }
}

pub(crate) struct PreparedCudaForward {
    pub(crate) program: crate::resident_program::ResidentProgram,
    pub(crate) owned_accesses: Vec<(ResidentLease, DeviceAccess)>,
    pub(crate) cohort_values: Option<Arc<CudaForwardCohortValues>>,
    pub(crate) cohort_value_accesses: Vec<(ResidentLease, DeviceAccess)>,
    pub(crate) greedy: [cubecl::server::Handle; 2],
    pub(crate) greedy_accesses: Vec<(ResidentLease, DeviceAccess)>,
    pub(crate) state_accesses: Vec<(ResidentLease, DeviceAccess)>,
    pub(crate) logit_accesses: Vec<(ResidentLease, DeviceAccess)>,
}

fn same_state_requirements(
    a: &[crate::resident_program::StateRequirement],
    b: &[crate::resident_program::StateRequirement],
) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            a.slot == b.slot
                && a.physical == b.physical
                && a.bytes == b.bytes
                && match (a.layout, b.layout) {
                    (
                        crate::resident_program::StateLayout::Vector { elements: a },
                        crate::resident_program::StateLayout::Vector { elements: b },
                    ) => a == b,
                    (
                        crate::resident_program::StateLayout::Kv {
                            slab: aslab,
                            capacity: acap,
                        },
                        crate::resident_program::StateLayout::Kv {
                            slab: bslab,
                            capacity: bcap,
                        },
                    ) => aslab == bslab && acap == bcap,
                    _ => false,
                }
        })
}

pub(crate) async fn prepare_program(
    service: Arc<ResidentService>,
    admission: CudaForwardAdmission,
    source: VerifiedTensorSource,
    source_owner: Option<Arc<dyn std::any::Any + Send + Sync>>,
    cohort_width: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<PreparedCudaForward> {
    service.ensure_usable()?;
    if service.source_identity() != admission.source()
        || source.source_identity() != admission.source()
    {
        return Err(Error::State(
            "CUDA forward preparation source/service differs".into(),
        ));
    }
    let greedy_sizes = crate::resident_greedy::workspace_bytes(admission.vocabulary())?;
    let greedy_bytes = greedy_sizes.into_iter().try_fold(0usize, |sum, bytes| {
        sum.checked_add(bytes)
            .ok_or_else(|| Error::Limit("CUDA forward greedy workspace overflow".into()))
    })?;
    let cohort_value_bytes = admission.cohort_value_bytes(cohort_width)?;
    let cold_bytes = admission.preflight().owned_bytes()?;
    let remaining = service.remaining_capacity()?;
    let required = admission
        .total_bytes()?
        .checked_add(greedy_bytes)
        .and_then(|bytes| bytes.checked_add(cohort_value_bytes))
        .ok_or_else(|| Error::Limit("CUDA forward complete admission overflow".into()))?;
    let required = required.max(cold_bytes);
    if required > remaining {
        return Err(Error::resource(
            "CUDA provider memory / VRAM budget",
            "CUDA forward cold/root/greedy preparation",
            required as u64,
            remaining as u64,
        ));
    }

    // The complete fit decision above precedes cleanup, queue work and allocation.
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("CUDA forward preparation cancelled".into()));
    }
    let gemm = CudaForwardGemm::prepare(
        service.clone(),
        &source,
        cohort_width,
        admission.layout.geometry(),
    )?;
    service
        .client()
        .memory_cleanup()
        .map_err(|error| Error::State(format!("CUDA forward cold allocator cleanup: {error}")))?;

    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: cold_bytes as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let cold = cold::CudaForwardCold::prepare(
        admission.layout.clone(),
        source,
        service.clone(),
        source_owner,
        &mut init,
        cancelled,
    )
    .await?;
    let (cohort_values, cohort_value_accesses) = if cohort_width == 0 {
        (None, Vec::new())
    } else {
        let (values, accesses) =
            CudaForwardCohortValues::allocate(&admission.layout, cohort_width, &mut init)?;
        (Some(values), accesses)
    };
    let mut greedy_accesses = Vec::with_capacity(greedy_sizes.len());
    let mut greedy_handles = Vec::with_capacity(greedy_sizes.len());
    for bytes in greedy_sizes {
        let lease = init.allocate_empty(bytes, BufferRole::Scratch)?;
        greedy_handles.push(lease.handle());
        greedy_accesses.push((lease, DeviceAccess::Write));
    }
    let greedy = greedy_handles
        .try_into()
        .map_err(|_| Error::State("CUDA forward greedy workspace coverage".into()))?;
    drop(init.complete_sync().await?);
    slot.finish()?;

    let (program, body) = body::CudaForwardBody::bind(cold, gemm)?;
    if !same_state_requirements(&program.budget().states, &admission.preflight().states)
        || program.budget().prior_state_bytes != admission.preflight().prior_state_bytes
        || program.budget().candidate_state_bytes != admission.preflight().candidate_state_bytes
    {
        return Err(Error::State(
            "CUDA forward bound program state preflight differs".into(),
        ));
    }
    Ok(PreparedCudaForward {
        program,
        owned_accesses: body.owned_accesses().to_vec(),
        cohort_values,
        cohort_value_accesses,
        greedy,
        greedy_accesses,
        state_accesses: body
            .accesses(crate::resident_program::ResidentOutput::StateOnly)
            .to_vec(),
        logit_accesses: body
            .accesses(crate::resident_program::ResidentOutput::Logits)
            .to_vec(),
    })
}

#[cfg(test)]
mod admission_matrix_tests {
    use super::CudaForwardAdmission;
    use crate::{CudaForwardWeightStorage, ResidentModel};
    use zerogravity_model_qwen35::plan::{QwenModelVariant, diagnostic_forward_plan};

    const MIB: usize = 1024 * 1024;
    const PROVIDER_CAP: usize = 4608 * MIB;

    #[test]
    fn public_32k_preflight_reports_current_storage_fit() {
        for (profile, storage) in [
            (
                QwenModelVariant::Qwen800M,
                CudaForwardWeightStorage::Resident,
            ),
            (QwenModelVariant::Qwen2B, CudaForwardWeightStorage::Streamed),
        ] {
            let plan = diagnostic_forward_plan(profile, 32_768).unwrap();
            let inventory =
                ResidentModel::cuda_forward_source_preflight(plan, 1, 0, PROVIDER_CAP).unwrap();
            eprintln!("forward-32k-current-fit profile={profile:?} inventory={inventory:?}");
            assert_eq!(inventory.context_capacity_tokens, 32_768);
            assert_eq!(inventory.root_count, 4);
            assert_eq!(inventory.cohort_width, 0);
            assert_eq!(inventory.weight_storage, storage);
            assert!(inventory.source_weight_bytes > 0);
            assert_eq!(
                inventory.resident_weight_bytes,
                match storage {
                    CudaForwardWeightStorage::Resident => inventory.source_weight_bytes,
                    CudaForwardWeightStorage::Streamed => 0,
                }
            );
            assert!(inventory.model_admission_bytes <= inventory.fit_capacity_bytes);
            assert!(inventory.total_admission_bytes <= PROVIDER_CAP);
        }
    }

    #[test]
    fn b4_activation_reserve_is_bounded_and_priced_before_residency() {
        for profile in [QwenModelVariant::Qwen800M, QwenModelVariant::Qwen2B] {
            let plan = diagnostic_forward_plan(profile, 1024).unwrap();
            let admission = CudaForwardAdmission::analyze(plan.clone())
                .unwrap()
                .with_roots(10)
                .unwrap();
            let cohort = admission.cohort_value_bytes(4).unwrap();
            assert!(cohort > 0 && cohort <= 64 * MIB);
            let greedy = crate::resident_greedy::workspace_bytes(admission.vocabulary())
                .unwrap()
                .into_iter()
                .sum::<usize>()
                + 10 * 12;
            let fit_cap = PROVIDER_CAP - cohort - greedy;
            let fitted = CudaForwardAdmission::analyze(plan)
                .unwrap()
                .with_roots(10)
                .unwrap()
                .fit(fit_cap)
                .unwrap();
            assert!(fitted.total_bytes().unwrap() <= fit_cap);
        }
    }

    #[test]
    fn placement_streams_only_weights_and_preserves_two_root_inventory() {
        let plan = diagnostic_forward_plan(QwenModelVariant::Qwen800M, 1024).unwrap();
        let resident = CudaForwardAdmission::analyze(plan.clone()).unwrap();
        let streamed = CudaForwardAdmission::analyze(plan.clone())
            .unwrap()
            .fit(512 * MIB)
            .unwrap();
        assert!(resident.layout.resident_weights());
        assert!(!streamed.layout.resident_weights());
        assert_eq!(streamed.preflight().source_bytes, 0);
        assert_eq!(
            streamed.preflight().workspace_bytes,
            resident.preflight().workspace_bytes + crate::resident_leases::source_page::PAGE_BYTES
        );
        assert_eq!(
            streamed.preflight().prior_state_bytes,
            resident.preflight().prior_state_bytes
        );
        assert_eq!(
            streamed.preflight().candidate_state_bytes,
            resident.preflight().candidate_state_bytes
        );
        assert_eq!(
            streamed.preflight().parameter_bytes,
            resident.preflight().parameter_bytes
        );
        assert!(streamed.total_bytes().unwrap() <= 512 * MIB);
        assert!(CudaForwardAdmission::analyze(plan).unwrap().fit(1).is_err());
    }

    #[test]
    fn placement_prices_every_original_root_before_selecting_residency() {
        let plan = diagnostic_forward_plan(QwenModelVariant::Qwen800M, 1024).unwrap();
        let base = CudaForwardAdmission::analyze(plan.clone()).unwrap();
        let state = base.preflight().prior_state_bytes;
        for roots in [2, 4, 6, 18] {
            let sized = CudaForwardAdmission::analyze(plan.clone())
                .unwrap()
                .with_roots(roots)
                .unwrap();
            assert_eq!(
                sized.total_bytes().unwrap(),
                base.total_bytes().unwrap() + (roots - 2) * state
            );
        }
        let complete = CudaForwardAdmission::analyze(plan.clone())
            .unwrap()
            .with_roots(6)
            .unwrap()
            .total_bytes()
            .unwrap();
        let streamed = CudaForwardAdmission::analyze(plan.clone())
            .unwrap()
            .with_roots(6)
            .unwrap()
            .fit(complete - 1)
            .unwrap();
        assert!(!streamed.layout.resident_weights());
        let floor = streamed.total_bytes().unwrap();
        assert!(
            CudaForwardAdmission::analyze(plan.clone())
                .unwrap()
                .with_roots(6)
                .unwrap()
                .fit(floor - 1)
                .is_err()
        );
        assert!(
            CudaForwardAdmission::analyze(plan.clone())
                .unwrap()
                .with_roots(1)
                .is_err()
        );
        assert!(
            CudaForwardAdmission::analyze(plan)
                .unwrap()
                .with_roots(usize::MAX)
                .is_err()
        );
    }

    #[test]
    fn source_semantic_context_matrix_reports_two_root_fit() {
        let capacities = [512usize, 1024, 2048, 4096, 8192, 16_384, 32_768];
        for profile in [QwenModelVariant::Qwen800M, QwenModelVariant::Qwen2B] {
            let mut previous = 0usize;
            for capacity in capacities {
                let plan = diagnostic_forward_plan(profile, capacity).unwrap();
                let admission = CudaForwardAdmission::analyze(plan).unwrap();
                let total = admission.total_bytes().unwrap();
                eprintln!(
                    "forward-context-fit profile={profile:?} capacity={capacity} total_bytes={total} total_mib={:.3} fits_4608={}",
                    total as f64 / MIB as f64,
                    total <= PROVIDER_CAP
                );
                assert!(total <= PROVIDER_CAP);
                assert!(total >= previous);
                previous = total;
            }
        }
    }
}
