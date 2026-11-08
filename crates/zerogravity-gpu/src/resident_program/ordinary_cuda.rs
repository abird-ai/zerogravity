//! Production ordinary-CUDA physical components.
//!
//! This module deliberately owns no model loop and no StateRoot publication.
//! It composes the existing pinned ResidentService/PhysicalQueue/flight owner;
//! every submitted native object is retained by that original flight until
//! actual completion or quarantine.
#![allow(dead_code)] // Private unselected implementation until the full body lands.

use crate::cuda::bf16_gemm::Bf16GemmShape;
use crate::resident_leases::{
    ResidentInitialization, ResidentLease, ResidentService, ResidentSubmission,
    ordinary_projection_upload::BoundSourceMatrices,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
mod arithmetic;
use arithmetic::ProjectionUse;
mod gemm;
use gemm::OrdinaryCudaGemm;
#[cfg(test)]
pub(crate) mod blas_reference;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
mod attention;
pub(super) mod body;
mod cold;
mod execute;
mod geometry;
mod grouped_gdn;
mod layout;
pub(crate) mod rotary;

use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        SourceNumericalContract, SourceOperator, SourceOperatorKind, SourceOperatorTemplate,
        SourceRowSelection,
    },
    device_events::{BufferRole, DeviceAccess},
    resource_arbiter::{AllocationClass, AllocationRequest},
};

#[cfg(test)]
fn server_error(error: impl std::fmt::Display) -> Error {
    Error::State(format!("ordinary CUDA cuBLAS owner: {error}"))
}

fn exact_view(
    lease: &ResidentLease,
    offset: usize,
    bytes: usize,
    role: &str,
) -> Result<cubecl::server::Handle> {
    let end = offset
        .checked_add(bytes)
        .filter(|end| *end <= lease.bytes())
        .ok_or_else(|| Error::Limit(format!("ordinary CUDA {role} view exceeds lease")))?;
    Ok(lease
        .handle()
        .offset_start(offset as u64)
        .offset_end((lease.bytes() - end) as u64))
}

/// One source-derived full projection. Cold ownership binds the call-free
/// template; request-local row geometry never owns or reloads weights.
pub(crate) struct OrdinaryCudaProjection {
    template: SourceOperatorTemplate,
    weights: Arc<BoundSourceMatrices>,
    arithmetic: ProjectionUse,
}
impl OrdinaryCudaProjection {
    pub(crate) async fn prepare(
        service: &Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        source: VerifiedTensorSource,
        template: &SourceOperatorTemplate,
        arithmetic: ProjectionUse,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        if template.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1 {
            return Err(Error::Unsupported(
                "ordinary CUDA projection numerical contract differs".into(),
            ));
        }
        if !matches!(
            template.kind(),
            SourceOperatorKind::Projection { selected_row: None }
        ) || template.source() != source.source_identity()
            || template.matrices().is_empty()
            || !template.parameters().is_empty()
        {
            return Err(Error::State(
                "ordinary CUDA projection template/source contract differs".into(),
            ));
        }
        let weights = init
            .upload_ordinary_projection(service, source, template.matrices(), cancelled)
            .await?;
        Self::prepare_with_weights(template, weights, arithmetic)
    }

    pub(crate) fn prepare_with_weights(
        template: &SourceOperatorTemplate,
        weights: Arc<BoundSourceMatrices>,
        arithmetic: ProjectionUse,
    ) -> Result<Arc<Self>> {
        if template.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1
            || !matches!(
                template.kind(),
                SourceOperatorKind::Projection { selected_row: None }
            )
            || template.source() != weights.source_identity()
            || template.matrices().is_empty()
            || !template.parameters().is_empty()
        {
            return Err(Error::State(
                "ordinary CUDA projection shared weights/template differs".into(),
            ));
        }
        weights.require_members(template.matrices())?;
        Ok(Arc::new(Self {
            template: template.clone(),
            weights,
            arithmetic,
        }))
    }

    fn operation(&self, descriptor: &SourceOperator) -> Result<(Bf16GemmShape, usize)> {
        descriptor.validate()?;
        let rows = descriptor.call().rows().len();
        let SourceOperatorKind::Projection { selected_row } = descriptor.kind() else {
            return Err(Error::State(
                "ordinary CUDA projection runtime descriptor kind differs".into(),
            ));
        };
        let expected_selected = match self.template.row_selection() {
            SourceRowSelection::AllActive => None,
            SourceRowSelection::LastActive => Some(
                rows.checked_sub(1)
                    .ok_or_else(|| Error::State("ordinary CUDA projection empty call".into()))?,
            ),
        };
        if descriptor.numerical_contract() != self.template.numerical_contract()
            || descriptor.call().source() != self.template.source()
            || descriptor.call().rows().vocabulary() != self.template.vocabulary()
            || rows > self.template.max_rows()
            || descriptor.matrices() != self.template.matrices()
            || descriptor.parameters() != self.template.parameters()
            || selected_row != expected_selected
        {
            return Err(Error::State(
                "ordinary CUDA projection runtime/template semantics differ".into(),
            ));
        }
        self.weights.require_members(descriptor.matrices())?;
        let active_rows = usize::from(selected_row.is_none())
            .checked_mul(rows)
            .and_then(|all| all.checked_add(usize::from(selected_row.is_some())))
            .ok_or_else(|| Error::Limit("ordinary CUDA projection active rows".into()))?;
        let operation =
            Bf16GemmShape::new(self.weights.rows(), active_rows, self.weights.columns())?;
        if descriptor.input_elements()?.as_slice() != [rows * self.weights.columns()]
            || descriptor.output_elements()?.as_slice() != [active_rows * self.weights.rows()]
        {
            return Err(Error::State(
                "ordinary CUDA projection operator/packed geometry differs".into(),
            ));
        }
        let input_offset = selected_row
            .unwrap_or(0)
            .checked_mul(self.weights.columns())
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary CUDA projection input offset overflow".into()))?;
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
                "ordinary CUDA projection input/output lease extent differs".into(),
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

    async fn enqueue(
        &self,
        descriptor: &SourceOperator,
        gemm: &OrdinaryCudaGemm,
        flight: &mut ResidentSubmission<'_>,
        input: &ResidentLease,
        output: &ResidentLease,
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
            )?;
            return Ok(1);
        }
        let page = page
            .ok_or_else(|| Error::State("ordinary CUDA streamed projection has no page".into()))?;
        flight.require_accesses([(input, DeviceAccess::Read), (output, DeviceAccess::Write)])?;
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

pub(crate) struct OrdinaryCudaAdmission {
    layout: Arc<layout::OrdinaryCudaLayout>,
    roots: usize,
}
impl OrdinaryCudaAdmission {
    pub(crate) fn analyze(
        plan: Arc<zerogravity_core::accelerator::program::ForwardPlan>,
    ) -> Result<Self> {
        Ok(Self {
            layout: layout::OrdinaryCudaLayout::analyze(plan)?,
            roots: 2,
        })
    }
    pub(crate) fn with_roots(mut self, roots: usize) -> Result<Self> {
        if roots < 2 {
            return Err(Error::Limit(
                "ordinary CUDA requires prior/candidate roots".into(),
            ));
        }
        self.roots = roots;
        self.total_bytes()?;
        Ok(self)
    }
    pub(crate) fn fit(mut self, provider_capacity: usize) -> Result<Self> {
        if self.total_bytes()? > provider_capacity {
            Arc::make_mut(&mut self.layout).stream_weights()?;
        }
        let total = self.total_bytes()?;
        if total > provider_capacity {
            return Err(Error::Limit(format!(
                "ordinary CUDA workspace/{}-root plan does not fit: {total} > {provider_capacity}",
                self.roots
            )));
        }
        Ok(self)
    }
    pub(crate) fn total_bytes(&self) -> Result<usize> {
        let extra = self
            .layout
            .preflight()
            .prior_state_bytes
            .checked_mul(self.roots - 2)
            .ok_or_else(|| Error::Limit("ordinary CUDA root inventory overflow".into()))?;
        self.layout
            .total_admission_bytes()?
            .checked_add(extra)
            .ok_or_else(|| Error::Limit("ordinary CUDA complete root inventory overflow".into()))
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
}

pub(crate) struct PreparedOrdinaryCuda {
    pub(crate) program: crate::resident_program::ResidentProgram,
    pub(crate) owned_accesses: Vec<(ResidentLease, DeviceAccess)>,
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
    admission: OrdinaryCudaAdmission,
    source: VerifiedTensorSource,
    source_owner: Option<Arc<dyn std::any::Any + Send + Sync>>,
    cancelled: Option<&AtomicBool>,
) -> Result<PreparedOrdinaryCuda> {
    service.ensure_usable()?;
    if service.source_identity() != admission.source()
        || source.source_identity() != admission.source()
    {
        return Err(Error::State(
            "ordinary CUDA preparation source/service differs".into(),
        ));
    }
    let cold_bytes = admission.preflight().owned_bytes()?;
    if admission.total_bytes()? > service.remaining_capacity()?
        || cold_bytes > service.remaining_capacity()?
    {
        return Err(Error::Limit(
            "ordinary CUDA complete cold/two-root admission exceeds model capacity".into(),
        ));
    }

    // The complete fit decision above precedes cleanup, queue work and allocation.
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("ordinary CUDA preparation cancelled".into()));
    }
    let gemm = OrdinaryCudaGemm::prepare(service.clone(), &source)?;
    service
        .client()
        .memory_cleanup()
        .map_err(|error| Error::State(format!("ordinary CUDA cold allocator cleanup: {error}")))?;

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
    let cold = cold::OrdinaryCudaCold::prepare(
        admission.layout.clone(),
        source,
        service.clone(),
        source_owner,
        &mut init,
        cancelled,
    )
    .await?;
    drop(init.complete_sync().await?);
    slot.finish()?;

    let (program, body) = body::OrdinaryCudaBody::bind(cold, gemm)?;
    if !same_state_requirements(&program.budget().states, &admission.preflight().states)
        || program.budget().prior_state_bytes != admission.preflight().prior_state_bytes
        || program.budget().candidate_state_bytes != admission.preflight().candidate_state_bytes
    {
        return Err(Error::State(
            "ordinary CUDA bound program state preflight differs".into(),
        ));
    }
    Ok(PreparedOrdinaryCuda {
        program,
        owned_accesses: body.owned_accesses().to_vec(),
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
    use super::OrdinaryCudaAdmission;
    use zerogravity_model_qwen35::plan::{QwenSourceProfile, diagnostic_ordinary_plan};

    const MIB: usize = 1024 * 1024;
    const PROVIDER_CAP: usize = 4608 * MIB;

    #[test]
    fn placement_streams_only_weights_and_preserves_two_root_inventory() {
        let plan = diagnostic_ordinary_plan(QwenSourceProfile::Qwen08, 1024).unwrap();
        let resident = OrdinaryCudaAdmission::analyze(plan.clone()).unwrap();
        let streamed = OrdinaryCudaAdmission::analyze(plan.clone())
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
        assert!(
            OrdinaryCudaAdmission::analyze(plan)
                .unwrap()
                .fit(1)
                .is_err()
        );
    }

    #[test]
    fn placement_prices_every_original_root_before_selecting_residency() {
        let plan = diagnostic_ordinary_plan(QwenSourceProfile::Qwen08, 1024).unwrap();
        let base = OrdinaryCudaAdmission::analyze(plan.clone()).unwrap();
        let state = base.preflight().prior_state_bytes;
        for roots in [2, 4, 6, 18] {
            let sized = OrdinaryCudaAdmission::analyze(plan.clone())
                .unwrap()
                .with_roots(roots)
                .unwrap();
            assert_eq!(
                sized.total_bytes().unwrap(),
                base.total_bytes().unwrap() + (roots - 2) * state
            );
        }
        let complete = OrdinaryCudaAdmission::analyze(plan.clone())
            .unwrap()
            .with_roots(6)
            .unwrap()
            .total_bytes()
            .unwrap();
        let streamed = OrdinaryCudaAdmission::analyze(plan.clone())
            .unwrap()
            .with_roots(6)
            .unwrap()
            .fit(complete - 1)
            .unwrap();
        assert!(!streamed.layout.resident_weights());
        let floor = streamed.total_bytes().unwrap();
        assert!(
            OrdinaryCudaAdmission::analyze(plan.clone())
                .unwrap()
                .with_roots(6)
                .unwrap()
                .fit(floor - 1)
                .is_err()
        );
        assert!(
            OrdinaryCudaAdmission::analyze(plan.clone())
                .unwrap()
                .with_roots(1)
                .is_err()
        );
        assert!(
            OrdinaryCudaAdmission::analyze(plan)
                .unwrap()
                .with_roots(usize::MAX)
                .is_err()
        );
    }

    #[test]
    fn source_semantic_context_matrix_reports_exact_two_root_fit() {
        let capacities = [512usize, 1024, 2048, 4096, 8192, 16_384, 32_768];
        let expected = [
            (
                QwenSourceProfile::Qwen08,
                [
                    1_652_916_585usize,
                    1_667_662_185,
                    1_697_153_385,
                    1_756_135_785,
                    1_874_100_585,
                    2_110_030_185,
                    2_581_889_385,
                ],
                [true; 7],
            ),
            (
                QwenSourceProfile::Qwen2B,
                [
                    3_911_846_249usize,
                    3_926_591_849,
                    3_956_083_049,
                    4_015_065_449,
                    4_133_030_249,
                    4_368_959_849,
                    4_840_819_049,
                ],
                [true, true, true, true, true, true, false],
            ),
        ];
        for (profile, totals, fits) in expected {
            let mut previous = 0usize;
            for ((capacity, expected_total), expected_fit) in
                capacities.into_iter().zip(totals).zip(fits)
            {
                let plan = diagnostic_ordinary_plan(profile, capacity).unwrap();
                let admission = OrdinaryCudaAdmission::analyze(plan).unwrap();
                let total = admission.total_bytes().unwrap();
                eprintln!(
                    "ordinary-context-fit profile={profile:?} capacity={capacity} total_bytes={total} total_mib={:.3} fits_4608={}",
                    total as f64 / MIB as f64,
                    total <= PROVIDER_CAP
                );
                assert_eq!(total, expected_total);
                assert_eq!(total <= PROVIDER_CAP, expected_fit);
                assert!(total >= previous);
                previous = total;
            }
        }
    }
}
