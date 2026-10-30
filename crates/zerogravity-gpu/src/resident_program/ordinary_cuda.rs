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
    ordinary_projection_upload::PackedSourceMatrices,
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
    weights: Arc<PackedSourceMatrices>,
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
        weights: Arc<PackedSourceMatrices>,
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
        Ok(vec![
            (self.weights.lease().clone(), DeviceAccess::Read),
            (input.clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
        ])
    }

    fn enqueue(
        &self,
        descriptor: &SourceOperator,
        gemm: &OrdinaryCudaGemm,
        flight: &mut ResidentSubmission<'_>,
        input: &ResidentLease,
        output: &ResidentLease,
    ) -> Result<()> {
        let (operation, input_offset) = self.operation(descriptor)?;
        gemm.enqueue(
            flight,
            operation,
            self.arithmetic.reduction(operation),
            self.weights.lease(),
            input,
            input_offset,
            output,
            0,
        )
    }
}

pub(crate) struct OrdinaryCudaAdmission {
    layout: Arc<layout::OrdinaryCudaLayout>,
}
impl OrdinaryCudaAdmission {
    pub(crate) fn analyze(
        plan: Arc<zerogravity_core::accelerator::program::ForwardPlan>,
    ) -> Result<Self> {
        Ok(Self {
            layout: layout::OrdinaryCudaLayout::analyze(plan)?,
        })
    }
    pub(crate) fn total_bytes(&self) -> Result<usize> {
        self.layout.total_admission_bytes()
    }
    pub(crate) fn source(&self) -> zerogravity_core::accelerator::SourceIdentity {
        self.layout.plan().source()
    }
    pub(crate) fn capacity(&self) -> usize {
        self.layout.plan().prefix_capacity()
    }
    pub(crate) fn vocabulary(&self) -> usize {
        layout::VOCABULARY
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
