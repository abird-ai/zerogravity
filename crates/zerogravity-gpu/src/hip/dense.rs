//! Lower compact dense operations into the original prepared HIP actor.
//! Borrow the existing native set and workspace; never create a new owner.
use super::native::PreparedSourceNative;
use crate::{
    dense::{Command, DensePrecision, Transpose},
    resident_leases::{ResidentLease, ResidentService, ResidentSubmission},
    resident_program::BufferRange,
};
use std::sync::Arc;
use zerogravity_core::{
    Error, Result,
    device_events::{BufferRole, DeviceAccess},
};

pub(crate) struct Dense<'a> {
    native: &'a PreparedSourceNative,
    service: &'a Arc<ResidentService>,
    workspace: &'a ResidentLease,
}
impl<'a> Dense<'a> {
    pub(crate) fn new(
        native: &'a PreparedSourceNative,
        service: &'a Arc<ResidentService>,
        workspace: &'a ResidentLease,
        flight: &ResidentSubmission<'_>,
    ) -> Result<Self> {
        flight.require_accesses([(workspace, DeviceAccess::Write)])?;
        if !Arc::ptr_eq(workspace.service(), service)
            || workspace.role() != BufferRole::Scratch
            || workspace.bytes() != cubecl::hip::ROCBLAS_WORKSPACE_BYTES
        {
            return Err(Error::State(
                "native dense original workspace differs".into(),
            ));
        }
        Ok(Self {
            native,
            service,
            workspace,
        })
    }

    /// # Safety
    /// Each operand view belongs to an already validated original flight
    /// access. A solve RHS is initialized and retained with Write authority.
    /// Neither the native set nor any operand can retire before completion.
    pub(crate) unsafe fn enqueue(&self, command: Command<'_>) -> Result<()> {
        match command {
            Command::Gemm {
                geometry,
                left,
                right,
                output,
            } => {
                let sizes = geometry.operand_bytes()?;
                for (handle, bytes) in [left, right, output].into_iter().zip(sizes) {
                    BufferRange::new(handle)?.require_exact_len(bytes)?;
                }
                let op = match geometry.precision {
                    DensePrecision::Bf16
                        if geometry.batches == 1
                            && geometry.left == Transpose::None
                            && geometry.right == Transpose::Transposed =>
                    {
                        cubecl::hip::RocBlasGemm::new(geometry.n, geometry.k, geometry.m)
                    }
                    DensePrecision::F32 => cubecl::hip::RocBlasGemm::f32_batched(
                        geometry.m,
                        geometry.n,
                        geometry.k,
                        geometry.batches,
                        geometry.left.is_transposed(),
                        geometry.right.is_transposed(),
                    ),
                    _ => {
                        return Err(Error::Unsupported(
                            "native HIP dense precision/layout".into(),
                        ));
                    }
                }
                .map_err(|e| Error::State(format!("native dense geometry: {e}")))?;
                self.native.enqueue_rocblas(
                    self.service,
                    op,
                    [
                        right.clone(),
                        left.clone(),
                        output.clone(),
                        self.workspace.handle(),
                    ],
                )
            }
            Command::Solve {
                geometry,
                lower,
                rhs,
            } => {
                let sizes = geometry.operand_bytes()?;
                for (handle, bytes) in [lower, rhs].into_iter().zip(sizes) {
                    BufferRange::new(handle)?.require_exact_len(bytes)?;
                }
                let op = cubecl::hip::RocBlasUnitLowerSolve::new(
                    geometry.order,
                    geometry.rhs_columns,
                    geometry.batches,
                )
                .map_err(|e| Error::State(format!("native solve geometry: {e}")))?;
                self.native.enqueue_rocblas_solve(
                    self.service,
                    op,
                    [lower.clone(), rhs.clone(), self.workspace.handle()],
                )
            }
        }
    }
}
