//! Original-flight adapter for the shared allocation-free GEMM primitive.
use super::exact_view;
use crate::{
    cuda::bf16_gemm::{Bf16Gemm, Bf16GemmReduction, Bf16GemmShape},
    physical_service::CudaCompilerContextBudget,
    resident_leases::{ResidentLease, ResidentService, ResidentSubmission},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::{Error, Result, device_events::DeviceAccess};

pub(super) struct OrdinaryCudaGemm {
    service: Arc<ResidentService>,
    context: Arc<CudaCompilerContextBudget>,
    retired: AtomicBool,
}
impl OrdinaryCudaGemm {
    pub(super) fn prepare(
        service: Arc<ResidentService>,
        source: &VerifiedTensorSource,
        native_rows: usize,
    ) -> Result<Arc<Self>> {
        service.ensure_usable()?;
        if !matches!(native_rows, 0 | 4) {
            return Err(Error::Unsupported(
                "ordinary CUDA native decode rows must be zero or four".into(),
            ));
        }
        let context = service.admit_cuda_compiler_context(source)?;
        let preparation =
            context.begin_preparation(service.budget(), source, vec![service.clone()])?;
        // Arithmetic entry families only; geometry and reduction spans are scalar data.
        for rows in [8, 32] {
            for reduction in [
                Bf16GemmReduction::Ordered,
                Bf16GemmReduction::InterleavedPair { terms: 32 },
                Bf16GemmReduction::RoundedSegments { terms: 32 },
                Bf16GemmReduction::LaneStrided {
                    lanes: 16,
                    terms: 512,
                },
                Bf16GemmReduction::LaneBlocked {
                    lanes: 32,
                    terms: 32,
                },
            ] {
                Bf16Gemm::with_reduction(64, rows, 32, reduction)?
                    .prepare_code(service.pinned_cuda_queue()?)?;
            }
        }
        // Shared attention code uses the same admitted compiler/context owner.
        crate::cuda::bf16_attention::Bf16CausalAttention::new(1, 0, 1, 8, 2, 256, 0.0625)?
            .prepare_code(service.pinned_cuda_queue()?)?;
        if native_rows == 4 {
            crate::cuda::gdn::CachedConvBatch::prepare(4)?
                .prepare_code(service.pinned_cuda_queue()?)?;
        }
        preparation.complete()?;
        Ok(Arc::new(Self {
            service,
            context,
            retired: AtomicBool::new(false),
        }))
    }

    pub(super) fn require_service(&self, service: &Arc<ResidentService>) -> Result<()> {
        if !Arc::ptr_eq(&self.service, service) || self.retired.load(Ordering::Acquire) {
            return Err(Error::State(
                "ordinary CUDA GEMM service differs or retired".into(),
            ));
        }
        self.context.require(service.budget())?;
        service.ensure_usable()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue(
        &self,
        flight: &mut ResidentSubmission<'_>,
        shape: Bf16GemmShape,
        reduction: Bf16GemmReduction,
        weights: &ResidentLease,
        input: &ResidentLease,
        input_offset: usize,
        output: &ResidentLease,
        output_offset: usize,
    ) -> Result<()> {
        self.require_service(&self.service)?;
        if weights.bytes() != shape.weight_bytes() {
            return Err(Error::State(
                "ordinary CUDA GEMM weight extent differs".into(),
            ));
        }
        flight.require_accesses([
            (weights, DeviceAccess::Read),
            (input, DeviceAccess::Read),
            (output, DeviceAccess::Write),
        ])?;
        let launch = Bf16Gemm::with_reduction(shape.columns, shape.rows, shape.inner, reduction)?
            .prepare(
            self.service.client(),
            weights.handle(),
            exact_view(input, input_offset, shape.input_bytes(), "GEMM input")?,
            exact_view(output, output_offset, shape.output_bytes(), "GEMM output")?,
        )?;
        flight.retain_completion_owner(self.context.clone())?;
        launch.launch_on(self.service.client())
    }

    pub(super) fn enqueue_page(
        &self,
        shape: Bf16GemmShape,
        reduction: Bf16GemmReduction,
        weights: cubecl::server::Handle,
        input: cubecl::server::Handle,
        output: cubecl::server::Handle,
        output_stride: usize,
    ) -> Result<()> {
        self.require_service(&self.service)?;
        Bf16Gemm::with_output_stride(
            shape.columns,
            shape.rows,
            shape.inner,
            reduction,
            output_stride,
        )?
        .prepare(self.service.client(), weights, input, output)?
        .launch_on(self.service.client())
    }

    pub(super) fn retire(&self) -> Result<()> {
        self.require_service(&self.service)?;
        if self.service.model_events()? != 0 {
            return Err(Error::State(
                "ordinary CUDA GEMM retirement has unfinished events".into(),
            ));
        }
        self.retired
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("ordinary CUDA GEMM retirement raced".into()))?;
        // No per-model provider/workspace exists. Code/context remain charged to
        // the original actor until process exit; this is not context retirement.
        Ok(())
    }
}
