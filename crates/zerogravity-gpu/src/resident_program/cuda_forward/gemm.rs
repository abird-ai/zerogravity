//! Original-flight adapter for the shared allocation-free GEMM primitive.
use crate::{
    cuda::bf16_gemm::{Bf16Gemm, Bf16GemmInputLayout, Bf16GemmReduction, Bf16GemmShape},
    physical_service::CudaCompilerContextBudget,
    resident_leases::{ResidentLease, ResidentService, ResidentSubmission},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::{
    Error, Result, device_events::DeviceAccess, placement::PreparedTensorLayout,
    provider::ProviderRequirements,
};

/// Original physical input orientation, for resident and streamed GEMM.
pub(super) fn input_storage(
    shape: Bf16GemmShape,
    layout: Bf16GemmInputLayout,
) -> Result<PreparedTensorLayout> {
    match layout {
        Bf16GemmInputLayout::RowMajor => {
            PreparedTensorLayout::dense(2, shape.inner * 2, shape.rows)
        }
        Bf16GemmInputLayout::InnerMajor => {
            PreparedTensorLayout::dense(2, shape.rows * 2, shape.inner)
        }
    }
}

pub(super) struct CudaForwardGemm {
    service: Arc<ResidentService>,
    context: Arc<CudaCompilerContextBudget>,
    retired: AtomicBool,
}
impl CudaForwardGemm {
    pub(super) const REQUIREMENTS: ProviderRequirements =
        ProviderRequirements::original_queue(0, 0);

    /// Bind once per original call/cohort, before any native enqueue.
    pub(super) fn bind_flight(self: &Arc<Self>, flight: &mut ResidentSubmission<'_>) -> Result<()> {
        flight.retain_completion_owner(self.clone())?;
        self.require_service(&self.service)?;
        flight.require_provider(&self.service, Self::REQUIREMENTS, None, &[])
    }

    pub(super) fn prepare(
        service: Arc<ResidentService>,
        source: &VerifiedTensorSource,
        cohort_width: usize,
        geometry: crate::cuda::text_forward::Geometry,
    ) -> Result<Arc<Self>> {
        service.ensure_usable()?;
        if !matches!(cohort_width, 0 | 4 | 8) {
            return Err(Error::Unsupported(
                "CUDA forward native decode rows must be zero, four or eight".into(),
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
                Bf16Gemm::with_reduction_certified(64, rows, 32, reduction)?
                    .prepare_code(service.pinned_cuda_queue()?)?;
            }
        }
        Bf16Gemm::with_reduction_certified(64, 65, 2048, Bf16GemmReduction::Ordered)?
            .prepare_code(service.pinned_cuda_queue()?)?;
        // Shared attention code uses the same admitted compiler/context owner.
        crate::cuda::bf16_attention::Bf16CausalAttention::new(1, 0, 1, 8, 2, 256, 0.0625)?
            .prepare_code(service.pinned_cuda_queue()?)?;
        if matches!(cohort_width, 4 | 8) {
            for terms in [256, 512] {
                for rows in [2, 4] {
                    Bf16Gemm::with_reduction_input_layout_certified(
                        64,
                        rows,
                        32,
                        Bf16GemmReduction::LaneStrided { lanes: 16, terms },
                        Bf16GemmInputLayout::InnerMajor,
                    )?
                    .prepare_code(service.pinned_cuda_queue()?)?;
                }
                if cohort_width == 8 {
                    Bf16Gemm::with_reduction_input_layout_certified(
                        64,
                        8,
                        32,
                        Bf16GemmReduction::LaneStrided { lanes: 16, terms },
                        Bf16GemmInputLayout::InnerMajor,
                    )?
                    .prepare_code(service.pinned_cuda_queue()?)?;
                }
            }
            for rows in [2, 4] {
                Bf16Gemm::with_reduction_certified(
                    64,
                    rows,
                    32,
                    Bf16GemmReduction::LaneBlocked {
                        lanes: 32,
                        terms: 32,
                    },
                )?
                .prepare_code(service.pinned_cuda_queue()?)?;
            }
            Bf16Gemm::with_reduction_certified(
                64,
                4,
                32,
                Bf16GemmReduction::LaneStrided {
                    lanes: 16,
                    terms: 512,
                },
            )?
            .prepare_code(service.pinned_cuda_queue()?)?;
            if cohort_width == 8 {
                Bf16Gemm::with_reduction_certified(64, 8, 2048, Bf16GemmReduction::Ordered)?
                    .prepare_code(service.pinned_cuda_queue()?)?;
                Bf16Gemm::with_reduction_certified(
                    64,
                    8,
                    64,
                    Bf16GemmReduction::InterleavedPair { terms: 32 },
                )?
                .prepare_code(service.pinned_cuda_queue()?)?;
                Bf16Gemm::with_reduction_certified(
                    64,
                    8,
                    32,
                    Bf16GemmReduction::LaneStrided {
                        lanes: 16,
                        terms: 512,
                    },
                )?
                .prepare_code(service.pinned_cuda_queue()?)?;
                Bf16Gemm::with_reduction_certified(
                    64,
                    8,
                    128,
                    Bf16GemmReduction::RoundedSegments { terms: 64 },
                )?
                .prepare_code(service.pinned_cuda_queue()?)?;
            }
            crate::cuda::bf16_attention::Bf16CausalAttentionBatch4::new(
                0,
                1,
                geometry.query_heads,
                geometry.kv_heads,
                256,
                0.0625,
            )?
            .prepare_code(service.pinned_cuda_queue()?)?;
            crate::cuda::gdn::CachedConvBatch::prepare(4)?
                .prepare_code(service.pinned_cuda_queue()?)?;
            crate::cuda::gdn::CachedDecodeBatch::prepare(4)?
                .prepare_code(service.pinned_cuda_queue()?)?;
            crate::cuda::gdn::CachedGatedBatch::prepare(4)?
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
                "CUDA forward GEMM service differs or retired".into(),
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
        status: &ResidentLease,
    ) -> Result<()> {
        self.enqueue_with_input_layout(
            flight,
            shape,
            reduction,
            Bf16GemmInputLayout::RowMajor,
            weights,
            input,
            input_offset,
            output,
            output_offset,
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue_with_input_layout(
        &self,
        flight: &mut ResidentSubmission<'_>,
        shape: Bf16GemmShape,
        reduction: Bf16GemmReduction,
        input_layout: Bf16GemmInputLayout,
        weights: &ResidentLease,
        input: &ResidentLease,
        input_offset: usize,
        output: &ResidentLease,
        output_offset: usize,
        status: &ResidentLease,
    ) -> Result<()> {
        self.require_service(&self.service)?;
        if weights.bytes() != shape.weight_bytes() {
            return Err(Error::State(
                "CUDA forward GEMM weight extent differs".into(),
            ));
        }
        flight.require_accesses([
            (weights, DeviceAccess::Read),
            (input, DeviceAccess::Read),
            (output, DeviceAccess::Write),
            (status, DeviceAccess::Write),
        ])?;
        let input_storage = input_storage(shape, input_layout)?;
        let output_storage = PreparedTensorLayout::dense(2, shape.columns * 2, shape.rows)?;
        let launch = Bf16Gemm::with_reduction_input_layout_certified(
            shape.columns,
            shape.rows,
            shape.inner,
            reduction,
            input_layout,
        )?
        .prepare_certified(
            self.service.client(),
            weights.handle(),
            input.typed_view(input_offset, input_storage)?,
            output.typed_view(output_offset, output_storage)?,
            status.handle(),
        )?;
        flight.retain_completion_owner(self.context.clone())?;
        launch.launch_on(self.service.client())
    }

    #[allow(clippy::too_many_arguments)] // Physical source/output/status bindings stay explicit.
    pub(super) fn enqueue_page(
        &self,
        shape: Bf16GemmShape,
        reduction: Bf16GemmReduction,
        weights: cubecl::server::Handle,
        input: cubecl::server::Handle,
        output: cubecl::server::Handle,
        output_stride: usize,
        status: cubecl::server::Handle,
    ) -> Result<()> {
        self.enqueue_page_with_input_layout(
            shape,
            reduction,
            Bf16GemmInputLayout::RowMajor,
            weights,
            input,
            output,
            output_stride,
            status,
        )
    }

    #[allow(clippy::too_many_arguments)] // Physical source/output/status bindings stay explicit.
    pub(super) fn enqueue_page_with_input_layout(
        &self,
        shape: Bf16GemmShape,
        reduction: Bf16GemmReduction,
        input_layout: Bf16GemmInputLayout,
        weights: cubecl::server::Handle,
        input: cubecl::server::Handle,
        output: cubecl::server::Handle,
        output_stride: usize,
        status: cubecl::server::Handle,
    ) -> Result<()> {
        self.require_service(&self.service)?;
        Bf16Gemm::with_output_stride_input_layout_certified(
            shape.columns,
            shape.rows,
            shape.inner,
            reduction,
            output_stride,
            input_layout,
        )?
        .prepare_certified(self.service.client(), weights, input, output, status)?
        .launch_on(self.service.client())
    }

    pub(super) fn retire(&self) -> Result<()> {
        self.require_service(&self.service)?;
        if self.service.model_events()? != 0 {
            return Err(Error::State(
                "CUDA forward GEMM retirement has unfinished events".into(),
            ));
        }
        self.retired
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("CUDA forward GEMM retirement raced".into()))?;
        // No per-model provider/workspace exists. Code/context remain charged to
        // the original actor until process exit; this is not context retirement.
        Ok(())
    }
}
