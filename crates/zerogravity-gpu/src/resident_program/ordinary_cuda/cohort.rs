//! Aligned independent-session batching over the ordinary CUDA body.
use super::{
    OrdinaryCudaGemm, OrdinaryCudaProjection, attention, body::OrdinaryCudaBody, exact_view,
};
use crate::{
    cuda::{
        bf16_gemm::{Bf16GemmInputLayout, Bf16GemmReduction, Bf16GemmShape},
        bf16_rows::{
            DenseRows, InnerMajorRowGather, RowSegment, TwoSegmentRows,
            launch_indexed_rows_to_inner_major, launch_row_major_to_inner_major,
            launch_two_segment_rows,
        },
        gdn::{
            CONV_STATE_BYTES, CachedConvBatch, CachedDecodeBatch, CachedDecodePrivateBatch,
            CachedGatedBatch, GdnBuffer, GdnPhase, GdnRuntimeBuffer, GdnRuntimeCacheSlot,
            state_transport::{
                CONV_ELEMENTS, RECURRENT_ELEMENTS, StateCopy, StateTransport, plane_handle,
            },
        },
        ordinary_text::{
            Access, BufferRole as OrdinaryRole, LaunchBinding, StatelessRowExtent, Step,
        },
    },
    resident_leases::{ResidentLease, ResidentSubmission as PhysicalSubmission},
    resident_program::{
        ResidentBufferLayout, ResidentPhysicalType, StateBinding, StateBuffers,
        cohort_rows::CohortRowSpans,
    },
};
use cubecl::server::Handle;
use std::collections::BTreeMap;
use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        ForwardPlan, PlanNode, PlanOperation, SourceCall, SourceCallMode, SourceOperator,
        SourceOperatorKind, SourceOperatorTemplate, SourceRowSelection, SourceRows, StateSlot,
        StateVersion, TextCallGeometry,
    },
    device_events::DeviceAccess,
};

pub(crate) struct OrdinaryCudaCohort {
    pub(crate) logits: Option<ResidentLease>,
    pub(crate) status: Handle,
    pub(crate) proposals: Vec<Vec<(StateVersion, StateBuffers)>>,
    pub(crate) dispatches: usize,
    pub(crate) fused_dispatches: usize,
}

impl OrdinaryCudaProjection {
    fn independent_operation(
        &self,
        descriptors: &[SourceOperator],
        row_counts: &[usize],
    ) -> Result<(Bf16GemmShape, Bf16GemmReduction)> {
        let lanes = descriptors.len();
        if !matches!(lanes, 2 | 4 | 8)
            || row_counts.len() != lanes
            || self.template.row_selection() != SourceRowSelection::AllActive
        {
            return Err(Error::State(
                "ordinary CUDA independent projection cohort geometry differs".into(),
            ));
        }
        let mut lane_shape: Option<Bf16GemmShape> = None;
        let mut reduction = None;
        for (lane, descriptor) in descriptors.iter().enumerate() {
            let (shape, input_offset) = self.operation(descriptor)?;
            if input_offset != 0 || shape.rows != row_counts[lane] {
                return Err(Error::State(
                    "ordinary CUDA independent projection lane span differs".into(),
                ));
            }
            let lane_reduction = self.arithmetic.reduction(shape);
            if let Some(first) = lane_shape {
                if shape.columns != first.columns || shape.inner != first.inner {
                    return Err(Error::State(
                        "ordinary CUDA independent projection lane geometry differs".into(),
                    ));
                }
            } else {
                lane_shape = Some(shape);
            }
            if reduction.is_some_and(|first| first != lane_reduction) {
                return Err(Error::Unsupported(
                    "ordinary CUDA independent projection reductions differ".into(),
                ));
            }
            reduction = Some(lane_reduction);
        }
        let lane_shape = lane_shape.ok_or_else(|| {
            Error::State("ordinary CUDA independent projection cohort empty".into())
        })?;
        let reduction = reduction.ok_or_else(|| {
            Error::State("ordinary CUDA independent projection reduction absent".into())
        })?;
        let physical_rows = row_counts.iter().try_fold(0usize, |total, rows| {
            total
                .checked_add(*rows)
                .ok_or_else(|| Error::Limit("ordinary CUDA independent projection rows".into()))
        })?;
        Ok((
            Bf16GemmShape::new(lane_shape.columns, physical_rows, lane_shape.inner)?,
            reduction,
        ))
    }

    /// Share one projection over independent source calls while retaining each
    /// call's exact reduction contract. `row_counts` is mathematical source
    /// geometry; the summed row count is only physical GEMM batching geometry.
    #[allow(clippy::too_many_arguments)]
    async fn enqueue_independent_rows(
        &self,
        descriptors: &[SourceOperator],
        row_counts: &[usize],
        gemm: &OrdinaryCudaGemm,
        flight: &mut PhysicalSubmission<'_>,
        input: &ResidentLease,
        input_layout: Bf16GemmInputLayout,
        output: &ResidentLease,
        status: &ResidentLease,
        page: Option<&std::sync::Arc<crate::resident_leases::source_page::ResidentSourcePage>>,
    ) -> Result<usize> {
        let (operation, reduction) = self.independent_operation(descriptors, row_counts)?;
        if input.bytes() < operation.input_bytes() || output.bytes() < operation.output_bytes() {
            return Err(Error::State(
                "ordinary CUDA independent projection cohort extent differs".into(),
            ));
        }
        // Reduction grouping belongs to each original source call. Never derive
        // it from the widened physical operation: row count can select a
        // different arithmetic family even when the GEMM rows are independent.
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
            Error::State("ordinary CUDA streamed independent cohort has no source page".into())
        })?;
        flight.require_accesses([
            (input, DeviceAccess::Read),
            (output, DeviceAccess::Write),
            (status, DeviceAccess::Write),
        ])?;
        let input = exact_view(
            input,
            0,
            operation.input_bytes(),
            "independent cohort GEMM input",
        )?;
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
                let shape = Bf16GemmShape::new(source_rows.len(), operation.rows, operation.inner)?;
                let offset = (base + source_rows.start) * 2;
                let extent = crate::cuda::bf16_gemm::output_span(
                    shape.columns,
                    shape.rows,
                    operation.columns,
                )?;
                let target = exact_view(output, offset, extent, "independent cohort GEMM output")?;
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

    #[allow(clippy::too_many_arguments)]
    async fn enqueue_prefill_cohort_rows(
        &self,
        descriptors: &[SourceOperator],
        spans: &CohortRowSpans,
        gemm: &OrdinaryCudaGemm,
        flight: &mut PhysicalSubmission<'_>,
        input: &ResidentLease,
        output: &ResidentLease,
        status: &ResidentLease,
        page: Option<&std::sync::Arc<crate::resident_leases::source_page::ResidentSourcePage>>,
    ) -> Result<usize> {
        let mut row_counts = [0usize; 8];
        if spans.len() > row_counts.len() {
            return Err(Error::State(
                "ordinary CUDA prefill projection cohort width differs".into(),
            ));
        }
        for (lane, count) in row_counts.iter_mut().take(spans.len()).enumerate() {
            *count = spans.lane_rows(lane)?;
        }
        self.enqueue_independent_rows(
            descriptors,
            &row_counts[..spans.len()],
            gemm,
            flight,
            input,
            Bf16GemmInputLayout::RowMajor,
            output,
            status,
            page,
        )
        .await
    }
}

impl OrdinaryCudaBody {
    fn exact_handle_view(
        handle: &Handle,
        offset: usize,
        bytes: usize,
        role: &str,
    ) -> Result<Handle> {
        let available = usize::try_from(handle.size_in_used())
            .map_err(|_| Error::Limit(format!("{role} handle extent")))?;
        let end = offset
            .checked_add(bytes)
            .filter(|end| *end <= available)
            .ok_or_else(|| Error::State(format!("{role} handle range")))?;
        Ok(handle
            .clone()
            .offset_start(offset as u64)
            .offset_end((available - end) as u64))
    }

    fn cohort_value_handle(
        &self,
        request: &ForwardPlan,
        id: zerogravity_core::accelerator::program::ValueId,
        per_row: usize,
        rows: usize,
    ) -> Result<Handle> {
        self.cohort_value_handle_range(request, id, per_row, 0, rows)
    }

    fn cohort_value_handle_range(
        &self,
        request: &ForwardPlan,
        id: zerogravity_core::accelerator::program::ValueId,
        per_row: usize,
        first_row: usize,
        rows: usize,
    ) -> Result<Handle> {
        let shape = request
            .values()
            .get(id.0 as usize)
            .ok_or_else(|| Error::State("ordinary CUDA cohort value ID absent".into()))?;
        let first = first_row
            .checked_mul(per_row)
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort row offset".into()))?;
        let elements = per_row
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort value extent".into()))?;
        let end = first
            .checked_add(elements)
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort value end".into()))?;
        if per_row == 0 || rows == 0 || end > shape.capacity_elements()? {
            return Err(Error::State(
                "ordinary CUDA cohort active rows exceed static capacity".into(),
            ));
        }
        let offset = first
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort BF16 offset".into()))?;
        let bytes = elements
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort BF16 bytes".into()))?;
        exact_view(
            self.cold().value_lease(id.0 as usize)?,
            offset,
            bytes,
            "ordinary CUDA cohort BF16 rows",
        )
    }

    #[allow(clippy::too_many_arguments)] // Keep cohort row/state/source bindings explicit.
    fn enqueue_gdn_convolution_cohort(
        &self,
        request: &ForwardPlan,
        node_index: usize,
        node: &zerogravity_core::accelerator::program::PlanNode,
        descriptor: &zerogravity_core::accelerator::program::SourceOperator,
        states: &[&StateBinding],
        row_base: usize,
    ) -> Result<(usize, usize)> {
        let ([packed_qkvz, _packed_ba], [conv_output, z]) =
            (node.inputs.as_slice(), node.outputs.as_slice())
        else {
            return Err(Error::State(
                "ordinary CUDA cohort GDN conv SSA arity".into(),
            ));
        };
        if states.len() != 4 || !self.cold().layout().groups().direct() {
            return Err(Error::Unsupported(
                "ordinary CUDA direct four-row GDN conv cohort".into(),
            ));
        }
        let lanes = states.len();
        let inputs = descriptor.input_elements()?;
        let outputs = descriptor.output_elements()?;
        let [conv_weights] = self.cold().node_parameters(node_index)? else {
            return Err(Error::State(
                "ordinary CUDA cohort GDN conv parameters".into(),
            ));
        };
        let state_layout = ResidentBufferLayout::new(ResidentPhysicalType::F32, CONV_ELEMENTS)?;
        let service = self.cold().service();
        let client = service.client();
        let cache = self.cold().gdn_internal(GdnRuntimeBuffer::CacheArena)?;
        let status = self.cold().status().handle();
        let mut dispatches = 0usize;

        for (row, state) in states.iter().enumerate() {
            let (StateBuffers::Vector(prior), StateBuffers::Vector(candidate)) =
                (&state.prior, &state.candidate)
            else {
                return Err(Error::State(
                    "ordinary CUDA cohort GDN conv state layout".into(),
                ));
            };
            prior.require_layout(state_layout)?;
            candidate.require_layout(state_layout)?;
            let slot = GdnRuntimeCacheSlot::new(row + 1)?;
            let native = plane_handle(cache, slot.convolution_offset(), CONV_STATE_BYTES)?;
            StateTransport::prepare(
                service,
                StateCopy::ConvIn,
                prior.handle.clone(),
                native,
                self.cold().status(),
            )?
            .launch(service)?;
            dispatches += 1;
        }

        let full_packed =
            self.cohort_value_handle_range(request, *packed_qkvz, inputs[0], row_base, lanes)?;
        let batch_bytes = inputs[0]
            .checked_mul(lanes)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort GDN scratch bytes".into()))?;
        let output_scratch = exact_view(
            self.cold().gdn_internal(GdnRuntimeBuffer::HChunks)?,
            row_base
                .checked_mul(inputs[0])
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| Error::Limit("ordinary GDN batch conv scratch offset".into()))?,
            batch_bytes,
            "ordinary GDN batch conv output scratch",
        )?;
        let full_conv =
            self.cohort_value_handle_range(request, *conv_output, outputs[0], row_base, lanes)?;
        let full_z = self.cohort_value_handle_range(request, *z, outputs[1], row_base, lanes)?;
        let indices_lease = self.cold().gdn_internal(GdnRuntimeBuffer::CacheIndices)?;
        let indices = exact_view(indices_lease, 0, 16, "ordinary GDN batch cache indices")?;
        CachedConvBatch::prepare(4)?.launch(
            client,
            &full_packed,
            &output_scratch,
            &conv_weights.lease().handle(),
            &cache.handle(),
            &indices,
        )?;
        dispatches += 1;

        dispatches += launch_two_segment_rows(
            client,
            TwoSegmentRows {
                rows: lanes,
                first: RowSegment {
                    stride: inputs[0],
                    offset: 0,
                    elements: outputs[0],
                },
                second: RowSegment {
                    stride: inputs[0],
                    offset: outputs[0],
                    elements: outputs[1],
                },
            },
            &output_scratch,
            &full_packed,
            &full_conv,
            &full_z,
            &status,
        )?;

        for (row, state) in states.iter().enumerate() {
            let StateBuffers::Vector(candidate) = &state.candidate else {
                unreachable!();
            };
            let slot = GdnRuntimeCacheSlot::new(row + 1)?;
            let native = plane_handle(cache, slot.convolution_offset(), CONV_STATE_BYTES)?;
            StateTransport::prepare(
                service,
                StateCopy::ConvOut,
                native,
                candidate.handle.clone(),
                self.cold().status(),
            )?
            .launch(service)?;
            dispatches += 1;
        }
        Ok((dispatches, 0))
    }

    #[allow(clippy::too_many_arguments)] // Keep cohort row/state/source bindings explicit.
    fn enqueue_gdn_recurrent_cohort(
        &self,
        request: &ForwardPlan,
        node_index: usize,
        node: &zerogravity_core::accelerator::program::PlanNode,
        descriptor: &zerogravity_core::accelerator::program::SourceOperator,
        states: &[&StateBinding],
        row_base: usize,
        cohort_rows: usize,
        values: &super::OrdinaryCudaCohortValues,
    ) -> Result<(usize, usize)> {
        let ([_conv_output, z, packed_ba], [output]) =
            (node.inputs.as_slice(), node.outputs.as_slice())
        else {
            return Err(Error::State(
                "ordinary CUDA cohort GDN recurrent SSA arity".into(),
            ));
        };
        if !matches!(states.len(), 4 | 8)
            || (states.len() == 8 && (row_base != 0 || cohort_rows != 8))
            || !self.cold().layout().groups().direct()
        {
            return Err(Error::Unsupported(
                "ordinary CUDA direct four/eight-row GDN recurrent cohort".into(),
            ));
        }
        let inputs = descriptor.input_elements()?;
        let outputs = descriptor.output_elements()?;
        let [a_log, dt_bias, gamma] = self.cold().node_parameters(node_index)? else {
            return Err(Error::State(
                "ordinary CUDA cohort GDN recurrent parameters".into(),
            ));
        };
        let recurrent_layout =
            ResidentBufferLayout::new(ResidentPhysicalType::F32, RECURRENT_ELEMENTS)?;
        let service = self.cold().service();
        let client = service.client();
        let cache = self.cold().gdn_internal(GdnRuntimeBuffer::CacheArena)?;
        let status = self.cold().status().handle();
        let mut dispatches = 0usize;
        let mut native_states = Vec::with_capacity(states.len());

        for (row, state) in states.iter().enumerate() {
            let (StateBuffers::Vector(prior), StateBuffers::Vector(candidate)) =
                (&state.prior, &state.candidate)
            else {
                return Err(Error::State(
                    "ordinary CUDA cohort GDN recurrent state layout".into(),
                ));
            };
            prior.require_layout(recurrent_layout)?;
            candidate.require_layout(recurrent_layout)?;
            let native = if row < 4 {
                let slot = GdnRuntimeCacheSlot::new(row + 1)?;
                plane_handle(cache, slot.recurrent_offset(), RECURRENT_ELEMENTS * 4)?
            } else {
                values.gdn_private_recurrent_handle(row)?
            };
            StateTransport::prepare(
                service,
                StateCopy::RecurrentIn,
                prior.handle.clone(),
                native.clone(),
                self.cold().status(),
            )?
            .launch(service)?;
            dispatches += 1;
            native_states.push(native);
        }

        let rows = states.len();
        let packed_row_bytes = GdnBuffer::PackedQkvz.bytes(GdnPhase::CachedDecode);
        let conv_scratch_bytes = packed_row_bytes
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("ordinary GDN batch decode conv scratch".into()))?;
        let core_bytes = GdnBuffer::Core
            .bytes(GdnPhase::CachedDecode)
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("ordinary GDN batch decode core scratch".into()))?;
        let rstd_bytes = GdnBuffer::Rstd
            .bytes(GdnPhase::CachedDecode)
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("ordinary GDN batch gated rstd scratch".into()))?;
        let scratch = self.cold().gdn_internal(GdnRuntimeBuffer::HChunks)?;
        let conv_scratch = exact_view(
            scratch,
            row_base
                .checked_mul(packed_row_bytes)
                .ok_or_else(|| Error::Limit("ordinary GDN batch decode conv offset".into()))?,
            conv_scratch_bytes,
            "ordinary GDN batch decode conv scratch",
        )?;
        let core_offset = packed_row_bytes
            .checked_mul(cohort_rows)
            .ok_or_else(|| Error::Limit("ordinary GDN batch core offset".into()))?;
        let core_scratch = exact_view(
            scratch,
            core_offset,
            core_bytes,
            "ordinary GDN batch decode core scratch",
        )?;
        let rstd_offset = core_offset
            .checked_add(core_bytes)
            .ok_or_else(|| Error::Limit("ordinary GDN batch rstd offset".into()))?;
        let rstd_scratch = exact_view(
            scratch,
            rstd_offset,
            rstd_bytes,
            "ordinary GDN batch gated rstd scratch",
        )?;
        let full_ba =
            self.cohort_value_handle_range(request, *packed_ba, inputs[2], row_base, rows)?;
        let ba_total = inputs[2]
            .checked_mul(rows)
            .and_then(|words| words.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary GDN batch BA bytes".into()))?;
        let ba_view = ba_total
            .checked_sub(32)
            .ok_or_else(|| Error::Limit("ordinary GDN batch BA view".into()))?;
        let a = Self::exact_handle_view(&full_ba, 32, ba_view, "ordinary GDN batch A")?;
        let beta = Self::exact_handle_view(&full_ba, 0, ba_view, "ordinary GDN batch beta")?;
        let full_z = self.cohort_value_handle_range(request, *z, inputs[1], row_base, rows)?;
        let full_output =
            self.cohort_value_handle_range(request, *output, outputs[0], row_base, rows)?;
        if rows == 8 {
            CachedDecodePrivateBatch::prepare(8)?.launch(
                client,
                &conv_scratch,
                &a,
                &beta,
                &a_log.lease().handle(),
                &dt_bias.lease().handle(),
                &core_scratch,
                &native_states,
                &native_states,
                &values.gdn_private_indices()?.handle(),
            )?;
        } else {
            let recurrent_cache = exact_view(
                cache,
                CONV_STATE_BYTES,
                cache
                    .bytes()
                    .checked_sub(CONV_STATE_BYTES)
                    .ok_or_else(|| Error::Limit("ordinary GDN recurrent cache view".into()))?,
                "ordinary GDN batch recurrent cache",
            )?;
            let indices_lease = self.cold().gdn_internal(GdnRuntimeBuffer::CacheIndices)?;
            let indices = exact_view(indices_lease, 0, 16, "ordinary GDN batch cache indices")?;
            CachedDecodeBatch::prepare(4)?.launch(
                client,
                &conv_scratch,
                &a,
                &beta,
                &a_log.lease().handle(),
                &dt_bias.lease().handle(),
                &core_scratch,
                &recurrent_cache,
                &indices,
            )?;
        }
        dispatches += 1;

        let core_row_bytes = GdnBuffer::Core.bytes(GdnPhase::CachedDecode);
        let output_row_bytes = outputs[0]
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("ordinary GDN gated output row bytes".into()))?;
        let z_row_bytes = inputs[1]
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("ordinary GDN gated z row bytes".into()))?;
        let rstd_row_bytes = GdnBuffer::Rstd.bytes(GdnPhase::CachedDecode);
        for first in (0..rows).step_by(4) {
            let core = Self::exact_handle_view(
                &core_scratch,
                first * core_row_bytes,
                4 * core_row_bytes,
                "ordinary GDN gated core group",
            )?;
            let output = Self::exact_handle_view(
                &full_output,
                first * output_row_bytes,
                4 * output_row_bytes,
                "ordinary GDN gated output group",
            )?;
            let z = Self::exact_handle_view(
                &full_z,
                first * z_row_bytes,
                4 * z_row_bytes,
                "ordinary GDN gated z group",
            )?;
            let rstd = Self::exact_handle_view(
                &rstd_scratch,
                first * rstd_row_bytes,
                4 * rstd_row_bytes,
                "ordinary GDN gated rstd group",
            )?;
            CachedGatedBatch::prepare(4)?.launch(
                client,
                &core,
                &output,
                &gamma.lease().handle(),
                &z,
                &rstd,
            )?;
            dispatches += 1;
        }
        dispatches += attention::certify(client, &full_output, &status, outputs[0] * rows)?;

        for (state, native) in states.iter().zip(native_states) {
            let StateBuffers::Vector(candidate) = &state.candidate else {
                unreachable!();
            };
            StateTransport::prepare(
                service,
                StateCopy::RecurrentOut,
                native,
                candidate.handle.clone(),
                self.cold().status(),
            )?
            .launch(service)?;
            dispatches += 1;
        }
        Ok((dispatches, 0))
    }

    /// Concatenate row-independent ordinary RMS work only when every original
    /// source call has the same Torch reduction class. This shares physical
    /// launch geometry, not numerical/session/root authority.
    fn enqueue_prefill_stateless_rms(
        &self,
        node_index: usize,
        node: &PlanNode,
        template: &SourceOperatorTemplate,
        calls: &[&SourceCall],
        spans: &CohortRowSpans,
        values: &super::OrdinaryCudaCohortValues,
    ) -> Result<Option<(usize, usize)>> {
        let (geometry, residual) = match template.kind() {
            SourceOperatorKind::Rms(geometry) => (geometry, false),
            SourceOperatorKind::ResidualRms(geometry) => (geometry, true),
            _ => return Ok(None),
        };
        if template.row_selection() != SourceRowSelection::AllActive
            || !matches!(
                template.state_access(),
                zerogravity_core::accelerator::program::SourceStateAccess::None
            )
            || spans.len() != calls.len()
        {
            return Err(Error::State(
                "ordinary CUDA stateless RMS cohort semantics differ".into(),
            ));
        }
        let program = self.cold().program();
        let hidden = program.geometry().hidden_width;
        if geometry.width != hidden {
            return Err(Error::State(
                "ordinary CUDA stateless RMS hidden width differs".into(),
            ));
        }

        let mut reduction = None;
        for (lane, call) in calls.iter().enumerate() {
            let descriptor = template.bind_runtime_call(call)?;
            let rows = spans.lane_rows(lane)?;
            let expected = rows
                .checked_mul(hidden)
                .ok_or_else(|| Error::Limit("ordinary CUDA stateless RMS lane extent".into()))?;
            let inputs = descriptor.input_elements()?;
            let outputs = descriptor.output_elements()?;
            let shape_matches = if residual {
                inputs.as_slice() == [expected, expected]
                    && outputs.as_slice() == [expected, expected]
            } else {
                inputs.as_slice() == [expected] && outputs.as_slice() == [expected]
            };
            if !shape_matches {
                return Err(Error::State(
                    "ordinary CUDA stateless RMS lane geometry differs".into(),
                ));
            }
            let class = program.rms_reduction_class(rows)?;
            if reduction.is_some_and(|first| first != class) {
                return Ok(None);
            }
            reduction.get_or_insert(class);
        }
        let reduction = reduction.ok_or_else(|| {
            Error::State("ordinary CUDA stateless RMS reduction class absent".into())
        })?;
        let bound_rows = template
            .max_rows()
            .checked_mul(spans.len())
            .ok_or_else(|| Error::Limit("ordinary CUDA stateless RMS row bound".into()))?;
        let physical = StatelessRowExtent::bounded(spans.total_rows(), bound_rows)?;
        let parameters = self.cold().node_parameters(node_index)?;
        let [gamma] = parameters else {
            return Err(Error::State(
                "ordinary CUDA stateless RMS gamma arity".into(),
            ));
        };

        if residual {
            let ([input, residual], [output, carried]) =
                (node.inputs.as_slice(), node.outputs.as_slice())
            else {
                return Err(Error::State(
                    "ordinary CUDA stateless residual RMS SSA arity".into(),
                ));
            };
            let residual_lease = values.lease(*residual)?;
            let carried_lease = values.lease(*carried)?;
            if residual_lease.id() != carried_lease.id()
                || residual_lease.allocation_generation() != carried_lease.allocation_generation()
            {
                return Err(Error::State(
                    "ordinary CUDA stateless residual RMS carried owner differs".into(),
                ));
            }
            self.enqueue_primitive(
                program.residual_rms_stateless(physical, reduction, geometry.epsilon)?,
                vec![
                    LaunchBinding::new(
                        OrdinaryRole::Input,
                        Access::Read,
                        values.packed_row_bounded_handle(*input, hidden, spans, bound_rows)?,
                    ),
                    LaunchBinding::new(
                        OrdinaryRole::Residual,
                        Access::ReadWrite,
                        values.packed_row_bounded_handle(*residual, hidden, spans, bound_rows)?,
                    ),
                    LaunchBinding::new(
                        OrdinaryRole::RawGamma,
                        Access::Read,
                        gamma.lease().handle(),
                    ),
                    LaunchBinding::new(
                        OrdinaryRole::Output,
                        Access::Write,
                        values.packed_row_bounded_handle(*output, hidden, spans, bound_rows)?,
                    ),
                ],
            )?;
        } else {
            let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                return Err(Error::State("ordinary CUDA stateless RMS SSA arity".into()));
            };
            self.enqueue_primitive(
                program.rms_stateless(physical, reduction, geometry.epsilon)?,
                vec![
                    LaunchBinding::new(
                        OrdinaryRole::Input,
                        Access::Read,
                        values.packed_row_bounded_handle(*input, hidden, spans, bound_rows)?,
                    ),
                    LaunchBinding::new(
                        OrdinaryRole::RawGamma,
                        Access::Read,
                        gamma.lease().handle(),
                    ),
                    LaunchBinding::new(
                        OrdinaryRole::Output,
                        Access::Write,
                        values.packed_row_bounded_handle(*output, hidden, spans, bound_rows)?,
                    ),
                ],
            )?;
        }
        Ok(Some((1, 0)))
    }

    /// Same-base prompt-chunk cohort: private causal/stateful execution per
    /// lane, with only reduction-compatible row-independent projection GEMMs
    /// shared physically. A caller may request state-only execution for an
    /// intermediate chunk by omitting the logits target.
    pub(crate) async fn execute_prefill_cohort(
        &self,
        flight: &mut PhysicalSubmission<'_>,
        requests: &[ForwardPlan],
        states: &[Vec<StateBinding>],
        logits: Option<ResidentLease>,
        values: &super::OrdinaryCudaCohortValues,
    ) -> Result<OrdinaryCudaCohort> {
        let lanes = requests.len();
        if !matches!(lanes, 2 | 4 | 8) || states.len() != lanes || values.width() < lanes {
            return Err(Error::Unsupported(
                "ordinary CUDA prefill cohort requires two, four or eight private lanes".into(),
            ));
        }
        let calls = requests
            .iter()
            .map(|request| self.validate_request(request))
            .collect::<Result<Vec<_>>>()?;
        let first = calls[0];
        let row_counts = calls
            .iter()
            .map(|call| call.rows().len())
            .collect::<Vec<_>>();
        let spans = CohortRowSpans::new(&row_counts)?;
        if first.mode() != SourceCallMode::Prefill
            || calls.iter().any(|call| {
                call.mode() != SourceCallMode::Prefill
                    || call.rows().base() != first.rows().base()
                    || !(1..=64).contains(&call.rows().len())
                    || call.rows().capacity() != first.rows().capacity()
                    || call.rows().vocabulary() != first.rows().vocabulary()
                    || call.source() != first.source()
            })
        {
            return Err(Error::Unsupported(
                "ordinary CUDA prefill cohort requires same-base prompt chunks".into(),
            ));
        }
        let state_maps = calls
            .iter()
            .zip(states)
            .map(|(call, state)| self.validate_states(call, state))
            .collect::<Result<Vec<_>>>()?;
        let output = if logits.is_some() {
            crate::resident_program::ResidentOutput::Logits
        } else {
            crate::resident_program::ResidentOutput::StateOnly
        };
        flight.require_accesses(
            self.accesses(output)
                .iter()
                .map(|(lease, access)| (lease, *access)),
        )?;
        if let Some(logits) = &logits {
            flight.require_accesses([(logits, DeviceAccess::Write)])?;
        }
        for lease in values.leases() {
            flight.require_accesses([(lease, DeviceAccess::Write)])?;
        }
        for row in states {
            for binding in row {
                for buffer in binding.prior.buffers() {
                    flight.require_accesses([(buffer.original_lease()?, DeviceAccess::Read)])?;
                }
                for buffer in binding.candidate.buffers() {
                    flight.require_accesses([(buffer.original_lease()?, DeviceAccess::Write)])?;
                }
            }
        }
        flight.retain_completion_owner(self.cold().clone())?;
        flight.retain_completion_owner(self.gemm().clone())?;

        let client = self.cold().service().client();
        self.clear_and_bind_call(client, first, false)?;
        let lane_steps = calls
            .iter()
            .map(|call| {
                Step::new(
                    call.rows().len(),
                    call.rows().base(),
                    call.rows().capacity(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let cold = self.cold().plan();
        let mut dispatches = 0usize;
        let mut fused_dispatches = 0usize;
        let mut gate_alias = (0..lanes)
            .map(|_| BTreeMap::<u32, u32>::new())
            .collect::<Vec<_>>();
        let nodes = if logits.is_some() {
            cold.nodes()
        } else {
            cold.state_nodes()
        };

        for (node_index, node) in nodes.iter().enumerate() {
            let PlanOperation::SourceOperatorTemplate {
                descriptor: template,
                ..
            } = &node.operation
            else {
                return Err(Error::State(
                    "ordinary CUDA prefill cohort contains non-source template".into(),
                ));
            };
            if matches!(template.kind(), SourceOperatorKind::Projection { .. }) {
                let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State(
                        "ordinary CUDA prefill cohort projection SSA arity".into(),
                    ));
                };
                let projection = self.cold().projection(node_index)?;
                if template.row_selection() == SourceRowSelection::AllActive {
                    let descriptors = calls
                        .iter()
                        .map(|call| template.bind_runtime_call(call))
                        .collect::<Result<Vec<_>>>()?;
                    let launched = projection
                        .enqueue_prefill_cohort_rows(
                            &descriptors,
                            &spans,
                            self.gemm(),
                            flight,
                            values.lease(*input)?,
                            values.lease(*output)?,
                            self.cold().status(),
                            self.cold().source_page(),
                        )
                        .await?;
                    dispatches += launched;
                    fused_dispatches += launched;
                    continue;
                }
                if template.row_selection() != SourceRowSelection::LastActive {
                    return Err(Error::State(
                        "ordinary CUDA prefill cohort projection row selection".into(),
                    ));
                }
                let hidden = self.cold().layout().geometry().hidden_width;
                let mut last_rows = Vec::with_capacity(lanes);
                for lane in 0..lanes {
                    let descriptor = template.bind_runtime_call(calls[lane])?;
                    let inputs = descriptor.input_elements()?;
                    let lane_rows = row_counts[lane];
                    if inputs.as_slice() != [lane_rows * hidden] {
                        return Err(Error::State(
                            "ordinary CUDA prefill cohort head input geometry".into(),
                        ));
                    }
                    last_rows.push(
                        spans
                            .row_span(lane)?
                            .end
                            .checked_sub(1)
                            .ok_or_else(|| Error::State("empty cohort lane span".into()))?,
                    );
                }
                let source = values.packed_row_handle(*input, hidden, &spans)?;
                let packed_bytes = lanes
                    .checked_mul(hidden)
                    .and_then(|elements| elements.checked_mul(2))
                    .ok_or_else(|| Error::Limit("ordinary CUDA prefill head bytes".into()))?;
                let packed = exact_view(
                    values.row_scratch(),
                    0,
                    packed_bytes,
                    "ordinary CUDA prefill head inner-major rows",
                )?;
                dispatches += launch_indexed_rows_to_inner_major(
                    client,
                    InnerMajorRowGather::new(spans.total_rows(), hidden, &last_rows)?,
                    &source,
                    &packed,
                    &self.cold().status().handle(),
                )?;
                let logits = logits.as_ref().ok_or_else(|| {
                    Error::State("ordinary CUDA prefill head has no logits target".into())
                })?;
                let launched = projection
                    .enqueue_decode_rows(
                        lanes,
                        self.gemm(),
                        flight,
                        values.row_scratch(),
                        Bf16GemmInputLayout::InnerMajor,
                        logits,
                        self.cold().status(),
                        self.cold().source_page(),
                    )
                    .await?;
                dispatches += launched;
                fused_dispatches += launched;
                continue;
            }

            if let Some((launched, fused)) = self
                .enqueue_prefill_stateless_rms(node_index, node, template, &calls, &spans, values)?
            {
                dispatches += launched;
                fused_dispatches += fused;
                continue;
            }

            for lane in 0..lanes {
                let descriptor = template.bind_runtime_call(calls[lane])?;
                let (launched, fused) = self
                    .enqueue_node_with_values(
                        flight,
                        cold,
                        node_index,
                        node,
                        &descriptor,
                        &state_maps[lane],
                        lane_steps[lane],
                        lane,
                        &mut gate_alias[lane],
                        values,
                        &spans,
                    )
                    .await?;
                dispatches += launched;
                fused_dispatches += fused;
            }
        }
        if let Some(page) = self.cold().source_page() {
            page.finish()?;
        }
        if gate_alias.iter().any(|aliases| !aliases.is_empty()) {
            return Err(Error::State(
                "ordinary CUDA prefill cohort attention gate alias not consumed".into(),
            ));
        }
        let expected = cold.proposed_domains();
        let proposals = states
            .iter()
            .map(|row| {
                expected
                    .iter()
                    .map(|version| {
                        let state = row
                            .iter()
                            .find(|state| state.slot == version.slot)
                            .ok_or_else(|| {
                                Error::State("ordinary CUDA prefill cohort proposal absent".into())
                            })?;
                        Ok((*version, state.candidate.clone()))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(OrdinaryCudaCohort {
            logits,
            status: self.cold().status().handle(),
            proposals,
            dispatches,
            fused_dispatches,
        })
    }

    /// One physical traversal for cached-decode rows. Stateless work is
    /// row-batched; position-sensitive/stateful operators retain each session's
    /// private causal position and root/state.
    pub(crate) async fn execute_decode_cohort(
        &self,
        flight: &mut PhysicalSubmission<'_>,
        requests: &[ForwardPlan],
        states: &[Vec<StateBinding>],
        logits: ResidentLease,
        values: &super::OrdinaryCudaCohortValues,
    ) -> Result<OrdinaryCudaCohort> {
        if !matches!(requests.len(), 2 | 4 | 8)
            || states.len() != requests.len()
            || values.width() < requests.len()
        {
            return Err(Error::Unsupported(
                "ordinary CUDA decode cohort requires two, four or eight rows".into(),
            ));
        }
        let calls = requests
            .iter()
            .map(|request| self.validate_request(request))
            .collect::<Result<Vec<_>>>()?;
        let first = calls[0];
        if first.mode() != SourceCallMode::Decode
            || first.rows().len() != 1
            || calls.iter().any(|call| {
                call.mode() != SourceCallMode::Decode
                    || call.rows().len() != 1
                    || call.rows().capacity() != first.rows().capacity()
                    || call.rows().vocabulary() != first.rows().vocabulary()
                    || call.source() != first.source()
            })
        {
            return Err(Error::Unsupported(
                "ordinary CUDA decode cohort requires compatible cached rows".into(),
            ));
        }
        let geometry = TextCallGeometry::new(
            first.source(),
            first.rows().vocabulary(),
            first.rows().capacity(),
        )?;
        let ids = calls
            .iter()
            .map(|call| call.rows().ids()[0])
            .collect::<Vec<_>>();
        let positions = calls
            .iter()
            .map(|call| call.rows().base())
            .collect::<Vec<_>>();
        // This synthetic call owns only dense row shape/token packing for
        // position-independent operators. Position-sensitive operators bind
        // `positions` explicitly below; no causal/root authority comes from it.
        let batch_call = SourceCall::new(
            SourceRows::new(geometry, 0, ids.into())?,
            SourceCallMode::Prefill,
        )?;
        let state_maps = calls
            .iter()
            .zip(states)
            .map(|(call, state)| self.validate_states(call, state))
            .collect::<Result<Vec<_>>>()?;
        flight.require_accesses(
            self.accesses(crate::resident_program::ResidentOutput::Logits)
                .iter()
                .map(|(lease, access)| (lease, *access)),
        )?;
        flight.require_accesses([(&logits, DeviceAccess::Write)])?;
        for row in states {
            for binding in row {
                for buffer in binding.prior.buffers() {
                    flight.require_accesses([(buffer.original_lease()?, DeviceAccess::Read)])?;
                }
                for buffer in binding.candidate.buffers() {
                    flight.require_accesses([(buffer.original_lease()?, DeviceAccess::Write)])?;
                }
            }
        }
        flight.retain_completion_owner(self.cold().clone())?;
        flight.retain_completion_owner(self.gemm().clone())?;

        let client = self.cold().service().client();
        self.clear_and_bind_call(client, &batch_call, false)?;
        let batch_step = Step::new(requests.len(), 0, first.rows().capacity())?;
        let cold = self.cold().plan();
        let mut dispatches = 0usize;
        let mut fused_dispatches = 0usize;
        let mut gate_alias = BTreeMap::<u32, u32>::new();

        for (node_index, node) in cold.nodes().iter().enumerate() {
            let PlanOperation::SourceOperatorTemplate {
                descriptor: template,
                ..
            } = &node.operation
            else {
                return Err(Error::State(
                    "ordinary CUDA cohort contains non-source template".into(),
                ));
            };
            if template.row_selection() == SourceRowSelection::LastActive {
                if !matches!(template.kind(), SourceOperatorKind::Projection { .. })
                    || node.inputs.len() != 1
                    || node.outputs.len() != 1
                {
                    return Err(Error::State(
                        "ordinary CUDA cohort last-active operator differs".into(),
                    ));
                }
                let input = self.cold().value_lease(node.inputs[0].0 as usize)?;
                let (input, input_layout) = if matches!(requests.len(), 2 | 4 | 8) {
                    flight.require_accesses([(values.row_scratch(), DeviceAccess::Write)])?;
                    let hidden = self.cold().layout().geometry().hidden_width;
                    let bytes = hidden
                        .checked_mul(requests.len())
                        .and_then(|elements| elements.checked_mul(2))
                        .ok_or_else(|| {
                            Error::Limit("ordinary CUDA native head input bytes".into())
                        })?;
                    let source =
                        exact_view(input, 0, bytes, "ordinary CUDA native head row-major input")?;
                    let packed = exact_view(
                        values.row_scratch(),
                        0,
                        bytes,
                        "ordinary CUDA native head inner-major input",
                    )?;
                    dispatches += launch_row_major_to_inner_major(
                        client,
                        DenseRows {
                            rows: requests.len(),
                            columns: hidden,
                        },
                        &source,
                        &packed,
                        &self.cold().status().handle(),
                    )?;
                    (values.row_scratch(), Bf16GemmInputLayout::InnerMajor)
                } else {
                    (input, Bf16GemmInputLayout::RowMajor)
                };
                let launched = self
                    .cold()
                    .projection(node_index)?
                    .enqueue_decode_rows(
                        requests.len(),
                        self.gemm(),
                        flight,
                        input,
                        input_layout,
                        &logits,
                        self.cold().status(),
                        self.cold().source_page(),
                    )
                    .await?;
                dispatches += launched;
                fused_dispatches += launched;
                continue;
            }
            let stateful = !matches!(
                template.state_access(),
                zerogravity_core::accelerator::program::SourceStateAccess::None
            );
            if stateful {
                if let SourceOperatorKind::CausalConvSplit { domain, .. } = template.kind()
                    && self.cold().layout().groups().direct()
                    && requests.len() >= 4
                {
                    let bound = state_maps
                        .iter()
                        .map(|states| {
                            states
                                .get(&StateSlot::Conv(domain))
                                .copied()
                                .ok_or_else(|| {
                                    Error::State("ordinary CUDA cohort conv state absent".into())
                                })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    for row_base in (0..requests.len()).step_by(4) {
                        let descriptor = template.bind_runtime_call(calls[row_base])?;
                        let (launched, fused) = self.enqueue_gdn_convolution_cohort(
                            cold,
                            node_index,
                            node,
                            &descriptor,
                            &bound[row_base..row_base + 4],
                            row_base,
                        )?;
                        dispatches += launched;
                        fused_dispatches += fused;
                    }
                    continue;
                }
                if let SourceOperatorKind::GatedDelta { domain, .. } = template.kind()
                    && self.cold().layout().groups().direct()
                    && requests.len() >= 4
                {
                    let bound = state_maps
                        .iter()
                        .map(|states| {
                            states
                                .get(&StateSlot::Recurrent(domain))
                                .copied()
                                .ok_or_else(|| {
                                    Error::State(
                                        "ordinary CUDA cohort recurrent state absent".into(),
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    if requests.len() == 8 {
                        let descriptor = template.bind_runtime_call(calls[0])?;
                        let (launched, fused) = self.enqueue_gdn_recurrent_cohort(
                            cold,
                            node_index,
                            node,
                            &descriptor,
                            &bound,
                            0,
                            requests.len(),
                            values,
                        )?;
                        dispatches += launched;
                        fused_dispatches += fused;
                    } else {
                        for row_base in (0..requests.len()).step_by(4) {
                            let descriptor = template.bind_runtime_call(calls[row_base])?;
                            let (launched, fused) = self.enqueue_gdn_recurrent_cohort(
                                cold,
                                node_index,
                                node,
                                &descriptor,
                                &bound[row_base..row_base + 4],
                                row_base,
                                requests.len(),
                                values,
                            )?;
                            dispatches += launched;
                            fused_dispatches += fused;
                        }
                    }
                    continue;
                }
                if let SourceOperatorKind::FlashAttention { domain, .. } = template.kind()
                    && requests.len() >= 4
                {
                    let ([query], [output]) = (node.inputs.as_slice(), node.outputs.as_slice())
                    else {
                        return Err(Error::State(
                            "ordinary CUDA cohort attention SSA arity".into(),
                        ));
                    };
                    let kv = state_maps
                        .iter()
                        .map(|states| {
                            let state =
                                states.get(&StateSlot::Kv(domain)).copied().ok_or_else(|| {
                                    Error::State(
                                        "ordinary CUDA cohort attention state absent".into(),
                                    )
                                })?;
                            let StateBuffers::Kv { keys, values } = &state.candidate else {
                                return Err(Error::State(
                                    "ordinary CUDA cohort attention KV layout".into(),
                                ));
                            };
                            Ok((keys, values))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    for row_base in (0..requests.len()).step_by(4) {
                        let descriptor = template.bind_runtime_call(calls[row_base])?;
                        let bases = std::array::from_fn(|lane| positions[row_base + lane]);
                        let inputs = descriptor.input_elements()?;
                        let outputs = descriptor.output_elements()?;
                        let query =
                            self.cohort_value_handle_range(cold, *query, inputs[0], row_base, 4)?;
                        let output =
                            self.cohort_value_handle_range(cold, *output, outputs[0], row_base, 4)?;
                        dispatches += self.cold().attention().enqueue_cohort(
                            client,
                            &descriptor,
                            bases,
                            &query,
                            &kv[row_base..row_base + 4],
                            &output,
                            self.cold().status(),
                        )?;
                    }
                    continue;
                }
                for row in 0..requests.len() {
                    let descriptor = template.bind_runtime_call(calls[row])?;
                    let single_step = Step::new(1, positions[row], first.rows().capacity())?;
                    let (launched, fused) = self
                        .enqueue_node(
                            flight,
                            cold,
                            node_index,
                            node,
                            &descriptor,
                            &state_maps[row],
                            single_step,
                            row,
                            None,
                            None,
                            &mut gate_alias,
                        )
                        .await?;
                    dispatches += launched;
                    fused_dispatches += fused;
                }
            } else {
                if matches!(template.kind(), SourceOperatorKind::Projection { .. }) {
                    let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice())
                    else {
                        return Err(Error::State(
                            "ordinary CUDA independent decode projection SSA arity".into(),
                        ));
                    };
                    let descriptors = calls
                        .iter()
                        .map(|call| template.bind_runtime_call(call))
                        .collect::<Result<Vec<_>>>()?;
                    let row_counts = [1usize; 8];
                    let row_counts = &row_counts[..calls.len()];
                    let projection = self.cold().projection(node_index)?;
                    let (operation, reduction) =
                        projection.independent_operation(&descriptors, row_counts)?;
                    let source = self.cold().value_lease(input.0 as usize)?;
                    let (projection_input, input_layout) =
                        if matches!(
                            reduction,
                            Bf16GemmReduction::LaneStrided {
                                lanes: 16,
                                terms: 256 | 512
                            }
                        ) && matches!(operation.rows, 2 | 4 | 8)
                            && values.row_scratch().bytes() >= operation.input_bytes()
                        {
                            flight
                                .require_accesses([(values.row_scratch(), DeviceAccess::Write)])?;
                            let source = exact_view(
                                source,
                                0,
                                operation.input_bytes(),
                                "ordinary CUDA exact decode row-major input",
                            )?;
                            let packed = exact_view(
                                values.row_scratch(),
                                0,
                                operation.input_bytes(),
                                "ordinary CUDA exact decode inner-major input",
                            )?;
                            dispatches += launch_row_major_to_inner_major(
                                client,
                                DenseRows {
                                    rows: operation.rows,
                                    columns: operation.inner,
                                },
                                &source,
                                &packed,
                                &self.cold().status().handle(),
                            )?;
                            (values.row_scratch(), Bf16GemmInputLayout::InnerMajor)
                        } else {
                            (source, Bf16GemmInputLayout::RowMajor)
                        };
                    let launched = projection
                        .enqueue_independent_rows(
                            &descriptors,
                            row_counts,
                            self.gemm(),
                            flight,
                            projection_input,
                            input_layout,
                            self.cold().value_lease(output.0 as usize)?,
                            self.cold().status(),
                            self.cold().source_page(),
                        )
                        .await?;
                    dispatches += launched;
                    fused_dispatches += launched;
                    continue;
                }
                let descriptor = template.bind_runtime_call(&batch_call)?;
                let row_positions =
                    matches!(template.kind(), SourceOperatorKind::QkRmsRotaryGated(_))
                        .then_some(positions.as_slice());
                let (launched, fused) = self
                    .enqueue_node(
                        flight,
                        cold,
                        node_index,
                        node,
                        &descriptor,
                        &state_maps[0],
                        batch_step,
                        0,
                        row_positions,
                        None,
                        &mut gate_alias,
                    )
                    .await?;
                dispatches += launched;
                fused_dispatches += fused;
            }
        }
        if let Some(page) = self.cold().source_page() {
            page.finish()?;
        }
        if !gate_alias.is_empty() {
            return Err(Error::State(
                "ordinary CUDA cohort attention gate alias not consumed".into(),
            ));
        }
        let expected = cold.proposed_domains();
        let proposals = states
            .iter()
            .map(|row| {
                expected
                    .iter()
                    .map(|version| {
                        let state = row
                            .iter()
                            .find(|state| state.slot == version.slot)
                            .ok_or_else(|| {
                                Error::State("ordinary CUDA cohort proposal absent".into())
                            })?;
                        Ok((*version, state.candidate.clone()))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(OrdinaryCudaCohort {
            logits: Some(logits),
            status: self.cold().status().handle(),
            proposals,
            dispatches,
            fused_dispatches,
        })
    }
}
