//! Compose head-independent GDN leaves with explicit grouped source/state views.
use super::{attention, body::CudaForwardBody, exact_view};
use crate::{
    cuda::{
        gdn::{
            GdnRuntimeBuffer as B, GdnRuntimePartition,
            head_tile::{HeadCopy, HeadGroups},
            state_transport::{
                CONV_ELEMENTS, RECURRENT_ELEMENTS, StateCopy, StateTransport, plane_handle,
            },
        },
        text_forward::Step,
    },
    resident_program::{ResidentBufferLayout, ResidentPhysicalType, StateBinding, StateBuffers},
};
use cubecl::{client::Client, server::Handle};
use zerogravity_core::{
    Error, Result,
    accelerator::program::{ForwardPlan, PlanNode, SourceOperator},
};

#[allow(clippy::too_many_arguments)]
fn copy_segments(
    client: &Client,
    groups: HeadGroups,
    first: usize,
    rows: usize,
    words: usize,
    z: bool,
    full: &Handle,
    tile: &Handle,
    scatter: bool,
) -> Result<usize> {
    let full_stride = (2 * groups.key + groups.value * if z { 2 } else { 1 }) * words;
    let tile_stride = if z { 64 } else { 48 } * words;
    let mut full_offset = 0;
    let mut launches = 0;
    for segment in 0..if z { 4 } else { 3 } {
        let key = segment < 2;
        launches += HeadCopy {
            rows,
            heads: 16,
            first_head: first,
            repeat: if key { groups.repeat() } else { 1 },
            words_per_head: words,
            full_stride,
            full_offset,
            tile_stride,
            tile_offset: segment * 16 * words,
        }
        .launch(client, full, tile, scatter)?;
        full_offset += if key { groups.key } else { groups.value } * words;
    }
    Ok(launches)
}

fn copy_coefficients(
    client: &Client,
    groups: HeadGroups,
    first: usize,
    rows: usize,
    full: &Handle,
    tile: &Handle,
) -> Result<usize> {
    let mut launches = 0;
    for segment in 0..2 {
        launches += HeadCopy {
            rows,
            heads: 8,
            first_head: first / 2,
            repeat: 1,
            words_per_head: 1,
            full_stride: groups.value,
            full_offset: segment * groups.value / 2,
            tile_stride: 16,
            tile_offset: segment * 8,
        }
        .launch(client, full, tile, false)?;
    }
    Ok(launches)
}

fn copy_values(
    client: &Client,
    groups: HeadGroups,
    first: usize,
    rows: usize,
    full: &Handle,
    tile: &Handle,
    scatter: bool,
) -> Result<usize> {
    HeadCopy {
        rows,
        heads: 16,
        first_head: first,
        repeat: 1,
        words_per_head: 64,
        full_stride: groups.value * 64,
        full_offset: 0,
        tile_stride: 16 * 64,
        tile_offset: 0,
    }
    .launch(client, full, tile, scatter)
}

impl CudaForwardBody {
    #[allow(clippy::too_many_arguments)] // Keep row/state/source bindings explicit.
    pub(super) fn enqueue_grouped_convolution(
        &self,
        request: &ForwardPlan,
        node_index: usize,
        node: &PlanNode,
        descriptor: &SourceOperator,
        state: &StateBinding,
        step: Step,
        workspace_row: usize,
    ) -> Result<(usize, usize)> {
        let ([qkvz, ba], [conv, z]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
            return Err(Error::State("CUDA grouped convolution SSA arity".into()));
        };
        let inputs = descriptor.input_elements()?;
        let outputs = descriptor.output_elements()?;
        let [weights] = self.cold().node_parameters(node_index)? else {
            return Err(Error::State(
                "CUDA grouped convolution parameter arity".into(),
            ));
        };
        let (StateBuffers::Vector(prior), StateBuffers::Vector(candidate)) =
            (&state.prior, &state.candidate)
        else {
            return Err(Error::State("CUDA grouped convolution state layout".into()));
        };
        let groups = self.cold().layout().groups();
        let layout = ResidentBufferLayout::new(ResidentPhysicalType::F32, groups.channels() * 3)?;
        prior.require_layout(layout)?;
        candidate.require_layout(layout)?;
        let runtime = self.gdn_plan(step, descriptor.call().mode())?;
        self.write_gdn_controls(&runtime)?;
        let client = self.cold().service().client();
        let cache = self.cold().gdn_internal(B::CacheArena)?;
        let view = runtime.convolution_state();
        let native = plane_handle(cache, view.offset(), view.bytes())?;
        let state_tile = self.cold().gdn_conv_tile()?.handle();
        let handles = self.gdn_internal_handles()?;
        let get = |role| {
            handles
                .get(&role)
                .ok_or_else(|| Error::State("CUDA grouped tile absent".into()))
        };
        let full_input = self.value_handle_at(request, *qkvz, inputs[0], workspace_row)?;
        let full_ba = self.value_handle_at(request, *ba, inputs[1], workspace_row)?;
        let full_conv = self.value_handle_at(request, *conv, outputs[0], workspace_row)?;
        let full_z = self.value_handle_at(request, *z, outputs[1], workspace_row)?;
        let mut dispatches = 0;
        for tile in 0..groups.tiles() {
            let first = tile * 16;
            dispatches += copy_segments(
                client,
                groups,
                first,
                step.rows,
                64,
                true,
                &full_input,
                get(B::PackedQkvz)?,
                false,
            )?;
            dispatches += copy_coefficients(
                client,
                groups,
                first,
                step.rows,
                &full_ba,
                get(B::PackedBa)?,
            )?;
            dispatches += copy_segments(
                client,
                groups,
                first,
                1,
                256,
                false,
                &weights.lease().handle(),
                get(B::ConvWeights)?,
                false,
            )?;
            dispatches += copy_segments(
                client,
                groups,
                first,
                3,
                128,
                false,
                &prior.handle,
                &state_tile,
                false,
            )?;
            StateTransport::prepare(
                self.cold().service(),
                StateCopy::ConvIn,
                state_tile.clone(),
                native.clone(),
                self.cold().status(),
            )?
            .launch(self.cold().service())?;
            dispatches += 1 + runtime
                .prepare_partition_launches(client, &handles, GdnRuntimePartition::Convolution)?
                .launch()?;
            dispatches += copy_segments(
                client,
                groups,
                first,
                step.rows,
                64,
                false,
                &full_conv,
                get(B::ConvOutput)?,
                true,
            )?;
            dispatches += copy_values(
                client,
                groups,
                first,
                step.rows,
                &full_z,
                get(B::ZCompact)?,
                true,
            )?;
            StateTransport::prepare(
                self.cold().service(),
                StateCopy::ConvOut,
                native.clone(),
                state_tile.clone(),
                self.cold().status(),
            )?
            .launch(self.cold().service())?;
            dispatches += 1 + copy_segments(
                client,
                groups,
                first,
                3,
                128,
                false,
                &candidate.handle,
                &state_tile,
                true,
            )?;
        }
        debug_assert_eq!(state_tile.size_in_used(), (CONV_ELEMENTS * 4) as u64);
        for (handle, elements) in [(&full_conv, outputs[0]), (&full_z, outputs[1])] {
            dispatches +=
                attention::certify(client, handle, &self.cold().status().handle(), elements)?;
        }
        Ok((dispatches, 0))
    }

    #[allow(clippy::too_many_arguments)] // Keep row/state/source bindings explicit.
    pub(super) fn enqueue_grouped_recurrent(
        &self,
        request: &ForwardPlan,
        node_index: usize,
        node: &PlanNode,
        descriptor: &SourceOperator,
        state: &StateBinding,
        step: Step,
        workspace_row: usize,
    ) -> Result<(usize, usize)> {
        let ([conv, z, ba], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
            return Err(Error::State("CUDA grouped recurrent SSA arity".into()));
        };
        let inputs = descriptor.input_elements()?;
        let outputs = descriptor.output_elements()?;
        let [a, dt, gamma] = self.cold().node_parameters(node_index)? else {
            return Err(Error::State(
                "CUDA grouped recurrent parameter arity".into(),
            ));
        };
        let (StateBuffers::Vector(prior), StateBuffers::Vector(candidate)) =
            (&state.prior, &state.candidate)
        else {
            return Err(Error::State("CUDA grouped recurrent state layout".into()));
        };
        let groups = self.cold().layout().groups();
        let layout =
            ResidentBufferLayout::new(ResidentPhysicalType::F32, groups.value * 128 * 128)?;
        prior.require_layout(layout)?;
        candidate.require_layout(layout)?;
        let runtime = self.gdn_plan(step, descriptor.call().mode())?;
        self.write_gdn_controls(&runtime)?;
        let client = self.cold().service().client();
        let cache = self.cold().gdn_internal(B::CacheArena)?;
        let native = plane_handle(
            cache,
            runtime.slot().recurrent_offset(),
            RECURRENT_ELEMENTS * 4,
        )?;
        let mut handles = self.gdn_internal_handles()?;
        handles.insert(B::GatedGamma, gamma.lease().handle());
        let full_conv = self.value_handle_at(request, *conv, inputs[0], workspace_row)?;
        let full_z = self.value_handle_at(request, *z, inputs[1], workspace_row)?;
        let full_ba = self.value_handle_at(request, *ba, inputs[2], workspace_row)?;
        let full_output = self.value_handle_at(request, *output, outputs[0], workspace_row)?;
        let mut dispatches = 0;
        for tile in 0..groups.tiles() {
            let first = tile * 16;
            let get = |role| {
                handles
                    .get(&role)
                    .ok_or_else(|| Error::State("CUDA grouped tile absent".into()))
            };
            dispatches += copy_segments(
                client,
                groups,
                first,
                step.rows,
                64,
                false,
                &full_conv,
                get(B::ConvOutput)?,
                false,
            )?;
            dispatches += copy_values(
                client,
                groups,
                first,
                step.rows,
                &full_z,
                get(B::ZCompact)?,
                false,
            )?;
            dispatches += copy_coefficients(
                client,
                groups,
                first,
                step.rows,
                &full_ba,
                get(B::PackedBa)?,
            )?;
            let prior_view = exact_view(
                prior.original_lease()?,
                first * 128 * 128 * 4,
                RECURRENT_ELEMENTS * 4,
                "grouped recurrent prior",
            )?;
            let candidate_view = exact_view(
                candidate.original_lease()?,
                first * 128 * 128 * 4,
                RECURRENT_ELEMENTS * 4,
                "grouped recurrent candidate",
            )?;
            StateTransport::prepare(
                self.cold().service(),
                StateCopy::RecurrentIn,
                prior_view,
                native.clone(),
                self.cold().status(),
            )?
            .launch(self.cold().service())?;
            handles.insert(
                B::ALog,
                exact_view(a.lease(), first * 4, 16 * 4, "grouped A_log")?,
            );
            handles.insert(
                B::DtBias,
                exact_view(dt.lease(), first * 2, 16 * 2, "grouped dt_bias")?,
            );
            dispatches += 1 + runtime
                .prepare_partition_launches(client, &handles, GdnRuntimePartition::Recurrent)?
                .launch()?;
            let tile = handles
                .get(&B::Gated)
                .ok_or_else(|| Error::State("CUDA grouped output tile absent".into()))?;
            dispatches += copy_values(client, groups, first, step.rows, &full_output, tile, true)?;
            StateTransport::prepare(
                self.cold().service(),
                StateCopy::RecurrentOut,
                native.clone(),
                candidate_view,
                self.cold().status(),
            )?
            .launch(self.cold().service())?;
            dispatches += 1;
        }
        dispatches += attention::certify(
            client,
            &full_output,
            &self.cold().status().handle(),
            outputs[0],
        )?;
        Ok((dispatches, 0))
    }
}
