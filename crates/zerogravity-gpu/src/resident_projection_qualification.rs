//! H23 host-only equivalence over actual admitted immutable GPU bindings.
//! This diagnostic neither allocates/uploads GPU data nor launches any kernel.
use super::*;
use std::{hint::black_box, time::Instant};

fn failure(detail: impl std::fmt::Display) -> Error {
    Error::State(format!("H23 prepared projection: {detail}"))
}
fn same(a: &Handle, b: &Handle) -> Result<bool> {
    let ar = BufferRange::new(a)?;
    let br = BufferRange::new(b)?;
    Ok(ar.service == br.service
        && ar.memory == br.memory
        && ar.bytes == br.bytes
        && a.size() == b.size()
        && a.offset_start == b.offset_start
        && a.offset_end == b.offset_end
        && a.stream == b.stream)
}
fn workspace(a: &DeviceActivation) -> [Handle; 7] {
    [
        a.parameters.clone(),
        a.scaled.clone(),
        a.codes.clone(),
        a.scales.clone(),
        a.offsets.clone(),
        a.block_status.clone(),
        a.row_status.clone(),
    ]
}

/// Frozen H22 reconstruction, independent of the new cold helper/cache loop.
fn legacy(
    program: &ResidentProgram,
    client: &Client,
    index: usize,
    tensor: &SourceTensor,
) -> Result<DeviceActivation> {
    let bindings = program.legacy()?;
    let t = &bindings.tensors[&tensor.name];
    let node = &program.plan.nodes()[index];
    let flags = &bindings.node_status[index];
    DeviceActivation::from_bound(
        client,
        &t.layout.ok_or_else(|| failure("legacy quant layout"))?,
        program.values[node.inputs[0].0 as usize].handle.clone(),
        t.scales.clone(),
        t.biases.clone(),
        [
            bindings.scratch[0].clone(),
            bindings.scratch[1].clone(),
            bindings.scratch[2].clone(),
            bindings.scratch[3].clone(),
            bindings.scratch[4].clone(),
            flags
                .block
                .as_ref()
                .ok_or_else(|| failure("legacy block status"))?
                .clone(),
            flags
                .row
                .as_ref()
                .ok_or_else(|| failure("legacy row status"))?
                .clone(),
        ],
    )
}

fn negative_constructors(
    client: &Client,
    layout: ActivationLayout,
    a: &DeviceActivation,
) -> Result<usize> {
    let build = |layout: &ActivationLayout, input: Handle| {
        DeviceActivation::from_bound(
            client,
            layout,
            input,
            a.weight_scales.clone(),
            a.biases.clone(),
            workspace(a),
        )
    };
    let mut malformed = layout;
    malformed.blocks += 1;
    if build(&malformed, a.input.clone()).is_ok() {
        return Err(failure("malformed layout accepted"));
    }
    let mut malformed = layout;
    malformed.signed_scale = true;
    malformed.with_bias = true;
    if build(&malformed, a.input.clone()).is_ok() {
        return Err(failure("malformed source variant accepted"));
    }
    // Keep a legitimate positive byte range: the old constructor's unchecked
    // size_in_used must never receive malformed/underflowing Handle offsets.
    let range = BufferRange::new(&a.input)?;
    let available = range.bytes.end - range.bytes.start;
    let required = (layout.columns * 4) as u64;
    if required <= 4 || available < required {
        return Err(failure("short-input fixture geometry"));
    }
    let short = a.input.clone().offset_end(available - required + 4);
    BufferRange::new(&short)?;
    if build(&layout, short).is_ok() {
        return Err(failure("short input accepted"));
    }
    struct ForeignService;
    let mut foreign = a.input.clone();
    foreign.service.service = std::any::TypeId::of::<ForeignService>();
    if build(&layout, foreign).is_ok() {
        return Err(failure("foreign service accepted"));
    }
    Ok(4)
}

impl ResidentProgram {
    /// Compare all ten cached handle coordinates to H22 and explicit typed-node
    /// ownership. Timing covers host construction/lookup only, never inference.
    pub(crate) fn qualify_projection_bindings(&self, client: &Client) -> Result<()> {
        let bindings = self.legacy()?;
        if bindings.projections.len() != self.plan.nodes().len()
            || self.plan.source() != self._source_owner.source_identity()
            || self.preflight.source != self.plan.source()
        {
            return Err(failure("node cache/source coverage differs"));
        }
        let mut quantized = Vec::new();
        let mut f32_nodes = 0usize;
        let mut negative_cases = 0usize;
        let mut statuses: Vec<BufferRange> = Vec::new();
        let mut sources = BTreeMap::new();
        let mut repeated_sources = 0usize;
        let mut handle_checks = 0usize;
        for (index, node) in self.plan.nodes().iter().enumerate() {
            let tensor = match &node.operation {
                PlanOperation::Project { tensor } if tensor.kind != QuantKind::F32 => tensor,
                operation => {
                    f32_nodes += usize::from(matches!(operation, PlanOperation::Project { .. }));
                    if bindings.projections[index].is_some() {
                        return Err(failure("nonquantized operation has a binding"));
                    }
                    continue;
                }
            };
            let cached = bindings.projections[index]
                .as_ref()
                .ok_or_else(|| failure("quantized operation has no binding"))?;
            let reconstructed = legacy(self, client, index, tensor)?;
            let t = &bindings.tensors[&tensor.name];
            let flags = &bindings.node_status[index];
            let explicit = [
                self.values[node.inputs[0].0 as usize].handle.clone(),
                t.scales.clone(),
                t.biases.clone(),
                bindings.scratch[0].clone(),
                bindings.scratch[1].clone(),
                bindings.scratch[2].clone(),
                bindings.scratch[3].clone(),
                bindings.scratch[4].clone(),
                flags
                    .block
                    .as_ref()
                    .ok_or_else(|| failure("missing block status"))?
                    .clone(),
                flags
                    .row
                    .as_ref()
                    .ok_or_else(|| failure("missing row status"))?
                    .clone(),
            ];
            for ((cached, legacy), explicit) in cached
                .handles()
                .iter()
                .zip(reconstructed.handles().iter())
                .zip(&explicit)
            {
                if cached.service != client.service_id()
                    || !same(cached, legacy)?
                    || !same(cached, explicit)?
                {
                    return Err(failure(format!(
                        "node {index} cached/H22/typed coordinates differ"
                    )));
                }
                handle_checks += 1;
            }
            // Shared source metadata stays shared; operation statuses remain
            // distinct even when a source is used by multiple SSA operations.
            if let Some((scales, biases)) = sources.get(tensor.name.as_str()) {
                if !same(scales, &cached.weight_scales)? || !same(biases, &cached.biases)? {
                    return Err(failure("shared tensor metadata rebound between operations"));
                }
                repeated_sources += 1;
            } else {
                sources.insert(
                    tensor.name.as_str(),
                    (cached.weight_scales.clone(), cached.biases.clone()),
                );
            }
            for status in [&cached.block_status, &cached.row_status] {
                let range = BufferRange::new(status)?;
                if statuses.iter().any(|prior| prior.overlaps(&range)) {
                    return Err(failure("per-operation statuses overlap"));
                }
                for original in explicit.iter().take(8) {
                    if range.overlaps(&BufferRange::new(original)?) {
                        return Err(failure(
                            "operation status overlaps source/input/shared workspace",
                        ));
                    }
                }
                statuses.push(range);
            }
            negative_cases += negative_constructors(
                client,
                t.layout.ok_or_else(|| failure("quant layout"))?,
                cached,
            )?;
            quantized.push((index, tensor));
        }
        let cached_count = bindings.projections.iter().filter(|p| p.is_some()).count();
        if quantized.is_empty()
            || cached_count != quantized.len()
            || statuses.len() != cached_count * 2
            || handle_checks != cached_count * 10
        {
            return Err(failure("quantized/status/coordinate totals differ"));
        }
        let vector_bytes = std::mem::size_of_val(bindings.projections.as_slice());
        let binding_bytes = std::mem::size_of::<DeviceActivation>()
            .checked_mul(cached_count)
            .ok_or_else(|| Error::Limit("H23 host cache footprint overflow".into()))?;
        let host_cache_bytes = vector_bytes
            .checked_add(binding_bytes)
            .ok_or_else(|| Error::Limit("H23 host cache footprint overflow".into()))?;
        for &(index, tensor) in &quantized {
            black_box(legacy(self, client, index, tensor)?);
            black_box(&bindings.projections[index]);
        }
        let measure = |cached: bool| -> Result<u128> {
            let start = Instant::now();
            for _ in 0..100 {
                for &(index, tensor) in &quantized {
                    if cached {
                        let binding = black_box(bindings).projections[black_box(index)]
                            .as_deref()
                            .ok_or_else(|| failure("cache changed during host measurement"))?;
                        black_box(binding);
                    } else {
                        black_box(legacy(
                            black_box(self),
                            black_box(client),
                            black_box(index),
                            black_box(tensor),
                        )?);
                    }
                }
            }
            Ok(start.elapsed().as_nanos())
        };
        let mut legacy_ns = [0u128; 5];
        let mut cached_ns = [0u128; 5];
        for round in 0..5 {
            if round % 2 == 0 {
                legacy_ns[round] = measure(false)?;
                cached_ns[round] = measure(true)?;
            } else {
                cached_ns[round] = measure(true)?;
                legacy_ns[round] = measure(false)?;
            }
        }
        println!(
            "H23_PREPARED_PROJECTION_BINDINGS_PASS node_slots={} quantized_operations={cached_count} f32_operations={f32_nodes} unique_quant_source_tensors={} repeated_source_operations={repeated_sources} exact_handle_coordinate_checks={handle_checks} unique_per_operation_statuses={} malformed_constructor_refusals={negative_cases} host_cache_element_bytes={} host_vector_element_bytes={vector_bytes} host_binding_payload_bytes={binding_bytes} logical_host_cache_bytes={host_cache_bytes} host_vec_header_bytes={} allocator_bookkeeping=unmeasured no_gpu_allocations_or_uploads=true host_binding_only=true full_binding_loops_per_round=100 rounds=5 alternating_order=true legacy_round_ns={legacy_ns:?} cached_round_ns={cached_ns:?} timing_threshold=none model_throughput_claim=none launch_prepare_validation=unchanged",
            bindings.projections.len(),
            sources.len(),
            statuses.len(),
            std::mem::size_of::<Option<Box<DeviceActivation>>>(),
            std::mem::size_of_val(&bindings.projections)
        );
        Ok(())
    }
}
