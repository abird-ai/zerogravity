//! Diagnostic-only physical dynamic-activation qualification.
//! CPU answers are comparison oracles; no expected code/scale is uploaded or
//! returned as an engine output. Uses the same physical ownership boundaries.
use super::*;
use crate::device_activation::DeviceActivation;
use zerogravity_artifacts::quant::{f16_to_f32, f32_to_f16_rne};

fn compare_source_f32(
    label: &str,
    actual: &[u8],
    expected: &[f32],
    allow_nan_payload: bool,
) -> Result<usize> {
    if actual.len() != expected.len() * 4 {
        return Err(Error::State(format!(
            "activation qualification {label} byte geometry mismatch"
        )));
    }
    for (i, (bytes, &expected)) in actual.chunks_exact(4).zip(expected).enumerate() {
        let actual = f32::from_le_bytes(bytes.try_into().expect("checked source-float bytes"));
        if actual.to_bits() != expected.to_bits()
            && !(allow_nan_payload && actual.is_nan() && expected.is_nan())
        {
            return Err(Error::State(format!(
                "activation qualification {label}[{i}] bits {:08x} != source {:08x}",
                actual.to_bits(),
                expected.to_bits(),
            )));
        }
    }
    Ok(expected.len())
}

impl CubeResidentProvider {
    /// Physically compare every consumed code/composite and the source
    /// quantizer's physically rounded intermediates. Counts are audited FP32
    /// and code integers, never numerical engine outputs or StateRoot writes.
    pub async fn qualify_device_activation(
        &self,
        region: MatVecRegion<'_>,
    ) -> Result<(usize, usize)> {
        if !self.selected.contains(region.tensor_name)
            || region.rows > self.max_rows
            || !matches!(
                region.kind,
                QuantKind::Q4_0 | QuantKind::Q4_1 | QuantKind::Q8_0 | QuantKind::Q6K
            )
        {
            return Err(Error::Unsupported(
                "activation qualification requires a selected bounded source codec".into(),
            ));
        }
        // Source key validation rejects an unbound source identity and
        // malformed source shape before any physical submission/allocation.
        ResidencyKey::from_region(region)?;
        let (host, unused_packed) = StaticSource::from_region(region)?;
        drop(unused_packed);
        let step = host.dynamic_device(region.activation)?;
        // Independent existing CPU source path is used ONLY as a test oracle.
        let expected = host.dynamic(region.activation)?;
        let (expected_codes, expected_scales, expected_offsets): (&[i32], &[f32], &[f32]) =
            match &expected {
                DynamicStep::Q4(p) => (&p.activations, &p.scales, &[]),
                DynamicStep::Mixed(p) => (&p.activation, &p.scales, &p.offsets),
            };
        let layout = &step.layout;
        if expected_codes.len() != layout.columns
            || expected_scales.len() != layout.composite_count
            || expected_offsets.len()
                != if layout.with_bias {
                    layout.composite_count
                } else {
                    0
                }
        {
            return Err(Error::State(
                "activation qualification oracle geometry mismatch".into(),
            ));
        }
        // Original FP32 schedule, including first-on-equal signed max, a
        // separately rounded reciprocal, and Q8_1's UNROUNDED d sum product.
        let mut expected_parameters = Vec::with_capacity(layout.blocks * 4);
        let mut expected_scaled = Vec::with_capacity(layout.columns);
        for (block, input) in step.input.chunks_exact(layout.block_width).enumerate() {
            let mut largest = 0.0f32;
            let mut selected = 0.0f32;
            for &value in input {
                if value.abs() > largest {
                    largest = value.abs();
                    selected = value;
                }
            }
            let inverse = if largest == 0.0 {
                0.0
            } else if layout.signed_scale {
                -127.0f32 / selected
            } else {
                127.0f32 / largest
            };
            let d = if layout.signed_scale {
                if largest == 0.0 {
                    0.0
                } else {
                    1.0f32 / inverse
                }
            } else {
                largest / 127.0f32
            };
            let execution_scale = if layout.signed_scale {
                d
            } else {
                f16_to_f32(f32_to_f16_rne(d))
            };
            let sum_scale = if layout.signed_scale {
                0.0
            } else {
                let sum: i32 = expected_codes[block * 32..block * 32 + 32].iter().sum();
                f16_to_f32(f32_to_f16_rne(sum as f32 * d))
            };
            expected_parameters.extend_from_slice(&[inverse, d, execution_scale, sum_scale]);
            expected_scaled.extend(input.iter().map(|&x| x * inverse));
        }

        let client = self.device.client();
        source_accumulation::ensure_capability(&client, SourceAccumulation::WideF64)?;
        if !client
            .properties()
            .features
            .supports_type(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U64))
        {
            return Err(Error::Unsupported(
                "activation qualification requires shader UInt64".into(),
            ));
        }
        let dynamic_bytes = DeviceActivation::upload_bytes(layout)?;
        let output_bytes = layout.workspace_bytes;
        if dynamic_bytes.checked_add(output_bytes)
            != Some(DeviceActivation::admission_bytes(layout)?)
        {
            return Err(Error::State(
                "activation qualification admission cohort mismatch".into(),
            ));
        }
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: u64::try_from(dynamic_bytes)
                        .map_err(|_| Error::Limit("activation fixture input overflow".into()))?,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: u64::try_from(output_bytes)
                        .map_err(|_| Error::Limit("activation fixture output overflow".into()))?,
                },
            ],
            None,
        )?;
        let mut worker_slot = ticket.wait().await?;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        resident.ensure_transient_room(dynamic_bytes, output_bytes)?;
        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let buffers =
            DeviceActivation::new(&client, layout, &step.input, &step.scales, &step.biases)?;
        in_flight.retain_gpu_buffers(&client, &buffers.handles())?;
        buffers.launch_prepare(&client, layout)?;
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;
        let results = client
            .read_async(vec![
                buffers.parameters.clone(),
                buffers.scaled.clone(),
                buffers.codes.clone(),
                buffers.scales.clone(),
                buffers.offsets.clone(),
                buffers.block_status.clone(),
                buffers.row_status.clone(),
            ])
            .await
            .map_err(|e| Error::State(format!("activation fixture readback failed: {e}")))?;
        let offset_count = if layout.with_bias {
            layout.composite_count
        } else {
            1
        };
        let expected_lengths = [
            layout.blocks * 4 * 4,
            layout.columns * 4,
            layout.columns * 4,
            layout.composite_count * 4,
            offset_count * 4,
            layout.blocks * 4,
            layout.rows * 4,
        ];
        if results.len() != expected_lengths.len()
            || results
                .iter()
                .zip(expected_lengths)
                .any(|(bytes, expected)| bytes.len() != expected)
            || expected_lengths.iter().sum::<usize>() != output_bytes
        {
            return Err(Error::State(
                "activation fixture incomplete physical cohort".into(),
            ));
        }
        for (label, bytes) in [("block", &results[5]), ("row", &results[6])] {
            for (i, bytes) in bytes.chunks_exact(4).enumerate() {
                let flags = u32::from_le_bytes(bytes.try_into().expect("checked status bytes"));
                if flags != 0 {
                    return Err(Error::State(format!(
                        "activation qualification {label} status[{i}]={flags:#x}"
                    )));
                }
            }
        }
        let mut floats =
            compare_source_f32("parameters", &results[0], &expected_parameters, false)?;
        // 0*infinite reciprocal is source-valid for tiny finite blocks. Its
        // unobservable intermediate NaN payload/sign is not an output promise.
        floats += compare_source_f32("scaled", &results[1], &expected_scaled, true)?;
        for (i, (bytes, &expected)) in results[2].chunks_exact(4).zip(expected_codes).enumerate() {
            let actual = i32::from_le_bytes(bytes.try_into().expect("checked code bytes"));
            if actual != expected {
                return Err(Error::State(format!(
                    "activation qualification code[{i}] {actual} != source {expected}"
                )));
            }
        }
        floats += compare_source_f32("composite scale", &results[3], expected_scales, false)?;
        if layout.with_bias {
            floats += compare_source_f32("composite offset", &results[4], expected_offsets, false)?;
        }
        // The no-bias dummy offset is allocated/read/accounted but not used
        // or initialized by source math. Its uninterpreted bytes are not a
        // claimed numerical comparison or an engine-consumable result.
        in_flight.complete(output_bytes)?;
        physical_work.complete()?;
        worker_slot.finish()?;
        Ok((floats, expected_codes.len()))
    }
}
