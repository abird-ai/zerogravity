//! Remaining source elementwise math under the existing physical authority.
use super::*;
use zerogravity_core::accelerator::{BinaryOperation, BinaryRegion};

impl CubeResidentProvider {
    pub async fn try_binary_async(&self, r: BinaryRegion<'_>) -> Result<RegionExecution> {
        if !self.selected.contains(r.operation) {
            return Ok(RegionExecution::Unsupported);
        }
        r.validate()?;
        let n = r.left.len();
        if !r.source_identity.is_bound() || r.operation.is_empty() || n > 65_536 {
            return Err(Error::Unsupported(
                "source binary requires bound identity and <=65536 channels".into(),
            ));
        }
        let client = self.device.client();
        source_accumulation::ensure_capability(&client, SourceAccumulation::WideF64)?;
        let gated = r.kind == BinaryOperation::SigmoidGate;
        if gated
            && !client
                .properties()
                .features
                .supports_type(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U64))
        {
            return Err(Error::Unsupported(
                "source sigmoid requires shader UInt64".into(),
            ));
        }
        let table_count = crate::source_transcendentals::TABLES.len();
        let dynamic_bytes = n * 8
            + if gated {
                n * (crate::elementwise::SIGMOID_SCRATCH * 4 + 8) + table_count * 8
            } else {
                0
            };
        let output_bytes = n * 4;
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: dynamic_bytes as u64,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: output_bytes as u64,
                },
            ],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        resident.ensure_transient_room(dynamic_bytes, output_bytes)?;
        slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let mut flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let left = client.create_from_slice(f32::as_bytes(r.left));
        let right = client.create_from_slice(f32::as_bytes(r.right));
        let output = client.empty(output_bytes);
        flight.retain_gpu_buffers(&client, &[left.clone(), right.clone(), output.clone()])?;
        let grid = CubeCount::Static((n as u32).div_ceil(32), 1, 1);
        if gated {
            let tables =
                client.create_from_slice(f64::as_bytes(&crate::source_transcendentals::TABLES));
            let scratch = client.empty(n * crate::elementwise::SIGMOID_SCRATCH * 4);
            let shifted = client.empty(n * 8);
            flight
                .retain_gpu_buffers(&client, &[tables.clone(), scratch.clone(), shifted.clone()])?;
            // SAFETY: complete bounded input/output/scratch, disjoint per lane.
            unsafe {
                for stage in 0..crate::elementwise::SIGMOID_STAGES {
                    crate::elementwise::sigmoid_gate_step::launch_unchecked(
                        &client,
                        grid.clone(),
                        CubeDim::new_1d(32),
                        BufferArg::from_raw_parts(right.clone(), n),
                        BufferArg::from_raw_parts(left.clone(), n),
                        BufferArg::from_raw_parts(tables.clone(), table_count),
                        BufferArg::from_raw_parts(
                            scratch.clone(),
                            n * crate::elementwise::SIGMOID_SCRATCH,
                        ),
                        BufferArg::from_raw_parts(shifted.clone(), n),
                        BufferArg::from_raw_parts(output.clone(), n),
                        n,
                        stage,
                    );
                }
            }
        } else {
            // SAFETY: equal validated lengths and fully retained handles.
            unsafe {
                crate::elementwise::residual_add::launch_unchecked(
                    &client,
                    grid,
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(left, n),
                    BufferArg::from_raw_parts(right, n),
                    BufferArg::from_raw_parts(output.clone(), n),
                    n,
                );
            }
        }
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;
        let bytes = client
            .read_async(vec![output])
            .await
            .map_err(|e| Error::State(format!("binary physical readback failed: {e}")))?;
        if bytes.len() != 1 || bytes[0].len() != output_bytes {
            return Err(Error::State("incomplete binary readback".into()));
        }
        let values: Vec<f32> = bytes[0]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("checked FP32")))
            .collect();
        r.validate_completed(&values)?;
        flight.complete(output_bytes)?;
        physical_work.complete()?;
        slot.finish()?;
        Ok(RegionExecution::Completed(values))
    }
}
