//! Source quantization and GEMV form one GPU island with one readback cohort.
use super::*;
use crate::device_activation::DeviceActivation;

impl PreparedStep {
    pub(super) async fn execute_device(
        p: &DeviceStep,
        client: &Client,
        source: &ResidentHandle,
        flight: &mut InFlightRegion<'_, ResidentWeightLease>,
        numeric_mode: SourceAccumulation,
    ) -> Result<ReadbackReady> {
        let l = &p.layout;
        let prepared = DeviceActivation::new(client, l, &p.input, &p.scales, &p.biases)?;
        let output = client.empty(l.rows * 4);
        flight.retain_gpu_buffers(client, &[source.handle.clone(), output.clone()])?;
        flight.retain_gpu_buffers(client, &prepared.handles())?;
        prepared.launch_prepare(client, l)?;
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;
        let grid = CubeCount::Static((l.rows as u32).div_ceil(32), 1, 1);
        let threads = CubeDim::new_1d(32);
        // SAFETY: source and every activation/composite/output dimension was
        // preflight-validated, admitted and retained. Quant outputs never
        // cross the host before the existing GEMV consumes them.
        unsafe {
            match p.kind {
                QuantKind::Q4_0 => crate::quantization::q4_0::source_q4_q8_gemv::launch_unchecked(
                    client,
                    grid,
                    threads,
                    BufferArg::from_raw_parts(source.handle.clone(), source.elements),
                    BufferArg::from_raw_parts(prepared.scales.clone(), l.composite_count),
                    BufferArg::from_raw_parts(prepared.codes.clone(), l.columns),
                    BufferArg::from_raw_parts(output.clone(), l.rows),
                    l.blocks,
                    l.rows,
                    numeric_mode == SourceAccumulation::WideF64,
                ),
                QuantKind::Q4_1 => crate::quantization::q4_1::q4_1_q8_1_gemv::launch_unchecked(
                    client,
                    grid,
                    threads,
                    BufferArg::from_raw_parts(source.handle.clone(), source.elements),
                    BufferArg::from_raw_parts(prepared.scales.clone(), l.composite_count),
                    BufferArg::from_raw_parts(prepared.offsets.clone(), l.composite_count),
                    BufferArg::from_raw_parts(prepared.codes.clone(), l.columns),
                    BufferArg::from_raw_parts(output.clone(), l.rows),
                    l.blocks,
                    l.rows,
                    numeric_mode == SourceAccumulation::WideF64,
                ),
                QuantKind::Q8_0 => crate::quantization::q8_0::q8_0_q8_0_gemv::launch_unchecked(
                    client,
                    grid,
                    threads,
                    BufferArg::from_raw_parts(source.handle.clone(), source.elements),
                    BufferArg::from_raw_parts(prepared.scales.clone(), l.composite_count),
                    BufferArg::from_raw_parts(prepared.codes.clone(), l.columns),
                    BufferArg::from_raw_parts(output.clone(), l.rows),
                    l.blocks,
                    l.rows,
                    numeric_mode == SourceAccumulation::WideF64,
                ),
                QuantKind::Q6K => crate::quantization::q6_k::q6_k_q8_k_gemv::launch_unchecked(
                    client,
                    grid,
                    threads,
                    BufferArg::from_raw_parts(source.handle.clone(), source.elements),
                    BufferArg::from_raw_parts(prepared.scales.clone(), l.composite_count),
                    BufferArg::from_raw_parts(prepared.codes.clone(), l.columns),
                    BufferArg::from_raw_parts(output.clone(), l.rows),
                    l.blocks,
                    l.rows,
                    numeric_mode == SourceAccumulation::WideF64,
                ),
                _ => {
                    return Err(Error::Unsupported(
                        "device-quantized source GEMV codec".into(),
                    ));
                }
            }
        }
        let results = client
            .read_async(vec![
                output,
                prepared.block_status.clone(),
                prepared.row_status.clone(),
            ])
            .await
            .map_err(|e| Error::State(format!("device quantization/GEMV readback failed: {e}")))?;
        if results.len() != 3
            || results[0].len() != l.rows * 4
            || results[1].len() != l.blocks * 4
            || results[2].len() != l.rows * 4
        {
            return Err(Error::State(
                "incomplete device quantization/GEMV cohort".into(),
            ));
        }
        for (kind, bytes) in [("block", &results[1]), ("row", &results[2])] {
            if let Some((i, flags)) = bytes
                .chunks_exact(4)
                .enumerate()
                .map(|(i, b)| (i, u32::from_le_bytes(b.try_into().expect("checked status"))))
                .find(|(_, flags)| *flags != 0)
            {
                return Err(Error::State(format!(
                    "device quantization {kind} status offset={i} flags={flags}"
                )));
            }
        }
        let values: Vec<f32> = results[0]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("checked FP32")))
            .collect();
        if values.iter().any(|x| !x.is_finite()) {
            return Err(Error::State("nonfinite device quantized GEMV".into()));
        }
        Ok(ReadbackReady {
            values,
            dynamic_admitted_bytes: DeviceActivation::admission_bytes(l)?
                - DeviceActivation::status_bytes(l)?,
            dynamic_h2d_bytes: DeviceActivation::upload_bytes(l)?,
            completed_d2h_bytes: l.rows * 4 + DeviceActivation::status_bytes(l)?,
        })
    }
}
