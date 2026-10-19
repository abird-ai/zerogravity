//! Physical dynamic-Q8 preparation buffers and ordered source kernel launches.
//!
//! The enclosing region owns admission, fair scheduling, source identity,
//! submission, completion/readback and quarantine. This helper never computes
//! live quantization on the host or publishes a result. Every handle returned
//! by `handles` must be retained by the existing InFlightRegion BEFORE calling
//! `launch_prepare`; both status arrays must join its actual readback cohort.
use crate::activation_quant::{self, ActivationLayout, PARAMETER_WORDS};
use crate::activation_quant_fused::{BlockBindings, PreparedBlocks};
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::Error;
use zerogravity_core::Result;

/// Ten original handles: input/static metadata and seven quant work buffers.
/// Source bytes and final GEMV output are separately owned by the caller.
pub(crate) struct DeviceActivation {
    layout: ActivationLayout,
    prepared_blocks: Option<PreparedBlocks>,
    pub input: Handle,
    pub weight_scales: Handle,
    pub biases: Handle,
    pub parameters: Handle,
    pub scaled: Handle,
    pub codes: Handle,
    pub scales: Handle,
    pub offsets: Handle,
    pub block_status: Handle,
    pub row_status: Handle,
}

impl DeviceActivation {
    /// Check public layout fields against their canonical source geometry.
    /// This is metadata validation only, with no activation arithmetic.
    fn validate_layout(layout: &ActivationLayout) -> Result<()> {
        let kind = if layout.signed_scale {
            if layout.with_bias {
                return Err(Error::Malformed("Q8_K has no bias sum".into()));
            }
            QuantKind::Q6K
        } else if layout.with_bias {
            QuantKind::Q4_1
        } else {
            // Q4_0 and Q8_0 have identical activation layouts.
            QuantKind::Q8_0
        };
        let expected = ActivationLayout::new(kind, layout.columns, layout.rows)?;
        if layout.block_width != expected.block_width
            || layout.blocks != expected.blocks
            || layout.composite_count != expected.composite_count
            || layout.workspace_bytes != expected.workspace_bytes
        {
            return Err(Error::Malformed(
                "device activation layout differs from source geometry".into(),
            ));
        }
        Ok(())
    }

    /// Includes input, source scales, actual/dummy biases and all seven
    /// workspace buffers. Excludes compressed source and final GEMV output.
    pub fn admission_bytes(layout: &ActivationLayout) -> Result<usize> {
        Self::validate_layout(layout)?;
        layout
            .workspace_bytes
            .checked_add(Self::upload_bytes(layout)?)
            .ok_or_else(|| Error::Limit("device activation admission byte overflow".into()))
    }

    /// Actual H2D input/static metadata bytes; includes the no-bias dummy.
    /// This differs from admitted bytes because device work buffers start
    /// empty and are filled by the source kernels, never uploaded as answers.
    pub fn upload_bytes(layout: &ActivationLayout) -> Result<usize> {
        Self::validate_layout(layout)?;
        let bias_words = if layout.with_bias {
            layout.composite_count
        } else {
            1
        };
        layout
            .columns
            .checked_add(layout.composite_count)
            .and_then(|n| n.checked_add(bias_words))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Limit("device activation upload byte overflow".into()))
    }

    /// Both GPU-written status buffers must be physically read back along
    /// with the real matrix output before any Completed/Ready publication.
    pub fn status_bytes(layout: &ActivationLayout) -> Result<usize> {
        Self::validate_layout(layout)?;
        layout
            .blocks
            .checked_add(layout.rows)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Limit("device activation status byte overflow".into()))
    }

    /// Fallible validation can run BEFORE the enclosing physical submission.
    /// Static metadata originates in the existing miss-only source decoder;
    /// it is not rescanned per token. GPU composite status also checks it.
    pub fn validate_inputs(
        layout: &ActivationLayout,
        input: &[f32],
        weight_scales: &[f32],
        biases: &[f32],
    ) -> Result<()> {
        Self::validate_layout(layout)?;
        if input.len() != layout.columns
            || input.iter().any(|x| !x.is_finite())
            || weight_scales.len() != layout.composite_count
            || biases.len()
                != if layout.with_bias {
                    layout.composite_count
                } else {
                    0
                }
        {
            return Err(Error::Malformed(
                "device activation input/static metadata geometry or finite mismatch".into(),
            ));
        }
        Ok(())
    }

    /// Allocate exactly the admitted footprint and upload only original
    /// input/static metadata. Retain `handles()` before launching anything.
    /// Host finite/shape checks repeat the preflight but never quantize input.
    pub fn new(
        client: &Client,
        layout: &ActivationLayout,
        input: &[f32],
        weight_scales: &[f32],
        biases: &[f32],
    ) -> Result<Self> {
        Self::validate_inputs(layout, input, weight_scales, biases)?;
        crate::activation_quant_fused::ensure_selected_capability(client)?;
        // Resolve every checked byte calculation before the first allocation.
        Self::admission_bytes(layout)?;
        let parameters_bytes = layout.blocks * PARAMETER_WORDS * 4;
        let columns_bytes = layout.columns * 4;
        let composite_bytes = layout.composite_count * 4;
        let offsets_bytes = if layout.with_bias { composite_bytes } else { 4 };
        let mut bound = Self {
            layout: *layout,
            prepared_blocks: None,
            input: client.create_from_slice(f32::as_bytes(input)),
            weight_scales: client.create_from_slice(f32::as_bytes(weight_scales)),
            biases: if layout.with_bias {
                client.create_from_slice(f32::as_bytes(biases))
            } else {
                client.create_from_slice(f32::as_bytes(&[0.0]))
            },
            parameters: client.empty(parameters_bytes),
            scaled: client.empty(columns_bytes),
            codes: client.empty(columns_bytes),
            scales: client.empty(composite_bytes),
            offsets: client.empty(offsets_bytes),
            block_status: client.empty(layout.blocks * 4),
            row_status: client.empty(layout.rows * 4),
        };
        bound.prepared_blocks = bound.bind_blocks(client)?;
        Ok(bound)
    }

    /// Bind already-resident input/static metadata and seven admitted scratch
    /// buffers. This performs no upload, allocation or live numerical work.
    pub(crate) fn from_bound(
        client: &Client,
        layout: &ActivationLayout,
        input: Handle,
        weight_scales: Handle,
        biases: Handle,
        workspace: [Handle; 7],
    ) -> Result<Self> {
        Self::validate_layout(layout)?;
        let [
            parameters,
            scaled,
            codes,
            scales,
            offsets,
            block_status,
            row_status,
        ] = workspace;
        let expected = [
            (&input, layout.columns * 4),
            (&weight_scales, layout.composite_count * 4),
            (
                &biases,
                if layout.with_bias {
                    layout.composite_count * 4
                } else {
                    4
                },
            ),
            (&parameters, layout.blocks * PARAMETER_WORDS * 4),
            (&scaled, layout.columns * 4),
            (&codes, layout.columns * 4),
            (&scales, layout.composite_count * 4),
            (
                &offsets,
                if layout.with_bias {
                    layout.composite_count * 4
                } else {
                    4
                },
            ),
            (&block_status, layout.blocks * 4),
            (&row_status, layout.rows * 4),
        ];
        if expected.iter().any(|(handle, bytes)| {
            handle.service != client.service_id() || handle.size_in_used() < *bytes as u64
        }) {
            return Err(Error::State(
                "resident activation binding service/size mismatch".into(),
            ));
        }
        let mut bound = Self {
            layout: *layout,
            prepared_blocks: None,
            input,
            weight_scales,
            biases,
            parameters,
            scaled,
            codes,
            scales,
            offsets,
            block_status,
            row_status,
        };
        bound.prepared_blocks = bound.bind_blocks(client)?;
        Ok(bound)
    }

    fn bind_blocks(&self, client: &Client) -> Result<Option<PreparedBlocks>> {
        PreparedBlocks::for_cuda(
            client,
            &self.layout,
            1,
            &BlockBindings {
                input: &self.input,
                parameters: &self.parameters,
                scaled: &self.scaled,
                codes: &self.codes,
                block_status: &self.block_status,
            },
        )
    }

    /// Retain all originals through existing physical completion/quarantine.
    /// The caller additionally retains its compressed source and GEMV output.
    pub fn handles(&self) -> [Handle; 10] {
        [
            self.input.clone(),
            self.weight_scales.clone(),
            self.biases.clone(),
            self.parameters.clone(),
            self.scaled.clone(),
            self.codes.clone(),
            self.scales.clone(),
            self.offsets.clone(),
            self.block_status.clone(),
            self.row_status.clone(),
        ]
    }

    /// CUDA uses two cold-bound launches. Other explicitly selected backends
    /// retain the original seven/five-stage source lowering. No error fallback.
    pub fn launch_prepare(&self, client: &Client, layout: &ActivationLayout) -> Result<usize> {
        if layout.columns != self.layout.columns
            || layout.rows != self.layout.rows
            || layout.block_width != self.layout.block_width
            || layout.blocks != self.layout.blocks
            || layout.composite_count != self.layout.composite_count
            || layout.workspace_bytes != self.layout.workspace_bytes
            || layout.signed_scale != self.layout.signed_scale
            || layout.with_bias != self.layout.with_bias
            || self
                .handle_refs()
                .iter()
                .any(|h| h.service != client.service_id())
        {
            return Err(Error::State(
                "device activation launch layout or service changed".into(),
            ));
        }
        let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
        let threads = CubeDim::new_1d(32);
        let parameter_count = layout.blocks * PARAMETER_WORDS;
        let bias_count = if layout.with_bias {
            layout.composite_count
        } else {
            1
        };
        let mut dispatches = 0;
        // SAFETY: all source layouts, element counts, grids and service IDs
        // are validated above. Each stage sees the preceding physical stores.
        // The enclosing region must retain all ten originals BEFORE this call.
        unsafe {
            if let Some(prepared) = &self.prepared_blocks {
                dispatches += prepared.launch(client)?;
            } else {
                activation_quant::block_parameters::launch_unchecked(
                    client,
                    grid(layout.blocks),
                    threads,
                    BufferArg::from_raw_parts(self.input.clone(), layout.columns),
                    BufferArg::from_raw_parts(self.parameters.clone(), parameter_count),
                    BufferArg::from_raw_parts(self.block_status.clone(), layout.blocks),
                    layout.blocks,
                    layout.block_width,
                    layout.signed_scale,
                );
                dispatches += 1;
                activation_quant::block_scales::launch_unchecked(
                    client,
                    grid(layout.blocks),
                    threads,
                    BufferArg::from_raw_parts(self.parameters.clone(), parameter_count),
                    BufferArg::from_raw_parts(self.block_status.clone(), layout.blocks),
                    layout.blocks,
                    layout.signed_scale,
                );
                dispatches += 1;
                activation_quant::scaled_inputs::launch_unchecked(
                    client,
                    grid(layout.columns),
                    threads,
                    BufferArg::from_raw_parts(self.input.clone(), layout.columns),
                    BufferArg::from_raw_parts(self.parameters.clone(), parameter_count),
                    BufferArg::from_raw_parts(self.scaled.clone(), layout.columns),
                    layout.columns,
                    layout.block_width,
                );
                dispatches += 1;
                activation_quant::codes::launch_unchecked(
                    client,
                    grid(layout.columns),
                    threads,
                    BufferArg::from_raw_parts(self.scaled.clone(), layout.columns),
                    BufferArg::from_raw_parts(self.codes.clone(), layout.columns),
                    layout.columns,
                );
                dispatches += 1;
                if !layout.signed_scale {
                    activation_quant::sum_products::launch_unchecked(
                        client,
                        grid(layout.blocks),
                        threads,
                        BufferArg::from_raw_parts(self.codes.clone(), layout.columns),
                        BufferArg::from_raw_parts(self.parameters.clone(), parameter_count),
                        layout.blocks,
                    );
                    dispatches += 1;
                    activation_quant::sum_halves::launch_unchecked(
                        client,
                        grid(layout.blocks),
                        threads,
                        BufferArg::from_raw_parts(self.parameters.clone(), parameter_count),
                        BufferArg::from_raw_parts(self.block_status.clone(), layout.blocks),
                        layout.blocks,
                        layout.with_bias,
                    );
                    dispatches += 1;
                }
            }
            activation_quant::composites::launch_unchecked(
                client,
                grid(layout.rows),
                threads,
                BufferArg::from_raw_parts(self.weight_scales.clone(), layout.composite_count),
                BufferArg::from_raw_parts(self.biases.clone(), bias_count),
                BufferArg::from_raw_parts(self.parameters.clone(), parameter_count),
                BufferArg::from_raw_parts(self.scales.clone(), layout.composite_count),
                BufferArg::from_raw_parts(self.offsets.clone(), bias_count),
                BufferArg::from_raw_parts(self.row_status.clone(), layout.rows),
                layout.blocks,
                layout.rows,
                layout.with_bias,
            );
            dispatches += 1;
        }
        Ok(dispatches)
    }

    fn handle_refs(&self) -> [&Handle; 10] {
        [
            &self.input,
            &self.weight_scales,
            &self.biases,
            &self.parameters,
            &self.scaled,
            &self.codes,
            &self.scales,
            &self.offsets,
            &self.block_status,
            &self.row_status,
        ]
    }
}
