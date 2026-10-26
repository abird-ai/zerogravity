//! Candidate block-local fusion of the original dynamic Q8 preparation.
//!
//! The caller retains the original scratch leases, queue slot and completion
//! guard. No allocation, upload, composite scale, status merge or publication
//! occurs here. Physical qualification precedes production selection.

use crate::quantization::activation::{ActivationLayout, PARAMETER_WORDS};
use crate::source_float::{round_source, widen_source};
use cubecl::{client::Client, prelude::*, server::Handle};
use cubecl_common::device::ServiceId;
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::{Error, Result};

const LANES: usize = 32;

/// Existing request-major buffers; no additional persistent scratch.
pub(crate) struct BlockBindings<'a> {
    pub input: &'a Handle,
    pub parameters: &'a Handle,
    pub scaled: &'a Handle,
    pub codes: &'a Handle,
    pub block_status: &'a Handle,
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    blocks: usize,
    columns: usize,
    parameters: usize,
}

/// Typed server identity, never adapter names or a guessed runtime ordinal.
pub(crate) fn is_cuda_service(client: &Client) -> bool {
    is_cuda_server(client.service_id().service)
}

fn is_cuda_server(service: std::any::TypeId) -> bool {
    #[cfg(feature = "cuda")]
    {
        service
            == std::any::TypeId::of::<
                <cubecl::cuda::CudaRuntime as cubecl_runtime::runtime::Runtime>::Server,
            >()
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = service;
        false
    }
}

/// Check cold hardware eligibility before the owner's first initialization.
/// Other explicitly selected backends retain their original source lowering.
pub(crate) fn ensure_selected_capability(client: &Client) -> Result<()> {
    if !is_cuda_service(client) {
        return Ok(());
    }
    crate::source_accumulation::ensure_capability(
        client,
        crate::source_accumulation::SourceAccumulation::WideF64,
    )?;
    if !client
        .properties()
        .features
        .supports_type(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U64))
        || client.properties().hardware.max_units_per_cube < LANES as u32
        || client.properties().hardware.max_shared_memory_size < (3 + 32) * 4
    {
        return Err(Error::Unsupported(
            "fused activation U64/workgroup capability".into(),
        ));
    }
    Ok(())
}

fn same_service(expected: ServiceId, actual: ServiceId) -> Result<()> {
    if expected != actual {
        return Err(Error::State("prepared activation service changed".into()));
    }
    Ok(())
}

/// Cold views of original admitted buffers. This object is no lease or Ready
/// capability; the original program/attempt retains every owning allocation.
pub(crate) struct PreparedBlocks {
    service: ServiceId,
    geometry: Geometry,
    width: usize,
    signed_scale: bool,
    with_bias: bool,
    handles: [Handle; 5],
}

impl PreparedBlocks {
    /// None means an explicitly selected non-CUDA backend, never a recoverable
    /// CUDA error. CUDA shape/capability/binding errors propagate unchanged.
    pub(crate) fn for_cuda(
        client: &Client,
        layout: &ActivationLayout,
        batch: usize,
        binding: &BlockBindings<'_>,
    ) -> Result<Option<Self>> {
        if !is_cuda_service(client) {
            return Ok(None);
        }
        Self::prepare_cuda(client, layout, batch, binding).map(Some)
    }

    pub(crate) fn prepare_cuda(
        client: &Client,
        layout: &ActivationLayout,
        batch: usize,
        binding: &BlockBindings<'_>,
    ) -> Result<Self> {
        validate(client, layout, batch, binding)?;
        Ok(Self {
            service: client.service_id(),
            geometry: geometry(layout, batch)?,
            width: layout.block_width,
            signed_scale: layout.signed_scale,
            with_bias: layout.with_bias,
            handles: [
                binding.input.clone(),
                binding.parameters.clone(),
                binding.scaled.clone(),
                binding.codes.clone(),
                binding.block_status.clone(),
            ],
        })
    }

    /// One launch, cached arguments only. Runtime uncertainty is handled by
    /// the original armed submission; it never retries a different lowering.
    pub(crate) fn launch(&self, client: &Client) -> Result<usize> {
        same_service(self.service, client.service_id())?;
        let g = self.geometry;
        // SAFETY: immutable original views passed cold validation; exactly one
        // full workgroup owns each block. The enclosing attempt pins all leases.
        unsafe {
            prepare_blocks::launch_unchecked(
                client,
                CubeCount::Static(g.blocks as u32, 1, 1),
                CubeDim::new_1d(LANES as u32),
                BufferArg::from_raw_parts(self.handles[0].clone(), g.columns),
                BufferArg::from_raw_parts(self.handles[1].clone(), g.parameters),
                BufferArg::from_raw_parts(self.handles[2].clone(), g.columns),
                BufferArg::from_raw_parts(self.handles[3].clone(), g.columns),
                BufferArg::from_raw_parts(self.handles[4].clone(), g.blocks),
                self.width,
                self.signed_scale,
                self.with_bias,
            );
        }
        Ok(1)
    }
}

fn geometry(layout: &ActivationLayout, batch: usize) -> Result<Geometry> {
    if ![1, 2, 4, 8].contains(&batch) || (layout.signed_scale && layout.with_bias) {
        return Err(Error::Malformed(
            "fused activation batch/codec layout".into(),
        ));
    }
    let kind = if layout.signed_scale {
        QuantKind::Q6K
    } else if layout.with_bias {
        QuantKind::Q4_1
    } else {
        QuantKind::Q8_0
    };
    let expected = ActivationLayout::new(kind, layout.columns, layout.rows)?;
    if layout.block_width != expected.block_width
        || layout.blocks != expected.blocks
        || layout.composite_count != expected.composite_count
        || layout.workspace_bytes != expected.workspace_bytes
    {
        return Err(Error::Malformed(
            "fused activation noncanonical layout".into(),
        ));
    }
    let count = |n: usize| {
        n.checked_mul(batch)
            .filter(|&n| u32::try_from(n).is_ok() && n.checked_mul(4).is_some())
            .ok_or_else(|| Error::Limit("fused activation shader/byte domain".into()))
    };
    let blocks = count(layout.blocks)?;
    let columns = count(layout.columns)?;
    let parameters = blocks
        .checked_mul(PARAMETER_WORDS)
        .filter(|&n| u32::try_from(n).is_ok() && n.checked_mul(4).is_some())
        .ok_or_else(|| Error::Limit("fused activation parameter domain".into()))?;
    Ok(Geometry {
        blocks,
        columns,
        parameters,
    })
}

/// Preflight before the original submission is begun. Buffer aliases are
/// rejected even when their visible sizes satisfy the individual contracts.
pub(crate) fn validate(
    client: &Client,
    layout: &ActivationLayout,
    batch: usize,
    binding: &BlockBindings<'_>,
) -> Result<()> {
    if !is_cuda_service(client) {
        return Err(Error::Unsupported(
            "fused activation CUDA-only lowering".into(),
        ));
    }
    let g = geometry(layout, batch)?;
    let buffers = [
        (binding.input, g.columns),
        (binding.parameters, g.parameters),
        (binding.scaled, g.columns),
        (binding.codes, g.columns),
        (binding.block_status, g.blocks),
    ];
    if buffers.iter().any(|(handle, n)| {
        handle.service != client.service_id()
            || !handle.offset_start.unwrap_or(0).is_multiple_of(4)
            || handle.size_in_used() < (*n * 4) as u64
    }) {
        return Err(Error::State("fused activation binding service/size".into()));
    }
    for (index, (a, _)) in buffers.iter().enumerate() {
        let a = (*a).clone().binding();
        let ar = a.range();
        for (b, _) in &buffers[index + 1..] {
            let b = (*b).clone().binding();
            let br = b.range();
            if a.memory.id() == b.memory.id() && ar.start < br.end && br.start < ar.end {
                return Err(Error::State("fused activation overlapping bindings".into()));
            }
        }
    }
    ensure_selected_capability(client)?;
    Ok(())
}

/// One actual launch; composite formation and both original status merges
/// remain separate. All owning leases must already be pinned by the caller.
#[cfg(all(test, feature = "qualification", feature = "cuda"))]
pub(crate) fn launch(
    client: &Client,
    layout: &ActivationLayout,
    batch: usize,
    binding: &BlockBindings<'_>,
) -> Result<usize> {
    PreparedBlocks::prepare_cuda(client, layout, batch, binding)?.launch(client)
}

/// A workgroup owns a complete source block. Only shared memory crosses a
/// barrier: CubeCL sync_cube does not promise storage visibility across lanes.
#[cube(launch_unchecked)]
fn prepare_blocks(
    input: &[f32],
    parameters: &mut [f32],
    scaled: &mut [f32],
    codes: &mut [i32],
    status: &mut [u32],
    #[comptime] width: usize,
    #[comptime] signed_scale: bool,
    #[comptime] with_bias: bool,
) {
    let block = usize::cast_from(CUBE_POS_X);
    let lane = usize::cast_from(UNIT_POS);
    let mut shared = Shared::<[u32]>::new_slice(3usize);
    let mut block_codes = Shared::<[i32]>::new_slice(32usize);
    if lane == 0usize {
        let mut largest = 0u32;
        let mut selected = 0u32;
        let mut flags = 0u32;
        for at in 0..width {
            let bits = u32::reinterpret(input[block * width + at]);
            let magnitude = bits & 0x7fffffffu32;
            if magnitude >= 0x7f800000u32 {
                flags |= crate::quantization::activation::INPUT_NONFINITE;
            } else if magnitude > largest {
                largest = magnitude;
                selected = bits;
            }
        }
        let mut inverse = 0u32;
        let mut d = 0u32;
        if largest != 0u32 {
            if signed_scale {
                inverse = crate::quantization::activation::divide_bits(0xc2fe0000u32, selected);
                d = crate::quantization::activation::divide_bits(0x3f800000u32, inverse);
            } else {
                inverse = crate::quantization::activation::divide_bits(0x42fe0000u32, largest);
                d = crate::quantization::activation::divide_bits(largest, 0x42fe0000u32);
            }
        }
        let mut execution = d;
        if !signed_scale {
            execution = crate::quantization::activation::half_roundtrip_bits(d);
        }
        if (execution & 0x7fffffffu32) >= 0x7f800000u32 {
            flags |= crate::quantization::activation::SCALE_NONFINITE;
        }
        parameters[block * 4] = f32::reinterpret(inverse);
        parameters[block * 4 + 1] = f32::reinterpret(d);
        parameters[block * 4 + 2] = f32::reinterpret(execution);
        parameters[block * 4 + 3] = 0.0f32;
        shared[0] = inverse;
        shared[1] = d;
        shared[2] = flags;
        if signed_scale {
            status[block] = flags;
        }
    }
    sync_cube();
    let inverse_bits = shared[0];
    let inverse = f32::reinterpret(inverse_bits);
    for part in 0..width / 32usize {
        let at = part * 32usize + lane;
        let index = block * width + at;
        let x = input[index];
        let sign = (u32::reinterpret(x) ^ inverse_bits) & 0x80000000u32;
        // The observable raw word forces source F32 rounding before coding.
        let bits = u32::reinterpret(round_source(widen_source(x) * widen_source(inverse), sign));
        let code = crate::quantization::activation::quantized_code_bits(bits);
        scaled[index] = f32::reinterpret(bits);
        codes[index] = code;
        if !signed_scale {
            block_codes[at] = code;
        }
    }
    if !signed_scale {
        sync_cube();
        if lane == 0usize {
            let mut sum = 0i32;
            for at in 0..32usize {
                sum += block_codes[at];
            }
            let a = f32::cast_from(sum);
            let d_bits = shared[1];
            let d = f32::reinterpret(d_bits);
            let sign = (u32::reinterpret(a) ^ d_bits) & 0x80000000u32;
            let bits = u32::reinterpret(round_source(widen_source(a) * widen_source(d), sign));
            let half = crate::quantization::activation::half_roundtrip_bits(bits);
            parameters[block * 4 + 3] = f32::reinterpret(half);
            let mut flags = shared[2];
            if with_bias && (half & 0x7fffffffu32) >= 0x7f800000u32 {
                flags |= crate::quantization::activation::SUM_NONFINITE;
            }
            status[block] = flags;
        }
    }
}

#[cfg(all(test, feature = "qualification", feature = "cuda"))]
mod qualification;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_selection_requires_exact_cuda_server_type_without_starting_runtime() {
        struct ForeignServer;
        assert!(!is_cuda_server(std::any::TypeId::of::<ForeignServer>()));
        #[cfg(feature = "cuda")]
        assert!(is_cuda_server(std::any::TypeId::of::<
            <cubecl::cuda::CudaRuntime as cubecl_runtime::runtime::Runtime>::Server,
        >()));
    }

    #[test]
    fn cached_service_guard_refuses_other_runtime_or_device_before_launch() {
        use cubecl_common::device::DeviceId;
        struct OriginalServer;
        struct OtherServer;
        let original = ServiceId::of::<OriginalServer>(DeviceId::new(0, 0));
        same_service(original, original).unwrap();
        assert!(
            same_service(
                original,
                ServiceId::of::<OriginalServer>(DeviceId::new(0, 1))
            )
            .is_err()
        );
        assert!(same_service(original, ServiceId::of::<OtherServer>(DeviceId::new(0, 0))).is_err());
    }

    #[test]
    fn geometry_preserves_real_request_and_block_boundaries() {
        for kind in [
            QuantKind::Q4_0,
            QuantKind::Q4_1,
            QuantKind::Q8_0,
            QuantKind::Q6K,
        ] {
            for batch in [1, 2, 4, 8] {
                let l = ActivationLayout::new(kind, 1024, 8).unwrap();
                let g = geometry(&l, batch).unwrap();
                assert_eq!(g.columns, 1024 * batch);
                assert_eq!(g.blocks * l.block_width, g.columns);
                assert_eq!(g.parameters, 4 * g.blocks);
                let mut seen = vec![false; g.columns];
                for block in 0..g.blocks {
                    for lane in 0..32 {
                        for part in 0..l.block_width / 32 {
                            let index = block * l.block_width + part * 32 + lane;
                            assert!(!seen[index]);
                            assert_eq!(index / l.columns, block / l.blocks);
                            seen[index] = true;
                        }
                    }
                }
                assert!(seen.into_iter().all(|v| v));
            }
        }
    }

    #[test]
    fn geometry_refuses_padding_and_forged_codec_or_shader_counts() {
        let l = ActivationLayout::new(QuantKind::Q4_1, 64, 8).unwrap();
        for batch in [0, 3, 5, 7, 9, usize::MAX] {
            assert!(geometry(&l, batch).is_err());
        }
        for bad in [
            ActivationLayout { blocks: 1, ..l },
            ActivationLayout {
                block_width: 256,
                ..l
            },
            ActivationLayout {
                composite_count: 1,
                ..l
            },
            ActivationLayout {
                workspace_bytes: 1,
                ..l
            },
            ActivationLayout {
                signed_scale: true,
                ..l
            },
        ] {
            assert!(geometry(&bad, 2).is_err());
        }
        let huge = ActivationLayout::new(QuantKind::Q8_0, 1 << 29, 1).unwrap();
        assert!(geometry(&huge, 8).is_err());
    }
}
