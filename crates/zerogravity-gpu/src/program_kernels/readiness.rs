//! Exact generated constructors used by unchecked launch; no BufferArg/handle.
use super::*;
use crate::resident_program::StateRequirement;
use cubecl::client::Client;
use cubecl::ir::{DeviceProperties, TargetProperties};
use cubecl_core::prelude::{BufferCompilationArg, CubeKernel};
use std::sync::Arc;

/// Settings of the actual `ResidentProgram::certify` launch: `threads()`, the
/// `launch_unchecked` attribute (`Unchecked`, default `U32`).
fn state_certificate_settings() -> KernelSettings {
    KernelSettings::new(
        *CubeDim::new_1d(32),
        ExecutionMode::Unchecked,
        AddressType::U32,
    )
}

/// Exact generated definition of one state plane certificate. Two read-only
/// `BufferArg`s give `inplace: None`; offset/stride are runtime scalars (not in
/// the key); `(rows, width)` is `(1, elements)`, as `certify` launches it.
/// Properties/target reach codegen only and are no part of the kernel key.
fn state_certificate(
    properties: Arc<DeviceProperties>,
    target: Arc<TargetProperties>,
    elements: usize,
) -> certify_finite_strided::CertifyFiniteStrided {
    let buffer = || BufferCompilationArg { inplace: None };
    certify_finite_strided::CertifyFiniteStrided::new(
        state_certificate_settings(),
        properties,
        target,
        buffer(),
        buffer(),
        (),
        (),
        1usize,
        elements,
    )
}

/// Actual plane count and F32 word extent per plane of one source-issued
/// requirement. A KV domain is two planes of one `slab*capacity` extent; total
/// requirement bytes/4 would cover both planes and is never a plane extent. The
/// extent must be nonzero, not overflowing and exactly the source bytes, and
/// inside the kernel's own `rows > MAX_WORD_OPS / width` admission, so no plane
/// the actual certificate would reject with `VIEW_OUT_OF_BOUNDS` is prepared.
fn state_planes(requirement: &StateRequirement) -> Result<(usize, usize)> {
    let (planes, elements) = requirement.layout.planes()?;
    let bytes = elements
        .checked_mul(4)
        .and_then(|n| n.checked_mul(planes))
        .filter(|n| *n != 0)
        .ok_or_else(|| Error::Limit("state certificate plane extent".into()))?;
    if elements > MAX_WORD_OPS {
        return Err(Error::Limit("state certificate plane extent".into()));
    }
    if bytes != requirement.bytes {
        return Err(Error::State(
            "state certificate plane bytes differ from source requirement".into(),
        ));
    }
    Ok((planes, elements))
}

/// Prepare the state-plane finite certificates the resident runtime launches for
/// every candidate row. Borrowed source requirements only, one actual candidate
/// call per plane, so a KV pair counts twice under one compile key; the original
/// actor's `is_loaded` test charges no second compilation for a loaded key.
pub(crate) fn prepare_state_certificates(
    client: &Client,
    queue: &cubecl::cuda::OwnedCudaQueue,
    states: &[StateRequirement],
) -> Result<usize> {
    if states.is_empty() {
        return Err(Error::State("state certificate requirements absent".into()));
    }
    let properties = client.properties_shared();
    let target = client.target_properties_shared();
    let mut count = 0;
    for requirement in states {
        let (planes, elements) = state_planes(requirement)?;
        for _ in 0..planes {
            // SAFETY: same admitted original compiler actor/scope as the cold
            // caller; the definition is the actual unchecked launch's.
            unsafe {
                queue.prepare_kernel(Box::new(state_certificate(
                    properties.clone(),
                    target.clone(),
                    elements,
                )))
            }
            .map_err(|e| Error::State(format!("original state certificate readiness: {e}")))?;
            count += 1;
        }
    }
    Ok(count)
}

pub(crate) fn prepare_short_hf_code(
    client: &Client,
    queue: &cubecl::cuda::OwnedCudaQueue,
    cohort: bool,
    profile: zerogravity_core::accelerator::program::FullModelProfile,
) -> Result<usize> {
    let settings = |threads| {
        KernelSettings::new(
            *CubeDim::new_1d(threads),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
    };
    let a = || BufferCompilationArg { inplace: None };
    let p = || client.properties_shared();
    let t = || client.target_properties_shared();
    let mut count = 0;
    let mut prepare = |kernel: Box<dyn CubeKernel>| -> Result<()> {
        // SAFETY: original caller retains the admitted physical compiler scope;
        // these are the same generated definitions/compilation args as launch.
        unsafe { queue.prepare_kernel(kernel) }
            .map_err(|e| Error::State(format!("original integer code readiness: {e}")))?;
        count += 1;
        Ok(())
    };
    prepare(Box::new(clear_step_status::ClearStepStatus::new(
        settings(32),
        p(),
        t(),
        a(),
    )))?;
    if cohort {
        let phases =
            crate::vllm_cuda_qualification::cohort_native::CohortPhases::for_profile(profile);
        let width_words = profile.hidden_size() / 2;
        for (rows, width) in [
            (188, 1024),
            (188, 8),
            (4, 1024),
            (4, 8),
            (4, 4096),
            (4, 3072),
            (4, 512),
            (1, 192512),
            (1, 4096),
            (1, 3),
        ]
        .into_iter()
        .filter(|(rows, _)| *rows != 188 || phases.contains(true))
        {
            prepare(Box::new(copy_strided_words::CopyStridedWords::new(
                settings(32),
                p(),
                t(),
                a(),
                a(),
                a(),
                (),
                (),
                (),
                (),
                rows,
                width,
                false,
            )))?;
        }
        for mode in 0..3 {
            prepare(Box::new(cohort_copy::copy_words::CopyWords::new(
                settings(128),
                p(),
                t(),
                a(),
                a(),
                a(),
                mode,
            )))?;
        }
        let clear_words: &[usize] = match profile {
            zerogravity_core::accelerator::program::FullModelProfile::Qwen08 => &[6, 16, 40],
            zerogravity_core::accelerator::program::FullModelProfile::Qwen2B => &[32],
        };
        for &words in clear_words {
            prepare(Box::new(cohort_copy::clear_prefix::ClearPrefix::new(
                settings(64),
                p(),
                t(),
                a(),
                a(),
                words,
            )))?;
        }
        if phases.contains(true) {
            prepare(Box::new(cohort_copy::gather_rows::GatherRows::new(
                settings(32),
                p(),
                t(),
                a(),
                a(),
                a(),
                a(),
                188,
                width_words,
            )))?;
        }
        prepare(Box::new(
            cohort_copy::gather_feedback_rows::GatherFeedbackRows::new(
                settings(32),
                p(),
                t(),
                a(),
                a(),
                a(),
                a(),
                a(),
                a(),
                a(),
                width_words,
            ),
        ))?;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resident_program::{StateLayout, StateRequirement};
    use cubecl::ir::{DeviceIdentity, HardwareProperties, MemoryDeviceProperties, VectorSize};
    use cubecl_common::profile::TimingMethod;
    use cubecl_runtime::{id::KernelId, kernel::KernelMetadata};
    use zerogravity_core::accelerator::program::{ConvDomain, StateSlot};

    fn requirement(layout: StateLayout, bytes: usize) -> StateRequirement {
        StateRequirement {
            slot: StateSlot::Conv(ConvDomain(0)),
            layout,
            bytes,
        }
    }
    /// Stand-in device only; `KernelId` carries no property or target data, so
    /// the generated key equality below is exact without a client or device.
    fn properties() -> (Arc<DeviceProperties>, Arc<TargetProperties>) {
        (
            Arc::new(DeviceProperties::new(
                cubecl::features::Features::default(),
                MemoryDeviceProperties::new(1024, 32),
                HardwareProperties {
                    load_width: 32,
                    vector_register_count: None,
                    plane_size_min: 32,
                    plane_size_max: 32,
                    max_bindings: u32::MAX,
                    max_shared_memory_size: 32 * 1024,
                    max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
                    max_units_per_cube: 1024,
                    max_cube_dim: (1024, 1024, 64),
                    num_streaming_multiprocessors: None,
                    num_cpu_cores: None,
                    last_level_cache_size: None,
                    num_tensor_cores: None,
                    min_tensor_cores_dim: None,
                    max_vector_size: VectorSize::MAX,
                    cube_mma_reserved_shared_memory: 0,
                },
                TimingMethod::System,
                DeviceIdentity::default(),
            )),
            Arc::new(TargetProperties::default()),
        )
    }
    fn key(
        properties: Arc<DeviceProperties>,
        target: Arc<TargetProperties>,
        elements: usize,
    ) -> KernelId {
        state_certificate(properties, target, elements).id()
    }
    #[test]
    fn compiler_context_closed_key_formats_fit_admitted_metadata_region() {
        use cubecl_runtime::{id::KernelId, kernel::BufferIOAttr};
        use std::{mem::size_of, sync::Arc};
        let generated = [
            size_of::<clear_step_status::ClearStepStatus>(),
            size_of::<copy_strided_words::CopyStridedWords>(),
            size_of::<cohort_copy::gather_feedback_rows::GatherFeedbackRows>(),
            size_of::<cohort_copy::gather_rows::GatherRows>(),
            size_of::<certify_finite_strided::CertifyFiniteStrided>(),
        ]
        .into_iter()
        .max()
        .unwrap();
        // Four map-entry extents cover capacity/slack/control bytes; generated
        // keys have only bounded scalar/buffer compilation args. Pinned keys
        // retain the separately charged complete source and<=32 ABI slots.
        let entry = size_of::<(KernelId, usize, usize, CubeDim, Option<Arc<[BufferIOAttr]>>)>();
        let required = 4 * (entry + 1)
            + generated
            + 32 * size_of::<(usize, BufferIOAttr)>()
            + crate::vllm_ptx_inline::PinnedKernel::KEY_METADATA_BYTES
            + 256
            + size_of::<[usize; 8]>();
        assert!(
            required <= 4096,
            "closed source/key/map metadata={required}"
        );
        println!(
            "CUDA_READINESS_KEY_METADATA required={required} admitted=4096 generated_inline_max={generated}"
        );
    }

    #[test]
    fn source_requirement_planes_are_vector_or_two_plane_kv_extents() {
        // Current admitted source geometry: one vector extent per delta plane and
        // one `slab*capacity` extent per KV plane, not total bytes / 4.
        let convolution = requirement(StateLayout::Vector { elements: 18_432 }, 18_432 * 4);
        assert_eq!(state_planes(&convolution).unwrap(), (1, 18_432));
        let recurrent = requirement(StateLayout::Vector { elements: 262_144 }, 262_144 * 4);
        assert_eq!(state_planes(&recurrent).unwrap(), (1, 262_144));
        let kv = requirement(
            StateLayout::Kv {
                slab: 512,
                capacity: 512,
            },
            2 * 512 * 512 * 4,
        );
        assert_eq!(state_planes(&kv).unwrap(), (2, 512 * 512));
        // Both KV planes are one extent, so one compile key covers both.
        assert_eq!(
            state_planes(&kv).unwrap().1,
            state_planes(&recurrent).unwrap().1
        );
        // Total KV bytes / 4 is both planes and would prepare a key no launch uses.
        assert_ne!(state_planes(&kv).unwrap().1, kv.bytes / 4);
        let (properties, target) = properties();
        assert_eq!(
            key(
                properties.clone(),
                target.clone(),
                state_planes(&kv).unwrap().1
            ),
            key(properties, target, 262_144)
        );
    }
    #[test]
    fn state_plane_zero_overflow_and_byte_mismatch_are_rejected() {
        let empty = requirement(StateLayout::Vector { elements: 0 }, 0);
        assert!(matches!(state_planes(&empty), Err(Error::Limit(_))));
        let overflow = requirement(
            StateLayout::Kv {
                slab: usize::MAX,
                capacity: 2,
            },
            0,
        );
        assert!(matches!(state_planes(&overflow), Err(Error::Limit(_))));
        let too_wide = requirement(
            StateLayout::Vector {
                elements: usize::MAX,
            },
            usize::MAX,
        );
        assert!(matches!(state_planes(&too_wide), Err(Error::Limit(_))));
        // The kernel's own `rows > MAX_WORD_OPS / width` admission rejects it.
        let beyond_kernel = requirement(
            StateLayout::Vector {
                elements: MAX_WORD_OPS + 1,
            },
            (MAX_WORD_OPS + 1) * 4,
        );
        assert!(matches!(state_planes(&beyond_kernel), Err(Error::Limit(_))));
        // A KV requirement carrying one plane's bytes is not the layout extent.
        let total = requirement(
            StateLayout::Kv {
                slab: 512,
                capacity: 512,
            },
            512 * 512 * 4,
        );
        assert!(matches!(state_planes(&total), Err(Error::State(_))));
    }
    #[test]
    fn state_certificate_key_matches_the_actual_unchecked_launch_arguments() {
        let (properties, target) = properties();
        let buffer = || BufferCompilationArg { inplace: None };
        // Literal reconstruction of `ResidentProgram::certify`: `threads()` is 32,
        // `launch_unchecked` is Unchecked with the attribute's default U32, both
        // read-only `BufferArg`s give `inplace: None`, and the comptime extent is
        // `(rows = 1, width = elements)`.
        let launch = |elements: usize, settings: KernelSettings, buffer: BufferCompilationArg| {
            certify_finite_strided::CertifyFiniteStrided::new(
                settings,
                properties.clone(),
                target.clone(),
                buffer.clone(),
                buffer,
                (),
                (),
                1usize,
                elements,
            )
            .id()
        };
        for elements in [18_432usize, 262_144] {
            let prepared = state_certificate(properties.clone(), target.clone(), elements).id();
            assert_eq!(
                prepared,
                launch(
                    elements,
                    KernelSettings::new(
                        *CubeDim::new_1d(32),
                        ExecutionMode::Unchecked,
                        AddressType::U32,
                    ),
                    buffer(),
                )
            );
            // Every key field is distinguished, so a drift cannot pass unnoticed.
            for other in [
                launch(
                    elements,
                    KernelSettings::new(
                        *CubeDim::new_1d(64),
                        ExecutionMode::Unchecked,
                        AddressType::U32,
                    ),
                    buffer(),
                ),
                launch(
                    elements,
                    KernelSettings::new(
                        *CubeDim::new_1d(32),
                        ExecutionMode::Checked,
                        AddressType::U32,
                    ),
                    buffer(),
                ),
                launch(
                    elements,
                    KernelSettings::new(
                        *CubeDim::new_1d(32),
                        ExecutionMode::Unchecked,
                        AddressType::U64,
                    ),
                    buffer(),
                ),
                launch(elements + 1, state_certificate_settings(), buffer()),
                launch(
                    elements,
                    state_certificate_settings(),
                    BufferCompilationArg { inplace: Some(0) },
                ),
            ] {
                assert_ne!(prepared, other, "extent {elements} key not distinguished");
            }
        }
        assert_ne!(
            key(properties.clone(), target.clone(), 18_432),
            key(properties, target, 262_144)
        );
    }
    #[test]
    fn compiler_context_original_runtime_configuration_is_supported() {
        use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
        let config = CubeClRuntimeConfig::get();
        // This reads the effective original config; no environment toggles,
        // Client/device/model access or persistence-store open.
        assert!(
            !config.compilation.cache,
            "persisted PTX mode requires a separate bounded policy"
        );
        println!("CUDA_READINESS_ORIGINAL_CONFIG persisted_ptx=false");
    }
}
