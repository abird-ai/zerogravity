use super::*;
use cubecl::device::{AmdDevice, CpuDevice, CudaDevice, MetalDevice, WgpuBackend, WgpuDevice};
use std::cell::Cell;
use zerogravity_core::Error;

fn selections() -> [(Device, Provider); 3] {
    [
        (Device::Cuda(CudaDevice::new(0)), Provider::Cuda),
        (Device::Hip(AmdDevice::new(0)), Provider::Hip),
        (
            Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::Vulkan)),
            Provider::Vulkan,
        ),
    ]
}

#[test]
fn explicit_selection_preserves_provider_and_compiled_features() {
    for (device, provider) in selections() {
        assert_eq!(Provider::selected(&device).unwrap(), provider);
        assert_eq!(
            validate_build_provider(&device).is_ok(),
            match provider {
                Provider::Cuda => cfg!(feature = "cuda"),
                Provider::Hip => cfg!(feature = "hip"),
                Provider::Vulkan => cfg!(feature = "vulkan"),
            }
        );
    }
}

#[test]
fn ambiguous_or_other_backends_refuse_before_any_loader() {
    for device in [
        Device::Cpu(CpuDevice),
        Device::Metal(MetalDevice::DefaultDevice),
        Device::Wgpu(WgpuDevice::new(Default::default())),
        Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::Metal)),
        Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::Dx12)),
        Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::Gl)),
        Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::WebGpu)),
    ] {
        assert!(matches!(
            validate_with(
                &device,
                |_| panic!("unexpected provider probe"),
                || panic!("unexpected HIP loader")
            ),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn compiled_out_provider_refuses_before_runtime_probe_or_fallback() {
    for (device, expected) in selections() {
        let error = validate_with(
            &device,
            |provider| {
                assert_eq!(provider, expected);
                false
            },
            || panic!("unlinked provider reached HIP loader"),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)));
        assert!(error.to_string().contains("not compiled"));
    }
}

#[test]
fn missing_hip_libraries_return_error_without_selecting_vulkan() {
    let hip = Device::Hip(AmdDevice::new(0));
    let calls = Cell::new(0);
    let error = validate_with(
        &hip,
        |provider| {
            assert_eq!(provider, Provider::Hip);
            true
        },
        || {
            calls.set(calls.get() + 1);
            false
        },
    )
    .unwrap_err();
    assert_eq!(calls.get(), 1);
    assert!(matches!(error, Error::Unsupported(_)));
    assert!(error.to_string().contains("HIP_PATH/ROCM_PATH"));
    assert_eq!(hip, Device::Hip(AmdDevice::new(0)));
}

#[test]
fn only_linked_hip_queries_hip_libraries() {
    for (device, _) in selections() {
        let calls = Cell::new(0);
        validate_with(
            &device,
            |_| true,
            || {
                calls.set(calls.get() + 1);
                true
            },
        )
        .unwrap();
        assert_eq!(calls.get(), usize::from(matches!(device, Device::Hip(_))));
    }
}

#[cfg(feature = "hip")]
#[test]
fn actual_hip_library_lookup_remains_fallible_without_a_client() {
    let device = Device::Hip(AmdDevice::new(0));
    let result = validate_provider(&device);
    let available = cubecl::hip::is_runtime_available();
    eprintln!("HIP runtime/RTC libraries loadable: {available}");
    assert_eq!(result.is_ok(), available);
    if !available {
        assert_service_refuses(device);
    }
    if let Err(error) = result {
        assert!(matches!(error, Error::Unsupported(_)));
        assert!(error.to_string().contains("HIP runtime/RTC libraries"));
    }
}

fn assert_service_refuses(device: Device) {
    use crate::resident_leases::ResidentService;
    use zerogravity_core::{accelerator::SourceIdentity, resource_arbiter::ResourceOwner};
    let result = ResidentService::new(
        device,
        SourceIdentity::for_new_allocation().unwrap(),
        ResourceOwner::new("cold-backend", "no-client").unwrap(),
        1024 * 1024,
        1024 * 1024,
    );
    assert!(matches!(result, Err(Error::Unsupported(_))));
}

#[test]
fn lower_service_refuses_an_unsupported_provider_before_client_creation() {
    assert_service_refuses(Device::Cpu(CpuDevice));
}
